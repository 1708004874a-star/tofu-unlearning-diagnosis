#!/usr/bin/env python
"""段A第5-6步编排：训5个方法，每个训完立刻定操作点A、（NPO 另做 Gate 1b）、清理，再训下一个。
全程无人值守——执行清单_v4.md §5 第4条本来要求删除前停机问人，这里用"from_pretrained 完整性校验
通过才删"替代人工确认（2026-09-04 用户明确批准的一次性例外，见对话）。
跑在 open-unlearning/ 目录下：nohup python ../scripts/lib/run_pipeline.py &
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

CONFIG_DIR = "/root/scripts/configs"
TARGET_REPO = "open-unlearning/tofu_Llama-3.2-1B-Instruct_full"
RETAIN_LOGS = "saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json"
GATE1A_UTILITY_BASELINE = 0.599397  # Gate 1a 自评的 model_utility，操作点可行性判定要用这个不用下载日志的 0.5992
THRESHOLD = 0.396407
SAVE_STEPS = 10
GATE1B_REF = {"model_utility": 0.46, "forget_truth_ratio": 0.7}  # docs/repro.md 1B/forget10/NPO 行
GATE1B_HARD_STOP_DELTA = 0.15
UTILITY_FLOOR_PRIMARY = 0.80
UTILITY_FLOOR_CHECK = 0.90
DISK_STOP_FREE_GB = 8  # workspace 是共享网络卷，判断爆盘只能看 du 实占，不能看 df 的 Avail

TRAINER_MAP = {
    "NPO": ("NPO", "unlearn/tofu/default.yaml"),
    "RMU": ("RMU", "unlearn/tofu/default.yaml"),
    "DPO": ("DPO", "unlearn/tofu/idk.yaml"),
    "GA": ("GradAscent", "unlearn/tofu/default.yaml"),
    "GD": ("GradDiff", "unlearn/tofu/default.yaml"),
}
METHOD_ORDER = ["NPO", "RMU", "DPO", "GA", "GD"]

STATE_FILE = Path("/root/pipeline_state.json")
LOG_PATH = Path("/root/pipeline.log")
LOG = open(LOG_PATH, "a", buffering=1)


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.write(line + "\n")


def halt(msg):
    log(f"!!! HALT: {msg}")
    LOG.write("PIPELINE_HALTED\n")
    LOG.flush()
    sys.exit(1)


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def workspace_used_gb():
    out = subprocess.run(
        ["du", "-s", "--block-size=1G", "/workspace"], capture_output=True, text=True
    )
    return int(out.stdout.split()[0])


def check_disk():
    used = workspace_used_gb()
    free = 85 - used  # 85G 配额；实占用 du，不用 df（网络卷 df 的 Avail 是整个机房的量，见 §)
    log(f"disk: workspace 实占 {used}G / 85G，估算剩余 {free}G")
    if free < DISK_STOP_FREE_GB:
        halt(f"磁盘剩余估算 {free}G < {DISK_STOP_FREE_GB}G 阈值")


def run(cmd, log_file):
    log(f"RUN ({log_file}): {' '.join(cmd)}")
    with open(log_file, "w") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return p.returncode


def train_method(method_key):
    trainer_cfg, experiment = TRAINER_MAP[method_key]
    task_name = f"tofu_Llama-3.2-1B-Instruct_forget10_{trainer_cfg}"
    base_cmd = [
        "python", "src/train.py", "--config-name=unlearn.yaml",
        f"experiment={experiment}", f"trainer={trainer_cfg}",
        f"task_name={task_name}", "model=Llama-3.2-1B-Instruct",
        "forget_split=forget10", "retain_split=retain90",
        f"model.model_args.pretrained_model_name_or_path={TARGET_REPO}",
        f"model.tokenizer_args.pretrained_model_name_or_path={TARGET_REPO}",
        f"retain_logs_path={RETAIN_LOGS}",
        "trainer.args.gradient_checkpointing=true",
        "trainer.args.save_strategy=steps",
        # +：finetune.yaml 里根本没有 save_steps 这个键（库默认 save_strategy='no'，从不需要它），
        # OmegaConf struct 模式下覆写不存在的键必须用 + 显式新增，直接赋值会报 "not in struct"
        f"+trainer.args.save_steps={SAVE_STEPS}",
        "trainer.args.eval_strategy=no",
        "trainer.args.eval_on_start=false",
        # do_eval 是独立开关（finetune.yaml 硬编码 True），不受 eval_strategy 控制——
        # train.py 训完会无条件跑一次完整 TOFU 评测，跟本管道自己的 eval.py 子进程评测
        # （cheap_v4/full_v4/utility_v4，写到 saves/eval/ 下）完全无关、纯属多余，
        # 而且这次评测用库默认的 per_device_eval_batch_size=16 且撞上我们的 bf16
        # upcast patch，在训练刚结束、显存还没释放时必炸（2026-09-05 NPO 撞见，见 limitations.md）
        "trainer.args.do_eval=false",
    ]
    for pd, ga, label in [(8, 4, "8x4"), (4, 8, "4x8_fallback")]:
        cmd = base_cmd + [
            f"trainer.args.per_device_train_batch_size={pd}",
            f"trainer.args.gradient_accumulation_steps={ga}",
        ]
        log_file = f"/root/train_{task_name}_{label}.log"
        t0 = time.time()
        rc = run(cmd, log_file)
        dt = time.time() - t0
        if rc == 0:
            log(f"{method_key}: 训练完成，{dt/60:.1f} 分钟，batch={pd}x{ga}")
            if label == "4x8_fallback":
                log(f"{method_key}: per_device=8 OOM，已退到 4x8 —— 不停机，按§5末尾上报")
            return task_name
        text = open(log_file, errors="replace").read()
        if "CUDA out of memory" in text or "OutOfMemoryError" in text:
            log(f"{method_key}: per_device=8 OOM（{dt/60:.1f}分钟后），清掉半成品重试 4x8")
            out_dir = Path(f"saves/unlearn/{task_name}")
            if out_dir.exists():
                shutil.rmtree(out_dir)  # 避免两次不同 batch size 的产物混进同一个 output_dir（§2.15）
            continue
        halt(f"{method_key} 训练失败（非 OOM，退出码 {rc}），见 {log_file}")
    halt(f"{method_key} 训练在 4x8 下仍失败，见日志")


def eval_one(model_path, eval_cfg, out_task):
    cmd = [
        "python", "src/eval.py", f"--config-dir={CONFIG_DIR}",
        "experiment=eval/tofu/default.yaml", f"eval={eval_cfg}",
        "model=Llama-3.2-1B-Instruct",
        f"model.model_args.pretrained_model_name_or_path={model_path}",
        f"model.tokenizer_args.pretrained_model_name_or_path={TARGET_REPO}",
        f"task_name={out_task}",
        f"retain_logs_path={RETAIN_LOGS}",
    ]
    log_file = f"/root/eval_{out_task}.log"
    rc = run(cmd, log_file)
    if rc != 0:
        halt(f"评测失败：{out_task}，见 {log_file}")
    return json.load(open(f"saves/eval/{out_task}/TOFU_EVAL.json"))


def find_checkpoints(task_name):
    d = Path(f"saves/unlearn/{task_name}")
    return sorted(
        int(p.name.split("-")[1]) for p in d.glob("checkpoint-*") if p.is_dir()
    )


def operating_point_sweep(task_name):
    ckpts = find_checkpoints(task_name)
    log(f"{task_name}: 发现 {len(ckpts)} 个 ckpt: {ckpts}")
    trajectory = []
    A = None  # step 数字，或 "final"，或 None（never_crossed）
    for step in ckpts:
        model_path = f"saves/unlearn/{task_name}/checkpoint-{step}"
        out_task = f"opA_{task_name}_step{step}"
        d = eval_one(model_path, "cheap_v4", out_task)
        rouge = d["forget_Q_A_ROUGE"]["agg_value"]
        trajectory.append({"step": step, "forget_Q_A_ROUGE": rouge})
        log(f"  step {step}: forget_Q_A_ROUGE={rouge:.6f} (阈值 {THRESHOLD})")
        if A is None and rouge <= THRESHOLD:
            A = step
            log(f"  -> 首个越线点：step {step}")
    final_task = f"opA_{task_name}_final"
    d_final = eval_one(f"saves/unlearn/{task_name}", "cheap_v4", final_task)
    rouge_final = d_final["forget_Q_A_ROUGE"]["agg_value"]
    trajectory.append({"step": "final", "forget_Q_A_ROUGE": rouge_final})
    log(f"  final: forget_Q_A_ROUGE={rouge_final:.6f}")
    if A is None and rouge_final <= THRESHOLD:
        A = "final"
        log("  -> 首个越线点：final")
    return trajectory, A, ckpts


def model_path_for(task_name, step):
    if step == "final":
        return f"saves/unlearn/{task_name}"
    return f"saves/unlearn/{task_name}/checkpoint-{step}"


def utility_at(task_name, step, label):
    out_task = f"util_{task_name}_{label}"
    d = eval_one(model_path_for(task_name, step), "utility_v4", out_task)
    u = d["model_utility"]["agg_value"]
    log(f"  utility[{label}] (step={step}): model_utility={u:.6f}")
    return u


def gate_1b(task_name):
    d = eval_one(f"saves/unlearn/{task_name}", "full_v4", f"gate1b_{task_name}_final")
    ours = {
        "model_utility": d["model_utility"]["agg_value"],
        "forget_truth_ratio": d["forget_truth_ratio"]["agg_value"],
    }
    log(f"Gate 1b: ours={ours} ref={GATE1B_REF} (repro.md 1B/forget10/NPO)")
    for k in GATE1B_REF:
        diff = abs(ours[k] - GATE1B_REF[k])
        log(f"  {k}: |diff|={diff:.4f} (hard_stop_delta={GATE1B_HARD_STOP_DELTA})")
        if diff > GATE1B_HARD_STOP_DELTA:
            halt(f"Gate 1b 不过：{k} 差 {diff:.4f} > {GATE1B_HARD_STOP_DELTA}")
    log("Gate 1b: 通过（在 hard_stop_delta 以内）")
    return ours


def verify_loadable(path):
    """删之前的完整性校验：把要留的 ckpt 完整加载一遍，代替人工确认（2026-09-04 例外）。"""
    code = (
        "from transformers import AutoModelForCausalLM; "
        f"AutoModelForCausalLM.from_pretrained('{path}'); print('OK')"
    )
    p = subprocess.run(["python", "-c", code], capture_output=True, text=True)
    ok = p.returncode == 0 and "OK" in p.stdout
    if not ok:
        log(f"完整性校验失败：{path}\n{p.stdout}\n{p.stderr}")
    return ok


def cleanup(task_name, A):
    d = Path(f"saves/unlearn/{task_name}")
    if A is not None:
        keep_path = model_path_for(task_name, A)
        if not verify_loadable(keep_path):
            halt(f"{task_name}: 操作点 A（step={A}）完整性校验没过，不删任何东西，人工检查")
        log(f"{task_name}: A（step={A}）完整性校验通过")
    else:
        log(f"{task_name}: 无操作点 A（never_crossed），全部清理")

    for ckpt_dir in list(d.glob("checkpoint-*")):  # 先物化成 list，边删边 glob 有风险
        if A != "final" and str(A) == ckpt_dir.name.split("-")[1]:
            continue  # 这是 A 本身，保留
        log(f"  删除 {ckpt_dir}")
        shutil.rmtree(ckpt_dir)

    if A != "final":
        # 顶层最终权重文件（不在 checkpoint-*/ 下的那份）：A 不是 final 时也要删
        for pat in ("*.safetensors", "*.safetensors.index.json", "pytorch_model*.bin"):
            for f in d.glob(pat):
                log(f"  删除 {f}")
                f.unlink()
    check_disk()


def run_method(method_key):
    log(f"===== {method_key} 开始 =====")
    task_name = train_method(method_key)

    ts = json.load(open(f"saves/unlearn/{task_name}/trainer_state.json"))
    log(f"trainer_state: max_steps={ts.get('max_steps')} epoch={ts.get('epoch')}")

    gate1b_result = None
    if method_key == "NPO":
        gate1b_result = gate_1b(task_name)

    trajectory, A, ckpts = operating_point_sweep(task_name)

    u_first = utility_at(task_name, ckpts[0], "first_ckpt") if ckpts else None
    u_final = utility_at(task_name, "final", "final")
    if A is None:
        u_A = None
    elif A == "final":
        u_A = u_final
    else:
        u_A = utility_at(task_name, A, "A")

    feasible = A is not None and u_A is not None and u_A >= GATE1A_UTILITY_BASELINE * UTILITY_FLOOR_PRIMARY
    feasible_check = A is not None and u_A is not None and u_A >= GATE1A_UTILITY_BASELINE * UTILITY_FLOOR_CHECK
    if A is not None:
        a_rouge = next(t["forget_Q_A_ROUGE"] for t in trajectory if t["step"] == A)
        overshoot = THRESHOLD - a_rouge
    else:
        overshoot = None

    log(f"{method_key}: A={A} feasible={feasible} feasible_check={feasible_check} overshoot={overshoot}")
    if not feasible:
        log(f"{method_key}: 判定不可行 —— 报告后继续（§5 第3条），不算硬停")

    archive_path = None
    if A is not None:
        eval_one(model_path_for(task_name, A), "full_v4", f"archiveA_{task_name}")
        archive_path = f"saves/eval/archiveA_{task_name}/TOFU_EVAL.json"
        log(f"{method_key}: A 的 full_v4 存档完成 -> {archive_path}")

    result = {
        "method": method_key, "task_name": task_name, "trainer_state": ts,
        "gate_1b": gate1b_result, "trajectory": trajectory, "A": A,
        "utility_first": u_first, "utility_A": u_A, "utility_final": u_final,
        "feasible": feasible, "feasible_check": feasible_check, "overshoot": overshoot,
        "archive_A_path": archive_path,
    }
    Path(f"../results/pipeline/{task_name}.json").parent.mkdir(parents=True, exist_ok=True)
    Path(f"../results/pipeline/{task_name}.json").write_text(json.dumps(result, indent=2, default=str))

    cleanup(task_name, A)
    log(f"===== {method_key} 完成 =====")
    return result


def main():
    check_disk()
    all_results = []
    for m in METHOD_ORDER:
        r = run_method(m)
        all_results.append(r)
        save_state({"completed": [x["method"] for x in all_results], "results": all_results})
    log("===== 全部 5 个方法完成 =====")
    save_state({"completed": [x["method"] for x in all_results], "results": all_results, "done": True})


if __name__ == "__main__":
    main()
