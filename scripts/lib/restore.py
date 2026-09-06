#!/usr/bin/env python
"""段A第7步：restore（换回）/ transplant（移植），含两个精确等式测试。

纯 state_dict/张量操作，不实例化模型对象；实际推理评测复用已验证的 src/eval.py 子进程路径
（跟 Gate 1a、run_pipeline.py 走同一条路，不引入新的 Hydra 组合方式）。

跑在 open-unlearning/ 目录下：python ../scripts/lib/restore.py test
临时模型目录按 §4段A9 写在 container disk（/root/tmp_restore），写一份→评→立刻删。
"""
import argparse
import glob
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

CONFIG_DIR = "/root/scripts/configs"
TARGET_REPO = "open-unlearning/tofu_Llama-3.2-1B-Instruct_full"
RETAIN_LOGS = "saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json"
TMP_ROOT = Path("/root/tmp_restore")

TARGET_LOCAL_DIR = "/workspace/.cache/huggingface/models--open-unlearning--tofu_Llama-3.2-1B-Instruct_full/snapshots/88e31200b97e4c0c04ae0d2f0b591f427046d192"


def load_sd(path):
    sd = {}
    for f in sorted(glob.glob(f"{path}/*.safetensors")):
        sd.update(load_file(f))
    return sd


def delta_param_names(sd_base, sd_variant):
    """base 相对 variant，哪些参数 Δθ≠0（逐元素严格比较）。base/variant 的 key 集合必须一致。"""
    if set(sd_base) != set(sd_variant):
        only_a = set(sd_base) - set(sd_variant)
        only_b = set(sd_variant) - set(sd_base)
        raise ValueError(f"key 集合不一致：only_in_base={only_a} only_in_variant={only_b}")
    names = []
    for k in sd_base:
        if sd_base[k].shape != sd_variant[k].shape:
            raise ValueError(f"shape 不一致：{k}")
        if not torch.equal(sd_base[k], sd_variant[k]):
            names.append(k)
    return names


def build_variant(skeleton_dir, source_sd, swap_names, out_dir):
    """以 skeleton_dir 为骨架（config/tokenizer 等非权重文件+权重的默认值），
    把 swap_names 对应的张量换成 source_sd 的值，写到 out_dir。out_dir 不能已存在。"""
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} 已存在，先清理（§4段A9：任一时刻最多一个临时模型）")
    skeleton_sd = load_sd(skeleton_dir)
    if set(skeleton_sd) != set(source_sd):
        raise ValueError("skeleton 与 source 的 key 集合不一致")
    new_sd = dict(skeleton_sd)
    for k in swap_names:
        new_sd[k] = source_sd[k].clone()
    out_dir.mkdir(parents=True)
    save_file(new_sd, str(out_dir / "model.safetensors"), metadata={"format": "pt"})
    for f in Path(skeleton_dir).iterdir():
        if f.is_file() and f.suffix != ".safetensors":
            shutil.copy(f, out_dir / f.name)
    return out_dir


def eval_cheap(model_dir, task_name, log_dir="/root"):
    cmd = [
        "python", "src/eval.py", f"--config-dir={CONFIG_DIR}",
        "experiment=eval/tofu/default.yaml", "eval=cheap_v4",
        "model=Llama-3.2-1B-Instruct",
        f"model.model_args.pretrained_model_name_or_path={model_dir}",
        f"model.tokenizer_args.pretrained_model_name_or_path={TARGET_REPO}",
        f"task_name={task_name}",
        f"retain_logs_path={RETAIN_LOGS}",
    ]
    log_file = f"{log_dir}/restore_{task_name}.log"
    with open(log_file, "w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        raise RuntimeError(f"eval 失败：{task_name}，见 {log_file}")
    return json.load(open(f"saves/eval/{task_name}/TOFU_EVAL.json"))


CHEAP_METRICS = ["forget_Q_A_ROUGE", "extraction_strength", "retain_Q_A_ROUGE"]  # 2026-09-05 补 retain_Q_A_ROUGE


def exact_equal(d1, d2):
    """cheap 的两项指标（含 value_by_index）逐位相同。返回 (bool, 差异说明)。"""
    for m in CHEAP_METRICS:
        s1 = json.dumps(d1[m], sort_keys=True)
        s2 = json.dumps(d2[m], sort_keys=True)
        if s1 != s2:
            return False, f"{m} 不相同"
    return True, None


def run_equality_tests():
    target_sd = load_sd(TARGET_LOCAL_DIR)
    rmu_ckpt_dir = "saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget10_RMU/checkpoint-60"
    rmu_sd = load_sd(rmu_ckpt_dir)

    names = delta_param_names(target_sd, rmu_sd)
    print(f"[restore.py] RMU 相对 target 的 Δθ≠0 参数数：{len(names)}")
    for n in sorted(names):
        print("   ", n)

    target_baseline = json.load(open("saves/eval/target_cheap_baseline/TOFU_EVAL.json"))
    rmu_A_baseline = json.load(
        open("saves/eval/opA_tofu_Llama-3.2-1B-Instruct_forget10_RMU_step60/TOFU_EVAL.json")
    )

    # 测试 1：把 RMU 所有 Δθ≠0 的层换回 target → 应与 target 自评 cheap 逐位相同
    restored_dir = TMP_ROOT / "test1_restored"
    if restored_dir.exists():
        shutil.rmtree(restored_dir)
    build_variant(rmu_ckpt_dir, target_sd, names, restored_dir)
    result1 = eval_cheap(str(restored_dir), "restore_test1_rmu_to_target")
    ok1, why1 = exact_equal(result1, target_baseline)
    shutil.rmtree(restored_dir)
    print(f"[restore.py] 测试1（restore：RMU→target）: {'PASS' if ok1 else 'FAIL: ' + why1}")

    # 测试 2：把 RMU 所有 Δθ≠0 的层移植进 target → 应与 RMU 自评 cheap（A=step60）逐位相同
    transplanted_dir = TMP_ROOT / "test2_transplanted"
    if transplanted_dir.exists():
        shutil.rmtree(transplanted_dir)
    build_variant(TARGET_LOCAL_DIR, rmu_sd, names, transplanted_dir)
    result2 = eval_cheap(str(transplanted_dir), "restore_test2_target_to_rmu")
    ok2, why2 = exact_equal(result2, rmu_A_baseline)
    shutil.rmtree(transplanted_dir)
    print(f"[restore.py] 测试2（transplant：target→RMU）: {'PASS' if ok2 else 'FAIL: ' + why2}")

    if not (ok1 and ok2):
        print("[restore.py] 任一测试不过 = 脚本有 bug，不许往下走（§4段A7）。")
        sys.exit(1)
    print("[restore.py] 两个等式测试全部通过。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["test"])
    args = parser.parse_args()
    if args.mode == "test":
        run_equality_tests()
