#!/usr/bin/env python
"""段A第9步：NPO、DPO 各扫16层 + 联合换回（探索性，见 constants-v4.yaml
restoration.exploratory_joint_restore）。forget/retain 两轴，含归一化。

复用 gate2.py 已验证的 layer_param_names / layer_norm_delta / by_index / bootstrap_ci_lower。
16 层全测，不按任一轴恢复量挑"前三层"——那会把"压制层和损伤层是不是同一批"预先答掉。

归一化只在聚合层面做（单一标量分母 = agg_value(target) − agg_value(遗忘后)），不逐样本相除
——避免分母接近 0 的题目把归一化炸掉。配对 bootstrap 全程走原始差值，归一化是算完 bootstrap
之后对 mean_recovery/ci_lower/ci_upper 三个数一起除的后处理，不影响显著性判定（除以同一个
正数不改变 CI 是否跨零）。

联合换回：0-15 全部层一起换回 target，embed_tokens/model.norm 保持遗忘后权重，跟 16 个单层
的恢复量一起报，只报数据不判断（探索性，不进主判据、不进跨方法比较，Bonferroni 分母仍锁 16）。

启动断言：save_steps 在 run_pipeline.py 里硬编码、不读 yaml，是两个真相来源；这里读一次
constants.yaml 的 training.save_steps 并跟硬编码值比对，对不上就抛错。

跑在 open-unlearning/ 目录下：
  python ../scripts/lib/layer_sweep.py NPO --layer 0      # 单层计时
  python ../scripts/lib/layer_sweep.py NPO                # 全量：16 层 + 联合换回
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from restore import load_sd, build_variant, eval_cheap, TMP_ROOT, TARGET_LOCAL_DIR  # noqa: E402
from gate2 import (  # noqa: E402
    layer_param_names,
    layer_norm_delta,
    by_index,
    bootstrap_ci_lower,
    N_LAYERS,
)

CONSTANTS_YAML = "/root/scripts/constants.yaml"
EXPECTED_SAVE_STEPS = 10  # run_pipeline.py:SAVE_STEPS 的硬编码值，跟 yaml 对比用（两个真相来源）

METHODS = {
    "NPO": {
        "ckpt": "saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget10_NPO/checkpoint-30",
        "baseline_task": "opA_tofu_Llama-3.2-1B-Instruct_forget10_NPO_step30",
    },
    "DPO": {
        "ckpt": "saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget10_DPO/checkpoint-20",
        "baseline_task": "opA_tofu_Llama-3.2-1B-Instruct_forget10_DPO_step20",
    },
}


def assert_save_steps_consistent():
    from omegaconf import OmegaConf

    c = OmegaConf.load(CONSTANTS_YAML)
    yaml_value = c.training.save_steps
    if yaml_value != EXPECTED_SAVE_STEPS:
        raise RuntimeError(
            f"save_steps 不一致：constants.yaml 写的是 {yaml_value}，"
            f"但 run_pipeline.py 硬编码是 {EXPECTED_SAVE_STEPS}（两个真相来源没对齐，"
            f"检查是不是有一边改了没同步再往下跑）"
        )


def bootstrap_axis(restored_result, baseline_values, metric_name):
    restored_values = by_index(restored_result, metric_name)
    diffs = [r - g for r, g in zip(restored_values, baseline_values)]
    lower, upper, mean_recovery = bootstrap_ci_lower(diffs)
    return {
        "mean_recovery": mean_recovery,
        "ci_lower": lower,
        "ci_upper": upper,
        "significant": lower > 0,
        "exact_no_change": all(d == 0.0 for d in diffs),
    }


def normalize(axis_result, denom):
    return {
        **axis_result,
        "normalized_recovery": axis_result["mean_recovery"] / denom,
        "normalized_ci_lower": axis_result["ci_lower"] / denom,
        "normalized_ci_upper": axis_result["ci_upper"] / denom,
        "denom": denom,
    }


def restore_layers(base_ckpt_dir, target_sd, layer_indices, task_name):
    names = []
    for L in layer_indices:
        names.extend(layer_param_names(target_sd, L))
    out_dir = TMP_ROOT / f"sweep_{task_name}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    build_variant(base_ckpt_dir, target_sd, names, out_dir)
    result = eval_cheap(str(out_dir), task_name)
    shutil.rmtree(out_dir)
    return result


def run_single_layer(method, layer_idx, target_sd, method_sd, forget_baseline, retain_baseline,
                      denom_forget, denom_retain):
    task_name = f"sweep_{method}_layer{layer_idx}"
    result = restore_layers(METHODS[method]["ckpt"], target_sd, [layer_idx], task_name)
    forget_axis = normalize(bootstrap_axis(result, forget_baseline, "forget_Q_A_ROUGE"), denom_forget)
    retain_axis = normalize(bootstrap_axis(result, retain_baseline, "retain_Q_A_ROUGE"), denom_retain)
    norm_delta = layer_norm_delta(target_sd, method_sd, layer_idx)
    return {"forget_axis": forget_axis, "retain_axis": retain_axis, "norm_delta": norm_delta}


def run_joint_restore(method, target_sd, forget_baseline, retain_baseline, denom_forget, denom_retain):
    task_name = f"sweep_{method}_joint_all16"
    result = restore_layers(METHODS[method]["ckpt"], target_sd, range(N_LAYERS), task_name)
    forget_axis = normalize(bootstrap_axis(result, forget_baseline, "forget_Q_A_ROUGE"), denom_forget)
    retain_axis = normalize(bootstrap_axis(result, retain_baseline, "retain_Q_A_ROUGE"), denom_retain)
    return {"forget_axis": forget_axis, "retain_axis": retain_axis}


def load_baselines(method):
    ckpt = METHODS[method]["ckpt"]
    baseline = eval_cheap(ckpt, METHODS[method]["baseline_task"])
    forget_baseline = by_index(baseline, "forget_Q_A_ROUGE")
    retain_baseline = by_index(baseline, "retain_Q_A_ROUGE")
    target_baseline = json.load(open("saves/eval/target_cheap_baseline/TOFU_EVAL.json"))
    denom_forget = (target_baseline["forget_Q_A_ROUGE"]["agg_value"]
                    - baseline["forget_Q_A_ROUGE"]["agg_value"])
    denom_retain = (target_baseline["retain_Q_A_ROUGE"]["agg_value"]
                    - baseline["retain_Q_A_ROUGE"]["agg_value"])
    return forget_baseline, retain_baseline, denom_forget, denom_retain


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=list(METHODS))
    parser.add_argument("--layer", type=int, default=None,
                         help="只测这一层（计时用），不传则跑16层+联合换回")
    args = parser.parse_args()

    assert_save_steps_consistent()

    method = args.method
    target_sd = load_sd(TARGET_LOCAL_DIR)
    method_sd = load_sd(METHODS[method]["ckpt"])
    forget_baseline, retain_baseline, denom_forget, denom_retain = load_baselines(method)
    print(f"denom_forget={denom_forget:.6f}  denom_retain={denom_retain:.6f}")

    if args.layer is not None:
        t0 = time.time()
        r = run_single_layer(method, args.layer, target_sd, method_sd,
                              forget_baseline, retain_baseline, denom_forget, denom_retain)
        dt = time.time() - t0
        print(f"单层（layer {args.layer}）耗时：{dt:.1f} 秒")
        print(json.dumps(r, indent=2))
        return

    t_start = time.time()
    layer_results = {}
    for L in range(N_LAYERS):
        layer_results[L] = run_single_layer(method, L, target_sd, method_sd,
                                             forget_baseline, retain_baseline,
                                             denom_forget, denom_retain)
        fa, ra = layer_results[L]["forget_axis"], layer_results[L]["retain_axis"]
        print(f"  layer {L:2d}  forget: raw={fa['mean_recovery']:+.6f} "
              f"norm={fa['normalized_recovery']:+.4f} sig={fa['significant']}  |  "
              f"retain: raw={ra['mean_recovery']:+.6f} norm={ra['normalized_recovery']:+.4f} "
              f"sig={ra['significant']}  ||Δθ||={layer_results[L]['norm_delta']:.6f}")

    joint = run_joint_restore(method, target_sd, forget_baseline, retain_baseline,
                               denom_forget, denom_retain)
    sum_forget = sum(layer_results[L]["forget_axis"]["mean_recovery"] for L in range(N_LAYERS))
    sum_retain = sum(layer_results[L]["retain_axis"]["mean_recovery"] for L in range(N_LAYERS))
    print()
    print("=== 联合换回 vs 16层单测求和（探索性，不判断）===")
    print(f"16层单测求和   : forget_raw={sum_forget:+.6f} (norm={sum_forget/denom_forget:+.4f})  "
          f"retain_raw={sum_retain:+.6f} (norm={sum_retain/denom_retain:+.4f})")
    print(f"联合换回实测   : forget_raw={joint['forget_axis']['mean_recovery']:+.6f} "
          f"(norm={joint['forget_axis']['normalized_recovery']:+.4f})  "
          f"retain_raw={joint['retain_axis']['mean_recovery']:+.6f} "
          f"(norm={joint['retain_axis']['normalized_recovery']:+.4f})")
    print(f"总耗时：{(time.time()-t_start)/60:.1f} 分钟")

    out = {
        "method": method,
        "layer_results": layer_results,
        "joint_restore_exploratory": joint,
        "sum_of_singles": {"forget": sum_forget, "retain": sum_retain},
        "denom": {"forget": denom_forget, "retain": denom_retain},
    }
    out_path = Path(f"../results/layer_sweep_{method}.json")
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"落盘：{out_path}")


if __name__ == "__main__":
    main()
