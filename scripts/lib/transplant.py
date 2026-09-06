#!/usr/bin/env python
"""段A第10步：移植（transplant）。跟第9步"换回"方向相反——从 target 出发，只把某方法
"某一层"的遗忘后权重装进去，其余层保持 target，测这一层单独是否足以复现遗忘（充分性）。
第9步测的是必要性（换回后遗忘是否消失），两者对称，见 constants-v4.yaml
restoration.transplant_damage_definition。

damage = metric(target 自评) − metric(移植后)，逐样本配对 bootstrap，CI 下界 > 0 才算显著。
归一化分母跟第9步 recovery 用同一个（compute_denom，等价于 layer_sweep.load_baselines 的算法）。
显著性用 95% CI、不做 Bonferroni（transplant_significance.alpha=0.05，n=1）——本步每方法每轴
只有 top-1 一层算确认性检验，不是第9步16层同时检验的场景。

top-1 层（forget 轴排名，2026-09-06 实测，见 constants-v4.yaml transplant_layer_selection_axis /
transplant_top1_tie_note）：NPO=layer 8、DPO=layer 15、RMU=layer 1。
另外测两个探索性对照（都不进确认性分母）：
  - layer 7 锚点：NPO/DPO forget 轴均为次高，RMU 理论因果边界层，三方法独立证据都指向它
  - 全16层联合移植：embed_tokens/model.norm 保持 target 权重，跟第9步 exploratory_joint_restore
    镜像对称，作阳性对照——top-1 若测出零效果，用它分辨"这层真不充分"还是"代码没生效"

测量之前必须先过等式测试（transplant_equality_test）：全16层+embed_tokens+model.norm 一起移植
进 target，必须与该方法自己的遗忘后自评逐位相同。这验证的是 transplant 这个新代码路径本身
（build_variant 的 skeleton/source 方向和 §4段A7 两个测试里那一个相反，那两个测试只验证过
RMU 的 Δθ≠0 子集，没覆盖"target 骨架 + 任意层集合"这个更通用的调用方式）。不过 = 硬停。

跑在 open-unlearning/ 目录下：
  python ../scripts/lib/transplant.py NPO
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from restore import load_sd, build_variant, eval_cheap, exact_equal, TMP_ROOT, TARGET_LOCAL_DIR  # noqa: E402
from gate2 import layer_param_names, by_index, bootstrap_ci_lower, N_LAYERS  # noqa: E402

CONFIRMATORY_ALPHA = 0.05  # constants-v4.yaml: restoration.transplant_significance.alpha（n=1，不做 Bonferroni）
TOP1_LAYER = {"NPO": 8, "DPO": 15, "RMU": 1}  # constants-v4.yaml: transplant_layer_selection_axis 实测结果
ANCHOR_LAYER = 7  # constants-v4.yaml: transplant_anchor_layer

METHODS = {
    "NPO": {"ckpt": "saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget10_NPO/checkpoint-30",
            "baseline_task": "opA_tofu_Llama-3.2-1B-Instruct_forget10_NPO_step30"},
    "DPO": {"ckpt": "saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget10_DPO/checkpoint-20",
            "baseline_task": "opA_tofu_Llama-3.2-1B-Instruct_forget10_DPO_step20"},
    "RMU": {"ckpt": "saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget10_RMU/checkpoint-60",
            "baseline_task": "opA_tofu_Llama-3.2-1B-Instruct_forget10_RMU_step60"},
}


def transplant_variant(method_sd, layer_indices, task_name, extra_names=()):
    names = []
    for L in layer_indices:
        names.extend(layer_param_names(method_sd, L))
    names.extend(extra_names)
    out_dir = TMP_ROOT / task_name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    build_variant(TARGET_LOCAL_DIR, method_sd, names, out_dir)
    result = eval_cheap(str(out_dir), task_name)
    shutil.rmtree(out_dir)
    return result


def damage_axis(transplanted_result, target_baseline_values, metric_name, alpha=CONFIRMATORY_ALPHA):
    transplanted_values = by_index(transplanted_result, metric_name)
    diffs = [t - r for t, r in zip(target_baseline_values, transplanted_values)]
    lower, upper, mean_damage = bootstrap_ci_lower(diffs, alpha=alpha)
    return {
        "mean_damage": mean_damage,
        "ci_lower": lower,
        "ci_upper": upper,
        "significant": lower > 0,
        "exact_no_change": all(d == 0.0 for d in diffs),
    }


def normalize_damage(axis_result, denom):
    return {
        **axis_result,
        "normalized_damage": axis_result["mean_damage"] / denom,
        "normalized_ci_lower": axis_result["ci_lower"] / denom,
        "normalized_ci_upper": axis_result["ci_upper"] / denom,
        "denom": denom,
    }


def compute_denom(method_baseline, target_baseline):
    """跟 layer_sweep.load_baselines 完全同一个公式（constants-v4.yaml: transplant_normalization）。"""
    denom_forget = (target_baseline["forget_Q_A_ROUGE"]["agg_value"]
                    - method_baseline["forget_Q_A_ROUGE"]["agg_value"])
    denom_retain = (target_baseline["retain_Q_A_ROUGE"]["agg_value"]
                    - method_baseline["retain_Q_A_ROUGE"]["agg_value"])
    return denom_forget, denom_retain


def run_equality_test(method, target_sd, method_sd, method_baseline):
    all_names = list(target_sd.keys())
    task_name = f"transplant_{method}_equality_full"
    result = transplant_variant(method_sd, [], task_name, extra_names=all_names)
    ok, why = exact_equal(result, method_baseline)
    print(f"  等式测试（全量移植 vs {method} 自身遗忘后自评）: {'PASS' if ok else 'FAIL: ' + why}")
    if not ok:
        print(f"!!! {method} 移植等式测试不过，transplant 代码有 bug，硬停，不许往下走。")
        sys.exit(1)


def run_single_layer_transplant(method_sd, layer_idx, task_name, target_forget, target_retain,
                                 denom_forget, denom_retain, alpha=CONFIRMATORY_ALPHA):
    result = transplant_variant(method_sd, [layer_idx], task_name)
    forget_axis = normalize_damage(
        damage_axis(result, target_forget, "forget_Q_A_ROUGE", alpha=alpha), denom_forget)
    retain_axis = normalize_damage(
        damage_axis(result, target_retain, "retain_Q_A_ROUGE", alpha=alpha), denom_retain)
    return {"forget_axis": forget_axis, "retain_axis": retain_axis}


def run_joint16_transplant(method, method_sd, target_forget, target_retain, denom_forget, denom_retain):
    task_name = f"transplant_{method}_joint16"
    result = transplant_variant(method_sd, range(N_LAYERS), task_name)
    forget_axis = normalize_damage(
        damage_axis(result, target_forget, "forget_Q_A_ROUGE"), denom_forget)
    retain_axis = normalize_damage(
        damage_axis(result, target_retain, "retain_Q_A_ROUGE"), denom_retain)
    return {"forget_axis": forget_axis, "retain_axis": retain_axis}


def print_axes(label, r):
    fa, ra = r["forget_axis"], r["retain_axis"]
    print(f"  {label}  forget: damage={fa['mean_damage']:+.6f} norm={fa['normalized_damage']:+.4f} "
          f"CI=[{fa['ci_lower']:+.6f},{fa['ci_upper']:+.6f}] sig={fa['significant']}")
    print(f"  {label}  retain: damage={ra['mean_damage']:+.6f} norm={ra['normalized_damage']:+.4f} "
          f"CI=[{ra['ci_lower']:+.6f},{ra['ci_upper']:+.6f}] sig={ra['significant']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=list(METHODS))
    args = parser.parse_args()
    method = args.method
    t_start = time.time()

    target_sd = load_sd(TARGET_LOCAL_DIR)
    method_sd = load_sd(METHODS[method]["ckpt"])
    method_baseline = eval_cheap(METHODS[method]["ckpt"], METHODS[method]["baseline_task"])  # 缓存命中，不重跑

    print(f"=== {method} 移植前置：全量移植等式测试 ===")
    run_equality_test(method, target_sd, method_sd, method_baseline)

    target_baseline = json.load(open("saves/eval/target_cheap_baseline/TOFU_EVAL.json"))
    target_forget = by_index(target_baseline, "forget_Q_A_ROUGE")
    target_retain = by_index(target_baseline, "retain_Q_A_ROUGE")
    denom_forget, denom_retain = compute_denom(method_baseline, target_baseline)
    print(f"denom_forget={denom_forget:.6f}  denom_retain={denom_retain:.6f}")

    top1 = TOP1_LAYER[method]
    print(f"\n=== {method} top-1层（layer {top1}）—— 确认性检验，95% CI，不做 Bonferroni ===")
    top1_result = run_single_layer_transplant(
        method_sd, top1, f"transplant_{method}_layer{top1}",
        target_forget, target_retain, denom_forget, denom_retain)
    print_axes(f"layer {top1}", top1_result)

    print(f"\n=== {method} layer {ANCHOR_LAYER} 锚点（探索性，不进确认性分母）===")
    if top1 == ANCHOR_LAYER:
        print(f"  layer {ANCHOR_LAYER} 就是本方法的 top-1，跳过重复测试，复用上面的结果")
        anchor_result = top1_result
    else:
        anchor_result = run_single_layer_transplant(
            method_sd, ANCHOR_LAYER, f"transplant_{method}_layer{ANCHOR_LAYER}",
            target_forget, target_retain, denom_forget, denom_retain)
        print_axes(f"layer {ANCHOR_LAYER}", anchor_result)

    print(f"\n=== {method} 全16层联合移植（阳性对照/探索性，embed_tokens+model.norm 保持 target）===")
    joint_result = run_joint16_transplant(method, method_sd, target_forget, target_retain,
                                          denom_forget, denom_retain)
    print_axes("joint16", joint_result)

    out = {
        "method": method,
        "top1_layer": top1,
        "anchor_layer": ANCHOR_LAYER,
        "top1_result_confirmatory": top1_result,
        "anchor_result_exploratory": anchor_result,
        "joint16_result_exploratory": joint_result,
        "denom": {"forget": denom_forget, "retain": denom_retain},
    }
    out_path = Path(f"../results/transplant_{method}.json")
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"\n落盘：{out_path}")
    print(f"总耗时：{(time.time() - t_start) / 60:.1f} 分钟")


if __name__ == "__main__":
    main()
