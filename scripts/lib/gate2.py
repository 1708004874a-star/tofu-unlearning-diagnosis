#!/usr/bin/env python
"""段A第8步：段C Gate 2 —— RMU 阳性对照，四条判据（§4段C）。

依赖 restore.py 里已经过等式测试验证的 load_sd / build_variant / eval_cheap。
16 层各自单独"换回"（该层换成 target 权重，其余保持 RMU），cheap 评测（forget_Q_A_ROUGE +
extraction_strength + retain_Q_A_ROUGE），forget/retain 两个轴各自做一次逐样本配对恢复量的
bootstrap 显著性检验（§3）。

四条判据（只看 forget 轴——这是 Gate 2 本来要验证的"管道能否正确定位 RMU 已知因果层"）：
  1. 8-15 层 Δθ 严格为 0（逐元素比 state_dict，不用评测）
  2. 这些层的 restoration 恢复量严格为 0（换回等于没换）
  3. significant 的层全部落在 0-7 层内
  4. 0-7 层里至少 1 层 significant
前 3 条任一不过 = 真停机。第 4 条不过 = 走累积换回分支，不停机。

retain 轴（2026-09-05 补）不参与判据，只是给 RMU 补一份 catastrophic 轴数据，
跟 NPO/DPO 那边"哪层拿回被搞坏的 utility"的分析口径一致、可比（model_utility 没有
value_by_index，套不进配对 bootstrap，retain_Q_A_ROUGE 结构一样、已实测确认可以代替）。

跑在 open-unlearning/ 目录下：python ../scripts/lib/gate2.py
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from restore import load_sd, build_variant, eval_cheap, TMP_ROOT, TARGET_LOCAL_DIR  # noqa: E402

RMU_CKPT = "saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget10_RMU/checkpoint-60"
N_LAYERS = 16
BONFERRONI_N = 16
ALPHA = 0.05 / BONFERRONI_N  # family-wise 0.05，16 次检验
N_RESAMPLES = 20000  # constants-v4.yaml: restoration.significance.bootstrap_n（之前写成10000是我的错，未核对原文）
SEED = 0


def layer_param_names(sd, layer_idx):
    prefix = f"model.layers.{layer_idx}."
    return [k for k in sd if k.startswith(prefix)]


def layer_norm_delta(sd_target, sd_rmu, layer_idx):
    names = layer_param_names(sd_target, layer_idx)
    total = 0.0
    for k in names:
        d = (sd_target[k].float() - sd_rmu[k].float())
        total += (d * d).sum().item()
    return total ** 0.5


def by_index(eval_result, metric_name, field="rougeL_recall"):
    """value_by_index 是 dict[str(i)] -> {rouge1_recall, rougeL_f1, rougeL_recall, ...}；
    agg_value 实测确认就是 rougeL_recall 的均值，恢复量的逐样本比较要用同一个字段。"""
    v = eval_result[metric_name]["value_by_index"]
    return [v[str(i)][field] for i in range(len(v))]


def bootstrap_axis(restored_result, baseline_values, metric_name):
    """跑一个轴（forget_Q_A_ROUGE 或 retain_Q_A_ROUGE）的配对恢复量 bootstrap，返回结果 dict。"""
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


def bootstrap_ci_lower(diffs, n_resamples=N_RESAMPLES, alpha=ALPHA, seed=SEED):
    diffs = np.asarray(diffs, dtype=np.float64)
    n = len(diffs)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = diffs[idx].mean(axis=1)
    lower = float(np.quantile(means, alpha / 2))
    upper = float(np.quantile(means, 1 - alpha / 2))
    return lower, upper, float(diffs.mean())


def restore_one_layer(target_sd, rmu_sd, layer_idx, task_name):
    names = layer_param_names(target_sd, layer_idx)
    out_dir = TMP_ROOT / f"gate2_layer{layer_idx}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    build_variant(RMU_CKPT, target_sd, names, out_dir)
    result = eval_cheap(str(out_dir), task_name)
    shutil.rmtree(out_dir)
    return result


def main():
    target_sd = load_sd(TARGET_LOCAL_DIR)
    rmu_sd = load_sd(RMU_CKPT)

    # RMU checkpoint-60（A 点）还在盘上，cheap_v4 新加的 retain_Q_A_ROUGE 用同一个 output_dir
    # 补跑一次——overwrite=false 的缓存机制会跳过已算过的 forget_Q_A_ROUGE/extraction_strength，
    # 只补算新指标（§2.1），不用重新扫层。
    print("=== 补跑 RMU baseline 的 retain_Q_A_ROUGE（复用已有 output_dir，只补新指标）===")
    rmu_baseline = eval_cheap(
        RMU_CKPT, "opA_tofu_Llama-3.2-1B-Instruct_forget10_RMU_step60"
    )
    rmu_forget_rouge = by_index(rmu_baseline, "forget_Q_A_ROUGE")
    rmu_retain_rouge = by_index(rmu_baseline, "retain_Q_A_ROUGE")

    # 判据 1：8-15 层 Δθ 严格为 0（逐元素，不用评测）
    print("=== 判据1：8-15层 Δθ 严格为0（逐元素比对，不评测）===")
    crit1_ok = True
    layer_norms = {}
    for L in range(N_LAYERS):
        names = layer_param_names(target_sd, L)
        maxdiff = max(
            (target_sd[k].float() - rmu_sd[k].float()).abs().max().item()
            for k in names
        )
        layer_norms[L] = layer_norm_delta(target_sd, rmu_sd, L)
        is_zero = maxdiff == 0.0
        if L >= 8 and not is_zero:
            crit1_ok = False
        print(f"  layer {L:2d}: max_abs_diff={maxdiff:.8f}  ||Δθ||={layer_norms[L]:.6f}  "
              f"{'(应为0)' if L >= 8 else ''}")
    embed_diff = (target_sd["model.embed_tokens.weight"].float()
                  - rmu_sd["model.embed_tokens.weight"].float()).abs().max().item()
    print(f"  embed_tokens: max_abs_diff={embed_diff:.8f}")
    print(f"判据1: {'PASS' if crit1_ok else 'FAIL'}")
    if not crit1_ok:
        print("!!! 判据1不过，真停机（§5第2条）。不往下跑。")
        sys.exit(1)

    # 逐层 restoration + 两个轴的恢复量（forget 轴决定 Gate 2 判据；catastrophic/retain 轴
    # 只为了让阳性对照跟 NPO/DPO 有同一套可比数据，不参与本次 Gate 2 的通过/不通过判定）
    print()
    print("=== 逐层 restoration，forget_Q_A_ROUGE + retain_Q_A_ROUGE 各做一次配对恢复量 bootstrap ===")
    layer_results = {}
    for L in range(N_LAYERS):
        task_name = f"gate2_rmu_layer{L}"
        result = restore_one_layer(target_sd, rmu_sd, L, task_name)
        forget_axis = bootstrap_axis(result, rmu_forget_rouge, "forget_Q_A_ROUGE")
        retain_axis = bootstrap_axis(result, rmu_retain_rouge, "retain_Q_A_ROUGE")
        layer_results[L] = {
            "forget_axis": forget_axis,
            "retain_axis": retain_axis,
            "norm_delta": layer_norms[L],
        }
        print(f"  layer {L:2d}  forget: recovery={forget_axis['mean_recovery']:+.6f} "
              f"CI=[{forget_axis['ci_lower']:+.6f},{forget_axis['ci_upper']:+.6f}] "
              f"sig={forget_axis['significant']}  |  "
              f"retain: recovery={retain_axis['mean_recovery']:+.6f} "
              f"CI=[{retain_axis['ci_lower']:+.6f},{retain_axis['ci_upper']:+.6f}] "
              f"sig={retain_axis['significant']}  ||Δθ||={layer_norms[L]:.6f}")

    out_path = Path("../results/gate2_rmu_layer_sweep.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(layer_results, open(out_path, "w"), indent=2)
    print(f"\n落盘：{out_path}")

    # 判据2：8-15层 restoration 恢复量严格为0（换回等于没换）—— 判据本身只看 forget 轴
    print()
    print("=== 判据2：8-15层 restoration 恢复量严格为0（forget 轴）===")
    crit2_ok = all(layer_results[L]["forget_axis"]["exact_no_change"] for L in range(8, 16))
    for L in range(8, 16):
        print(f"  layer {L}: exact_no_change={layer_results[L]['forget_axis']['exact_no_change']}")
    print(f"判据2: {'PASS' if crit2_ok else 'FAIL'}")

    # 顺带的确定性检查：8次restoration是否逐位相同（对RMU baseline，也就是彼此之间）
    print()
    print("=== 顺带检查：8-15层的8次restoration是否逐位相同（评测确定性）===")
    determinism_ok = crit2_ok  # exact_no_change 就是"跟 RMU baseline 逐位相同"
    print(f"determinism_ok(8次结果与RMU baseline逐位相同): {determinism_ok}")
    if not determinism_ok:
        print("!!! 8-15层理论Δθ=0但restoration后结果不同 = 评测本身不确定，"
              "影响所有后续比较，立刻停机上报（§4段C'顺带的确定性检查'）。")

    # 判据3、4（forget 轴）
    print()
    print("=== 判据3/4：significant层的分布（forget 轴）===")
    sig_layers = [L for L in range(N_LAYERS) if layer_results[L]["forget_axis"]["significant"]]
    print(f"significant层: {sig_layers}")
    retain_sig_layers = [L for L in range(N_LAYERS) if layer_results[L]["retain_axis"]["significant"]]
    print(f"（附：retain 轴 significant层，不参与判据: {retain_sig_layers}）")
    crit3_ok = all(L <= 7 for L in sig_layers)
    crit4_ok = any(L <= 7 for L in sig_layers)
    print(f"判据3（significant全部落在0-7）: {'PASS' if crit3_ok else 'FAIL'}")
    print(f"判据4（0-7至少1层significant）: {'PASS' if crit4_ok else 'FAIL（走累积换回分支，不停机）'}")

    print()
    hard_stop = not (crit1_ok and crit2_ok and crit3_ok and determinism_ok)
    if hard_stop:
        print("!!! Gate 2 前3条（含确定性检查）任一不过，真停机，等指示（§5第2条）。")
        sys.exit(1)
    if not crit4_ok:
        print(">>> Gate 2 判据4不过：0-7层没有单层显著，按§4段C分支走累积换回"
              "（先0-7全换回，若显著则二分0-3/4-7，再二分），不停机，report后继续。")
    else:
        print(">>> Gate 2 通过（判据1-4全过）。")


if __name__ == "__main__":
    main()
