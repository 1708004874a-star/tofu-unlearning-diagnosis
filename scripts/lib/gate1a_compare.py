#!/usr/bin/env python
"""Gate 1a 比对（执行清单_v4.md §3 / constants-v4.yaml gate_1a.metric_buckets，2026-09-04 规则版）。
分桶按规则不按枚举：TOFU_EVAL.json 里实际落盘的每一项都要分到三桶之一，没有例外。
跑在 open-unlearning/ 目录下：python ../scripts/lib/gate1a_compare.py
"""
import json

REF = "saves/eval/tofu_Llama-3.2-1B-Instruct_full/evals_forget10/TOFU_EVAL.json"
OURS = "saves/eval/gate1a_target_full/TOFU_EVAL.json"

TOLERANCE_PROB = 0.01
TOLERANCE_ROUGE = 0.02
BORDERLINE_FRAC = 0.20  # 偏差落在容差的 (1-frac)*tol ~ tol 之间 = borderline

# constants-v4.yaml gate_1a.metric_buckets 的规则，逐字对应：
ROUGE_EXPLICIT = {"extraction_strength", "exact_memorization", "model_utility"}
UNRATED_EXPLICIT = {"forget_quality", "privleak"}


def classify(name):
    lname = name.lower()
    if name in ROUGE_EXPLICIT:
        return "tolerance_rouge"
    if name in UNRATED_EXPLICIT or lname.startswith("mia_"):
        return "unrated"
    if "rouge" in lname:
        return "tolerance_rouge"
    if "prob" in lname or "truth_ratio" in lname:
        return "tolerance_prob"
    return "UNMATCHED"


def agg(d, name):
    return d[name]["agg_value"]


def main():
    ref = json.load(open(REF))
    ours = json.load(open(OURS))
    keys = sorted(ours.keys())

    unmatched = [k for k in keys if classify(k) == "UNMATCHED"]
    if unmatched:
        raise SystemExit(
            f"三条规则都套不上的指标，不许自己猜，先问：{unmatched}"
        )

    print(f"{'metric':<28}{'bucket':<16}{'ours':>14}{'official':>14}{'|diff|':>10}  verdict")
    print("-" * 100)

    hard_stop, borderline = [], []
    for name in keys:
        bucket = classify(name)
        o, r = agg(ours, name), agg(ref, name)
        diff = abs(o - r)
        if bucket == "unrated":
            print(f"{name:<28}{bucket:<16}{o:>14.6f}{r:>14.6f}{diff:>10.6f}  记录，不设判据")
            continue
        tol = TOLERANCE_PROB if bucket == "tolerance_prob" else TOLERANCE_ROUGE
        lo = tol * (1 - BORDERLINE_FRAC)
        if diff <= lo:
            verdict = "PASS"
        elif diff <= tol:
            verdict = "BORDERLINE"
            borderline.append(name)
        else:
            verdict = "FAIL (hard stop)"
            hard_stop.append(name)
        print(f"{name:<28}{bucket:<16}{o:>14.6f}{r:>14.6f}{diff:>10.6f}  {verdict} (tol={tol})")

    if borderline or hard_stop:
        print("\n逐样本 diff（借 value_by_index，按差值降序，前 10 条）：")
        for name in set(borderline + hard_stop):
            ov = ours[name].get("value_by_index")
            rv = ref[name].get("value_by_index")
            if not ov or not rv:
                print(f"  [{name}] 没有 value_by_index，跳过逐样本 diff")
                continue
            rows = []
            for idx in ov:
                if idx not in rv:
                    continue
                a, b = ov[idx], rv[idx]
                common_numeric = [
                    k for k in a
                    if k in b and isinstance(a[k], (int, float)) and isinstance(b[k], (int, float))
                ]
                if not common_numeric:
                    continue
                k = max(common_numeric, key=lambda kk: abs(a[kk] - b[kk]))
                rows.append((idx, k, a[k], b[k], abs(a[k] - b[k])))
            rows.sort(key=lambda t: -t[4])
            print(f"  [{name}]")
            for idx, k, av, bv, d in rows[:10]:
                print(f"    idx={idx:<5} field={k:<16} ours={av:.6f} official={bv:.6f} diff={d:.6f}")

    print()
    if hard_stop:
        print(f"判定：HARD STOP —— {hard_stop} 超出容差。按 §5 第1条：停、列产物、等指示。")
    elif borderline:
        print(f"判定：BORDERLINE —— {borderline} 落在容差 80%-100% 之间，不自动放行，需人工定性。")
    else:
        print("判定：Gate 1a 通过（tolerance_prob/tolerance_rouge 两桶内全部 PASS）。")


if __name__ == "__main__":
    main()
