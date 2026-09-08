#!/usr/bin/env python
"""段A第10步·二级分析：移植的全16层扫描。见 constants-v4.yaml
restoration.transplant_full_sweep_significance。

起因：transplant.py 只测了每方法的 top-1 层（forget 轴必要性排名第一）+ layer 7 锚点，
NPO 的实测结果里 layer 7 的充分性(norm 15.77%)反而比 top-1 layer 8(norm 8.22%)更高——
必要性排名不能预测充分性排名，只抽查2层不够，得16层全测才知道真正的充分性最大值在哪层。

这是二级分析，跟 transplant.py 里 top-1 的确认性检验（95% CI，n=1，不做 Bonferroni）
两套并存、互不覆盖：
  - transplant.py 的结果（results/transplant_{method}.json）——原样保留，不重跑不改写
  - 本脚本的结果（results/transplant_full_sweep_{method}.json）——16层同时检验，
    Bonferroni alpha=0.05/16，CI=0.996875，跟第9步换回扫描同一套统计处理

RMU 的 8-15 层理论 Δθ=0、移植应为精确 no-op，但仍然实测、不 assert 跳过——transplant 是
新代码路径，实测才能验证层号有没有搞错、有没有读到上一个变体的缓存结果，这正是 Gate 2
当初设计"8-15层 restoration 恢复量严格为0"要抓的问题，同一类检查要在新代码路径上重做一次。

joint16 联合移植不重跑：跟 transplant.py 那次是同一个 task_name，output_dir 缓存机制下重跑
只会拿到同一个结果，直接读已经落盘的 results/transplant_{method}.json 里的值。

跑在 open-unlearning/ 目录下：
  python /root/scripts/lib/transplant_full_sweep.py NPO
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from restore import load_sd, eval_cheap, TARGET_LOCAL_DIR  # noqa: E402
from gate2 import by_index, N_LAYERS  # noqa: E402
from transplant import (  # noqa: E402
    METHODS, run_equality_test, run_single_layer_transplant, compute_denom, print_axes,
)

BONFERRONI_ALPHA = 0.05 / N_LAYERS  # constants-v4.yaml: transplant_full_sweep_significance.alpha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=list(METHODS))
    args = parser.parse_args()
    method = args.method
    t_start = time.time()

    target_sd = load_sd(TARGET_LOCAL_DIR)
    method_sd = load_sd(METHODS[method]["ckpt"])
    method_baseline = eval_cheap(METHODS[method]["ckpt"], METHODS[method]["baseline_task"])  # 缓存命中

    print(f"=== {method} 移植前置：全量移植等式测试（新脚本，独立跑一次，不假定复用之前的结果）===")
    run_equality_test(method, target_sd, method_sd, method_baseline)

    target_baseline = json.load(open("saves/eval/target_cheap_baseline/TOFU_EVAL.json"))
    target_forget = by_index(target_baseline, "forget_Q_A_ROUGE")
    target_retain = by_index(target_baseline, "retain_Q_A_ROUGE")
    denom_forget, denom_retain = compute_denom(method_baseline, target_baseline)
    print(f"denom_forget={denom_forget:.6f}  denom_retain={denom_retain:.6f}")

    print(f"\n=== {method} 全16层移植扫描（二级分析，Bonferroni alpha={BONFERRONI_ALPHA:.6f}，"
          f"CI=0.996875）——RMU 的 8-15 层同样实测，不 assert 跳过 ===")
    layer_results = {}
    for L in range(N_LAYERS):
        r = run_single_layer_transplant(
            method_sd, L, f"transplant_{method}_fullsweep_layer{L}",
            target_forget, target_retain, denom_forget, denom_retain,
            alpha=BONFERRONI_ALPHA)
        layer_results[L] = r
        print_axes(f"layer {L:2d}", r)

    existing = json.load(open(f"../results/transplant_{method}.json"))
    joint_result = existing["joint16_result_exploratory"]
    print(f"\n=== {method} 全16层联合移植（复用 transplant.py 第一轮已落盘的结果，未重跑）===")
    print_axes("joint16", joint_result)

    out = {
        "method": method,
        "analysis_level": "secondary（二级分析，Bonferroni，跟第9步换回扫描同一套统计处理）",
        "significance_note": (
            "alpha=0.05/16=0.003125，CI=0.996875，16层同时检验——不同于 "
            "transplant_{method}.json 里 top1_result_confirmatory 的 95% CI / n=1"
            "（那是预先按必要性排名选定单一层的确认性检验，本文件不覆盖、不改写那份结果）"
        ),
        "layer_results": layer_results,
        "joint16_result_reused_from_transplant_py": joint_result,
        "denom": {"forget": denom_forget, "retain": denom_retain},
    }
    out_path = Path(f"../results/transplant_full_sweep_{method}.json")
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"\n落盘：{out_path}")
    print(f"总耗时：{(time.time() - t_start) / 60:.1f} 分钟")


if __name__ == "__main__":
    main()
