#!/usr/bin/env python
"""段A第11步：出图（4张）。见 执行清单_v4.md §4段A11。

纯读盘 + matplotlib，不碰远端、不跑评测。数据来源：
  results/layer_sweep_NPO.json / layer_sweep_DPO.json   —— 第9步，16层单测（含 norm_delta）
  results/gate2_rmu_layer_sweep.json                     —— 第8步，RMU 16层单测（没有归一化字段，
                                                             这里用 transplant_RMU.json 的 denom 补算）
  results/transplant_{NPO,DPO,RMU}.json                  —— 第10步，top-1/layer7锚点/joint16
  results/transplant_full_sweep_{NPO,DPO,RMU}.json       —— 第10步二级分析，16层全测移植
                                                             （2026-09-06 新增，Bonferroni）
  results/archiveA/{NPO,RMU,DPO}.json                    —— 第6步，A点的 full eval 存档

图②用 retain_Q_A_ROUGE 代 model_utility（2026-09-05 决定，constants-v4.yaml
restoration.significance.metric_for_catastrophic_axis）——model_utility 没有 value_by_index，
retain_Q_A_ROUGE 结构相同、已实测确认可代替，图上标注清楚这个替代关系。

图④只用 forget 轴，48个点（3方法 × 16层）：横轴necessity(recovery，第9步换回扫描)，
纵轴sufficiency(damage，第10步二级分析·全16层移植扫描)，两边都是Bonferroni口径
（CI=0.996875，n_tests=16，见 constants-v4.yaml restoration.significance 和
transplant_full_sweep_significance）——统计上对称、可以画在同一张图上。
**这跟 transplant.py 原来那次 top-1 确认性检验（95% CI、n=1）是两套不同的统计处理，
互不覆盖：top-1 那次的结论原样保留在 results/transplant_{method}.json 里，不在这张图上
重复画出来，避免把两种显著性标准混进同一张图**（2026-09-06 决定）。

跑在本地 Mac 上：python3 scripts/lib/make_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
FIGDIR = RESULTS / "figures"
FIGDIR.mkdir(exist_ok=True)

METHODS = ["NPO", "RMU", "DPO"]
COLOR = {"NPO": "#1f77b4", "RMU": "#2ca02c", "DPO": "#d62728"}
N_LAYERS = 16

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def load(name):
    return json.load(open(RESULTS / name))


sweep = {
    "NPO": load("layer_sweep_NPO.json")["layer_results"],
    "DPO": load("layer_sweep_DPO.json")["layer_results"],
    "RMU": load("gate2_rmu_layer_sweep.json"),
}
transplant = {m: load(f"transplant_{m}.json") for m in METHODS}
transplant_full = {m: load(f"transplant_full_sweep_{m}.json") for m in METHODS}
archiveA = {m: json.load(open(RESULTS / "archiveA" / f"{m}.json")) for m in METHODS}

# RMU 的 sweep 没有归一化字段（gate2.py 没算），用 transplant_RMU.json 里已经算过的同一个 denom 补
RMU_DENOM_FORGET = transplant["RMU"]["denom"]["forget"]
RMU_DENOM_RETAIN = transplant["RMU"]["denom"]["retain"]


def norm_recovery(method, layer, axis):
    """axis: 'forget_axis' or 'retain_axis'，返回该层该轴的归一化 recovery。"""
    d = sweep[method][str(layer)][axis]
    if "normalized_recovery" in d:
        return d["normalized_recovery"]
    denom = RMU_DENOM_FORGET if axis == "forget_axis" else RMU_DENOM_RETAIN
    return d["mean_recovery"] / denom


def raw_delta(method, layer):
    return sweep[method][str(layer)]["norm_delta"]


def sig(method, layer, axis):
    return sweep[method][str(layer)][axis]["significant"]


# ---------------------------------------------------------------------------
# 图① 逐层恢复量(forget轴) vs ||Δθ_layer||，三个方法叠在一起
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
layers = list(range(N_LAYERS))
for m in METHODS:
    rec = [norm_recovery(m, L, "forget_axis") * 100 for L in layers]
    sigs = [sig(m, L, "forget_axis") for L in layers]
    ax1.plot(layers, rec, "-", color=COLOR[m], label=m, linewidth=1.5, zorder=2)
    ax1.scatter([L for L, s in zip(layers, sigs) if s], [r for r, s in zip(rec, sigs) if s],
                color=COLOR[m], marker="o", s=45, zorder=3)
    ax1.scatter([L for L, s in zip(layers, sigs) if not s], [r for r, s in zip(rec, sigs) if not s],
                facecolors="white", edgecolors=COLOR[m], marker="o", s=45, zorder=3)
ax1.axhline(0, color="gray", linewidth=0.7)
ax1.set_ylabel("Normalized Recovery % (forget axis, filled = significant)")
ax1.set_title("Figure 1: Per-Layer Recovery (Necessity) vs ‖Δθ_layer‖ — NPO / RMU / DPO")
ax1.legend(loc="upper right")

for m in METHODS:
    delta = [raw_delta(m, L) for L in layers]
    ax2.plot(layers, delta, "-o", color=COLOR[m], label=m, linewidth=1.5, markersize=4)
ax2.set_xlabel("Layer index (0-15)")
ax2.set_ylabel("‖Δθ_layer‖ (L2 norm, weight space)")
ax2.set_xticks(layers)
ax2.legend(loc="upper right")

fig.tight_layout()
fig.savefig(FIGDIR / "fig1_recovery_vs_delta.png", dpi=150)
plt.close(fig)


# ---------------------------------------------------------------------------
# 图② 换回哪层能修回"utility 损伤"（retain_Q_A_ROUGE 代 model_utility，2026-09-05 决定）
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4.5))
for m in METHODS:
    rec = [norm_recovery(m, L, "retain_axis") * 100 for L in layers]
    sigs = [sig(m, L, "retain_axis") for L in layers]
    ax.plot(layers, rec, "-", color=COLOR[m], label=m, linewidth=1.5, zorder=2)
    ax.scatter([L for L, s in zip(layers, sigs) if s], [r for r, s in zip(rec, sigs) if s],
               color=COLOR[m], marker="o", s=45, zorder=3)
    ax.scatter([L for L, s in zip(layers, sigs) if not s], [r for r, s in zip(rec, sigs) if not s],
               facecolors="white", edgecolors=COLOR[m], marker="o", s=45, zorder=3)
ax.axhline(0, color="gray", linewidth=0.7)
ax.set_xlabel("Layer index (0-15)")
ax.set_ylabel("Normalized Recovery % (retain axis, filled = significant)")
ax.set_xticks(layers)
ax.set_title("Figure 2: Which Layer Restores the Utility Damage\n"
              "(retain_Q_A_ROUGE as proxy for model_utility — the latter has no value_by_index, "
              "descriptive only, see constants-v4.yaml)",
              fontsize=10)
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(FIGDIR / "fig2_retain_recovery.png", dpi=150)
plt.close(fig)


# ---------------------------------------------------------------------------
# 图③ 三个方法在操作点 A 的 ES / MinK++ 对比
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 4.8))
x = range(len(METHODS))
w = 0.35
es = [archiveA[m]["extraction_strength"]["agg_value"] for m in METHODS]
mink = [archiveA[m]["mia_min_k_plus_plus"]["agg_value"] for m in METHODS]
ax.bar([i - w / 2 for i in x], es, width=w, label="extraction_strength", color="#9467bd")
ax.bar([i + w / 2 for i in x], mink, width=w, label="mia_min_k_plus_plus", color="#8c564b")
ax.set_xticks(list(x))
ax.set_xticklabels(METHODS)
ax.set_ylabel("agg_value")
ax.set_ylim(0, 1.12)
ax.set_title("Figure 3: ES / MinK++ Comparison at Operating Point A\n(insufficient-forgetting axis, descriptive only)",
             fontsize=11)
ax.legend()
for i, (e, k) in enumerate(zip(es, mink)):
    ax.text(i - w / 2, e + 0.02, f"{e:.3f}", ha="center", fontsize=8)
    ax.text(i + w / 2, k + 0.02, f"{k:.3f}", ha="center", fontsize=8)
fig.tight_layout()
fig.savefig(FIGDIR / "fig3_es_mink.png", dpi=150)
plt.close(fig)


# ---------------------------------------------------------------------------
# 图④ 必要性(recovery, forget轴) × 充分性(damage, forget轴) 散点 —— 48点二级分析
# 2026-09-06 改版：从"top-1+layer7锚点"6个点扩到16层全测的48个点。两轴都是Bonferroni
# 口径（CI=0.996875，n_tests=16），实心=充分性(y轴)过Bonferroni显著，空心=没过。
# 原 transplant.py 的 top-1 确认性检验（95% CI，n=1）不在这张图上重复画出，见图注。
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 6.5))
for m in METHODS:
    layer_res = transplant_full[m]["layer_results"]
    for L in range(N_LAYERS):
        x_val = norm_recovery(m, L, "forget_axis") * 100
        y_axis = layer_res[str(L)]["forget_axis"]
        y_val = y_axis["normalized_damage"] * 100
        is_sig = y_axis["significant"]
        if is_sig:
            ax.scatter(x_val, y_val, color=COLOR[m], marker="o", s=70,
                       edgecolors="black", linewidths=0.6, zorder=3)
        else:
            ax.scatter(x_val, y_val, facecolors="white", edgecolors=COLOR[m], marker="o",
                       s=70, linewidths=1.3, zorder=3)
ax.axhline(0, color="gray", linewidth=0.7)
ax.axvline(0, color="gray", linewidth=0.7)
ax.set_xlabel("Necessity: Normalized Recovery % (Step 9 restore sweep, forget axis, Bonferroni CI=0.996875)")
ax.set_ylabel("Sufficiency: Normalized Damage %\n(Step 10 secondary full 16-layer transplant sweep, "
              "forget axis, Bonferroni CI=0.996875)")
ax.set_title("Figure 4: Necessity × Sufficiency Scatter (3 methods × 16 layers = 48 points)\n"
             "Filled = sufficiency significant (Bonferroni)   Hollow = not significant",
             fontsize=10)
from matplotlib.lines import Line2D
legend_elems = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markeredgecolor="black",
                       markersize=9, label=m) for m, c in COLOR.items()]
ax.legend(handles=legend_elems, loc="best", fontsize=9)
fig.text(0.5, 0.085,
          "Note: both axes here use the secondary-analysis convention (16 simultaneous tests, "
          "Bonferroni, CI=0.996875).",
          ha="center", fontsize=7.5, color="dimgray")
fig.text(0.5, 0.05,
          "The original transplant.py top-1 single-layer confirmatory test (95% CI, n=1, one layer "
          "pre-selected by necessity ranking)",
          ha="center", fontsize=7.5, color="dimgray")
fig.text(0.5, 0.015,
          "stands independently and is not overwritten or mixed into this plot — see "
          "top1_result_confirmatory in transplant_{method}.json.",
          ha="center", fontsize=7.5, color="dimgray")
fig.tight_layout(rect=[0, 0.14, 1, 1])
fig.savefig(FIGDIR / "fig4_necessity_sufficiency.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("四张图已生成：")
for f in sorted(FIGDIR.glob("*.png")):
    print(" ", f)
