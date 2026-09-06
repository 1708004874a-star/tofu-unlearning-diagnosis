#!/usr/bin/env python
"""段A第11步：出图（4张）。见 执行清单_v4.md §4段A11。

纯读盘 + matplotlib，不碰远端、不跑评测。数据来源：
  results/layer_sweep_NPO.json / layer_sweep_DPO.json   —— 第9步，16层单测（含 norm_delta）
  results/gate2_rmu_layer_sweep.json                     —— 第8步，RMU 16层单测（没有归一化字段，
                                                             这里用 transplant_RMU.json 的 denom 补算）
  results/transplant_{NPO,DPO,RMU}.json                  —— 第10步，top-1/layer7锚点/joint16
  results/archiveA/{NPO,RMU,DPO}.json                    —— 第6步，A点的 full eval 存档

图②用 retain_Q_A_ROUGE 代 model_utility（2026-09-05 决定，constants-v4.yaml
restoration.significance.metric_for_catastrophic_axis）——model_utility 没有 value_by_index，
retain_Q_A_ROUGE 结构相同、已实测确认可代替，图上标注清楚这个替代关系。

图④只用 forget 轴：necessity/sufficiency 这套对称定义本来就是在 forget 轴上建立的
（transplant_damage_definition），且只有 top-1 层和 layer7 锚点两个点同时有第9步(recovery)
和第10步(damage)的数据，6个点（3方法 × 2层）。

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
ax1.set_ylabel("归一化恢复量 %（forget 轴，实心=显著）")
ax1.set_title("图① 逐层恢复量（必要性）vs ‖Δθ_layer‖ —— NPO / RMU / DPO")
ax1.legend(loc="upper right")

for m in METHODS:
    delta = [raw_delta(m, L) for L in layers]
    ax2.plot(layers, delta, "-o", color=COLOR[m], label=m, linewidth=1.5, markersize=4)
ax2.set_xlabel("层号（0-15）")
ax2.set_ylabel("‖Δθ_layer‖（L2 范数，权重空间）")
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
ax.set_xlabel("层号（0-15）")
ax.set_ylabel("归一化恢复量 %（retain 轴，实心=显著）")
ax.set_xticks(layers)
ax.set_title("图② 换回哪层能修回 utility 损伤\n"
              "(retain_Q_A_ROUGE 代 model_utility——后者无 value_by_index，只画不判，见 constants-v4.yaml)",
              fontsize=10)
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(FIGDIR / "fig2_retain_recovery.png", dpi=150)
plt.close(fig)


# ---------------------------------------------------------------------------
# 图③ 三个方法在操作点 A 的 ES / MinK++ 对比
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4.5))
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
ax.set_title("图③ 操作点 A 的 ES / MinK++ 对比（遗忘不足轴，只画不判）")
ax.legend()
for i, (e, k) in enumerate(zip(es, mink)):
    ax.text(i - w / 2, e + 0.02, f"{e:.3f}", ha="center", fontsize=8)
    ax.text(i + w / 2, k + 0.02, f"{k:.3f}", ha="center", fontsize=8)
fig.tight_layout()
fig.savefig(FIGDIR / "fig3_es_mink.png", dpi=150)
plt.close(fig)


# ---------------------------------------------------------------------------
# 图④ 必要性(recovery, forget轴) × 充分性(damage, forget轴) 散点
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 6))
for m in METHODS:
    t = transplant[m]
    top1_L = t["top1_layer"]
    anchor_L = t["anchor_layer"]
    points = [
        (top1_L, "top1", "o",
         norm_recovery(m, top1_L, "forget_axis"),
         t["top1_result_confirmatory"]["forget_axis"]["normalized_damage"]),
    ]
    if anchor_L != top1_L:
        points.append(
            (anchor_L, "anchor", "^",
             norm_recovery(m, anchor_L, "forget_axis"),
             t["anchor_result_exploratory"]["forget_axis"]["normalized_damage"])
        )
    for L, role, marker, x_val, y_val in points:
        ax.scatter(x_val * 100, y_val * 100, color=COLOR[m], marker=marker, s=110,
                   edgecolors="black", linewidths=0.6, zorder=3)
        ax.annotate(f"{m}-L{L}", (x_val * 100, y_val * 100),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
ax.axhline(0, color="gray", linewidth=0.7)
ax.axvline(0, color="gray", linewidth=0.7)
ax.set_xlabel("必要性：归一化恢复量 %（第9步，换回=减法，forget 轴）")
ax.set_ylabel("充分性：归一化损伤 %（第10步，移植=加法，forget 轴）")
ax.set_title("图④ 必要性 × 充分性散点\n○=各方法 top-1 层（forget 轴排名） △=layer 7 锚点（探索性）",
             fontsize=10)
from matplotlib.lines import Line2D
legend_elems = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markeredgecolor="black",
                       markersize=9, label=m) for m, c in COLOR.items()]
legend_elems += [Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markeredgecolor="black",
                        markersize=9, label="top-1"),
                 Line2D([0], [0], marker="^", color="w", markerfacecolor="gray", markeredgecolor="black",
                        markersize=9, label="layer7锚点")]
ax.legend(handles=legend_elems, loc="best", fontsize=8)
fig.tight_layout()
fig.savefig(FIGDIR / "fig4_necessity_sufficiency.png", dpi=150)
plt.close(fig)

print("四张图已生成：")
for f in sorted(FIGDIR.glob("*.png")):
    print(" ", f)
