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


# gate2 产物有两种 schema：2026-09-13 之前是扁平的 {"0":…,"15":…}，之后与
# layer_sweep_{m}.json 对齐成 {"layer_results":…, "denom":…}。两种都读得动，
# 这样盘上的旧产物和下次重跑的新产物都能出图。
_g2 = load("gate2_rmu_layer_sweep.json")
_g2_nested = "layer_results" in _g2

sweep = {
    "NPO": load("layer_sweep_NPO.json")["layer_results"],
    "DPO": load("layer_sweep_DPO.json")["layer_results"],
    "RMU": _g2["layer_results"] if _g2_nested else _g2,
}
transplant = {m: load(f"transplant_{m}.json") for m in METHODS}
transplant_full = {m: load(f"transplant_full_sweep_{m}.json") for m in METHODS}
archiveA = {m: json.load(open(RESULTS / "archiveA" / f"{m}.json")) for m in METHODS}

# constants-v4.yaml:141 overshoot_must_appear_on_figures: true
# ——「跨方法比恢复量的前提是 overshoot 相当」，所以每张恢复量图都必须把它标出来。
# 实测值来自 run_pipeline 的产物，不重算。
OVERSHOOT = {m: json.load(open(RESULTS / "pipeline"
                / f"tofu_Llama-3.2-1B-Instruct_forget10_{m}.json"))["overshoot"]
             for m in METHODS}
OVERSHOOT_FLAG = 0.10          # constants-v4.yaml:140，只标记不重跑


def m_label(m):
    """图例标签带 overshoot；超过 flag 的打 † 。"""
    ov = OVERSHOOT[m]
    return f"{m}  (overshoot {ov:.3f}{' †' if ov > OVERSHOOT_FLAG else ''})"


# 执行清单_v4.md:109「不可行的方法…仍在图上用空心点画出」——原设计用空心编码 feasibility，
# 但填充已被显著性占用。2026-09-13 决定：形状编码 feasibility，填充继续编码显著性，两维分离。
FEASIBLE = {m: json.load(open(RESULTS / "pipeline"
               / f"tofu_Llama-3.2-1B-Instruct_forget10_{m}.json"))["feasible"]
            for m in METHODS}
MARKER = {m: ("o" if FEASIBLE[m] else "s") for m in METHODS}


def panel_title(m):
    ov = OVERSHOOT[m]
    return (f"{m}    overshoot {ov:.3f}{' †' if ov > OVERSHOOT_FLAG else ''}"
            f"    {'feasible' if FEASIBLE[m] else 'infeasible'}")


# 2026-09-13：这段限定条件原先烤在 PNG 里，投影上约 4pt 读不到，而它承载的是全图
# 最重要的限定。改为只留在幻灯片正文 + speaker notes，图内不再重复。文本保留在此
# 供 deck 引用，不再 fig.text 上图。
FIG_NOTE = ("Each panel has its own y-axis: shapes are comparable across panels, heights are not. "
            "Filled = significant (Bonferroni). Circle = feasible, square = infeasible. "
            "† overshoot > %.2f (constants-v4.yaml:140). Single operating point per method — "
            "shape-invariance across doses is untested, so all layer-level claims are within-method."
            % OVERSHOOT_FLAG)

# 旧版 gate2 产物没有 denom，只能借 transplant_RMU.json 的（跨步骤依赖，已记入
# limitations.md）；新版自带 denom，优先用自己的。
_rmu_denom = _g2["denom"] if _g2_nested else transplant["RMU"]["denom"]
RMU_DENOM_FORGET = _rmu_denom["forget"]
RMU_DENOM_RETAIN = _rmu_denom["retain"]


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
# 图① small multiples（2026-09-13 改版）。原版三方法叠在一个坐标系上，但三者的
# denom（0.5320 / 0.7653 / 0.4235）和 overshoot（0.109 / 0.000 / 0.342）都不同，
# 叠图会诱导"谁恢复得更多"这类不合法的跨方法比较——RMU 视觉最高很大程度只是分母最小。
# 分面后：层间形状比较（合法）保留，高度比较（不合法）视觉上做不到。
# ---------------------------------------------------------------------------
# 2026-09-14：下半栏改回单面板。上半栏（归一化 recovery）分母各不相同、不可跨方法比，
# 所以必须分面；‖Δθ‖ 是权重空间 L2，三方法本来就在同一把尺子上——它是全图唯一一处
# 合法的跨方法量级比较，也正是"改得最多的层"这个对照的全部作用。分面给它三条独立 y 轴，
# 会把 DPO 真实 0.034 的起伏和 RMU 真实 0.612 的尖峰渲染成同样高度，对照就不对照任何东西了。
fig = plt.figure(figsize=(13, 6.4))
_gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.82], hspace=0.42, wspace=0.26)
axes = [[fig.add_subplot(_gs[0, c]) for c in range(3)]]
ax_delta = fig.add_subplot(_gs[1, :])
layers = list(range(N_LAYERS))
for c, m in enumerate(METHODS):
    ax = axes[0][c]
    rec = [norm_recovery(m, L, "forget_axis") * 100 for L in layers]
    sigs = [sig(m, L, "forget_axis") for L in layers]
    ax.plot(layers, rec, "-", color=COLOR[m], linewidth=1.4, zorder=2)
    ax.scatter([L for L, v in zip(layers, sigs) if v], [r for r, v in zip(rec, sigs) if v],
               color=COLOR[m], marker=MARKER[m], s=46, zorder=3)
    ax.scatter([L for L, v in zip(layers, sigs) if not v], [r for r, v in zip(rec, sigs) if not v],
               facecolors="white", edgecolors=COLOR[m], marker=MARKER[m], s=46,
               linewidths=1.2, zorder=3)
    ax.axhline(0, color="gray", linewidth=0.7)
    ax.set_title(panel_title(m), fontsize=12)
    # 上下排拆成两个 gridspec 之后 sharex 没了，上排会自动取 2.5 步长——层索引没有 2.5 层。
    # 用 layers[::3] 而不是 MaxNLocator(integer=True)，是为了跟附录 A4 那三格的刻度逐字一致。
    ax.set_xticks(layers[::3])
    ax.tick_params(labelsize=9)
    if c == 0:
        ax.set_ylabel("Normalized recovery %\n(forget axis)", fontsize=10)

for m in METHODS:
    delta = [raw_delta(m, L) for L in layers]
    ax_delta.plot(layers, delta, "-", color=COLOR[m], linewidth=1.5,
                  marker=MARKER[m], markersize=4.5, label=m)
    # 每个方法自己的峰在哪一层，共享轴下 DPO 那条几乎贴平，标出来才读得到
    pk = max(layers, key=lambda L: delta[L])
    ax_delta.annotate(f"L{pk}", xy=(pk, delta[pk]), xytext=(0, 6),
                      textcoords="offset points", ha="center",
                      fontsize=8.5, color=COLOR[m])
ax_delta.set_xticks(layers)
ax_delta.tick_params(labelsize=9)
ax_delta.set_xlabel("Layer index", fontsize=10.5)
ax_delta.set_ylabel("‖Δθ_layer‖  (L2, weight space)", fontsize=10.5)
ax_delta.set_title("Control — shared axis: Δθ is measured in weight space, so magnitudes "
                   "ARE comparable across methods", fontsize=10.5)
ax_delta.legend(fontsize=9, loc="upper right", ncol=3, frameon=True)
ax_delta.margins(y=0.16)

fig.savefig(FIGDIR / "fig1_recovery_vs_delta.png", dpi=150, bbox_inches="tight")
plt.close(fig)


# ---------------------------------------------------------------------------
# 图② 同样改成 small multiples（retain 轴）。retain_Q_A_ROUGE 代 model_utility 的
# 理由见 constants-v4.yaml restoration.significance.metric_for_catastrophic_axis，
# 不再印在图上——那是内部备注，不该占投影面积。
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharex=True)
for c, m in enumerate(METHODS):
    ax = axes[c]
    rec = [norm_recovery(m, L, "retain_axis") * 100 for L in layers]
    sigs = [sig(m, L, "retain_axis") for L in layers]
    ax.plot(layers, rec, "-", color=COLOR[m], linewidth=1.4, zorder=2)
    ax.scatter([L for L, v in zip(layers, sigs) if v], [r for r, v in zip(rec, sigs) if v],
               color=COLOR[m], marker=MARKER[m], s=46, zorder=3)
    ax.scatter([L for L, v in zip(layers, sigs) if not v], [r for r, v in zip(rec, sigs) if not v],
               facecolors="white", edgecolors=COLOR[m], marker=MARKER[m], s=46,
               linewidths=1.2, zorder=3)
    ax.axhline(0, color="gray", linewidth=0.7)
    ax.set_title(panel_title(m), fontsize=12)
    ax.set_xticks(layers[::3])
    ax.tick_params(labelsize=9)
    ax.set_xlabel("Layer index", fontsize=10)
    if c == 0:
        ax.set_ylabel("Normalized recovery %\n(retain axis)", fontsize=10)

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
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

DECKDIR = FIGDIR / "deck"
DECKDIR.mkdir(exist_ok=True)


def draw_fig4(with_provenance_note: bool):
    """图④。with_provenance_note=True 出分析版（图内带三行来路说明，进 results/figures/）；
    False 出投影派生版（去掉那三行，进 results/figures/deck/）。

    2026-09-13：投影版原先是 PIL 一次性裁剪出来的，不是任何脚本的产物，README 里
    「four output figures」跟实际对不上。改成这里一个有名字的派生步骤——同一份数据、
    同一段绘图代码，只是省掉在投影尺寸下约 4pt、根本读不到的那三行说明。
    """
    fig, ax = plt.subplots(figsize=(7.5, 6.5 if with_provenance_note else 5.9))
    for m in METHODS:
        layer_res = transplant_full[m]["layer_results"]
        for L in range(N_LAYERS):
            x_val = norm_recovery(m, L, "forget_axis") * 100
            y_axis = layer_res[str(L)]["forget_axis"]
            y_val = y_axis["normalized_damage"] * 100
            if y_axis["significant"]:
                ax.scatter(x_val, y_val, color=COLOR[m], marker=MARKER[m], s=70,
                           edgecolors="black", linewidths=0.6, zorder=3)
            else:
                ax.scatter(x_val, y_val, facecolors="white", edgecolors=COLOR[m],
                           marker=MARKER[m], s=70, linewidths=1.3, zorder=3)
    ax.axhline(0, color="gray", linewidth=0.7)
    ax.axvline(0, color="gray", linewidth=0.7)
    # 正文说 "the points would lie on a rising diagonal"，那条线得真的画出来，
    # 否则读者无从判断偏离多远。
    lim = max(
        max(norm_recovery(m, L, "forget_axis") * 100 for m in METHODS for L in range(N_LAYERS)),
        max(transplant_full[m]["layer_results"][str(L)]["forget_axis"]["normalized_damage"] * 100
            for m in METHODS for L in range(N_LAYERS)))
    ax.plot([0, lim], [0, lim], ls=":", c="gray", linewidth=1.1, zorder=1)

    # 2026-09-13：RMU 的 layers_must_be_zero（constants-v4.yaml:175）在必要性和充分性
    # 两轴上都精确为 0，8 个点完全重合在原点。不标出来，读者会把它们数成一个点，而
    # 标题里的 "48 points" 有 8 个是构造出来的零、不是测出来的零——这两种零不该混读。
    struct_zero = [L for L in range(N_LAYERS)
                   if norm_recovery("RMU", L, "forget_axis") == 0.0
                   and transplant_full["RMU"]["layer_results"][str(L)]["forget_axis"][
                       "normalized_damage"] == 0.0]
    if struct_zero:
        contiguous = struct_zero == list(range(struct_zero[0], struct_zero[-1] + 1))
        span = (f"L{struct_zero[0]}\u2013{struct_zero[-1]}" if contiguous
                else "L" + ",".join(str(L) for L in struct_zero))
        ax.annotate(
            f"RMU {span}: {len(struct_zero)} coincident points at the origin\n"
            "\u0394\u03b8 = 0 by construction, not by measurement",
            xy=(0.45, 0.45), xytext=(0.40 * lim, 0.30 * lim),
            fontsize=8, color="dimgray", ha="left", va="center",
            arrowprops=dict(arrowstyle="->", color="darkgray", linewidth=0.8,
                            shrinkA=2, shrinkB=2),
            zorder=2)
        # 原点那一簇本来就挤（17 个点落在 ±1.5 之内），圈出来才知道箭头指的是哪一堆
        ax.add_patch(Circle((0, 0), 0.9, fill=False, ec="darkgray",
                            ls=(0, (2, 2)), linewidth=0.8, zorder=2))

    ax.set_xlabel("Necessity: normalized recovery % (step 9 restore sweep, forget axis)", fontsize=11)
    ax.set_ylabel("Sufficiency: normalized damage %\n(step 10 full 16-layer transplant sweep, forget axis)",
                  fontsize=11)
    ax.tick_params(labelsize=9)
    if with_provenance_note:
        ax.set_title("Figure 4: Necessity × Sufficiency (3 methods × 16 layers = 48 points)\n"
                     "Filled = sufficiency significant (Bonferroni)   Hollow = not significant",
                     fontsize=10)

    elems = [Line2D([0], [0], marker=MARKER[m], color="w", markerfacecolor=c,
                    markeredgecolor="black", markersize=9,
                    label=f"{m} ({'feasible' if FEASIBLE[m] else 'infeasible'})")
             for m, c in COLOR.items()]
    elems.append(Line2D([0], [0], ls=":", color="gray", label="y = x"))
    ax.legend(handles=elems, loc="best", fontsize=9)

    if with_provenance_note:
        for y, txt in [
            (0.085, "Note: both axes use the secondary-analysis convention (16 simultaneous tests, "
                    "Bonferroni, CI=0.996875)."),
            (0.050, "The original top-1 single-layer confirmatory test (95% CI, n=1, one layer "
                    "pre-selected by necessity ranking)"),
            (0.015, "stands independently and is not mixed into this plot — see "
                    "top1_result_confirmatory in transplant_{method}.json."),
        ]:
            fig.text(0.5, y, txt, ha="center", fontsize=7.5, color="dimgray")
        fig.tight_layout(rect=[0, 0.14, 1, 1])
        out = FIGDIR / "fig4_necessity_sufficiency.png"
    else:
        fig.tight_layout()
        out = DECKDIR / "fig4_necessity_sufficiency.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


_deck_fig4 = draw_fig4(with_provenance_note=True)     # 分析产物
_deck_fig4 = draw_fig4(with_provenance_note=False)   # 投影派生版
# 原先这里 glob 整个 FIGDIR，把 fig5（make_fig5_norm_confound.py 的产物）也报成本脚本写的。
# 只列自己真正写出来的，才对得上 README 里"renders figures 1-4"那句话。
print("已生成：")
for name in ["fig1_recovery_vs_delta.png", "fig2_retain_recovery.png",
             "fig3_es_mink.png", "fig4_necessity_sufficiency.png"]:
    print(" ", FIGDIR / name)
print("  ", _deck_fig4, "(figure 4, projection variant)")
