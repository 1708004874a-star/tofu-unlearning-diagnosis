# TOFU unlearning diagnosis

This repository investigates why machine unlearning methods on the TOFU benchmark fail, and where inside the model their effects actually live. It builds on [locuslab/open-unlearning](https://github.com/locuslab/open-unlearning), pinned at commit `4ad738aaf60f6a4385f6e2506d01da99e76c31f3`, and adds a layer of causal analysis on top of the library's standard unlearning and evaluation code.

## Background

An unlearning method applied to a language model can fail in two different directions. It can forget too little, so the target information is still extractable, or it can forget too much, so unrelated ability collapses along with it (catastrophic forgetting). Knowing that a method failed is not the same as knowing why. This project treats "why" as a question of location: which transformer layers actually carry the forgetting effect, and which carry the collateral damage.

The main tool is layer-level surgery on model weights, run in two directions. `restore` puts a layer's original, pre-unlearning weights back into the unlearned model and checks whether the effect disappears, which tests necessity. `transplant` moves a layer's changed weights alone into an otherwise untouched model and checks whether the effect appears by itself, which tests sufficiency. Necessity and sufficiency turn out not to be the same question: one finding from this project is that the layer most necessary for an effect is not always the layer most sufficient to reproduce it on its own.

RMU is used as a built-in positive control. Its loss terms are computed entirely on layer 7's output activations, so layers 8 through 15 provably receive no gradient during training and their weights stay unchanged; that follows directly from the training code rather than from measurement. Because the correct answer is known in advance for RMU, it is used to check that the restore/transplant machinery finds effects exactly where they should exist and nowhere else, before that machinery is trusted on methods (NPO, DPO-idk, GA, GD) where the correct answer is not known ahead of time.

Every recovery or damage number is tested with paired bootstrap resampling over per-sample scores. Wherever a full 16-layer sweep is treated as one family of simultaneous tests, the significance threshold is Bonferroni-corrected; a single, pre-registered layer is not.

## Status

Segment A is complete for NPO, RMU, and DPO-idk, and is a self-contained pilot. It covers environment and fidelity gates, operating-point selection for each method, restore/transplant correctness checks, the RMU positive control, the 16-layer necessity and sufficiency sweeps, and the four summary figures. Five methods (NPO, RMU, DPO-idk, GA, GD) were trained; GA and GD are analyzed further in segment B. Segments B through F (cross-family sweeps, a second positive control restricted to RMU's down-projection matrices, Fisher information overlap, within-layer localization, and a relearning-speed probe) are optional extensions and have not been started.

## Key results

Segment A covers NPO, RMU and DPO-idk, each at its own operating point. Every number below is a mean over 400 paired per-sample differences and is tested by bootstrap; a full 16-layer sweep is Bonferroni-corrected, a single pre-registered layer is not.

### How far do single layers add up to the joint effect?

Each ratio is the sum of the 16 single-layer effects divided by the effect of operating on all 16 layers at once. On the necessity side it is the fraction of the joint restore that the single restores recover; on the sufficiency side, the fraction of the joint transplant that the single transplants reproduce.

| | necessity (restore) | sufficiency (transplant) |
|---|---|---|
| NPO | 0.37 | 0.94 |
| DPO-idk | 0.83 | 0.07 |
| RMU | ≥ 0.81 (estimated) | 0.04 |

NPO is the only method whose necessity ratio falls below its sufficiency ratio; DPO-idk and RMU are the other way round. These ratios say how far the measured single-layer effects add up. They are not evidence for any particular encoding structure: the two columns map onto structure in opposite directions, so no single label fits both, and none is claimed here.

RMU's necessity figure is a lower bound rather than a point estimate, because its joint restore was never run. Evidence in the same direction suggests the bound is tight: transplanting all 16 transformer blocks — without `embed_tokens` — already reproduces 99.996% of RMU's forgetting effect, the closest to complete of the three methods. That does not settle the restore direction, which is the distinction this whole analysis is about, so the bound is left as a bound.

### Does the damage separate forgetting from general ability?

The selectivity index is the forget-axis denominator divided by the retain-axis denominator, both taken at a method's own operating point.

| | selectivity index | forget denom | retain denom |
|---|---|---|---|
| NPO | 1.012 | 0.5320 | 0.5255 |
| DPO-idk | 1.086 | 0.7653 | 0.7044 |
| RMU | 1.892 | 0.4235 | 0.2238 |

Only RMU moves the two axes by appreciably different amounts, and RMU is also the only one of the three that clears the feasibility floor. The index is not dose-controlled — the methods overshoot their operating points by very different margins — so it is an auditable observation, not a calibrated measurement.

Both denominators are read from `denom.forget` and `denom.retain` in `results/transplant_full_sweep_{method}.json`; the single-layer and joint effects behind the ratios above come from `layer_sweep_{NPO,DPO}.json`, `gate2_rmu_layer_sweep.json` and the same `transplant_full_sweep_*` files.

## Repository layout

Tracked in git:

```
constants-v4.yaml            frozen thresholds, formulas, and statistical parameters (single source of truth)
scripts/
  constants.yaml               copy of constants-v4.yaml used by the remote runs
  lib/
    run_pipeline.py              trains all five methods, selects each one's operating point
    gate1a_compare.py            fidelity check against the official TOFU logs
    gate2.py                     RMU positive control
    restore.py                   layer restore/transplant, including the equality tests
    layer_sweep.py               16-layer necessity sweep (step 9)
    transplant.py                 top-1 sufficiency test, anchor layer, joint-16 control (step 10, phase 1)
    transplant_full_sweep.py     16-layer sufficiency sweep (step 10, phase 2)
    make_figures.py              renders figures 1-4, plus the projection variant of figure 4
  patches/                      two bf16 fixes applied on top of open-unlearning's evaluation code
results/
  layer_sweep_{NPO,DPO}.json, gate2_rmu_layer_sweep.json    necessity sweep results
  transplant_{NPO,DPO,RMU}.json                              sufficiency, phase 1 (confirmatory)
  transplant_full_sweep_{NPO,DPO,RMU}.json                   sufficiency, phase 2 (secondary sweep)
  archiveA/                      full evaluation at each method's operating point
  figures/                       the summary figures; deck/ holds the projection variant of figure 4
  limitations.md                 deviations from plan and why, including the full Gate 1a comparison
  env/                           environment snapshot (pip freeze, GPU, library versions)
```

Not tracked, kept locally:

- `open-unlearning/`: a clone of the upstream library at the pinned commit above, with the two patches from `scripts/patches/` applied on top of its evaluation code. It is large enough that it gets cloned fresh rather than committed here.
- `results/eval_archive/` and `results/unlearn_archive/`: full local mirrors of every remote evaluation and training artifact, kept as a backup rather than as part of the analysis itself.
- Planning and working notes used while running the pipeline.

## Reproducing this

The environment has to match `open-unlearning/docs/repro.md` closely enough to pass Gate 1a: same commit, the `torch` 2.4.1 and `transformers` 4.51.3 versions pinned in `open-unlearning/requirements.txt`, flash-attention 2.6.3, `bfloat16`, and the full 400-example evaluation set with no subsampling. `results/env/` records exactly what was used for the runs in this repository.

This reproduction passed Gate 1a: 23 of 27 metrics fell within tolerance of the official logs, and the remaining 4 (AUC- and p-value-based metrics without a stable tolerance) are recorded for reference rather than gated. The full comparison table is in `results/limitations.md`, along with every other place this run had to deviate from the original plan and why.

`constants-v4.yaml` is the authority on every threshold, formula, and statistical parameter the analysis scripts use. They read values from it rather than hardcoding them, so it is the right place to check before trusting a number in the results.

## Acknowledgments

Training and evaluation code is [locuslab/open-unlearning](https://github.com/locuslab/open-unlearning), with two small patches in `scripts/patches/` restoring bf16 numerical behavior that a `transformers` version upgrade had silently broken in the evaluation path.
