# RUNBOOK — 遗忘失败分析执行规范

本文件是执行规范，不含设计理由。设计理由见 `实验流程_遗忘失败分析_v3.md`。
所有数值常量来自 `scripts/constants.yaml`，本文件只引用键名，不复制数值。

**适用范围**：从 Gate 1a 开始。之前的环境搭建（flash-attn 编译、模型下载、依赖冲突）失败模式过杂，由人工完成，结果填入 S0 交接快照。

**路径约定**（`paths.*`）：所有命令以 `paths.cwd`（`open-unlearning/`）为工作目录，`src/train.py`、`src/eval.py`、`saves/` 都是相对它的路径。自写产物一律写到 `paths.results_dir`（`../results/`）；**不要写 `results/...`**，那会落进库目录。`saves/` 是库的默认训练输出，在库目录内，是 §0.5 "不改库"的唯一例外——它是 git-ignored 产物，不是代码。

---

## 0. 执行约定

### 0.1 幂等

**文件系统是唯一状态来源。不生成 `state.json` 作为真相。**

每步开头先做产物存在性检查：

```
若 <产物路径>.done 存在 → 跳过本步，继续下一步
否则 → 执行
```

写入规则（自写产物一律遵守）：

1. 写到 `<产物路径>.tmp`
2. `fsync` + `os.rename(<tmp>, <产物路径>)`（同文件系统内 rename 是原子的）
3. 再写空文件 `<产物路径>.done`

`.done` 必须最后写。中断在任何位置，下次运行都能正确判断该步未完成。

**库自己写出的文件**（`saves/eval/**/TOFU_EVAL.json`、`saves/unlearn/**/checkpoint-*/`）不走 tmp/rename 协议：由包装脚本校验（文件存在、JSON 可解析、ckpt 可 `from_pretrained`）后再补写对应的 `.done`。

需要人读的进度视图，由扫描文件系统**派生**：`python scripts/status.py`，只读，不作为判断依据。

### 0.2 恢复时的一致性校验

每次会话开始、以及每次训练步开始前，校验：

```
per_device_train_batch_size × gradient_accumulation_steps × num_processes == training.effective_batch_size
```

不等 → **STOP-ASK #6**。GPU 数变化会改变有效 batch，前后 ckpt 不可比。

`training.verified_from_trainer_state` 为 `false` 时，不得进入 S5（NPO 除外）和 S6 → **STOP-ASK #9**。

同一方法的所有下游分析必须使用同一次 run 的 ckpt。**禁止跨 run 混用**（GPU 训练非逐位确定，两次 run 不是同一条轨迹）。

### 0.3 STOP-ASK 协议

触发任一条件时：

1. 停止执行，不进入下一步
2. 写 `../results/stops/<gate_or_reason>.json`（schema 见 §1.7）
3. 向人报告：触发条件、实测值、阈值、已完成的产物清单
4. 等待人工指示。**不得**自行修改库代码、seed、学习率、阈值来绕过

### 0.4 脚本审查点

以下自写脚本在投入全量运行前，必须先在 **1 个方法 × 1 层（或 1 个方法 × 1 个 ckpt）** 上跑通，输出中间量，交人确认后才放全量：

- `scripts/fisher.py`（先 `fisher.smoke_test_samples` 条）
- `scripts/delta_theta.py`
- `scripts/restore.py`
- `scripts/transplant.py`
- `scripts/relearn.py`

理由：这类脚本可以"跑得通、结果错"。全量跑完再发现要重烧一天。

另有两个**精确等式测试**（`gate_2.exact_equality_tests`），自动判定、不需要人看中间量，`restore.py` / `transplant.py` 通过后才允许放全量。两个都在 RMU 上做（它的 Δθ 非零集最小）：

- `restore.py`：把 RMU 遗忘后模型所有 Δθ≠0 的层**全部**换回 target，用 `eval=tofu_cheap` 评测，结果必须与 target 的 `tofu_cheap` 自评（S3 产出的 `../results/eval/target_cheap`）**逐位相同**，含 `value_by_index`。
- `transplant.py`：把所有 Δθ≠0 的层**全部**移植进 target，结果必须与 RMU 遗忘后模型自身的 `tofu_cheap` 自评逐位相同。

任一不相等 → 脚本有 bug，不进 S11。

### 0.5 不得改动

- 不修改 `open-unlearning/` 下任何文件。配置定制通过 `scripts/configs/` + Hydra `--config-dir` 实现。
- `saves/` 在库目录内，属于允许写入的产物目录，不算改库。
- 若最终不得不改库文件，改动必须以 patch 形式存入 `scripts/patches/`，并在报告中列出。
- 自写代码一律在 `open-unlearning/` 之外。

---

## 1. 目录与产物契约

```
unlearning-analysis/
├── open-unlearning/                    # 只读（saves/ 例外）
├── scripts/
│   ├── constants.yaml
│   ├── configs/
│   │   ├── eval/{tofu_full,tofu_oppoint,tofu_cheap}.yaml
│   │   └── data/                       # relearning 用的自建数据配置
│   ├── patches/
│   ├── status.py
│   └── {fisher,delta_theta,restore,transplant,conflict,relearn,plots}.py
├── results/
│   ├── snapshot.json
│   ├── gates/{gate_1a,gate_1b,gate_2}.json
│   ├── eval/target_selfcheck/TOFU_EVAL.json        # Gate 1a，默认配置
│   ├── eval/target_cheap/TOFU_EVAL.json            # target 的 tofu_cheap 自评，S12 归一化与 §0.4 等式测试用
│   ├── eval/<method>/step_<n>/TOFU_EVAL.json       # 操作点轨迹，tofu_oppoint
│   ├── eval/<method>/opA_full/TOFU_EVAL.json       # 操作点 A 的全套留档（tofu_full），遗忘不足轴的指标都从这取
│   ├── eval/<method>/opB_full/TOFU_EVAL.json       # 操作点 B 同上（与 A 相同 ckpt 时省略）
│   ├── eval/<method>/restore_layer_<k>_cheap/      # 扫层原始输出
│   ├── eval/<method>/restore_layer_<k>_full/       # top-k 全套
│   ├── eval/<method>/transplant_layer_<k>_cheap/
│   ├── operating_point/_threshold.json
│   ├── operating_point/<method>.json
│   ├── fisher/{I_f.pt,I_r.pt,meta.json}
│   ├── delta/<method>.json
│   ├── restoration/<method>/layer_<k>.json
│   ├── transplant/<method>/layer_<k>.json
│   ├── conflict/GD.json
│   ├── relearn/splits/{forget10_ft40.jsonl,forget10_heldout360_indices.json}
│   ├── relearn/<method>.json
│   ├── stops/<reason>.json
│   ├── manifests/checkpoints_to_delete_<method>.json
│   └── figures/
└── notes/{pip_freeze.txt,gate_1b_deviation.md,eval_timing.md}
```

`<method>` ∈ `{GA, GD, NPO, SimNPO, DPO, RMU, RMU_downproj}`

### 1.1 snapshot.json

```json
{
  "repo_commit": "str",
  "conda_env": "str",
  "python_version": "str",
  "transformers_version": "str",
  "torch_version": "str",
  "gpu_model": "str",
  "gpu_count": 1,
  "per_device_train_batch_size": 4,
  "gradient_accumulation_steps": 8,
  "effective_batch_size": 32,
  "target_model_eval_path": "str",
  "retain_model_eval_path": "str",
  "repro_command": "str",
  "pip_freeze_path": "notes/pip_freeze.txt",
  "created_at": "ISO8601"
}
```

### 1.2 operating_point/&lt;method&gt;.json

```json
{
  "method": "str",
  "threshold_source": "retain90 forget_Q_A_ROUGE bootstrap 95% upper",
  "threshold_value": 0.0,
  "utility_baseline": 0.0,
  "utility_floor_primary": 0.0,
  "utility_floor_check": 0.0,
  "feasibility_rule": "model_utility(A) >= utility_baseline * floor",
  "trajectory": [{"step": 12, "ckpt_path": "str", "pass": "coarse|refine|final", "forget_Q_A_ROUGE": 0.0, "model_utility": 0.0}],
  "operating_point_A": {"step": 0, "ckpt_path": "str", "forget_Q_A_ROUGE": 0.0, "model_utility": 0.0},
  "operating_point_B": {"step": 0, "ckpt_path": "str", "model_utility": 0.0},
  "operating_point_B_check": {"step": 0, "ckpt_path": "str", "model_utility": 0.0},
  "overshoot": 0.0,
  "overshoot_flagged": false,
  "feasible": true,
  "feasible_check": true,
  "infeasible_reason": null
}
```

`operating_point_A` 不存在时为 `null`，`feasible: false`，`infeasible_reason: "never_crossed"`；A 存在但 utility 低于底线时 `feasible: false`，`infeasible_reason: "utility_below_floor_at_A"`。`feasible: false` 的方法退出跨方法参数比较的统计，但 Δθ 照算（`delta_theta.compute_for_infeasible`），图上用空心点。

### 1.3 delta/&lt;method&gt;.json

```json
{
  "method": "str",
  "feasible": true,
  "computed_at": "A|B",
  "ckpt_step": 0,
  "global": {"D": 0.0, "C": 0.0, "D_rand_mean": 0.0, "C_rand_mean": 0.0,
             "D_over_D_rand": 0.0, "C_over_C_rand": 0.0, "delta_norm_sq": 0.0},
  "by_module": [{"layer": 0, "module": "attn|mlp|other", "delta_norm": 0.0,
                 "D": 0.0, "C": 0.0, "D_over_D_rand": 0.0, "C_over_C_rand": 0.0,
                 "fisher_overlap": 0.0}]
}
```

不可行的方法可能有两份（`computed_at: "A"` 与 `"B"`），文件名加后缀 `<method>_A.json` / `<method>_B.json`。

### 1.4 restoration/&lt;method&gt;/layer_&lt;k&gt;.json

```json
{
  "method": "str", "layer": 0, "mode": "restore",
  "baseline_unlearned": {"forget_Q_A_ROUGE": 0.0, "extraction_strength": 0.0},
  "target": {"forget_Q_A_ROUGE": 0.0, "extraction_strength": 0.0},
  "after": {"forget_Q_A_ROUGE": 0.0, "extraction_strength": 0.0},
  "recovery": {"forget_Q_A_ROUGE": 0.0, "extraction_strength": 0.0},
  "recovery_normalized": {"forget_Q_A_ROUGE": 0.0, "extraction_strength": 0.0},
  "recovery_ci": {"forget_Q_A_ROUGE": [0.0, 0.0]},
  "ci_level_effective": 0.996875,
  "significant": false,
  "layer_delta_norm": 0.0
}
```

`transplant/` 同 schema，`mode: "transplant"`。

### 1.5 fisher/meta.json

```json
{
  "computed_on": "target model path",
  "batch_size": 1,
  "n_forget": 400, "n_retain": 400, "retain_sample_seed": 0,
  "retain_sample_indices_path": "str",
  "fisher_type": "empirical",
  "dtype": "float32", "device": "cpu"
}
```

### 1.6 gates/&lt;gate&gt;.json

```json
{
  "gate": "str", "passed": true, "checked_at": "ISO8601",
  "checks": [{"name": "str", "observed": 0.0, "reference": 0.0, "delta": 0.0,
              "tolerance": 0.0, "passed": true}],
  "notes": "str"
}
```

### 1.7 stops/&lt;reason&gt;.json

```json
{
  "reason": "str", "triggered_at": "ISO8601",
  "observed": {}, "threshold": {},
  "completed_artifacts": ["str"],
  "next_step_if_overridden": "str"
}
```

### 1.8 relearn/&lt;method&gt;.json

```json
{
  "method": "str", "start_ckpt": "操作点 A 路径",
  "finetune_file": "str", "heldout_indices_file": "str",
  "baseline_unlearned_heldout": 0.0, "target_heldout": 0.0,
  "curve": [{"step": 0, "heldout_forget_Q_A_ROUGE": 0.0, "recovery_normalized": 0.0}],
  "steps_to_50pct_gap": 0, "curve_auc": 0.0
}
```

---

## 2. STOP-ASK 清单

| # | 条件 | 判据来源 |
|---|---|---|
| 1 | Gate 1a 失败 | `gate_1a.tolerance_prob` / `tolerance_rouge` |
| 2 | Gate 1b 差值超过硬停线 | `gate_1b.hard_stop_delta` |
| 3 | 某方法可行操作点为空集 | `operating_point.feasibility_rule`；报告后该方法退出跨方法比较，其余继续 |
| 4 | Gate 2（阳性对照）失败 | `gate_2.criteria` |
| 5 | 磁盘剩余低于阈值 | `resources.disk_stop_free_gb` |
| 6 | 有效 batch 校验不通过 | `training.effective_batch_size` |
| 7 | 自写脚本首次全量运行前 / 精确等式测试不相等 | §0.4 |
| 8 | 需要删除任何训练产物且未预授权 | §S7，`resources.auto_delete_after_manifest` |
| 9 | `training.verified_from_trainer_state` 仍为 false 时要进 S5/S6 | §S4 |
| 10 | RMU_downproj 可训练矩阵数 ≠ 3 | `gate_2.RMU_downproj.expected_trainable_matrices` |

第 3 条是唯一"报告后可继续"的；其余全部停等指示。

---

## S0 — 交接快照（人工）

产出 `../results/snapshot.json`（schema §1.1）+ `notes/pip_freeze.txt`。

必须包含对上 Gate 1a 的完整命令原文。`repo_commit` 与 `meta.repo_commit` 不一致时，先与人确认再继续。

---

## S1 — 数据与配置就绪

**前置**：`../results/snapshot.json.done` 存在。

```bash
python setup_data.py --eval_logs
python setup_data.py --idk        # DPO 依赖 data/idk.jsonl，漏了直接崩
```

校验：
- `saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json` 存在，且 `forget_Q_A_ROUGE` 带 `value_by_index`；没有 `value_by_index` → 按 `operating_point.retain_per_sample_fallback` 用默认配置自评一次 retain90
- `saves/eval/tofu_Llama-3.2-1B-Instruct_full/TOFU_EVAL.json` 存在 → Gate 1a 用它作参照；不存在则改用 `gate_1a.fallback_reference`
- `data/idk.jsonl` 存在
- retain 日志或 idk 缺失 → STOP-ASK

**产物**：`../results/data_ready.done`

---

## S2 — Gate 1a：target model 自评（硬停）

**前置**：S1 完成。

不训练，直接评 target model，与官方日志比对。同模型、同评测代码（`gate_1a.eval_config: default`），偏差只应来自硬件与 batch 组成。

```bash
python src/eval.py --config-name=eval.yaml \
  experiment=eval/tofu/default \
  model=Llama-3.2-1B-Instruct \
  model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
  forget_split=forget10 holdout_split=holdout10 \
  retain_logs_path=saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json \
  paths.output_dir=../results/eval/target_selfcheck \
  task_name=target_selfcheck
```

**判定**：
- 概率类、truth ratio 类：`|观测 − 参照| ≤ gate_1a.tolerance_prob`
- ROUGE 类：`≤ gate_1a.tolerance_rouge`
- ROUGE 偏差落在容差的 `(1 − borderline_frac)` 到 100% 之间 → 不自动放行，diff 逐样本生成结果后交人定性
- 任一超限 → **STOP-ASK #1**

**产物**：`../results/gates/gate_1a.json`。其中的 `model_utility` 即操作点的 utility 基线（`operating_point.utility_baseline_source`）。

---

## S3 — 三份评测配置

**前置**：Gate 1a 通过。

复制 `open-unlearning/configs/eval/tofu.yaml` 到 `scripts/configs/eval/` 三份（`metrics.configs`）：

| 文件 | 指标 | 用途 |
|---|---|---|
| `tofu_full.yaml` | 默认全部，再取消注释 `exact_memorization`、`mia_min_k_plus_plus`；`mia_reference` 保持注释 | S6 第 10 步操作点留档、S13 |
| `tofu_oppoint.yaml` | 只留 `forget_Q_A_ROUGE`、`model_utility`（后者自动带 9 个分项） | S6 |
| `tofu_cheap.yaml` | 只留 `forget_Q_A_ROUGE`、`extraction_strength` | S11、S12、S14、S16、§0.4 |

**文件名不得是 `tofu.yaml`**——同名会 shadow 掉库里的文件，命令行看不出加载的是哪份。

调用方式：

```bash
python src/eval.py --config-dir=../scripts/configs ... eval=tofu_cheap ...
```

**三份都 dry run**，验证 `eval=<name>` 能压过 `experiment` 里的 `override /eval: tofu`，且各自输出的指标集正确。每份记录一次实际耗时到 `notes/eval_timing.md`，后面排期按它算。

**顺手**：用 `eval=tofu_cheap` 评一次 target，存 `../results/eval/target_cheap/`（S12 的归一化和 §0.4 等式测试都要它）。

**时限 30 分钟。** 超时则退回直接编辑 `open-unlearning/configs/eval/tofu.yaml`，并将 `git diff` 存入 `scripts/patches/eval_metrics.patch`。

**产物**：三份 yaml + `../results/eval/target_cheap/` + `../results/config_ready.done`

---

## S4 — Gate 1b：NPO 单方法（软）+ 步数回填

**前置**：S3 完成。

用 §S5 的完整参数跑 NPO（这一跑就是 S5 的 NPO run，不重跑）。**对 repro.md 比的是训练结束的最终权重，用默认评测配置**（`gate_1b.eval_config` / `compare_weights`）；操作点候选评测归 S6。

**判定**：
- `forget_Q_A_ROUGE` 或 `model_utility` 差值 > `gate_1b.hard_stop_delta` → **STOP-ASK #2**
- 以内 → 记录到 `notes/gate_1b_deviation.md`，继续
- `forget_quality` 不设判据（`gate_1b.forget_quality_gated: false`）

**步数回填**（`gate_1b.after_pass`）：读 `saves/unlearn/NPO_1b/checkpoint-*/trainer_state.json` 里的 `max_steps`，回填 `training.total_steps`、`steps_per_epoch = total_steps / num_train_epochs`、`expected_checkpoints = floor(total_steps / save_steps)`，把 `training.verified_from_trainer_state` 改为 `true`，并据此重算 S6 的粗评点列表。名义值是 120 步 / 30 个 ckpt，实际很可能是 130 / 32——以文件为准。

**产物**：`../results/gates/gate_1b.json`

---

## S5 — 七个 run 的训练

**前置**：Gate 1b 通过或已记录，`training.verified_from_trainer_state: true`。按 `training.method_order` 串行；**每个 run 完成后立即执行 S6、S7 再开下一个**。

```bash
python src/train.py --config-name=unlearn.yaml \
  --config-dir=../scripts/configs \
  experiment=unlearn/tofu/default \
  model=Llama-3.2-1B-Instruct \
  model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
  trainer=<TRAINER> \
  forget_split=forget10 retain_split=retain90 holdout_split=holdout10 \
  retain_logs_path=saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json \
  trainer.args.per_device_train_batch_size=4 \
  trainer.args.gradient_accumulation_steps=8 \
  trainer.args.gradient_checkpointing=true \
  trainer.args.save_strategy=steps \
  trainer.args.save_steps=4 \
  trainer.args.eval_strategy=no \
  task_name=<METHOD>_1b
```

**方法与配置对照**（顺序即 `training.method_order`）

| `<METHOD>` | `<TRAINER>` | `experiment` | 备注 |
|---|---|---|---|
| NPO | `NPO` | `unlearn/tofu/default` | Gate 1b 已跑，复用 |
| RMU | `RMU` | `unlearn/tofu/default` | 默认 `trainable_params_regex: .*`；阳性对照 ① |
| RMU_downproj | `RMU` | `unlearn/tofu/default` | 阳性对照 ②，见下 |
| GA | `GradAscent` | `unlearn/tofu/default` | |
| GD | `GradDiff` | `unlearn/tofu/default` | S10 要它的粗评 ckpt，S7 不得删 |
| SimNPO | `SimNPO` | `unlearn/tofu/default` | repro.md 无参考值 |
| DPO | `DPO` | **`unlearn/tofu/idk`** | 需 `data/idk.jsonl`；idk.yaml 的 `override /model` 写的是 3B，必须显式传 `model=Llama-3.2-1B-Instruct` |

**RMU_downproj 的追加参数**：`trainer.method_args.trainable_params_regex` 取 `gate_2.RMU_downproj.trainable_params_regex`，列表还是字符串照 `configs/trainer/RMU.yaml` 里注释那行的形式，整个 override 用引号包住。**不要给 `|` 加反斜杠**——`\|` 在 Python re 里是字面竖线，模式会匹配不到任何参数，训练等于没动。训练开始前打印 `requires_grad` 为 True 的参数名，必须恰好 `expected_trainable_matrices` 个；不是 → **STOP-ASK #10**。

它到不了操作点 A 时（S6 判定 `never_crossed`）：按 `gate_2.RMU_downproj.lr_ladder` 依次加 `trainer.args.learning_rate=` 重跑（task_name 加后缀 `_lr<value>`），它不参与跨方法比较，调 lr 不影响结论；ladder 用完仍到不了 → 记录后跳过，Gate 2 只剩 RMU 那一半。

**校验**：训练开始前核对有效 batch（§0.2）；结束后确认 `saves/unlearn/<METHOD>_1b/` 下 ckpt 数 == `training.expected_checkpoints`，不等则记录实际值；确认根目录有最终权重，没有则以最后一个 ckpt 为最终点。

**产物**：`saves/unlearn/<METHOD>_1b/checkpoint-*/` + `../results/train_<METHOD>.done`

---

## S6 — 操作点选择

**前置**：该方法 S5 完成，`training.verified_from_trainer_state: true`。评测一律 `eval=tofu_oppoint`，输出到 `../results/eval/<method>/step_<n>/`。

1. **阈值**（全局唯一，算一次存 `../results/operating_point/_threshold.json`）：取 retain90 日志 `forget_Q_A_ROUGE` 的 `value_by_index`，按 `operating_point.bootstrap_n` / `bootstrap_ci` 做均值的 bootstrap，取上界。文件里同时存 retain 均值、CI 两端、样本数。
2. `utility_baseline` 取 Gate 1a 自评的 `model_utility`。
3. **粗评**：每第 `eval_stride_coarse` 个 ckpt（回填后的实际步数列表），加上最终权重（若不与最后一个 ckpt 重合）。
4. **细评**：找到首次 `forget_Q_A_ROUGE ≤ 阈值` 的粗评点，评它与前一个粗评点之间的全部 ckpt（`eval_refine_count` 个）。第一个粗评点已经 ≤ 阈值时，区间是 `[step 0 = target, 第一个粗评点]`（`refine_interval_if_first_coarse_already_below`）。到最终权重仍未越线 → 不细评，A 为 `null`。
5. **操作点 A** = 已评测点中首个 `forget_Q_A_ROUGE ≤ 阈值` 的 ckpt。
6. **操作点 B** = 已评测点中 `model_utility ≥ utility_baseline × utility_floor_primary` 的最后一个 ckpt；**B_check** 同法用 `utility_floor_check`。B 不做细评。
7. **可行性**（`feasibility_rule`）：`feasible = A 存在 且 model_utility(A) ≥ utility_baseline × utility_floor_primary`；`feasible_check` 同法用 `utility_floor_check`。
8. `overshoot = 阈值 − A 的 ROUGE`；超过 `overshoot_flag` 置 `overshoot_flagged: true`（**只记录，不重跑**）。
9. `feasible: false` → 写 **STOP-ASK #3**（`infeasible_reason` 为 `never_crossed` 或 `utility_below_floor_at_A`），报告后该方法退出跨方法参数比较的统计，其余方法继续。

评测次数 = 粗评点数 + `eval_refine_count`（名义 `evals_per_method`）。

10. **全套留档**：对 A（不可行方法若 A 存在也留）跑一次 `eval=tofu_full` 存 `../results/eval/<method>/opA_full/`；B 与 A 不是同一 ckpt 时再跑一次存 `opB_full/`。遗忘不足轴的 `extraction_strength` / `exact_memorization` / `mia_min_k_plus_plus` / `privleak`、S13 的 `baseline_unlearned` 全套值、S17 验证二的替代 y 轴，都从这里取。必须在 S7 删 ckpt 之前完成。

**产物**：`../results/operating_point/<method>.json` + `../results/eval/<method>/opA_full/`（及 `opB_full/`）

---

## S7 — Checkpoint 清理

**前置**：该方法 S6 完成。

生成 `../results/manifests/checkpoints_to_delete_<method>.json`：

```json
{"method": "str",
 "keep": ["操作点A路径", "操作点B路径", "操作点B_check路径", "最终权重目录", "（仅 GD）全部粗评 ckpt 路径"],
 "delete": ["路径"], "bytes_freed_estimate": 0,
 "keep_loadable_verified": false, "confirmed_by_human": false}
```

**keep 规则**（`resources.delete_safety.never_delete`）：A、B、B_check、最终权重目录；**GD 另外保留全部粗评 ckpt**（S10 要用，约 10 × 2.5 GB）。不可行的方法同样保留 A（若存在）和 B。

删除的两种放行方式：
- `resources.auto_delete_after_manifest: false`（默认）→ `confirmed_by_human: true` 由人写入后方可删除，否则 **STOP-ASK #8**
- `auto_delete_after_manifest: true` → keep 列表每个 ckpt 都 `from_pretrained` 成功（`keep_loadable_verified: true`）且 delete 列表与 never_delete 无交集，agent 可直接删

磁盘剩余低于 `resources.disk_stop_free_gb` → **STOP-ASK #5**。

---

## S8 — Fisher

**前置**：全部 run 的 S5–S7 完成（GPU 空闲时可提前，它只依赖 target model）。在**原始 target model** 上计算，与遗忘后 ckpt 无关。

要求（`fisher.*`）：
- `batch_size: 1`，逐样本梯度平方后累加，最后除以样本数
- forget 用 forget10 全部 400 条；retain 从 retain90 按 `retain_sample_seed` 随机抽 400 条，抽样索引存 `retain_sample_indices_path`（S10 复用）
- **数据管线复用库的 dataset / collator**（`fisher.data_pipeline`）：chat template、answer-only 的 `-100` mask 与训练完全一致，不自行 tokenize
- 存 CPU、float32；记录 `fisher_type: "empirical"`

**§0.4 审查点**：先在 `smoke_test_samples` 条上跑通，打印若干参数的 Fisher 值量级，交人确认后放全量。

Fisher Overlap：各自归一化到和为 1 后 `Σ min(Î_f, Î_r)`，聚合到模块级。

**产物**：`../results/fisher/{I_f.pt, I_r.pt, meta.json}`

---

## S9 — Δθ 与定位精度

**前置**：S8 完成。对每个方法计算：`feasible: true` 在操作点 A；`feasible: false` 也算（`delta_theta.compute_for_infeasible`）——A 存在则在 A 算一份，另在 B 算一份，文件名带 `_A` / `_B` 后缀。

1. `Δθ = ckpt − target`，两个 bf16 `state_dict` 先 upcast 到 fp32（`delta_theta.dtype`）
2. `D = Σ I_r · Δθ²`，`C = Σ I_f · Δθ² / Σ I_f`
3. 随机方向对照：每模块生成与真实 `Δθ` 同范数的随机方向，采 `delta_theta.null_baseline_draws` 次取均值，得 `D_rand`、`C_rand`
4. **上报 `D/D_rand` 与 `C/C_rand`**，原始 `D`、`C` 一并存但不用于跨方法比较
5. 聚合到模块级（每层 attention / MLP 各一个数；embedding、lm_head、norm 归入 `other`）

**产物**：`../results/delta/<method>[_A|_B].json`

---

## S10 — 梯度冲突（仅 GD）

**前置**：S9 完成，GD 的粗评 ckpt 仍在（S7 keep）。

离线计算，**不进训练循环、不改 trainer**。对 GD 的每个粗评 ckpt（`gradient_conflict.checkpoints`，≈epoch 边界）各算一次 `g_f` 与 `g_r` 的余弦。

- forget 400 条、retain 用 S8 抽样的同一批 400 条（`retain_sample_indices_path`）
- `batch_size: gradient_conflict.batch_size`，多个 micro-batch 的梯度累加后除以样本数得均值梯度（这里要的是**均值梯度**，大 batch 正确——与 Fisher 的 `batch_size=1` 要求相反，不要混淆）
- 结果标注为**全 batch 梯度冲突**，非逐步 minibatch 冲突

**产物**：`../results/conflict/GD.json`

---

## S11 — Gate 2：阳性对照（硬停）

**前置**：RMU 与 RMU_downproj 的 S6 完成；§0.4 两个精确等式测试通过。**不依赖 S8/S9**——各层 `‖Δθ‖` 由 `restore.py` 自己算；**必须在 S12 之前执行**，这是对整条 restoration 流程的检验，不通过则其余方法的扫层无意义。

理论依据：RMU 两项 loss（forget 的激活-control vector MSE、retain 的 `EMBED_DIFF`）都建在 `model.layers.7` 输出激活上，loss 的计算图止于第 7 层，8–15 层的 `.grad` 为 `None`，AdamW 整个跳过（`weight_decay` 亦不作用）。RMU_downproj 只有 5–7 层的三个 `down_proj` 可训练，其余全部为零。

对 `gate_2.controls` 里每个对照，在其操作点 A 上：

1. 逐元素比较 state_dict，`layers_must_be_zero` 各层 `Δθ` 严格为 0（`delta_theta_tolerance: 0.0`）
2. 16 层全部做单层 restoration（`eval=tofu_cheap`，即 S12 对该方法的扫层，结果直接复用）；`layers_must_be_zero` 各层恢复量严格为 0
3. `significant: true` 的层全部落在 `nonzero_set` 内
4. `nonzero_set` 内至少 `min_significant_layers` 层 `significant: true`——否则 pipeline 什么都没检测到，gate 是空过的

四条任一不满足 → **STOP-ASK #4**。RMU_downproj 因 `never_crossed` 跳过时，Gate 2 只剩 RMU 那一半，`gates/gate_2.json` 的 `notes` 里注明。

**峰值层只记录，不作判据**（`peak_layer_is_observation_only: true`）。

**附带：数值确定性检查**（`restoration.significance.determinism_check`）。RMU 8–15 层的 8 次 restoration 跑的是同一个模型。预期 8 次结果**逐位相同，离散度为 0**。非 0 则记为数值噪声底并上报——评测不确定性会影响所有后续比较。显著性下限**不用**这 8 次的离散度，用 §S12 的 bootstrap CI。

**产物**：`../results/gates/gate_2.json`

---

## S12 — Layer Restoration 扫描

**前置**：Gate 2 通过。

全扫，不做收窄（`restoration.sweep_all_layers: true`）：`n_layers_1b` 层 × 每个 `feasible: true` 的方法（RMU、RMU_downproj 在 S11 已扫，复用）。

- 把遗忘后模型第 k 层参数换回 target model 的对应值，其余不动；按 `temp_model_policy`：写一个临时目录 → `eval=tofu_cheap` → 立即删，任一时刻最多一个临时目录（或进程内调评测器不落盘）
- `recovery` 与 `recovery_normalized` 按 `restoration.recovery_definition`，target 值取 `../results/eval/target_cheap`
- 记录 `layer_delta_norm`，用于恢复量对 `‖Δθ_layer‖` 的对照图
- **显著性**：逐样本配对恢复量（同一题：换层后 − 遗忘后）做 bootstrap，CI 水平用 `significance.ci_effective`（每方法 16 次检验的 Bonferroni），下界 > 0 才置 `significant: true`

**产物**：`../results/restoration/<method>/layer_<k>.json`

---

## S13 — 可疑层全套评测

**前置**：S12 完成。

按恢复量取 `restoration.full_eval_top_k` 层（每方法），用 `eval=tofu_full` 跑全套指标。层级确定后，对这几层再做模块级细分（attention / MLP 分别换回）。

**产物**：`../results/eval/<method>/restore_layer_<k>_full/TOFU_EVAL.json`

---

## S14 — 充分性移植

**前置**：S13 完成。

反方向：把遗忘后模型第 k 层移植进 **target model**，其余保持原样，`eval=tofu_cheap` 测遗忘效果转移多少。层集合同 S13（`transplant.layers`）。显著性同 S12。

必要（S12）与充分（S14）同时成立的层，记为压制点。

**产物**：`../results/transplant/<method>/layer_<k>.json`

---

## S15 — （已并入 S11）

RMU_downproj 的训练在 S5（`method_order` 第 3 位），操作点在 S6，判定在 S11。保留本编号只为避免引用漂移。

---

## S16 — Relearning（最小版）

**前置**：S9 完成（需要 `C/C_rand` 选方法）。参数全部在 `relearning.*`。

**方法选择**：取 `C/C_rand` 最高与最低的两个方法（`min_version_methods`），**排除 GA**（`excluded`）。

**划分**（一次，全局复用）：按 `split_seed` 从 forget10 抽 `finetune_subset_frac`（40 条）存 `finetune_file`；其余 360 条的索引存 `heldout_indices_file`。

**微调**（每个方法）：
- 起点 = 该方法的操作点 A ckpt（`start_from`）
- `trainer=finetune`，训练数据用 `--config-dir` 里自建的 `scripts/configs/data/` 配置指向 `finetune_file`（照库的 `TOFU_QA_*` 数据配置改 `hf_args` 为 `path=json` + `data_files`），**先 dry run**
- `learning_rate`、`per_device_train_batch_size=4`、`gradient_accumulation_steps=1`（有效 batch 4 → 10 步/epoch）、`max_steps=200`、`save_steps=10`、`eval_strategy=no`

**曲线**：对每个 ckpt（含 step 0 = 操作点 A）用 `eval=tofu_cheap` 照常评全部 400 条，只对 `heldout_indices_file` 里的 `value_by_index` 取平均得 held-out `forget_Q_A_ROUGE`——不需要自定义评测数据集。评完即删该 ckpt。target 的 held-out 值同法从 `../results/eval/target_cheap` 取。

**恢复速度**（`recovery_speed`）：`steps_to_50pct_gap` = 首个 held-out ROUGE ≥ 遗忘后 + 0.5 × (target − 遗忘后) 的步数，200 步内未达记 `">200"`；`curve_auc` 作备选。

**§0.4 审查点**：`relearn.py` 先在 1 个方法上跑到 step 20，出前 3 个点交人看。

**产物**：`../results/relearn/splits/` + `../results/relearn/<method>.json`

---

## S17 — 闭环验证图

**前置**：S9、S16 完成。

- **验证一**：`D/D_rand` vs `model_utility` 相对下降。`feasible: true` 的方法用实心点进拟合；`feasible: false` 的方法用空心点画上（A、B 各一个，`delta_theta.infeasible_marker`），不进拟合——它们正是"D 在崩溃处最不可信"的证据
- **验证二**：`C/C_rand` vs relearning 恢复速度（`steps_to_50pct_gap`，S16 的方法）
- S16 未执行时的替代：y 轴换 `extraction_strength` 或 `mia_min_k_plus_plus`，并在图注标明该替代测不到再微调维度

**产物**：`../results/figures/validation_{1,2}.{png,pdf}` + 底层数据 `.json`

---

## 附：3B

`resources.vram_3b_gb` 超出 24G 卡。3B 相关步骤在换到 48G 卡之前一律不执行；被要求执行时先 STOP-ASK 确认硬件。

若执行：仅 NPO + RMU 两个方法，`n_layers_3b` 层，其余流程同上。
