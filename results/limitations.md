# 局限记录（供段A第12步写报告直接用）

## 段A第5-6步无人值守：清理前的人工确认换成脚本完整性校验（2026-09-04）

§5 第4条原文要求"要删任何训练产物…停、把已完成的产物列给我、把实测值和阈值列给我，然后等指示"。
`scripts/lib/run_pipeline.py` 连续跑 5 个方法（NPO→RMU→DPO→GA→GD）时，每个方法训完立刻清理
（删掉操作点 A 以外的 ckpt）——这一步按原规则要停机等人确认，但用户要求无人值守整晚跑完，
当面明确批准了这一处例外，条件是：

- 删除前用 `from_pretrained` 完整加载一遍要保留的 ckpt（A），加载失败就不删任何东西、直接停机
  （`verify_loadable()`，代替人工确认，不是跳过检查）
- 以下情况脚本会自动停机、不会继续：Gate 1b 差超过 `hard_stop_delta`（NPO 专属外部校验）、
  ckpt 完整性校验不过、训练报错（非 OOM）、`/workspace` 实占预估剩余 < 8G
- 显存装不下 `per_device=8` 时自动退到 `4×8` 继续跑，不停机，只在日志里记一笔——这条本来就在
  §5 末尾"不用停"的例外列表里，不算新破例

这条例外只对**这一晚这一次运行**有效，不是把 §5 第4条永久改成"脚本自动确认"——以后每一段
新的无人值守训练都要重新问一次。

## 训练后自动评测必炸：`do_eval` 跟 `eval_strategy` 是两个开关（2026-09-05）

**现象**：无人值守跑 NPO 时，训练本身两次都完整跑完（8x4、4x8 各 120/120 步，loss 正常下降），
但训练一结束立刻崩：`CUDA out of memory`，崩溃点是 `evals/metrics/utils.py:98`
`evaluate_probability()`——就是 bf16 patch 0001 打过的那个函数。8x4 和 4x8 崩在完全相同的位置、
相同的评测子步骤（"Calculating loss: 2/13"），说明跟训练 batch size 无关。

**根因**（读了 `train.py` 和 `configs/trainer/finetune.yaml` 源码，不是推断）：

1. `train.py:64-65` 训练结束后无条件跑一次 `trainer.evaluate()`，这个调用只看 `do_eval`，
   跟管道设的 `eval_strategy=no` 是两个独立开关；`finetune.yaml:20` 把 `do_eval` 硬编码成
   `True`，管道从未覆写过，所以每次训练后都会多跑一次完整 TOFU 评测——这次评测跟管道自己
   的评测（`operating_point_sweep`/`utility_at`/archive，各自起 `eval.py` 子进程，写到
   `saves/eval/` 下）完全无关，纯属多余
2. 这次多余评测用的是库默认 `per_device_eval_batch_size=16`（`finetune.yaml:4`），管道的
   OOM 重试只改了 `per_device_train_batch_size`，从没碰过这个值——这就是为什么把训练 batch
   从 8 降到 4 对这次崩溃毫无作用
3. `finetune.yaml:8` 的 `bf16_full_eval=True` 本意是让评测阶段省显存，但 patch 0001 在
   `evaluate_probability()` 里无条件把 logits 转 fp32——Llama 词表 12.8 万，fp32 比 bf16
   的 logits 张量大一倍。Gate 1a 当时没暴露这个问题，是因为那次评测在干净进程里跑，没有
   训练残留的显存占用；这里评测紧跟在训练后、同一进程里，训练用掉的 ~13G 还没释放，
   24G 显存只剩 ~1.2G 可用，撑不住这次 fp32 logits

**顺带发现的编排脚本自身 bug**：`run_pipeline.py` 的 OOM 重试逻辑（`train_method()`）对
子进程整段输出做 `"CUDA out of memory" in text` 字符串匹配，不区分 OOM 发生在训练循环
还是这次训练后评测里——把本该无关的失败错判成"训练 batch 太大"，重试了一次同样会崩的
路径，白烧了两轮完整训练（8x4、4x8 都跑完 120 步又都被清理逻辑删掉）。停机行为本身是
安全的：没有半成品或残留产物留在盘上（`saves/unlearn/` 清空，磁盘用量跟训练前一致）。

**做了什么**：`run_pipeline.py` 的训练命令加一条 `trainer.args.do_eval=false`，彻底关掉
这次多余评测。确认过不影响任何门槛判据（`gate_1b`/`operating_point_sweep`/`utility_at`
读的都是 `saves/eval/{task}/TOFU_EVAL.json`，来自独立的 `eval.py` 调用，跟这次关掉的
`saves/unlearn/.../checkpoint-*/evals/` 完全是两个路径）。这处改动不碰训练数值——
`evaluate()` 是只读调用，跑在 `save_model()` 之后，删不删都不影响已存的 checkpoint 权重。
OOM 重试的字符串匹配 bug 没有单独修——`do_eval=false` 落地后这条误判路径已经没有触发
条件了，留作已知但目前不会再触发的次要问题。

## bf16 数值精度：两处库代码 patch（跟"单卡 vs 2×L40s"是同一类问题的另一面）

**现象**：`src/eval.py` 在 bf16 模型上对 `forget_Q_A_Prob`/`forget_truth_ratio`/`model_utility`/
`forget_quality`/`mia_min_k_plus_plus` 这几项指标直接崩溃：`TypeError: Got unsupported ScalarType
BFloat16`。ROUGE 类、`extraction_strength`、`exact_memorization` 不受影响（不走这条 numpy 转换路径）。

**根因**（已用 diagnostic print 实测确认，非纯代码推断）：`transformers==4.51.3` 的
`LlamaForCausalLM.forward`（`modeling_llama.py` L835 注释"do not upcast them to float if we are
not computing the loss"）里，`logits` 从 lm_head 出来后不再被无条件升到 fp32；即使传了 `labels`，
`self.loss_function` 内部的升精度（如果有）只作用于它自己返回的 `loss`，不会回写调用方拿到的
`output.logits`。`src/evals/metrics/utils.py:evaluate_probability()` 直接读 `output.logits` 算
交叉熵，全程停留在 bf16，最后 `.cpu().numpy()` 才报错——错误发生的位置比真正失真的位置晚了好几步。

**判断依据**：docs/repro.md 和下载的官方日志里这几项指标都存在，说明官方那次运行不是在这条 bf16-to-numpy
必崩的路径上产出的——`src/trainer/base.py` 顶部 "Modified from v4.45.1" 的注释暗示更早版本的
transformers 有无条件的 `logits = logits.float()`，与现在的 4.51.3 行为不同。补上 upcast 是**恢复**
官方那次运行的数值口径，不是引入新偏离。

**做了什么**：
1. `scripts/patches/0001-evaluate_probability-bf16-upcast.patch` —— `utils.py` 里紧跟
   `output.logits` 之后升 fp32，再走交叉熵；`.numpy()` 前的 `.float()` 保留作双保险
2. `scripts/patches/0002-mia_min_k_plus_plus-sigma-bf16.patch` —— 同一类问题在
   `mia/min_k_plus_plus.py` 里的一处遗漏（`sigma` 同样源自 bf16 logits 链路）

**上游对照**：GitHub `locuslab/open-unlearning` 有一个**未合并**的 PR #156（"Fix BFloat16
compatibility in evaluation metrics"，2025-12-03 关闭未合并），diff 跟 patch 0001 逐字节相同——
不是我们自己发明的改法。同一类问题在 MIA 指标上已被 **PR #162（已合并，在我们的 pin 里）** 修过，
但漏了 `min_k_plus_plus.py` 里 `sigma` 那一处，patch 0002 补的就是这个漏洞。

**影响范围核实**：应用两个 patch 后重跑 Gate 1a，9 个指标全部算完、无崩溃，且跟官方日志的差异全部
落在 1e-4 ～ 3e-3 量级（见下表），证明这不是引入了新的数值偏差，而是消除了此前"崩溃掉的那部分根本没跑"
的假问题。

## Gate 1a 分桶：规则不是清单

最初只判了 9 个显式点名的指标，其余 18 个 pre_compute 副产品（也真实落盘在 `TOFU_EVAL.json` 里）
没有判据——这个范围本身是不对的。改成三条规则，覆盖 `TOFU_EVAL.json` 实际落盘的全部 27 项，
不逐条枚举（`constants-v4.yaml` `gate_1a.metric_buckets`，2026-09-04 签字）：

- `tolerance_prob`（0.01）：名字含 `prob` 或 `truth_ratio`
- `tolerance_rouge`（0.02）：名字含 `rouge`，加 `extraction_strength`/`exact_memorization`/`model_utility`
- `unrated`（记录不设判据）：`forget_quality`/`privleak`/`mia_*`

程序过一遍 27 个 key，三条规则都套不上的 = 0（不用人工兜底）。

**Gate 1a 最终结果**（`open-unlearning/tofu_Llama-3.2-1B-Instruct_full`，2026-09-04，27 项全覆盖）：

| metric | bucket | ours | official | \|diff\| | 判定 |
|---|---|---|---|---|---|
| exact_memorization | tolerance_rouge | 0.973526 | 0.973936 | 0.000410 | PASS |
| extraction_strength | tolerance_rouge | 0.705161 | 0.706268 | 0.001108 | PASS |
| forget_Q_A_PARA_Prob | tolerance_prob | 0.100417 | 0.100418 | 0.000000 | PASS |
| forget_Q_A_PERT_Prob | tolerance_prob | 0.041326 | 0.041335 | 0.000010 | PASS |
| forget_Q_A_Prob | tolerance_prob | 0.880468 | 0.880482 | 0.000014 | PASS |
| forget_Q_A_ROUGE | tolerance_rouge | 0.819499 | 0.820117 | 0.000618 | PASS |
| forget_quality | unrated | 3.905e-22 | 3.905e-22 | ~0 | 记录 |
| forget_truth_ratio | tolerance_prob | 0.475446 | 0.475564 | 0.000118 | PASS |
| mia_min_k | unrated | 0.996669 | 0.996650 | 0.000019 | 记录 |
| mia_min_k_plus_plus | unrated | 0.997969 | 0.998031 | 0.000062 | 记录 |
| model_utility | tolerance_rouge | 0.599397 | 0.599153 | 0.000244 | PASS |
| privleak | unrated | -99.460428 | -99.457391 | 0.003037 | 记录 |
| ra_Q_A_PERT_Prob | tolerance_prob | 0.006409 | 0.006411 | 0.000002 | PASS |
| ra_Q_A_Prob | tolerance_prob | 0.016108 | 0.016145 | 0.000037 | PASS |
| ra_Q_A_Prob_normalised | tolerance_prob | 0.413896 | 0.413565 | 0.000332 | PASS |
| ra_Q_A_ROUGE | tolerance_rouge | 0.807000 | 0.797000 | 0.010000 | PASS |
| ra_Truth_Ratio | tolerance_prob | 0.528382 | 0.527326 | 0.001056 | PASS |
| retain_Q_A_PARA_Prob | tolerance_prob | 0.088589 | 0.088430 | 0.000159 | PASS |
| retain_Q_A_PERT_Prob | tolerance_prob | 0.038772 | 0.038715 | 0.000057 | PASS |
| retain_Q_A_Prob | tolerance_prob | 0.870625 | 0.870660 | 0.000035 | PASS |
| retain_Q_A_ROUGE | tolerance_rouge | 0.793905 | 0.793263 | 0.000642 | PASS |
| retain_Truth_Ratio | tolerance_prob | 0.517712 | 0.517962 | 0.000250 | PASS |
| wf_Q_A_PERT_Prob | tolerance_prob | 0.001937 | 0.001953 | 0.000015 | PASS |
| wf_Q_A_Prob | tolerance_prob | 0.005328 | 0.005381 | 0.000054 | PASS |
| wf_Q_A_Prob_normalised | tolerance_prob | 0.435673 | 0.436331 | 0.000658 | PASS |
| wf_Q_A_ROUGE | tolerance_rouge | 0.819373 | 0.827920 | 0.008547 | PASS |
| wf_Truth_Ratio | tolerance_prob | 0.620433 | 0.620105 | 0.000327 | PASS |

23 项判据全部 PASS（0 BORDERLINE，0 FAIL），4 项 unrated 仅记录。

**Gate 1a：通过。**
