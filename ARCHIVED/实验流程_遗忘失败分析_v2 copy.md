# 遗忘失败分析 — 两周实验流程 v2

**一句话目标**：不是比较哪个遗忘方法效果好，而是搞清楚它们**为什么失败**，并且用因果证据而不只是相关性来支撑。

失败分两种，两种都要能诊断：

- **灾难性遗忘** — 模型被搞坏了，通用能力崩塌
- **遗忘不足** — 模型看起来忘了，但知识还在，稍加微调就回来

---

## v2 相对 v1 的改动

| # | 改了什么 | 为什么 |
|---|---|---|
| 1 | 阶段 0 补 `setup_data.py --idk` | 缺了 DPO 直接崩 |
| 2 | DPO 换成 `unlearn/tofu/idk.yaml` | 官方脚本里 DPO 就不走 default |
| 3 | eval 命令补 `pretrained_model_name_or_path` | 原命令评的是 target model，不报错但结果无意义 |
| 4 | 新增「操作点对齐」 | 不对齐的话 $D$、$C$ 比的是"跑了多远"不是"改得准不准" |
| 5 | Fisher 代码改成 `batch_size=1` | 原代码累加的是 batch 平均梯度的平方，不是 Fisher |
| 6 | $D$、$C$ 加归一化和随机方向对照 | 原定义随 $\|\Delta\theta\|^2$ 增长，不是"精度" |
| 7 | Layer Restoration 加 $\Delta\theta$ 幅度对照 | 否则结论平凡（"改得最多的层最重要"） |
| 8 | 新增 RMU 阳性对照 | RMU 的压制层是配置里写死的，是免费的 ground truth |
| 9 | 新增阶段 5「闭环验证」 | 零成本把参数侧证据和行为侧结果接起来 |
| 10 | Relearning 改成子集微调 + held-out 评测 | 原设定测的是"重新背下来"不是"知识残留" |
| 11 | 梯度冲突降级为可选 | 成本 +50% 训练时间，师兄已否掉这个方向 |
| 12 | 排期重排，3B 明确缩范围 | v1 排期里根本没有 3B 的位置 |
| 13 | relearning 从「可砍」改为「保最小版」，且移到阶段 5 之前 | v2 初版自相矛盾：阶段 5 的验证二依赖它，排期还把它排在后一天 |

---

## 优先级（排期挤压时按这个砍）

阶段 2 的 Fisher 和阶段 4 的 relearning 曲线，正好是师兄两篇论文的核心工具。**差异化几乎全押在阶段 3（Layer Restoration + RMU 阳性对照 + 充分性移植）。**

所以时间不够时的砍法是：

1. **保**：阶段 1（操作点）→ 阶段 3（干预证据）→ 阶段 5（闭环）
2. **保最小版**：阶段 4 relearning 缩到 2 个方法（选法见阶段 4）
3. **压缩**：阶段 2 只留 Fisher Overlap 和归一化后的 $D$、$C$，不做梯度冲突
4. **砍**：3B 全量、梯度冲突、Activation Patching

注意这个顺序和 v1 的时间分配是反的。

**为什么 relearning 不能整个砍。** 遗忘不足那一轴的行为侧证据并不是只有它——`extraction_strength` 和 MIA 本来就是行为侧的（前者测实际生成，后者是攻击），Layer Restoration 给的还是更强的干预证据。所以砍掉 relearning，这一轴不会塌。

真正丢掉的是一个更窄但无可替代的东西：**对抗性维度**。ES/MIA 是在权重固定的前提下测的，Restoration 是你自己动权重，只有 relearning 是让**新的梯度更新**去把知识捞回来，对应「攻击者拿到模型后还能再微调」这个威胁模型。这个只有它能覆盖，所以留最小版而不是清零。

**依赖关系**：阶段 5 的验证二以 relearning 恢复速度为 y 轴，所以 **relearning 必须排在阶段 5 之前**。真跑不动时验证二有个免费替代，见阶段 5。

---

## 主流程

### 阶段 0 — 环境准备

```bash
conda create -n unlearning python=3.11
conda activate unlearning
pip install ".[lm-eval]"
pip install --no-build-isolation flash-attn==2.6.3

python setup_data.py --eval_logs   # retain model 的评测日志，forget_quality 依赖它
python setup_data.py --idk         # DPO 需要 data/idk.jsonl，漏了 DPO 直接崩
```

> `--eval` 也能用（argparse 前缀匹配到 `--eval_logs`），但 `--idk` 是独立的一项，README 的 Quickstart 里没写。

**跑正式实验之前必须改的三处配置**

1. **`save_strategy`** — 默认是 `'no'`，训练全程**一个中间 checkpoint 都不存**。要设 `trainer.args.save_strategy=epoch`。你的 $\Delta\theta$ 分析和操作点选择全靠这些中间权重，漏了后面全部重跑。这是最容易忘、代价最大的一步。
   - `save_only_model` 默认已经是 `True`，所以只存权重不存优化器状态，1B 一个 checkpoint 约 2.5 GB，3B 约 6.4 GB。10 个 epoch × 6 个方法 = 1B 单独就要 150 GB。租的机器盘小的话，算完 $\Delta\theta$ 就删掉中间的，只留操作点那个。

2. **有效 batch size 必须等于 32**。`configs/trainer/finetune.yaml` 默认 `per_device_train_batch_size=8` × `gradient_accumulation_steps=4` = 32，单卡直接用默认就是对的。**但不要把官方脚本里的 `per_device=4 accum=4` 抄到单卡上** —— 那是给 2 卡用的，单卡会变成 16，repro.md 的数就对不上，你会以为是环境装错了。24G 卡上 `per_device=8` 可能爆显存，那就用 `per_device=4 accum=8`。

3. **`eval_strategy`** — 默认 `epoch` + `eval_on_start=True`，训练过程中每个 epoch 会跑一次完整 TOFU 评测。有两个细节：
   - 它只在**单进程**下生效（`src/trainer/base.py` 里 `num_processes != 1` 会 warning 后直接返回空）。官方 2 卡脚本其实拿不到这些数。
   - 单卡上它是免费的操作点数据来源，但会让训练时间涨很多（10 次完整评测）。
   - **建议**：训练时设 `trainer.args.eval_strategy=no` 让训练跑快，之后对存下来的 10 个 checkpoint 离线跑**精简指标**评测来定操作点。可控性更好。

**目录结构**

```
unlearning-analysis/
├── open-unlearning/    # git clone 下来，不改它
├── scripts/            # 自己写的分析代码
├── results/            # 输出的 json 和图
└── notes/
```

自己的代码放在库外面。师兄要求不改库的原有实现，分目录相当于给这条约束一个物理边界。

---

### 阶段 1 — baseline 与操作点对齐

**这一步和分析方向无关，无论后面做什么都必须先有它。**

模型直接用官方微调好的，不要自己微调：

- Target model：`open-unlearning/tofu_Llama-3.2-1B-Instruct_full`
- Retain model：`open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90`

#### 六个方法与各自的配置

| 族 | 方法 | `experiment` | 注意 |
|---|---|---|---|
| 无约束 | GA | `unlearn/tofu/default` | |
| 带 retain 项 | GD | `unlearn/tofu/default` | |
| preference | NPO | `unlearn/tofu/default` | |
| preference | SimNPO | `unlearn/tofu/default` | **不在官方 baseline 脚本里**，repro.md 无参考值 |
| preference | DPO | **`unlearn/tofu/idk`** | 需要 `data/idk.jsonl`；配置里 `override /model` 写的是 3B，**必须显式传 `model=Llama-3.2-1B-Instruct`** |
| representation engineering | RMU | `unlearn/tofu/default` | `module_regex: model\.layers\.7` —— 压制层是已知的，见阶段 3b |

preference 族放三个是有意的：**同族表现应当接近**，这是可验证的观察点。

但要小心一个混淆：DPO 走的是 idk 数据（把答案换成"我不知道"），机制上是**输出替换**，而 NPO/SimNPO 是**偏好压低**。所以"DPO 和 NPO 差很远"不一定是发现，可能只是数据设定不同。这一点在报告里要讲清楚，别当成同族异常。

#### 遗忘命令（五个方法通用）

```bash
python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/tofu/default \
  model=Llama-3.2-1B-Instruct \
  model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
  trainer=GradAscent \
  forget_split=forget10 retain_split=retain90 holdout_split=holdout10 \
  retain_logs_path=saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json \
  trainer.args.per_device_train_batch_size=4 \
  trainer.args.gradient_accumulation_steps=8 \
  trainer.args.gradient_checkpointing=true \
  trainer.args.save_strategy=epoch \
  trainer.args.eval_strategy=no \
  task_name=ga_1b
```

DPO 只需换两处：`experiment=unlearn/tofu/idk trainer=DPO`。

输出在 `saves/unlearn/<task_name>/`，中间 checkpoint 在 `saves/unlearn/<task_name>/checkpoint-<step>/`。

#### 评测命令

```bash
python src/eval.py --config-name=eval.yaml \
  experiment=eval/tofu/default \
  model=Llama-3.2-1B-Instruct \
  model.model_args.pretrained_model_name_or_path=saves/unlearn/ga_1b \
  forget_split=forget10 holdout_split=holdout10 \
  retain_logs_path=saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json \
  paths.output_dir=saves/eval/ga_1b \
  task_name=ga_1b
```

三个最容易漏的：

- **`model.model_args.pretrained_model_name_or_path`** — 漏了会静默评测默认的 target model，数字看起来正常但完全没意义
- **`retain_logs_path`** — 漏了 `forget_quality` 算不出来
- **`holdout_split`** — MIA 是拿 forget 和 holdout 对比算的，配错了 MIA 全废（默认 `holdout10` 配 `forget10` 是对的，但只要动 `forget_split` 就必须同步改）

结果在 `saves/eval/<task_name>/TOFU_EVAL.json`。

**跑完对一遍 `docs/repro.md`**，数字对得上才说明环境和配置没问题。这是校验，不是结果。SimNPO 除外。

#### 操作点对齐（v2 新增，最关键的一步）

默认配置下**所有方法都是 lr 1e-5 跑 10 个 epoch**。这意味着 GA 在第 10 个 epoch 已经彻底崩了，而 NPO 可能刚开始遗忘。在这个状态下比较 $D$ 和 $C$，你比的是"走了多远"，不是"改得准不准"。

项目里那篇 [arXiv 2509.26427] 讲的正是这件事：ascent 类方法对学习率和微调时长极度敏感，且缺少明确的停止准则；看完最终指标再回头挑最好的 run，本身就是有偏的。

**做法**：对每个方法的 10 个 epoch checkpoint 都跑一次**精简评测**（只要 `forget_Q_A_ROUGE` + `model_utility`），得到一条轨迹，然后定两个操作点：

- **操作点 A（forget 侧对齐）**：`forget_Q_A_ROUGE` 首次降到 retain model 水平的那个 checkpoint。retain model 的值直接从 `saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json` 读。
- **操作点 B（utility 侧对齐）**：`model_utility` 仍保有 target model 的 90% 的最后一个 checkpoint。（OpenUnlearning 论文里用的是 80% 这个门槛，可以对齐它。）

**所有阶段 2、3、5 的分析都在操作点 A 上做，操作点 B 作为稳健性复核。**

有个方法可能**根本不存在可行操作点**——比如 GA 在 forget 降下来之前 utility 就先崩了，A 和 B 交不上。这本身就是"失败"的一个干净定义，值得单独作一张图。

#### 指标清单

灾难性遗忘那一轴：
- `model_utility` — 主指标
- `retain_Q_A_ROUGE` / `real_authors_Q_A_ROUGE` / `world_facts_Q_A_ROUGE` — 三个分项，用来看是局部损伤还是整体崩塌

遗忘不足那一轴：
- `forget_Q_A_ROUGE`
- `extraction_strength`
- `exact_memorization`
- `mia_min_k_plus_plus` / `mia_reference`

主指标 `forget_quality` 单独看，但**它和 ES/MIA 打架时以后者为准**。`forget_quality` 高而 `extraction_strength` 也高，就是典型的遗忘不足。

---

### 阶段 2 — 参数侧测量

作用不是结论，是**筛出哪几层最可疑**，作为阶段 3 的输入。全部在**操作点 A** 的 checkpoint 上做。

#### 2.1 Fisher information

**必须在遗忘之前算**，用原始 target model。Fisher 近似 Hessian 这件事只在最优点附近成立，跑崩的模型不满足。

```python
import torch
from torch.utils.data import DataLoader

fisher = {n: torch.zeros_like(p, device="cpu") for n, p in model.named_parameters()}
loader = DataLoader(dataset, batch_size=1, shuffle=False)   # 必须是 1

n_seen = 0
for batch in loader:
    model.zero_grad(set_to_none=True)
    loss = model(**batch).loss
    loss.backward()
    for n, p in model.named_parameters():
        if p.grad is not None:
            fisher[n] += (p.grad.detach() ** 2).cpu()
    n_seen += 1

for n in fisher:
    fisher[n] /= n_seen
```

**`batch_size` 必须是 1。** v1 的写法在 batch > 1 时累加的是「batch 平均梯度的平方」，而 Fisher 要的是「per-sample 梯度平方的期望」。两者差约 $B$ 倍，而且样本间方差被平均掉了——算出来的不是 Fisher。要么 `batch_size=1`，要么用 per-sample gradient（`torch.func.vmap` + `grad`）。

其他注意：

- 用真实标签算出来的是 **empirical Fisher**，不是 true Fisher。近似 Hessian 只在"模型已经拟合得很好 + 在最优点附近"成立。target model 满足这个条件，但**报告和论文里要写明是 empirical Fisher**。
- $I_f$ 用 forget10 全部 400 条；$I_r$ 从 retain90 里**随机抽 400 条**，保持样本量对称，方差可比，也省时间。
- 内存：1B 的 fisher 字典 fp32 约 5 GB，3B 约 12 GB，$I_f$ 和 $I_r$ 两份就是 24 GB **CPU 内存**。租机器时别只看显存。

#### 2.2 Fisher Overlap

先各自归一化成和为 1，再算重叠：

$$\hat{I}_f = \frac{I_f}{\sum_i I_f(i)}, \quad \hat{I}_r = \frac{I_r}{\sum_i I_r(i)}, \quad \text{Overlap} = \sum_i \min\big(\hat{I}_f(i),\, \hat{I}_r(i)\big) \in [0, 1]$$

重叠高的参数是最难办的——改了伤模型，不改忘不掉。

**聚合到模块级别看**（每层的 attention、MLP 各一个数），逐参数的散点没有意义。可以再补一个模块级的 top-k Jaccard 作为交叉验证。

#### 2.3 Δθ 定位精度（修正版）

$\Delta\theta = $ 操作点权重 $-$ 原始权重，两个 `state_dict` 相减。

两个基础量：

- $D = \sum_i I_r(i)\,\Delta\theta_i^2$ — 改动落在 retain 重要参数上的量
- $C = \dfrac{\sum_i I_f(i)\,\Delta\theta_i^2}{\sum_i I_f(i)}$ — 该改的改了多少

**但这两个量不能直接跨方法比。** 两者都随 $\|\Delta\theta\|^2$ 增长，所以走得远的方法 $D$ 大、$C$ 也大——这是"改得多"不是"改得准"。必须加对照：

```python
# 与真实 Δθ 每模块范数相同的随机方向
delta_rand = {}
for name, d in delta.items():
    g = torch.randn_like(d)
    delta_rand[name] = g * (d.norm() / (g.norm() + 1e-12))

D_rand = sum((I_r[n] * delta_rand[n] ** 2).sum() for n in delta_rand)
D_ratio = D / D_rand      # 报这个
```

**报 $D/D_{\text{rand}}$ 和 $C/C_{\text{rand}}$。** 大于 1 才说明"确实打在了 retain / forget 重要参数上"，等于 1 说明和乱改一个方向没区别。随机方向至少采 5 次取均值。

$D$ 小、$C$ 大是理想状态。$D$ 有推导撑着（retain loss 上升量的二阶近似），$C$ 只是必要条件——参数改了不等于知识删了，所以要靠阶段 3、4 补。

**同样聚合到模块级别。**

#### 2.4 梯度冲突（可选，降级）

师兄已经否掉这个作为主方向。它的成本又最高：GD 里两个梯度是合并后才 backward，要分别测就得每步多一次 backward，训练时间涨约 50%。

**建议只在 GD 一个方法上测，每 20 步一次**，作用是给"冲突确实存在，但不足以解释失败"提供一个佐证，占报告一页。不要六个方法都测。

---

### 阶段 3 — Layer Restoration（主证据）

**这是这两周的重点，也是相对师兄已有工作的主要差异化。**

做法：把遗忘后模型的某一层参数换回原始值，其余不动，重跑 eval 看知识回来多少。工程上就是 `state_dict` 替换 + 重跑 eval，几十行。

Llama-3.2-1B 有 16 层，3B 有 28 层。

**为什么它比阶段 2 强**：阶段 2 是测量，只能说"这层相关"；Restoration 是**干预**，能说"就是这层"。因果力度完全不同，这也是师兄那张表把它排在第 2 的原因。

#### 3a 必要性扫描 + Δθ 幅度对照

逐层换回，看恢复量。**但必须同时记录每层的 $\|\Delta\theta_{\text{layer}}\|$，并把恢复量对 $\|\Delta\theta_{\text{layer}}\|$ 作图。**

没有这个对照的话，结论是平凡的：换回某层知识回来了，很可能只是因为那层改动最大。

- **恢复量完全跟着 $\|\Delta\theta\|$ 走** → 结论就是"改得最多的层最重要"，没有信息量
- **有层 $\Delta\theta$ 不大，但一换回来知识就回来** → 这才是真发现：因果集中在某处，而参数改动是散开的
- **换回某层知识就回来** → 那层是压制点，遗忘只是把它按住了
- **全换回来才回来** → 知识是分布式存储的，没有单点

#### 3b RMU 阳性对照（v2 新增，几乎白送）

`configs/trainer/RMU.yaml` 里写着 `module_regex: model\.layers\.7` —— RMU 的损失锚死在第 7 层。也就是说**这一个方法的压制点你有 ground truth**。

- 如果 Layer Restoration 扫出来 RMU 的关键层就在第 7 层附近，**整条 pipeline 就被验证了**
- 扫不出来，说明方法本身有问题，其他五个方法的结论也就不可信

更妙的是 RMU 的 `trainable_params_regex` 是 `.*`（全参数更新），所以它的 $\Delta\theta$ 是散开的，但因果集中在一层。**这正好是 3a 那个对照最干净的正面例子**——"$\Delta\theta$ 大的层 ≠ 起作用的层"，用一个已知答案的方法演示出来，汇报时最有说服力。

#### 3c 充分性移植（v2 新增）

3a 测的是**必要性**（拿掉这层的改动，遗忘效果消失吗）。反方向测**充分性**：把遗忘后模型的某一层**移植进原始模型**，其余保持原样，看遗忘效果转移过来多少。

同样是 `state_dict` 替换，成本和 3a 一样，但因果结论强一档。必要 + 充分都成立的层，才是干净的"压制点"。

#### 成本控制（必读）

1B 有 16 层 × 6 方法 = 96 次 eval，3c 再翻倍。TOFU 全套评测（约 1000 条生成 + 6 种 MIA）在 4090 上一次十几到几十分钟，全跑要 20–40 GPU 小时，塞不进排期。

**两段式**：

1. **扫层阶段**只跑 `forget_Q_A_ROUGE` + `extraction_strength` 两个便宜指标，一次 2–4 分钟，96 次约 5 GPU 小时
2. **选出可疑层**后，只对这几层跑全套评测（含 `model_utility`、MIA）

另外，**先用阶段 2 的结果缩小范围**：不用盲扫所有层，先扫 $D$、$C$、Fisher Overlap 标出来的那几层，剩下的层用粗粒度采样。

层级找到之后，再对那几层做**模块级细分**（attention vs MLP）。

---

### 阶段 4 — Relearning（保最小版，且必须排在阶段 5 之前）

遗忘完之后拿少量数据再微调，每隔几步测一次，画恢复曲线。

**库里已经有 relearning stress test 实现，不用自己从头写。** 但官方设定（完整 forget10 上微调 1 epoch，lr 2e-5）有个问题：在全部 forget 数据上微调、再在同样的数据上测，测的是"重新背下来"，不是"知识本来就在"。

**改法**：

- 在 forget10 的一个**子集**（比如 10%，40 条）上微调
- 在**剩下 90%（360 条）**上评测

这样恢复才说明是知识残留而不是重新记忆。

两种微调数据分别对应两个失败模式：

- 用 **forget 数据**微调 → 测遗忘不足。几步就弹回来说明知识从来没走
- 用 **retain 数据**微调 → 测损伤深度。修得回来说明只是表层受损

**曲线形状比终点值重要**。两个方法终点一样，但一个第 5 步就跳回去、另一个爬了 200 步才上来，机理完全不同。

#### 最小版怎么选方法

跑不了六个方法时，缩到两个。**选法：等阶段 2 出结果，取 $C/C_{\text{rand}}$ 最高和最低的那两个方法。** 这样散点图上的两个点是有意义的两端，不是拍脑袋定的，验证二也才有解释力。

不要提前定死。如果一定要预判，NPO + DPO(idk) 是合理的猜测：DPO 走 idk 数据是最典型的输出替换，理论上回弹最快，和 NPO 的对比正好是这一轴要的东西。

**不要用 GA 做这一轴的对照。** GA 不回弹很可能是因为模型已经坏了（能力没了自然学不回来），而不是知识被删了——这个点在图上没法解读，是混淆的。GA 属于灾难性遗忘那一轴，归验证一管。

#### 成本

比看起来便宜：微调本身很快（40 条跑几十步，1B 上几分钟），开销在曲线上每隔几步要在 360 条 held-out 上评测一次。只跑 `forget_Q_A_ROUGE` 的话一个方法约一小时量级，六个方法一天能收工。所以「砍不砍」这个决定可以推迟到第 11 天再做。

---

### 阶段 5 — 闭环验证（v2 新增，零额外训练成本）

到这一步你手上已经有：$D$（Fisher 预测的 retain 损伤）、实测的 `model_utility` 下降、$C$、relearning 恢复速度。六个方法就是六个点。**直接画散点。**

**验证一：$D/D_{\text{rand}}$ vs `model_utility` 下降幅度**

- 相关性好 → 你的 Fisher 工具被验证了，可以说"参数级证据能预测行为级后果"
- 相关性差 → 二阶近似在大 $\Delta\theta$ 下失效。这本身也是结论，而且正好和 GA 崩掉对得上：$D$ 最不可信的地方，恰恰是灾难性遗忘最严重的地方

**验证二：$C/C_{\text{rand}}$ vs relearning 恢复速度**

$C$ 小应该预测"几步就弹回来"。对上了，说明参数覆盖不足确实是遗忘不足的原因。

这一张**依赖阶段 4**，所以 relearning 必须先跑完。最小版的两个点也能画，只是拟合不了趋势，只能作两端对照。

**免费替代（relearning 真跑不动时）**：把 y 轴换成 `extraction_strength` 或 `mia_min_k_plus_plus`，数据阶段 1 就有了，六个点齐全。

代价是它更弱：ES 是在单个操作点、权重固定下测的，会被输出压制骗过——而输出压制恰恰是被研究的现象本身，所以这张图有循环论证的风险。能跑 relearning 就跑，替代版只当兜底，并且在报告里说明它测不到再微调这个维度。

这两张图把参数侧和行为侧接起来了，正是最后那张对照表想要的东西。验证一完全免费（数据阶段 1–2 就有），验证二要阶段 4 撑着。**汇报时这是最有分量的两页。**

---

## 两周排期（修订）

| 时间 | 做什么 |
|---|---|
| 第 1–2 天 | 阶段 0 + 单个方法跑通，对上 repro.md（注意有效 batch = 32） |
| 第 3–4 天 | 六个方法 1B 全跑完（`save_strategy=epoch`），离线精简评测定操作点 |
| 第 5–6 天 | 阶段 2：Fisher $\times 2$、Overlap、$\Delta\theta$ 归一化 + 随机方向对照 |
| 第 7–10 天 | **阶段 3**：便宜指标扫层 → 可疑层全套评测 → RMU 阳性对照 → 充分性移植 |
| 第 11 天 | **阶段 4 relearning**：先跑 $C/C_{\text{rand}}$ 两端的 2 个方法，有余量再往中间补 |
| 第 12 天 | 阶段 5 闭环验证，出两张散点图 |
| 第 13 天 | 补实验、统一出图；有余量则做 3B 一致性验证 |
| 第 14 天 | 做 PPT |

头两天大概率都在填坑（装不上、跑不动、数字对不上），留出这个余量。

**顺序不能调**：阶段 4 必须在阶段 5 之前，否则第 12 天画验证二时没有 y 轴数据。（v2 初版把这两天写反了。）

**关于 3B**：v1 的"坑"那节写了必须跑 3B，但排期表里没有位置。v2 把它降级为**第 13 天的余量项**：2 个方法（建议 NPO + RMU）的跨模型一致性验证，不是全套六个。前面任何一段超时它就会被挤掉——那就在报告里写成 next step，不要两边都挂着。真实预期是大概率做不完。

**汇报形式是 PPT + 可视化结果**，所以从阶段 2 开始每张图都存好，别等最后一天补。

---

## 坑清单

**配置类**

1. **`save_strategy` 默认是 `'no'`** → 一个中间 checkpoint 都不存，过程分析和操作点选择全部要重跑
2. **`setup_data.py --idk` 没跑** → DPO 找不到 `data/idk.jsonl`，直接崩
3. **DPO 用了 `unlearn/tofu/default`** → 数据配置不对，跑出来的不是 idk 版 DPO
4. **`idk.yaml` 里 `override /model` 写的是 3B** → 必须显式传 `model=Llama-3.2-1B-Instruct`
5. **eval 漏 `pretrained_model_name_or_path`** → 静默评测默认 target model，不报错，数字看起来正常
6. **eval 漏 `retain_logs_path`** → `forget_quality` 算不出来
7. **`holdout_split` 和 `forget_split` 不匹配** → MIA 全废
8. **把官方脚本的 `per_device=4 accum=4` 抄到单卡** → 有效 batch 变 16，对不上 repro.md
9. **多卡时指望训练中评测** → `num_processes != 1` 时评测器直接返回空，只 warning 不报错

**方法类**

10. **没对齐操作点就比较 $D$、$C$** → 比的是"跑了多远"，结论无效
11. **$D$、$C$ 不归一化** → 测的是改动幅度不是定位精度
12. **Fisher 用 `batch_size > 1`** → 算出来的不是 Fisher
13. **Fisher 在遗忘之后才算** → 顺序反了，数值没有意义
14. **Layer Restoration 没有 $\Delta\theta$ 幅度对照** → 结论平凡
15. **Relearning 在全部 forget 数据上微调再在同样数据上测** → 测的是重新记忆，不是知识残留
16. **只跑 1B** → 师兄明确要跨模型一致的规律
17. **用 LoRA 跑 3B** → LoRA 只改低秩旁路，$\Delta\theta$ 的分布模式和全参数完全不同，结论推不过去

**资源**

- **显存**：1B 全参数微调约 14–16 GB，24G 卡够；3B 约 40–42 GB，建议 48G。开 `gradient_checkpointing` 能省一截激活值，代价是慢 20–30%。
- **CPU 内存**：Fisher 字典和参数一样大。3B 的 $I_f$ + $I_r$ 要 24 GB CPU 内存，别只看显存。
- **磁盘**：`save_strategy=epoch` 下 1B 六个方法约 150 GB。算完 $\Delta\theta$ 就删中间的，只留操作点那个。

---

## 报告里要主动交代的局限

先说比被问出来好：

1. **单 seed**。默认 `seed: 0`，六个方法各跑一次。方法间的小差异可能是噪声，只讨论跨族的大趋势，不解读同族内的细微差别。
2. **超参未调**。所有方法都用库的默认（lr 1e-5、10 epoch），操作点对齐部分缓解了这个问题，但不等于每个方法都在自己最好的状态。
3. **empirical Fisher**，不是 true Fisher；二阶近似只在 $\Delta\theta$ 小时可靠，对 GA 这类崩溃的方法最不可信——阶段 5 的验证一正好能量化这一点。
4. **DPO 与 NPO/SimNPO 数据设定不同**（idk vs 原答案），同族比较时不完全对等。
5. **TOFU 的 forget/retain 只是微调数据**，预训练语料动不了，所以"删干净"在这里是相对 retain model 而言的。
6. **relearning 只覆盖 2 个方法**（如果走的是最小版）。验证二是两端对照，不是趋势拟合；其余方法的遗忘不足证据来自 ES/MIA 和 Layer Restoration，这两者测不到"攻击者再微调"这个维度。

---
---

# 术语表

## 数据与模型

**unlearning（遗忘学习）** — 让训练好的模型「忘掉」指定的一批数据，同时保住其他能力。动机是隐私合规（被遗忘权）、版权、有害内容清除。

**forget set** — 要被忘掉的那批数据。

**retain set** — 要保住的那批数据，是 forget set 的补集。

**holdout set** — 既不在 forget 也不在 retain 里的第三份数据，模型从没见过。MIA 指标靠它做对照：如果模型对 forget 数据的行为和对 holdout 的行为一样，说明真忘了。

**TOFU** — 一个遗忘学习基准数据集。200 个**虚构**作者 × 每人 20 条问答 = 4000 条。用虚构人物是为了确保这些知识只来自微调、不来自预训练，这样才有干净的对照。

**forget10 / retain90 / holdout10** — TOFU 的划分。forget10 是要遗忘的 10%（400 条），retain90 是剩下的 90%（3600 条）。两者相加是 TOFU 的全部，但**不是模型训练数据的全部**——模型还有万亿级的预训练语料你碰不到。

**target model** — 在 forget + retain 全集上微调过的模型，也就是「什么都记得」的起点。

**retain model** — 只在 retain 上微调、从没见过 forget 数据的模型。它是**黄金标准**：理想的遗忘结果应该和它一样。TOFU 的 `forget_quality` 就是拿遗忘后模型和它做统计检验。

---

## 遗忘方法

**GA（Gradient Ascent，梯度上升）** — 最朴素的做法：在 forget 数据上做梯度**上升**，让 loss 变大。问题是完全不管 retain，跑久了模型会崩。

**GD（GradDiff，Gradient Difference）** — GA 的扩展版，**同一步里同时**做两件事：forget 上升 + retain 下降。

$$L = -L_{\text{forget}} + L_{\text{retain}}$$

不是分阶段，是两个梯度相加后一次更新。

**NPO（Negative Preference Optimization）** — 借用 DPO 的形式，把「忘掉」表述成「相对于参考模型，降低对 forget 样本的偏好」，而不是无限拉高 loss。因此比 GA 稳得多。

**SimNPO** — NPO 的简化版本。注意它不在库的官方 baseline 脚本里，`docs/repro.md` 没有它的参考数值。

**DPO（Direct Preference Optimization）** — 原本是对齐用的方法，也可以改造来做遗忘。在这个库里 DPO 走 idk 数据：把 forget 问题的答案换成「我不知道」类回复，所以机制上更接近**输出替换**，和 NPO 的「偏好压低」不完全同类。

**RMU（Representation Misdirection Unlearning）** — 直接在某一层扰乱 activation，机制和上面几个都不同，属于 representation engineering。库里默认锚在 `model.layers.7`，但参数更新是全参数的。**这个「已知压制层」的性质使它成为阶段 3 的天然阳性对照。**

**方法族** — 同一族的方法机制相近，表现应当接近。师兄要求按族看趋势，而不是逐个孤立看。

**操作点（operating point）** — 拿来做跨方法比较的那个统一状态。因为不同方法在同样的 lr 和 epoch 数下会停在遗忘-效用权衡曲线的完全不同位置，不先对齐操作点，任何跨方法的参数级比较都在比「走了多远」而不是「走得准不准」。

---

## 评测指标

**forget_quality** — TOFU 的招牌指标。把遗忘后模型和 retain model 的 truth ratio 分布做 KS 检验，越接近说明越像「从没学过」。有黄金标准做锚点，这是它比 NLL 强的地方。

**model_utility** — 模型整体能力有没有坏。由 retain、real_authors、world_facts 三个分项聚合（每个分项各有 probability、ROUGE、truth ratio 三个数，共九个取调和平均）。

**extraction_strength（ES）** — 给一段前缀，模型能不能续出剩下的内容。测的是**实际生成**，不是 teacher forcing 下的概率。

**exact_memorization（EM）** — 逐 token 精确记住了多少。

**MIA（Membership Inference Attack，成员推断攻击）** — 判断某条数据有没有参与过训练。如果攻击成功，说明知识还在。open-unlearning 集成了六种：LOSS、ZLib、Reference、GradNorm、MinK、MinK++。

**MinK++** — 最强的 MIA 变体之一。你用 NLL 报「忘掉了」，别人用 MinK++ 一测可能就把知识捞回来了。

**NLL（Negative Log-Likelihood，负对数似然）** — 就是 loss 本身。**为什么它作为遗忘指标不合格**：① 无上界，GA 定义上就是把它推高，多跑几步能到任意大；② 没有参照点，说不了「多高算忘干净」；③ 它就是 MIA 里最弱的 LOSS attack；④ 分不清「知识删了」和「模型崩了」。

---

## 分析工具

**Fisher information（费雪信息）** — 衡量**参数重要性**的一种算法。参数重要性是概念（「这个参数动一下，模型表现掉不掉」），Fisher 是算它的具体方法：

$$I_i = \mathbb{E}\left[\left(\frac{\partial L}{\partial \theta_i}\right)^2\right]$$

**per-sample** 梯度平方在数据集上求平均——注意是逐样本，不是 batch 平均后再平方。同一个公式，喂 forget 数据得到 $I_f$，喂 retain 数据得到 $I_r$。

出处是持续学习里的 **EWC**（Kirkpatrick et al., 2017），当年就是用它来防灾难性遗忘的。

**empirical Fisher vs true Fisher** — 用数据集里的真实标签算梯度得到的是 empirical Fisher；true Fisher 要从模型自己的预测分布里采样标签。实践中都用前者，但它近似 Hessian 的条件更严（要求模型已经拟合得很好且在最优点附近）。**报告里要写清用的是哪个。**

**Hessian（黑塞矩阵）** — 所有二阶偏导排成的矩阵。一阶导（梯度）告诉你坡朝哪边斜，二阶导告诉你坡**弯得多厉害**。弯得厉害 = 参数动一点 loss 就飙 = 重要。

严格的参数重要性应该用 Hessian，但它是 $n \times n$ 的，1B 模型有 $10^{18}$ 个数，算不动。**Fisher 的价值在于：在最优点附近它近似等于 Hessian 的对角线，而只需要梯度平方。二阶的信息，一阶的代价。**

**influence function（影响函数）** — Fisher 的严格版，问「某条训练样本对模型的贡献有多大」。形式是 $-H^{-1}\nabla_\theta L(z)$，要算 Hessian 的逆，LLM 上做不动。写论文时可以作为理论出处引用。

**Δθ** — 遗忘前后的参数变化量，两个 `state_dict` 相减。

**随机方向对照（null baseline）** — 造一个和真实 $\Delta\theta$ 每模块范数都相同、但方向随机的扰动，用它算出 $D_{\text{rand}}$、$C_{\text{rand}}$ 作分母。**没有这个分母，$D$ 和 $C$ 测的是改动幅度，不是定位精度。**

**gradient conflict（梯度冲突）** — forget 梯度和 retain 梯度的夹角。用余弦相似度量化：

$$\cos\theta = \frac{g_f \cdot g_r}{\|g_f\|\,\|g_r\|}$$

负值 = 两个目标直接对抗，「忘掉 A」和「保住 B」要求参数往相反方向调。**注意**：GA 也有冲突，只是它不算也不管——冲突是数据和模型本身的性质，不是算法造出来的。

局限：它只能诊断灾难性遗忘。一个什么都不做的方法冲突为零，模型完好，知识一点没少——这个指标会给它满分。这也是师兄否掉它作为主方向的原因。

**Layer Restoration** — 把某层参数换回原始值，看知识回不回来。**干预**，不是测量，所以能给因果结论。测的是**必要性**。

**层移植（transplant）** — 反方向：把遗忘后模型的某层放进原始模型，看遗忘效果转不转移。测的是**充分性**。必要 + 充分都成立，才是干净的压制点。

**阳性对照（positive control）** — 用一个答案已知的案例来验证你的方法本身管不管用。这里 RMU 就是：它的压制层写在配置里，如果扫不出来，说明整条 pipeline 有问题。

**Activation Patching** — 更精细的干预：模型运行时，在某层某个 token 位置，把遗忘后模型的中间结果替换成原模型的。分辨率是「第几层 × 第几个词」，产出一张热力图。

因果力度最强，但工程量最大（要挂 forward hook、跑 $L \times T$ 次前向）。**陷阱**：标准做法是同模型不同输入，你这里是跨模型，如果两个模型表征漂移太远，搬过去的激活值就是分布外噪声。GA 跑崩的模型尤其危险。两周内不建议做。

**relearning（再学习）** — 遗忘后用少量数据再微调，看知识多快回来。快速恢复 = 知识只是被压制，没被删除。**要在微调数据之外的 forget 样本上评测**，否则测的是重新记忆。

它和 ES/MIA、Layer Restoration 的分工：后两者都在权重固定（ES/MIA）或由你手动改权重（Restoration）的前提下取证，只有 relearning 让**新的梯度更新**参与进来，对应「攻击者拿到模型后还能再微调」这个威胁模型。这是它不可替代的地方，也是它值得占一个排期位的唯一理由。

**CKA / PCA / SVCCA / Tuned Lens / Probe** — 都是 representation（隐层表征）分析手法，输出通常是「横轴层号」的图。师兄已经做过这块，注意避让。

---

## 两个失败模式对照

| | 灾难性遗忘 | 遗忘不足 |
|---|---|---|
| 现象 | 模型能力崩塌、输出乱码 | 看起来忘了，实际知识还在 |
| 指标 | `model_utility` 掉 | `extraction_strength`、MIA 仍然高 |
| 参数侧证据 | $D/D_{\text{rand}}$ 大（改到 retain 重要参数） | $C/C_{\text{rand}}$ 小（该改的没改） |
| 行为侧证据 | retain 微调修不回来 | forget 微调几步就弹回来 |
| 干预证据 | 换回受损层能修复 utility | 换回某层知识就回来 |
| 主要工具 | Fisher（有二阶展开撑着） | relearning、Restoration |

**Fisher 强在左边，relearning 强在右边，两个互补。** 单靠一个都有说不圆的地方——这也是师兄批评梯度冲突「只适合灾难性遗忘」的原因。而 Layer Restoration 是唯一两边都能给因果证据的工具，这是把它排在第一优先级的理由。

---

## 参考

- **OpenUnlearning 技术报告** — arXiv 2506.12618。框架设计、指标 meta-evaluation（faithfulness / robustness）、relearning 与 quantization stress test 的官方设定。
- **DA 类遗忘方法的失败分析** — arXiv 2509.26427。forget/retain 之间的数据依赖导致 ascent 类方法系统性失败；关于停止准则缺失和超参敏感性的论证，是阶段 1「操作点对齐」的直接依据。
- **EWC** — Kirkpatrick et al., 2017。Fisher 作为参数重要性度量的出处。
- **TOFU** — Maini et al., 2024。基准本身。
