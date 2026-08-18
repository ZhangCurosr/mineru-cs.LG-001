---
title: "Beyond-Single-Turn-Confidence-Trajectory-Adapted-Uncertainty"
source: https://arxiv.org/pdf/2608.11552v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 07:31:09"
field: "大语言模型 Agent 不确定性感知"
keywords: ["uncertainty quantification", "LLM agents", "trajectory-level confidence", "tool-use agents", "self-consistency", "white-box scoring", "reflexive scoring"]
innovations: ["首次系统比较白盒/黑盒/反思式三类UQ方法在多轮Agent轨迹上的可迁移性", "提出轨迹等效率(TER)——基于独立LLM judge的结果等效判定，和6种动作结构一致性度量", "揭示白盒评分对跨轮聚合策略的极端敏感性（同base score不同聚合可造成>0.5 AUROC swing）"]
benchmarks: ["BFCL-v4", "τ²-bench (airline, telecom, retail)"]
---

# 论文速读：Beyond-Single-Turn-Confidence: Trajectory-Adapted Uncertainty Quantification for LLM Agents

## 一句话总结
本文系统评估了三种单轮 LLM 不确定性感知（UQ）方法族——白盒（token概率）、黑盒（轨迹一致性采样）、反思式（模型自评）——在多轮工具调用 Agent 轨迹层面的可迁移性，发现**轨迹等效率（TER）和行动集一致性（ASC）等黑盒方法整体最强，反思式是低成本实用基线，白盒对跨轮聚合策略极度敏感**，无单一方法在所有模型/数据集上稳定有效。

---

## 研究问题与动机

- **单轮 UQ 无法直接迁移到 Agent 轨迹**：现有 UQ 方法将置信度绑定在单次生成答案上，而 Agent 的观测单位是交互式轨迹（多轮工具调用、澄清提问、状态更新），误差会沿轨迹传播并影响最终结果。
- **误差累积与早期失误的级联效应**：Agent 在"何时澄清 vs 何时行动"等决策点均存在不确定性，单轮置信度无法捕捉这种跨轮传播。
- **现有工作缺乏系统跨模型对比**：虽有 SAUP（Zhao et al., 2025）、UProp（Duan et al., 2025）等轨迹级传播方法，以及 Oh et al. (2026) 的形式化框架，但尚无工作在同一条件下系统比较三类方法在多个 LLM 和多数据集上的表现。
- **部署安全性需求**：可靠的 UQ 信号使 Agent 能在执行有害或不可逆操作前选择退出或升级至人工，但当前信号是否足以作为安全闸门尚不明确。

---

## 核心贡献（创新点）

1. **首个系统性的多轮 Agent UQ 经验对比**：在 5 个 LLM × 4 个多轮工具调用数据集的完整网格上，统一评估白盒、黑盒、反思式三类方法，填补了跨模型/跨域对比空白。
2. **引入轨迹适配的黑盒一致性度量体系**：提出 6 种新的一致性评分器（NCP、FAC、ASC、ADC、AEC、TER），其中 TER 使用独立 LLM judge（gemini-flash-lite）进行轨迹结果等效判定，是新颖的"轨迹等效率"概念。
3. **揭示白盒评分对跨轮聚合策略的极端敏感性**：同一 base score（如 LNSP）在不同聚合规则（first/mean/min/last/early/late）下 AUROC 可相差超过 0.5，证明"随意选取聚合策略会导致错误结论"。
4. **发现黑盒自一致性整体最强但存在"卡死策略"陷阱**：TER 和 ASC 在多数据集上居首，但 failure-failure 配对仍保留中等一致性（0.40），说明一致性可能仅反映"重复失败的同一策略"而非正确性。
5. **提供计算预算-判别力权衡的量化分析**：通过 Table 2 对比三类方法在额外轨迹数、辅助模型调用数和 AUROC 上的 tradeoff，为实际部署提供选型依据。

---

## 方法详解

### 问题设定（Oh et al., 2026 形式化）
- 一个 episode 为 $T$ 步轨迹 $F_{\leq T}$，第 $i$ 步发出动作 $A_i$（工具调用或用户消息）并接收观察 $O_i$。
- 二元奖励 $r(F_{\leq T}) \in \{0, 1\}$ 记录轨迹成功，目标是用置信度分数 $C: \mathcal{F} \to [0, 1]$ 排序成功轨迹优先。

### 3.1 白盒评分器（White-Box）
基于每步动作 span（committed action tokens）的 token 概率，跨轮聚合：

- **SP**（Sequence Probability）：$\mathrm{SP}(y) = \prod_{j=1}^{L} p_j$，长度惩罚严重。
- **LNSP**（Length-Normalized SP）：$\mathrm{LNSP}(y) = (\prod p_j)^{1/L}$，per-token 几何平均。
- **ATN@K**（Average Token Negentropy@K）：$= \frac{1}{L}\sum_j (1 - \mathrm{TE}@K(t_j)/\log K)$，基于 top-K 截断熵。
- **PM**（Probability Margin）：$\frac{1}{L}\sum_j (p_{j,1} - p_{j,2})$，最大两token概率差。

**跨轮聚合策略**：$g_{\text{mean}}$（算术平均）、$g_{\text{min}}$（取最小值，对应最不确定步规则）、$g_{\text{first}}$、$g_{\text{last}}$、$g_{\text{early}}$（早期加权）、$g_{\text{late}}$（晚期加权）。

### 3.2 黑盒一致性评分器（Black-Box Consistency）
采样 $m=3$ 条额外轨迹 $\tilde{\mathbf{F}}$，测量与原轨迹的一致性：

- **NCP**（Non-Contradiction Probability）：使用 DeBERTa-large-MNLI 计算参考与采样终消息的双向 NLI 矛盾概率均值：$\mathrm{NCP}(y;\tilde{\mathbf{y}}) = 1 - \frac{1}{m}\sum_j \frac{p_c(y,\tilde{y}_j)+p_c(\tilde{y}_j,y)}{2}$。
- **FAC**（First-Action Consistency）：首行动作类型匹配率，保证 prefix-aligned 比较。
- **ASC**（Action-Set Consistency）：动作类型集合 Jaccard 相似度，对顺序和重复不敏感。
- **ADC**（Action-Distribution Consistency）：动作频率分布的 1-JSD（Jensen-Shannon 散度），保留重复信息。
- **AEC**（Action-Edit Consistency）：$1 - \text{normalized Levenshtein距离}$，对顺序敏感。
- **TER**（Trajectory Equivalence Rate）：用 gemini-flash-lite judge 判定采样轨迹与参考轨迹是否达成相同任务结果，judge 仅见 transcript、不知 hidden reward。

### 3.3 反思式评分器（Reflexive）
对最终动作前的 transcript prefix 单独请求模型自评一次：

- **P(True)**：要求模型输出"True"/"False"，取"True" token 概率（Kadavath et al., 2022）。
- **VC**（Verbalized Confidence）：要求模型输出 Yes/No 及 0–1 概率，Yes 时取所述概率，No 时取补数（Tian et al., 2023）。

---

## 实验与结果

### 实验设置
- **数据集**：BFCL-v4 多轮子集（200 任务）+ τ²-bench 三个文本域（airline 50、telecom 114、retail 114），二元成功标签。
- **模型**：Qwen2.5-7B、gpt-oss-20b、Qwen3.5-9B、MiniMax-M3（Together AI）+ gpt-4o-mini（OpenAI）。
- **协议**：greedy 轨迹（T=0）+ 3 条采样轨迹（T=0.7）；NLI 用 microsoft/deberta-large-mnli；TER judge 用 gemini-flash-lite。
- **评估指标**：AUROC（主）、AUPRC、PRR（预测拒绝比）、ECE（期望校准误差）。

### 核心结果（AUROC，Table 1）

| 方法族 | 均值 AUROC（峰值） | 关键发现 |
|---|---|---|
| **白盒** | 0.628 / **0.725**（SP_mean, Qwen3.5-9B/telecom） | LNSP_min 在 retail/gpt-4o-mini 达 0.72，但在 telecom 骤降至 0.24；91/240 格 < 0.5；SAUP-inspired 传播控制平均仅 0.434（< 0.53 的 SP_last） |
| **反思式** | 0.691 / **0.885**（VC, gpt-oss-20b/BFCL-v4） | BFCL-v4 表现最佳（P(True) 0.75–0.85）；airline 跨模型差异极大（P(True) 0.225–0.659），不跟踪任务成功率 |
| **黑盒一致性** | 0.705 / **0.849**（ASC）；消息级 0.686 / **0.868**（TER, Qwen3.5-9B/airline） | TER 在 airline/Qwen3.5-9B 达 0.868；ASC 在 BFCL-v4 多个模型居首；但 gpt-4o-mini/telecom 仅 0.342（11% 成功率导致 judge 比较多为失败-失败对） |

### 稳定性诊断
- **白盒聚合不稳定**：LNSP 在不同聚合规则下 AUROC  swings 极大（Figure 1）；mean 风格在 airline 几乎全部坍缩。
- **一致性采样数敏感**：TER AUROC 从 m=1（0.63）升至 m=3（0.68），airline Qwen3.5-9B 从 0.76 升至 0.87；但 Qwen2.5-7B/airline 随 m 增大反而下降（卡死策略）。
- **Failure-failure 一致性陷阱**（Appendix G）：failure-failure 配对的 TER 为 0.40（全 τ²-bench），AEC 为 0.52，说明一致性可能仅反映"重复同一错误策略"。
- **文本动作接口 vs 原生工具调用**（Appendix I）：gpt-4o-mini/retail 上 TER 0.759→0.761、AEC 0.678→0.681，结果跨接口稳定；但白盒评分依赖文本动作接口的 token 可观测性。
- **Agent-only prefix ablation**（Appendix H）：固定 transcript、仅重采样下一步动作，最高 AUROC 仅 0.588，说明下游轨迹动态贡献了额外信号。

---

## 相关工作脉络

1. **单轮 UQ 基础**：Kadavath et al. (2022) P(True)；Manakul et al. (2023) SelfCheckGPT；Farquhar et al. (2024) Semantic Entropy；Lin et al. (2024) GWC——本文证明这些方法在 Agent 轨迹上表现参差不齐。
2. **Oh et al. (2026) 轨迹级 UQ 形式化**：首次定义 trajectory-level 联合不确定性，本文在其框架基础上扩展为完整的三类方法对比实验。
3. **SAUP（Zhao et al., 2025）与 UProp（Duan et al., 2025）**：分别提出基于情境权重的步级不确定传播和分解当前/继承不确定性；本文的 SAUP-inspired 控制仅在 τ²-bench 平均 0.434 AUROC，远低于反射式和一致性方法。
4. **ULA（Han et al., 2024）与 ProbeCal（Liu et al., 2024）**：将 UQ 嵌入 thought-action-observation 循环或用于校准；本文关注事后分类性能而非在线决策。
5. **Agentic UQ（Zhang et al., 2026b）与澄清导向 Agent（Suri et al., 2026）**：近期工作将 UQ 作为控制信号；本文提供对比基线和实证参考。
6. **Xuan et al. (2026) 工具使用中的过度自信**：发现 evidence 工具诱导过自信、verification 工具可缓解；本文结果与其互补，强调 UQ 方法本身也需跨部署验证。

---

## 局限性与未来方向

- **基准范围有限**：仅覆盖 BFCL-v4 多轮子集和 τ²-bench 三个文本域，web browsing、embodied agents 或更开放任务未测试。
- **用户模拟器混淆**：部分失败源于模拟器而非 Agent；全轨迹采样衡量的是"联合 agent-user 系统"变异性而非 Agent 单独不确定性（prefix ablation 已部分诊断）。
- **动作接口局限**：白盒方法依赖文本动作接口以获取跨模型可比的 token 概率；原生工具调用虽在 retail/gpt-4o-mini 上一致，但并非所有模型/Provider 均支持。
- **统计功效不足**：airline 仅 50 任务，bootstrap 区间中位半宽 0.10，列内排名多为描述性结论。
- **Judge 依赖**：TER 依赖单一外部 judge（gemini-flash-lite），虽经数据库 ground truth 验证（precision 0.88, recall 0.93），但未测试所有域或更远距离的 judge。
- **未完成轨迹决策验证**：本文仅评估事后分类性能，未测试"在执行中基于 UQ 信号主动退出/澄清/升级"是否能真正改善结果。
- **观察不确定性与推理 token 未覆盖**：仅对 committed action span 计算白盒分数，observation 不确定性和 reasoning  token 的 UQ 留待未来。

---

## 研究启发与可借鉴点

1. **跨轮聚合策略需显式验证**：白盒方法不是"即插即用"，不同聚合规则（min vs mean vs last）可产生相反排序；建议在实际部署前进行 per-domain 聚合策略选择实验。
2. **TER 的轨迹等效 judge 设计可迁移**：使用独立轻量 judge 判定"结果等效"而非"文本相同"，能有效捕捉不同工具调用路径达成相同目标的场景，适用于多路径可达性的评估。
3. **failure-failure 一致性陷阱的识别机制**：将配对一致性按 success-success / success-failure / failure-failure 分组诊断（Appendix G 的方法），可区分"高一致性=稳健"与"高一致性=卡死"，对后续研究有直接参考价值。
4. **反射式作为零额外轨迹的低成本基线**：P(True)/VC 仅需一次额外前向 pass，在多数模型-数据集组合上表现稳健，适合计算预算受限的部署场景。
5. **文本动作接口 vs 原生工具调用的对照设计**：Appendix I 的接口消融实验展示了如何在保持实验公平性的同时评估接口敏感性，方法论可推广到其他 Agent benchmark 研究。

---

## 关键术语表

**Uncertainty Quantification (UQ)**：量化 LLM 输出不确定性的方法体系，本文聚焦轨迹级置信度估计。

**Trajectory（轨迹）**：Agent 与环境的完整交互序列，包含多轮动作 $A_i$ 和观察 $O_i$，是 Agent UQ 的基本观测单位。

**White-box Scorer（白盒评分器）**：利用模型内部 token 概率计算的置信度分数，需访问 logprobs。

**Black-box Consistency Scorer（黑盒一致性评分器）**：通过多次随机采样轨迹并测量结果一致性来估计不确定性，无需模型内部概率。

**Reflexive Scorer（反思式评分器）**：要求模型对自身已完成的轨迹 transcript 进行置信度自评。

**Trajectory Equivalence Rate (TER)**：使用独立 LLM judge 判定采样轨迹与参考轨迹是否达成相同任务结果的比率，是本文最强的黑盒一致性变体。

**Action-Set Consistency (ASC)**：基于动作类型集合 Jaccard 相似度的轨迹一致性度量，对顺序和重复不敏感但计算廉价。

**Prediction Rejection Ratio (PRR)**：选择性预测指标，衡量按置信度排序后拒绝低置信样本带来的成功率提升（相对随机拒绝）。

---

## 可复现要素

- **数据集**：BFCL-v4（Apache-2.0）和 τ²-bench（MIT）均公开可用，多轮子集和 retail/airline/telecom 域均可复现。
- **代码**：论文未明确声明开源代码仓库（参考 Bouchard & Chauhan, 2025 的 uqlm Python 包，但本文实验代码未列明）。
- **权重**：使用 Together AI 托管的 Qwen2.5-7B、gpt-oss-20b、Qwen3.5-9B、MiniMax-M3 及 OpenAI gpt-4o-mini；NLI 用 microsoft/deberta-large-mnli；TER judge 用 gemini-flash-lite。
- **关键超参**：greedy T=0，采样 T=0.7，采样数 m=3，top-K=5（ATN@K），用户模拟器固定为 gpt-4.1-mini T=0。
- **评估指标**：AUROC（主）、AUPRC、PRR、ECE；bootstrap 1000 resamples，任务聚类。

---
