---
title: "Beyond-Single-Turn-Confidence-Trajectory-Adapted-Uncertainty"
source: https://arxiv.org/pdf/2608.11552v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 07:31:05"
field: "大语言模型代理系统的安全与可靠性"
keywords: ["uncertainty quantification", "LLM agents", "trajectory-level confidence", "tool-use agents", "self-consistency", "white-box scoring", "reflexive confidence"]
innovations: ["系统评估三类单轮UQ方法在多轮Agent轨迹上的迁移效果，发现迁移性参差不齐", "提出轨迹级黑盒一致性度量（TER/ASC/ADC/AEC/FAC），揭示一致性陷阱", "揭示白盒方法对跨轮聚合策略的高度敏感性并提供成本-性能权衡指南"]
benchmarks: ["BFCL-v4", "τ²-bench (airline, telecom, retail)"]
---

# 论文速读：Beyond-Single-Turn-Confidence-Trajectory-Adapted-Uncertainty

## 一句话总结
本文系统评估了三种主流单轮不确定性量化（UQ）方法（白盒概率、黑盒一致性、反射式自评）在多轮工具调用 Agent 轨迹上的迁移效果，发现其迁移性参差不齐：白盒方法高度依赖跨轮聚合策略的选择，反射式方法是最强低成本基线，而黑盒自一致性整体最强，其中轨迹等价率（TER）和行动集一致性（ASC）表现最佳。

## 研究问题与动机
- **核心问题**：现有 LLM 的不确定性量化方法主要在单轮输出上评估，而 Agent 的运行单元是交互轨迹（多轮工具调用、状态更新、中间决策），早期错误会传播至最终结果，单轮 UQ 能否直接迁移到轨迹级需要实证检验。
- **现有方法不足**：现有 Agent UQ 工作（如 SAUP、UProp）聚焦不确定性传播建模，但缺少在同一框架下系统比较白盒、黑盒、反射式三类方法在多模型、多数据集上的表现，以及选择何种聚合/一致性度量对结果的影响。
- **评估缺口**：三类 UQ 方法从"单个生成响应"迁移到"交互式轨迹"后，各自在判别力（AUROC）、校准性（ECE）、选择性预测（PRR）上的退化模式不同，缺乏系统性对照实验。
- **实践价值**：为 Agent 部署时的 UQ 方法选型提供实证依据，帮助团队根据计算预算和部署场景选择最可靠的不确定性信号。

## 核心贡献（创新点）
- **系统性的轨迹级 UQ 基准评估**：在 5 个 LLM 和 4 个多轮工具调用数据集（BFCL-v4、τ²-bench 的 airline/telecom/retail）上统一评估白盒、黑盒、反射式三类 UQ 方法的轨迹成功预测能力，填补了 Agent UQ 领域缺少对照实验的空白。
- **提出多种轨迹适配的黑盒一致性度量**：将单轮一致性扩展到轨迹结构层面，设计了轨迹等价率（TER，基于 LLM judge）、行动集一致性（ASC）、行动分布一致性（ADC）、行动编辑一致性（AEC）和新颖的首动作一致性（FAC），弥补了此前仅在最终消息层面做一致性比较的局限。
- **揭示白盒方法对聚合策略的高度敏感性**：发现相同的白盒基础分数（如 LNSP）在不同跨轮聚合规则（first/mean/min/last/early/late）下 AUROC 可从 0.72 骤降至 0.23，证明白盒置信度不可作为开箱即用的方法。
- **提供成本-性能权衡的综合分析**：通过 Table 2 对比三类方法的额外计算开销（白盒零额外调用、反射式一次自评估、一致性需 m 次采样轨迹），为不同部署预算下的方法选型提供实用指导。
- **发现"固执策略"一致性陷阱**：通过失败-失败配对诊断（Appendix G），揭示了当模型始终重复相同错误计划时，一致性分数可能虚假偏高，说明仅凭一致性不足以区分"反复正确"和"反复错误"。

## 方法详解
- **轨迹设定**：遵循 Oh et al. (2026) 的 Agent UQ 形式化，一个 episode 是长度为 T 的轨迹 F≤T，每轮 i 发出动作 A_i（工具调用或用户消息）并接收观测 O_i，轨迹结束后二元奖励 r(F≤T) 记录成功。
- **白盒评分器（White-Box）**：
  - 单轮基础分数：序列概率 SP(y)=∏p_j、长度归一化序列概率 LNSP(y)=(∏p_j)^(1/L)、平均 token 负熵 ATN@K(y)=1/L Σ_j TN@K(t_j)，均在动作 span（committed action tokens）上计算。
  - 跨轮聚合：将每轮分数 s_i 通过 first（首轮）、mean（均值）、min（最小）、last（末轮）、early/late（位置加权）聚合为轨迹级分数。
- **黑盒一致性评分器（Black-Box Consistency）**：对参考轨迹采样 m=3 条替代轨迹 F̃，比较一致性：
  - 最终消息一致性 NCP：用 DeBERTa-large-MNLI 计算参考与采样最终消息之间的非矛盾概率均值。
  - 行动结构化一致性：FAC（首动作类型匹配率）、ASC（行动类型集合 Jaccard 相似度）、ADC（行动类型分布 Jensen-Shannon  divergence 补）、AEC（行动类型序列归一化编辑距离补）。
  -  judged 一致性 TER：用 gemini-flash-lite 作为 judge 判断两条完整轨迹是否达成任务等价的结果（只看不带 hidden instruction 的 transcript）。
- **反射式评分器（Reflexive）**：
  - P(True)：让模型对轨迹生成 "True"/"False"，取 "True" token 概率作为置信度。
  - Verbalized Confidence (VC)：让模型输出 Yes/No 判断及 0–1 概率，置信度为声明概率（Yes 时取原值，No 时取补）。
  - 两者均在最终动作之前的 transcript prefix 上评估，不利用后续用户确认信息。
- **评估指标**：主指标 AUROC，辅以 AUPRC、选择性预测风险覆盖率（PRR）和期望校准误差（ECE），使用 95% 任务聚类 bootstrap 区间。

## 实验与结果
- **数据集**：BFCL-v4 多轮子集（200 tasks）、τ²-bench 的 airline（50）、telecom（114）、retail（114），均以二元成功标签评估。
- **模型**：Qwen2.5-7B、gpt-oss-20b、Qwen3.5-9B、MiniMax-M3（Together AI）、gpt-4o-mini（OpenAI）。
- **关键结果（AUROC）**：
  - **白盒方法**：BFCL-v4 表现最佳（ATN@5_mean 达 0.71），但在 τ²-bench 上极不稳定——LNSP_mean 在 airline 上接近或低于随机（0.17–0.41），同一基础分数切换聚合规则可使 AUROC 从 0.725 骤降至 0.228（SP_mean，Qwen3.5-9B/telecom）。240 个白盒单元格中 91 个低于 0.5。
  - **反射式方法**：BFCL-v4 最强（P(True) 0.746–0.852，VC 0.69–0.88），telecom 上 P(True) 达 0.852（gpt-4o-mini）；airline 上差距最大（P(True) 0.225–0.659，VC 0.40–0.70）。
  - **黑盒一致性**：整体最强家族，TER 在 airline 上达 0.868（Qwen3.5-9B），ASC 在 airline 达 0.819（gpt-oss-20b）；但 TER 在 gpt-4o-mini/telecom（11% 成功率）降至 0.342。整体无单一方法在所有设置下占优。
  - **SAUP 启发式传播控制**：在 15 个 τ²-bench 单元格上平均 AUROC 仅 0.434，低于最优固定朴素聚合（0.53）。
- **最强结果**：反射式 VC 在 BFCL-v4/gpt-4o-mini 达到 0.88 AUROC；TER 在 airline/Qwen3.5-9B 达到 0.868 AUROC。

## 相关工作脉络
- **Oh et al. (2026)**：首次形式化 Agent 轨迹级不确定性，本文在其框架基础上扩展了跨轮聚合策略和多种黑盒一致性变体，填补了系统实证比较的空白。
- **SAUP (Zhao et al., 2025) / UProp (Duan et al., 2025)**：前者用情境依赖权重传播步级不确定性，后者区分当前决策不确定性和继承不确定性；本文将其 adapted 为 action-level 传播控制并验证，发现其对 τ²-bench 效果不佳（AUROC ~0.43）。
- **单轮 UQ 经典方法**：SelfCheckGPT (Manakul et al., 2023)、Semantic Entropy (Farquhar et al., 2024)、RED-CT (Farr et al., 2025) 等均在静态生成任务上验证，本文系统检验了它们向 Agent 轨迹的迁移边界。
- **Agent UQ 近期工作**：UALA (Han et al., 2024) 将不确定性纳入 thought-action-observation 循环；ProbeCal (Liu et al., 2024) 用执行轨迹校准工具调用 Agent；Agentic UQ (Zhang et al., 2026b) 用 verbalized uncertainty 引导记忆与反思。本文定位为通用评估而非方法设计。
- **多轮对话置信度**：Zhang et al. (2026a) 研究多轮对话中的逐轮校准与信息累积单调性；本文聚焦工具调用 Agent 而非纯对话场景，评估目标为轨迹成功而非逐轮置信度。

## 局限性与未来方向
- **数据集范围有限**：仅覆盖 BFCL-v4 多轮子集和 τ²-bench 的三个文本域，未测试 web browsing、embodied agent 或其他更开放的 Agent 环境。
- **模拟器混淆**：轨迹在 simulated user（gpt-4.1-mini, T=0）下生成，部分失败源于模拟器而非 Agent 本身；一致性评分器测量的是 agent-user 联合系统的变异性。
- **动作接口限制**：白盒方法依赖 text-action 接口以暴露 action-token log probabilities；native tool-calling 在各模型上支持不统一，文中仅用 gpt-4o-mini/retail 做了 ablation（附录 I），结论不能直接推广到所有模型。
- **统计功效不足**：部分数据集任务数少（如 airline 仅 50 任务），bootstrap 区间宽（中位半宽 0.10），列内排名多为描述性结论而非统计显著。
- **judge 依赖性**：TER 依赖单一辅助 judge 模型（gemini-flash-lite），虽然与 gemini-flash 替换时 91% 一致且 AUROC 变化 ≤0.062，但未测试更多 judge 变体。
- **未来方向**：建议探索步级标签（按动作类型/工具家族分层评估）、在推理过程中利用逐轮分数进行主动干预（abstention/clarification/escalation）、在更多样化的 Agent 环境中验证泛化性。

## 研究启发与可借鉴点
- **聚合策略调优是关键**：白盒方法并非不可用，但必须针对特定模型-数据集组合选择聚合规则（min/mean/early-weighted 等在 airline/retail/telecom 上表现截然不同），不可简单移植。建议后续工作研究自动化聚合策略选择或学习型聚合。
- **反射式评分是高 ROI 基线**：仅需一次额外 pass 过 transcript，即可获得 0.75–0.88 AUROC，适合计算预算有限的部署场景；可考虑结合领域 policy 定制 prompt 以提升校准性。
- **一致性需区分"成功-成功"与"失败-失败"**：Appendix G 揭示的一致性陷阱——模型反复犯同样错误时也会产生高一致性分数。后续研究可在一致性度量中引入任务可解性条件或失败模式感知。
- **采样数量影响显著**：Appendix D 显示 TER 和 ASC/AEC 随 m 增大持续提升（m=1→3 时 TER 从 0.63 升至 0.68），暗示 m>3 可能仍有提升空间，值得系统搜索最优采样预算。
- **与团队方向结合点**：若团队关注 Agent 安全性/可靠性，可将 TER 或 VC 作为 abort/escalation 信号嵌入 Agent loop；若关注工具调用优化，可研究 ATN@K 与工具参数不确定性的关联。

## 关键术语表
- **Uncertainty Quantification (UQ)**：量化 LLM 输出不确定性的方法体系，用于区分高/低置信度预测。
- **White-box scorer**：基于模型内部 token 概率计算的置信度评分器，如 SP、LNSP、ATN@K。
- **Black-box consistency scorer**：通过多次采样并比较输出一致性来估计不确定性的方法，无需访问内部概率。
- **Reflexive scorer**：要求模型对自身轨迹进行评估（如输出 True/False 及概率）的置信度估计方法。
- **Trajectory Equivalence Rate (TER)**：用 LLM judge 判断采样轨迹与参考轨迹是否达成等价任务结果的比例，是本文提出的新颖一致性度量。
- **Action-Set Consistency (ASC)**：比较两条轨迹使用的工具类型集合的 Jaccard 相似度，忽略顺序和重复。
- **Stuck-policy effect**：模型重复相同错误行为时产生虚假高一致性分数的现象，说明一致性不等于正确性。
- **Selective prediction (PRR)**：通过丢弃低置信度预测来提升保留样本成功率的指标，范围 [-1, 1]，正值表示优于随机拒绝。

## 可复现要素
- **数据集**：BFCL-v4（Apache-2.0 许可，通过 Berkeley Function Calling Leaderboard/Gorilla 发布）、τ²-bench airline/telecom/retail（MIT 许可）。论文未提供统一代码库，但所有数据集均可公开获取。
- **代码/权重**：论文未提及开源代码仓库；模型通过 Together AI（Qwen、MiniMax）和 OpenAI API 访问。
- **关键超参**：采样轨迹数 m=3，采样温度 T=0.7，greedy 参考轨迹温度 T=0；白盒 K=5 top logprobs；NLI 模型 microsoft/deberta-large-mnli；TER judge 为 gemini-flash-lite（T=0）；user simulator 为 gpt-4.1-mini（T=0）。
