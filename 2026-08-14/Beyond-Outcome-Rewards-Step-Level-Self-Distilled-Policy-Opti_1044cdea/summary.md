---
title: "Beyond-Outcome-Rewards-Step-Level-Self-Distilled-Policy-Opti"
source: https://arxiv.org/pdf/2608.12764v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:06:33"
field: "多轮搜索代理训练"
keywords: ["reinforcement learning", "self-distillation", "search agents", "process supervision", "GRPO", "tool use"]
innovations: ["提出Evidence Anchors作为开放搜索场景的步级特权信息，避免传统参考答案依赖", "将自蒸馏信号转化为步级优势权重而非优化目标，解耦更新方向与幅度", "证明step-level粒度（完整搜索动作）显著优于token-level粒度"]
benchmarks: ["BrowseComp", "GAIA", "FRAMES"]
---

# 论文速读：Beyond-Outcome-Rewards-Step-Level-Self-Distilled-Policy-Opti

## 一句话总结
本文提出 **Step-Level Self-Distilled Policy Optimization（SSPO）**，通过将自蒸馏信号转化为**步级优势权重**而非直接优化目标，解决了深度搜索代理中奖励过于稀疏、特权信息泄露导致学生"走捷径"的核心矛盾；在 Qwen3-8B 上于 BrowseComp、GAIA、FRAMES 三个基准上均持续超越 GRPO，以约 5% 的额外计算开销实现两倍于 GRPO 的训练效率。

## 研究问题与动机
- **奖励稀疏导致信用分配困难**：深度搜索代理的轨迹通常包含 20+ 步操作，但标准 RL 仅在轨迹结束时给出单一二元结果奖励，无法有效指导哪些搜索步骤是关键的。
- **现有自蒸馏方法直接迁移至搜索场景存在致命缺陷**：在数学/代码等单轮推理任务中可行的自蒸馏，在开放网页搜索中因教师（含特权信息）与学生（纯探索）行为分布差异巨大（教师平均 3.5 步 vs 学生 17.7 步），直接蒸馏会导致学生模仿教师的"捷径行为"，过早放弃工具调用。
- **开放搜索缺少天然的特权信息**：与数学任务有参考解答、代码任务有编译错误不同，开放搜索没有现成的特权信息，构造既 informative 又 safe to distill 的标注是一个开放挑战。
- **粒度不匹配**：Token 级信号与搜索代理"信息寻求动作"的自然单位不对齐，无法有效提供过程监督。

## 核心贡献（创新点）
- **提出 Evidence Anchors（证据锚点）**：用 SOTA LLM 从网页中为每个 QA 对提取紧凑的、步级粒度的证据片段作为特权信息，既不暴露完整答案路径，又能提供过程监督信号——与已有工作（使用参考答案或执行反馈作为特权信息）的本质区别在于适配了开放搜索无现成特权信息的场景。
- **提出 SSPO（步级自蒸馏策略优化）**：将教师-学生分歧转化为 GRPO 中的步级优势权重，仅应用于错误轨迹，解耦了"更新方向"（由结果奖励决定）与"更新幅度"（由教师信号调节）——与直接优化 KL 散度的 OPSD/RLSD 的本质区别在于避免了特权信息泄露导致的分布塌陷。
- **证明 step-level 优于 token-level**：系统性地消融表明，以完整搜索动作（thought + tool call）为单位计算优势权重，优于逐 token 粒度，使熵曲线更稳定、训练更有效。
- **在三个主流搜索基准上验证高效性**：SSPO 训练 100 步即超越 GRPO 训练 200 步，额外开销仅约 5%（一次额外的前向传播），实现了优异的样本效率。

## 方法详解
- **整体框架**：基于 ReAct 范式的搜索代理，在每个时间步 $\tau$ 生成思考 token $\mathtt{t}_\tau$ 和工具调用 $\mathtt{a}_\tau$，联合构成一个 agentic MDP 中的 action。
- **Evidence Anchors 构造**：用 Prompt 驱动 SOTA LLM 为每个 QA 对提取支持正确答案的关键证据片段（平均每个问题 5.24 个锚点），并为教师模型同时提供学生的错误答案作为上下文，防止教师重复相同错误。
- **步级优势权重公式**：
  - 对轨迹中每个步 $\tau$，计算教师-学生对数联合似然比：$\Delta_\tau^{\text{step}} = \text{sg}(\log \frac{P_T(\mathbf{t}_\tau, \mathbf{a}_\tau)}{P_S(\mathbf{t}_\tau, \mathbf{a}_\tau)})$
  - 步级权重：$w_\tau^{\text{step}} = \min(\exp(\text{sign}(A^{(i)}) \cdot \Delta_\tau^{\text{step}}), 1 + \epsilon)$
  - 修改后的优势：$\hat{A}_t^{(i)} = w_t A_t^{(i)}$（仅当 $R_{final} < 1$ 时应用）
- **设计要点**：
  - **stop-gradient** 避免教师被意外更新；
  - **仅应用于错误轨迹**，正确轨迹保持原始 GRPO 更新，保留多样性；
  - **$\epsilon = 0.2$** 上限裁剪防止极端梯度；
  - 教师每 50 步用当前学生参数重新初始化（EMA/同步策略）。
- **两种特权信息的互补作用**：Evidence Anchors 主要提供中间搜索步骤的评估信号（81.1% 步骤一致率），错误答案反馈主要用于放大最终步的惩罚（92.4% 轨迹一致率）。

## 实验与结果
- **模型与数据**：基于 Qwen3-8B，冷启动用 ~4000 条 WebExplorer 轨迹，On-Policy 学习用 DeepForge 中 ~6000 个 English QA 对（难度比 1.5:3.5:3.5:1.5）。
- **评估基准**：BrowseComp（OpenAI）、GAIA text-only subset、FRAMES，采用 Avg@4 度量（temperature=0.6, top-p=0.95）。
- **主要结果（Table 3）**：
  - Cold-Start: BrowseComp 11.7, GAIA 44.8, FRAMES 67.2, **平均 41.2**
  - GRPO: 13.6, 47.3, 69.8, **平均 43.6**（+2.4 相对冷启动）
  - **SSPO: 15.7, 49.3, 73.0, 平均 46.0**（+4.8 相对冷启动，显著优于 GRPO）
  - SSPO 在三项基准上全面超越 GRPO；训练 100 步即超越 GRPO 200 步的结果（Figure 1）。
- **效率**：额外开销仅 ~5% 的训练时间（一次教师 forward pass），平均搜索步数 20.5 vs GRPO 20.0（几乎持平）。
- **消融（Table 4）**：OPSD 直接蒸馏仅 11.8（低于 GRPO 的 12.8），Token-level RLSD/SRPO 适配为 11.8，SSPO（step-level）达 14.5，证明粒度与权重化设计均关键。
- **消融（Table 5）**：Evidence Anchors 单独即达 14.0，加上错误答案反馈提升至 14.5。

## 相关工作脉络
- **GRPO（DeepSeekMath/DeepSeek-R1 系列）**：搜索代理训练的默认基线，仅依赖结果奖励；SSPO 在其基础上叠加步级过程监督，但通过权重化而非直接优化避免信息泄露。
- **OPSD / Self-Distilled Reasoner（Zhao et al. [63]）**：在数学推理上的自蒸馏方法，依赖参考答案作为特权信息；本文将其扩展到无现成特权信息的开放搜索场景，并提出 Evidence Anchors 替代方案。
- **RLSD（Yang et al. [54]）**：指出直接 OPSD 会导致特权信息泄露，提出将蒸馏信号转为 advantage weights；本文沿此思路，但进一步将粒度从 token-level 提升到 step-level，并针对搜索代理做专门设计。
- **SRPO（Li et al. [17]）**：发现仅对错误轨迹应用自蒸馏信号效果更佳；本文直接沿用这一设计原则。
- **CriticSearch [62] / PPR [51] / SmartSearch [45]**：均引入外部 LLM 作为 per-step 评估器提供过程奖励；本文与之本质区别在于不依赖额外评分模型，而是通过自蒸馏信号直接获得步级权重。
- **WebSailor [20] / WebExplorer [26] / WebThinker [22]**：近期大规模搜索代理训练工作；本文在同一 scaffold（Qwen3-8B + ReAct）上证明 SSPO 相对于 GRPO 的增益，展示了过程监督的有效性。

## 局限性与未来方向
- **数据规模有限**：受限于 API 成本（Serper、Jina、LLM 服务），冷启动和 On-Policy 训练均仅使用数千条样本。
- **模型规模受限**：实验仅覆盖 8B 模型，未验证更大规模模型上的泛化性。
- **语言单一**：教师模型（GPT-OSS-120B）的 English-only chain-of-thought 导致训练和评估均限于英文，未涉及中文等多语言场景。
- **未来方向**：扩展至更多样化的代理场景、更大数据规模、更大模型，以及多语言设置；探索细粒度监督对现有 RLVR 方法的通用增益。

## 研究启发与可借鉴点
- **奖励方向与幅度解耦**：将结果奖励决定更新方向、教师信号调节更新幅度，是避免特权信息泄露的通用设计原则，可迁移至其他需要过程监督但缺乏完美标定的 RL 场景。
- **粒度对齐任务结构**：以"完整搜索动作"（thought + tool call）为单位而非逐 token 提供监督，是方法成功的关键；这对其他 multi-step agentic 任务（代码生成、数学证明）有借鉴价值。
- **特权信息构造策略**：Evidence Anchors 的构造方式（SOTA LLM + URL 验证 + 错误答案反馈）可推广到其他无现成参考答案的开放域任务。
- **仅对错误轨迹施加过程监督**：保留正确轨迹的原始 GRPO 更新以维持行为多样性，是一个简单但有效的训练技巧，可与其他 RLVR 方法结合。
- **低开销高收益**：仅 5% 额外前向传播开销即实现接近两倍于 GRPO 的样本效率，为资源受限场景下的高效过程监督提供了可行范式。

## 关键术语表
- **SSPO（Step-Level Self-Distilled Policy Optimization）**：本文提出的方法，将自蒸馏信号转化为步级优势权重，仅应用于错误轨迹，用于深度搜索代理训练。
- **Evidence Anchors（证据锚点）**：从网页中提取的紧凑步级证据片段，作为搜索代理自蒸馏中的特权信息，不暴露完整答案路径。
- **On-Policy Self-Distillation（OPSD）**：学生通过自身 rollouts 生成数据，同时用教师 logits 提供 token 级监督的学习范式；本文将其扩展到多轮搜索场景。
- **GRPO（Group Relative Policy Optimization）**：通过组内响应间的奖励比较来估计优势值，无需 critic 网络的策略梯度算法，是当前搜索代理训练的主流基线。
- **RLVR（Reinforcement Learning with Verifiable Rewards）**：用可验证奖励（如答案正确性）进行强化学习训练的大模型训练范式。
- **Privileged Information（特权信息）**：仅在教师侧可用的额外信息（如参考答案），用于增强教师分布；直接暴露给学生会导致信息泄露。
- **Step-level vs Token-level**：监督粒度——step-level 以完整搜索动作为单位计算权重，token-level 逐 token 计算；前者与搜索行为的自然单位对齐。
- **ReAct Paradigm**：将推理（Thought）与行动（Action/Tool Call）交替进行的代理交互范式，是本文搜索代理的基础架构。

## 可复现要素
- **数据集**：DeepForge [64] 中采样的 ~6000 个 English QA 对（公开），冷启动轨迹 ~4000 条（通过 WebExplorer [26] 管道收集）；Evidence Anchors 基于 DeepSeek-V3.2 构造。
- **代码/权重**：论文未明确声明代码开源状态，模型基于 Qwen3-8B（开源权重可获取）。
- **关键超参**：learning rate = 1e-6，batch size = 64，rollouts per question = 8，$\epsilon = 0.2$（权重裁剪上限），teacher 每 50 步重新初始化，max context length = 128K，max agent steps = 100。
- **工具**：Serper API（搜索）、Jina（browse 内容提取）。
- **环境**：两个 8-GPU 节点（各 140GB 显存）。
