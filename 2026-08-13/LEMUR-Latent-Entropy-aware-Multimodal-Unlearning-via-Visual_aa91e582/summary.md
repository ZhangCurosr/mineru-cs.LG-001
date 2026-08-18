---
title: "LEMUR-Latent-Entropy-aware-Multimodal-Unlearning-via-Visual"
source: https://arxiv.org/pdf/2608.11691v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:28:56"
field: "多模态大模型隐私安全"
keywords: ["machine unlearning", "multimodal large reasoning models", "reinforcement learning", "entropy dynamics", "inference-time intervention", "reasoning trace privacy", "visual anchoring"]
innovations: ["揭示RL训练MLRMs推理轨迹泄露的token-level两阶段熵签名现象", "提出LEMUR无训练推理时框架，通过熵调制视觉锚点注入重定向推理轨迹", "设计熵增强敏感性开关与动态阶段持续时间机制，精准控制干预窗口"]
benchmarks: ["MLLMU-Bench", "VQAv2"]
---

# 论文速读：LEMUR-Latent-Entropy-aware-Multimodal-Unlearning-via-Visual

## 一句话总结
本文针对RL训练的多模态大推理模型（MLRMs）中推理轨迹仍会泄露隐私信息的问题，提出了LEMUR——一种完全无训练、仅在推理时运行的机器遗忘框架，通过利用token-level熵动力学信号检测敏感推理段，并注入熵调制的视觉锚点来重定向推理轨迹，在显著降低答案和推理轨迹双重泄露的同时保持模型推理能力。

## 研究问题与动机
1. **RL训练引入独特的隐私漏洞**：RL后训练赋予MLRMs探索性思维链（CoT）能力，但这导致即使敏感事实在最终答案中已被遗忘，模型仍可能在`<think>...</think>`推理轨迹中复现该信息，且RL训练的模型比非推理基础模型面临更严重的轨迹泄露。
2. **现有遗忘方法存在双重不足**：答案级清理（如GAdiff、NPO）留下推理轨迹泄露；而激进扰动推理轨迹又会导致推理退化（重复/崩溃），现有方法缺乏对RL训练引入的多样化探索性推理轨迹的可信监控机制。
3. **缺乏解码时干预手段**：现有方法主要作用于权重微调或隐层激活，但无法精准定位推理轨迹中敏感内容的" deliberation → commit"两个阶段，无法在不破坏整体推理能力的情况下精确干预。
4. **熵信号的可检测性**：研究发现RL训练使记忆化属性在解码时呈现独特的两阶段熵特征（高熵 deliberation → 低熵 deterministic recital），这一信号在非推理基础模型中基本不存在，为精准的推理时干预提供了可行依据。

## 核心贡献（创新点）
1. **揭示了RL-MLRMs推理轨迹泄露的熵签名现象**：首次系统刻画了RL训练导致的token-level两阶段熵动力学特征（高熵犹豫→低熵确定复述），并证明这是推理轨迹泄露的关键信号，与基础MLLMs的混沌熵分布形成鲜明对比。
2. **提出LEMUR无训练推理时框架**：不同于现有微调方法（GAdiff、NPO、MMUnlearner）或激活干预方法（R-MUSE），LEMUR完全不修改模型参数，仅在解码时通过熵监控和视觉锚点注入实现推理轨迹重定向。
3. **设计熵增强敏感性开关机制**：结合词汇 forbidden token 概率质量检测与自适应熵阈值（区分 committed recital 与 deliberation 两个阶段），实现比单一词汇检测更精确的敏感段定位。
4. **熵调制的视觉锚点注入策略**：通过凸组合将视觉锚点（图像重 grounding）与安全回答锚点注入到受约束潜反馈中，注入强度$\gamma_t$随当前熵动态调整，高熵阶段强干预、低熵阶段弱干预，避免对已确定文本的扭曲。
5. **动态熵控制阶段持续时间**：使用指数移动平均基线估计自适应退出阈值，结合冷却窗口和最大长度限制，精准控制干预窗口与记忆化span匹配，避免反复振荡。

## 方法详解
**问题设定**：给定MLRM $\mathcal{M}_\theta$、遗忘集$\mathcal{D}_f$和保留集$\mathcal{D}_n$，目标是找到$\hat{\theta}$使遗忘集正确率最小化同时保持保留集性能不变；LEMUR保持$\hat{\theta} = \theta$，纯在推理时干预解码分布。

**关键公式与机制**：

1. **Token级熵计算**：$H_t(v) = -\sum_{v \in \mathcal{V}} p_t(v) \log p_t(v)$，用于度量每一步的不确定性。

2. **熵增强敏感性开关**：$g_t = [P_t^\Phi \geq \rho] \lor [H_t(v) \geq \tau \land P_t^\Phi \geq \rho_{\text{lo}}]$，其中第一项检测 committed recital（低熵确定阶段），第二项检测 deliberation（高熵犹豫阶段），$\rho_{\text{lo}} < \rho$降低高熵阶段的触发阈值。

3. **受约束潜反馈**：移除禁止词元后重新归一化 $\tilde{p}_t(v) = \frac{p_t(v) \mathcal{H}[v \notin \Phi_s]}{\sum_{u \notin \Phi_s} p_t(u)}$，然后计算期望嵌入 $\hat{e}_t = \sum_v \tilde{p}_t(v) \bar{E}[v]$ 作为软反馈。

4. **视觉锚点构建**：$e_{\text{vis}} = \frac{1}{|\mathcal{V}_{\text{vis}}|}\sum_{v \in \mathcal{V}_{\text{vis}}} \bar{E}[v]$（视觉特殊词元平均），$e_{\text{safe}}$ 为安全回答模板平均，组合锚点 $a = \beta e_{\text{vis}} + (1-\beta)e_{\text{safe}}$。

5. **熵调制注入**：$e_t = (1-\gamma_t)\hat{e}_t + \gamma_t a$，其中 $\gamma_t = \min(\gamma_{\text{max}}, \frac{H_t(v)}{\tau}\gamma)$，实现高熵强 steering、低熵弱 steering。

6. **动态阶段持续时间**：使用指数移动平均 $\bar{H}_t = (1-\eta)\bar{H}_{t-1} + \eta H_t(v)$ 作为基线，退出条件 $z_t = [\neg g_t \land H_t(v) \geq \kappa \bar{H}_t] \lor [t - t_0 \geq W_{\max}]$，配合冷却窗口C防止快速振荡。

## 实验与结果
**数据集**：基于 MLLMU-Bench 重构，使用 Qwen3.5-35B-A3B 作为教师模型蒸馏生成带推理链的 QA 对，分为 forget/retain/celebrity 三个 split，支持 classification、fill-in-blank、generation 三种任务。

**评估模型**：R1-Onevision-7B、Vision-R1-7B、OpenVLThinker-7B（均为 RL 训练 MLRM），以及作为可迁移性验证的 Qwen2.5-VL（非推理基础模型）。

**评估指标**：CLS Acc（分类准确率）、FIB Acc（填空准确率）、Gen TR（生成目标召回率）、SRL（Subject-level Reasoning Leakage，推理轨迹泄露率）、RRA（Reasoning Retention Ability，推理保持能力，由 Gemini-2.5-Pro 评估）。

**主要结果**：
- **Forget split（5%遗忘率，R1-Onevision-7B）**：LEMUR 达到 CLS Acc 26.0%、FIB Acc 1.0%、Gen TR 15.1%、SRL 11.3%，显著优于所有基线；R-MUSE 的 SRL 为 29.8%，LEMUR 降低至 11.3%。
- **Retain/Celebrity split**：LEMUR 保持接近 vanilla 水平（CLS Acc 59.1%/67.9%，SRL 57.6%/88.0%），而微调基线（GAdiff、NPO 等）出现可见的 ground 损失。
- **RRA 保持**：LEMUR 在 forget split 上 RRA 为 6.9，接近 vanilla 的 7.7，而 GAdiff 降至 4.3，NPO 为 6.2。
- **跨模型泛化**：在 Vision-R1-7B（5%/10%/15% 遗忘率）和 OpenVLThinker-7B 上均复现相同趋势，LEMUR 始终保持最强遗忘效果和最好效用保持。
- **跨架构迁移**：迁移到非 RL 的 Qwen2.5-VL 时，LEMUR 仍优于 R-MUSE（Forget split SRL 28.5% vs 30.9%），但熵信号减弱，视觉锚点成为主导修正信号。
- **通用领域验证**：在 VQAv2 蒸馏数据上，LEMUR 同样实现最低 Gen TR（22.7%）和 SRL（13.6%），证明熵签名是 RL 训练产物而非特定隐私数据产物。

**最强结果**：在 R1-Onevision-7B 5% 遗忘设置下，LEMUR 的 SRL 降至 11.3%（较 vanilla 61.6% 降低 50.3 个百分点），同时 RRA 仅从 7.7 降至 6.9，显著优于所有基线。

## 相关工作脉络
1. **多模态大推理模型（MLRMs）**：R1-Onevision（Yang et al. 2025）、Vision-R1（Huang et al. 2025）通过 RL 训练获得原生推理能力，区别于指令微调 MLLM（如 Qwen2-VL），其推理轨迹是显式生成的 CoT。
2. **机器遗忘（训练基础）**：GAdiff、NPO、KLMin 等通过梯度上升/负偏好优化压制遗忘内容，但存在过遗忘（over-forgetting）和灾难性遗忘扩散问题，且对 RL-MLRMs 的推理能力破坏显著。
3. **机器遗忘（训练免费）**：Liu et al. (2024) 的 embedding corruption、Huang et al. (2024) 的 offset unlearning、R-MUSE（Li et al. 2026）的激活 steering，这些方法作用于 prompt 或隐层激活，但未针对推理轨迹设计，答案级清理会留下轨迹泄露。
4. **MLLM 机器遗忘**：MMUnlearner（Huo et al. 2025）、Single-image unlearning（Li et al. 2024）等方法作用于最终答案，缺乏对 reasoning trace 的干预能力。
5. **本文定位差异**：LEMUR 是首个专为原生 RL 训练 MLRMs 设计的遗忘框架，也是首个利用解码时熵动力学进行推理轨迹重定向的方法，解决了"答案干净但轨迹泄露"的核心矛盾。

## 局限性与未来方向
1. **对非 RL 训练模型效果减弱**：Qwen2.5-VL 实验中熵信号不明显，ESS 组件作用有限，表明方法对 RL 诱导的熵签名有较强依赖性。
2. **蒸馏数据的局限性**：训练数据依赖单一轮次的 teacher 蒸馏（无 verifier 自校正循环），可能存在内容级泄露或链质量不一致问题。
3. **退化成重复文本的潜在风险**：少数 forget-split 生成出现 token 重复或截断，虽然不泄露受保护属性，但影响输出质量。
4. **超参数敏感性**：$\rho, \tau, \kappa, \gamma, W_{\max}, C$ 等参数需要调优，论文未提供系统性的灵敏度分析。
5. **未来方向**：将熵驱动的解码时视角扩展到更广泛的安全目标（如偏见消除、有害内容过滤），以及探索无需蒸馏的端到端推理时遗忘。

## 研究启发与可借鉴点
1. **熵动力学作为隐私泄露检测信号**：将 token-level 熵的两阶段模式（deliberation spike → collapse）作为敏感段定位的可靠信号，这一思路可迁移到纯文本 LLM 的推理轨迹隐私保护场景。
2. **无训练推理时干预范式**：LEMUR 的"监控-切换-注入-退出"四步框架避免了微调的计算开销和过遗忘风险，其解码时干预设计可直接借鉴到其他需要实时内容控制的任务（如实时 Moderation）。
3. **视觉锚点与语言锚点的组合策略**：用图像重 grounding 替代直接屏蔽，既保留了模型的感知一致性又阻断了敏感关联，这一"重定向而非删除"的思路对多模态内容安全具有启发价值。
4. **自适应退出机制**：基于指数移动平均基线的动态熵阈值和冷却窗口设计，为其他序列生成干预任务提供了可复用的"精准窗口控制"模板。
5. **与本团队的结合机会**：可将 LEMUR 的熵检测模块与本团队在推理加速/早退（early-exit）工作结合，利用同一熵信号同时实现隐私保护和计算效率优化。

## 关键术语表
**MLRM（Multimodal Large Reasoning Model）**：通过 RL 后训练获得原生推理能力（探索性 CoT）的多模态大模型，如 R1-Onevision、Vision-R1，区别于指令微调 MLLM。

**Token-level Entropy Signature**：RL 训练中记忆化属性在解码时呈现的两阶段熵模式——先经历高熵 deliberation（候选值竞争），后进入低熵 deterministic recital（确定复述）。

**Forget Set / Retain Set**：遗忘集包含需去除的敏感 subject 及其 QA 对；保留集包含需保持性能的普通数据和 celebrity 数据。

**Visual Anchor**：预训练视觉特殊词元（如 `<|vision_start|>`, `<|image_pad|>`, `<|vision_end|>`）的嵌入平均，用于将推理轨迹重新 grounding 到输入图像模态。

**SRL（Subject-level Reasoning Leakage）**：衡量推理轨迹中是否暴露查询 subject 的任何属性（排除 prompt 中已给定的值），是评估推理轨迹级遗忘的核心指标。

**RRA（Reasoning Retention Ability）**：由 Gemini-2.5-Pro 自动评估的推理保持能力，衡量生成文本的流畅度和自然度。

**Entropy-Modulated Injection**：注入强度 $\gamma_t$ 随当前 token 熵动态调整的策略，高熵阶段强 steering（属性未确定）、低熵阶段弱 steering（避免扭曲已确定的流畅文本）。

**Latent Decoding Regime**：LEMUR 在检测到敏感段后切换的解码模式，不再反馈离散采样 token，而是反馈连续嵌入（受约束潜反馈+锚点注入的组合），直到熵阈值判定退出。

## 可复现要素
- **数据集**：基于 MLLMU-Bench（Liu et al. 2025a）重构，使用 Qwen3.5-35B-A3B 蒸馏推理链；原始 MLLMU-Bench 包含虚构人物肖像图和私有 QA 对，论文未提供独立开源链接，但注明可在 MLLMU-Bench 基础上重建。
- **代码/权重**：论文未明确声明代码开源；模型使用 R1-Onevision-7B、Vision-R1-7B、OpenVLThinker-7B 的官方 checkpoint，LEMUR 作为推理时干预无需发布新权重。
- **关键超参**：$\rho$（词汇检测阈值）、$\rho_{\text{lo}}$（低熵阶段降低阈值）、$\tau$（熵参考阈值）、$\kappa$（退出阈值系数）、$\gamma$（注入强度参考值）、$\gamma_{\text{max}}$（注入强度上限）、$W_{\max}$（最大阶段长度）、$C$（冷却窗口步数）、$\eta$（EMA 更新率）、$\beta$（视觉/安全锚点平衡系数）——论文正文未给出具体数值，需在附录或代码中查找。
