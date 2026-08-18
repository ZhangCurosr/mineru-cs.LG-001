---
title: "LEMUR-Latent-Entropy-aware-Multimodal-Unlearning-via-Visual"
source: https://arxiv.org/pdf/2608.11691v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:28:25"
field: "多模态大模型隐私安全"
keywords: ["machine unlearning", "multimodal large reasoning models", "entropy-aware intervention", "privacy leakage", "inference-time unlearning", "visual anchoring"]
innovations: ["利用 token 级两阶段熵签名作为推理时干预信号，首次实现 RL 训练 MLRM 的无训练 unlearning", "熵调制的视觉锚点 latent injection 机制，在抑制 sensitive recall 的同时保持推理流畅性", "自适应熵阈值退出 + 冷却窗口防止离散/latent 解码振荡的动态相持续时间控制"]
benchmarks: ["MLLMU-Bench", "VQAv2"]
---

# 论文速读：LEMUR-Latent-Entropy-aware-Multimodal-Unlearning-via-Visual

## 一句话总结
针对 RL 训练的 multimodal large reasoning models (MLRMs) 在推理 trace 中存在隐私泄露的新风险，本文提出 LEMUR——一种完全无训练的推理时 unlearning 框架，通过利用 token 级熵信号识别敏感推理区间，并将轨迹重定向至视觉锚点的 latent embedding，实现对 reasoning-trace 和 answer 双层面的隐私擦除，同时保持模型推理能力与输出流畅性。

## 研究问题与动机
- **推理 trace 级隐私泄露**：即使敏感事实已成功从最终答案中遗忘，RL 训练的 MLRM 仍可能在其显式思考链（⟨think⟩…⟨/think⟩）中复现该信息，这是传统以答案为中心的 unlearning 方法完全遗漏的漏洞。
- **现有方法不匹配 RL 训练模型**：现有方法（微调、activation-steering、prompt 注入等）缺乏对 RL 诱导的多样化探索性推理轨迹的可靠监控机制，且其激活扰动对推理模型破坏性过大，会显著劣化其核心推理能力。
- **熵信号的独特价值**：RL 训练在解码过程中留下了独特的 token 级"两阶段熵签名"（高熵犹豫期 → 低熵确定复述期），这一特征在 non-reasoning base model 中基本不存在，为无训练干预提供了可靠信号源。
- **答案级清洗与 trace 干扰的张力**：仅清洗答案会留下 trace 泄露；激进扰动 trace 又会将推理退化为重复退化文本，二者之间缺乏精细的平衡机制。

## 核心贡献（创新点）
- **揭示 RL 推理模型的新型隐私风险**：首次系统证明 RL-trained MLRMs 在 reasoning trace 中的泄露严重性显著高于其 non-reasoning base MLLMs，且该泄露携带独特的 token 级熵签名——与已有工作仅关注最终答案输出形成本质区别。
- **提出首个面向原生 RL 训练 MLRM 的推理时 unlearning 框架 LEMUR**：通过熵动力学而非梯度更新实现推理干预，将离散自回归解码动态切换为 latent decoding 模式并注入受熵调控的视觉锚点——与既有微调型/activation-steering 方法在干预时机和作用层面完全不同。
- **熵感知的视觉锚点注入机制**：结合 constrained latent feedback（移除 forbidden mass 并保留可微性）与 entropy-modulated visual anchor injection（按当前熵值动态调节注入强度 γ_t），使得干预窗口精确对齐敏感 span 的犹豫起始点到确定结束点——避免了固定阈值方法对低熵区域的过度干扰。
- **动态熵控相持续时间机制**：基于自适应熵阈值 κ·H̄_t 和强制冷却窗口 W_max 共同决定 latent 阶段的退出点，防止离散/latent 解码之间的振荡退化——这是实现推理保真度保持的关键工程创新。
- **跨模型泛化验证**：在 Qwen2.5-VL（non-RL 模型）上的迁移实验表明，即使在没有显著熵跃迁信号的情况下，LEMUR 的视觉锚点路径仍保持有效性，说明方法不仅依赖 RL 特有熵信号，具有更广泛的适用性。

## 方法详解

**问题定义**：给定原始 MLRM M_θ、forget set D_f = {(I_j, T_j)} 和 retain set D_n，寻找 unlearned 模型 M_θ̂ 使得 A(θ̂; D_f) 最小化，同时 A(θ̂; D_n) ≈ A(θ; D_n)。LEMUR 保持 θ̂ = θ，纯在 inference 时通过干预解码分布实现。

**两阶段熵签名观察**（Figure 2）：
- 模型回忆记忆化敏感属性时，per-token 熵 H_t(v) = −Σ_v p_t(v)log p_t(v) 呈现"先升后塌"模式：初始犹豫阶段（多个候选值竞争，H_t 尖峰），然后 commitment 到记忆化 span 后 H_t 坍缩至接近零，直到 span 边界恢复。

**核心组件一：熵增强的敏感性开关**（Entropy-augmented Sensitivity Switching, ESS）

构建 forget-relevance 空间，维护 forbidden token 集合 Φ_s，追踪概率质量：
- P_t^Φ = Σ_{v∈Φ_s} p_t(v)

触发条件 g_t 结合两个信号：
- 词典信号（committed recital）：[P_t^Φ ≥ ρ]
- 熵增强信号（deliberation）：[H_t(v) ≥ τ ∧ P_t^Φ ≥ ρ_lo]，其中 ρ_lo < ρ，用于覆盖犹豫阶段中概率分散但语义相关的同义词集合

**核心组件二：熵感知视觉锚点注入**（Entropy-aware Visual Anchor Injection, VAI/EVAI）

进入敏感模式 S 后，执行以下步骤：

1. **约束 latent feedback**：移除 forbidden tokens 并重归一化
   - p̃_t(v) = p_t(v)·ℋ[v∉Φ_s] / Σ_{u∉Φ_s} p_t(u)
   - 计算期望 embedding：ê_t = Σ_v p̃_t(v)·Ē[v]

2. **视觉锚点构建**：
   - e_vis = (1/|V_vis|) Σ_{v∈V_vis} Ē[v]（预训练视觉专用 token 的平均 embedding，如 <|vision_start|>, <|image_pad|>, <|vision_end|>）
   - e_safe = (1/|S|) Σ_{w∈S} Ē[w]（固定拒绝模板的平均 embedding，如 "I'm not sure"、"I cannot identify this person from the image"）
   - 复合锚点：a = β·e_vis + (1−β)·e_safe

3. **熵控注入**：
   - e_t = (1−γ_t)·ê_t + γ_t·a
   - γ_t = min(γ_max, (H_t(v)/τ)·γ)，按当前步熵值线性缩放，高熵时强 steering，低熵时弱 steering

**核心组件三：动态熵控相持续时间**（Dynamic Entropy-controlled Phase Duration, DEPD）

- 维护指数移动平均熵（仅在 discrete 模式下更新）：H̄_t = (1−η)·H̄_{t−1} + η·H_t(v)
- 退出指示器：z_t = [¬g_t ∧ H_t(v) ≥ κ·H̄_t] ∨ [t − t_0 ≥ W_max]
- 冷却窗口：每次退出后至少 C 个离散步骤才能重新触发，防止 fast-latching 退化

## 实验与结果

**数据集与评估设置**：
- 基于 MLLMU-Bench 重建推理数据集（使用 Qwen3.5-35B-A3B 作为 teacher 蒸馏 ⟨think⟩/⟨answer⟩ 链）
- 评估三种任务类型：classification、fill-in-blank、generation
- 使用三个不同 forget ratio（5%、10%、15%）
- 主干模型：R1-Onevision-7B、Vision-R1-7B、OpenVLThinker-7B
- 迁移实验：Qwen2.5-VL（non-RL）

**评估指标**：CLS Acc（分类准确率）、FIB Acc（填空准确率）、Gen TR（生成目标召回率）、SRL（Subject-level Reasoning Leakage）、RRA（Reasoning Retention Ability，Gemini-2.5-Pro 评分）

**主要结果**（Onevision-R1-7B, 10% forget）：
- LEMUR 在 forget split 上取得最优遗忘效果：CLS Acc 26.0%（对比基线最低 R-MUSE 35.2%）、Gen TR 15.1%（对比 R-MUSE 21.3%）、SRL 11.3%（大幅低于所有基线，R-MUSE 29.8%、MMUnlearner 58.3%）
- Retain 和 Celebrity split 上性能保持与 vanilla 几乎一致（CLS Acc 59.1% vs 58.7%，RRA 7.2 vs 7.7）
- 在 Vision-R1-7B（5% forget）上：CLS Acc 21.9%，SRL 12.1%，同样显著优于所有基线
- 在 OpenVLThinker-7B 上 across 5%/10%/15% 三种 ratio，LEMUR 均取得最强 forget 效果和最佳 utility 保持

**关键提升幅度**：
- SRL 从基线最高 61.6%（Vanilla）降至 11.3%（Onevision-R1）/12.1%（Vision-R1），降低约 50+ 个百分点
- 相比 R-MUSE（最强推理感知基线），LEMUR 的 SRL 再降低约 18–20 个百分点

## 相关工作脉络
- **Thudi et al. (2022)**：SGD-based unlearning，梯度上升方法，需要权重更新，与 LEMUR 无训练范式本质不同。
- **Huo et al. (2025, MMUnlearner)**：针对 MLLM 的 unlearning 方法，但仅作用于最终答案层，无法处理 reasoning trace 泄露，SRL 指标接近 58%。
- **Li et al. (2026, R-MUSE)**：首个推理保持的 unlearning 方法，但针对 instruct MLLMs，作用于 hidden activations 而非解码过程本身，SRL 约 29.8%。
- **Wang et al. (2025)**：针对 reasoning model 的 unlearning，指出 trace perturbation 会导致退化重复，但未提供有效的干预信号。
- **Liu et al. (2025a, MLLMU-Bench)**：构建了多模态 unlearning 评测基准，本文在此基础上重建了带推理链的训练数据。
- **Pawelczyk, Neel, and Lakkaraju (2023)**：in-context unlearning，属于 training-free 类别，但作用于 prompt embedding 层，不涉及 reasoning trace 干预。

## 局限性与未来方向
- **熵信号的依赖性**：在 non-RL 模型（Qwen2.5-VL）上熵信号弱化，ESS 组件效果有限，主要依赖视觉锚点路径，说明当前框架对 RL 训练的熵签名有较强依赖。
- **Teacher 蒸馏的质量依赖**：推理链由 Qwen3.5-35B-A3B 蒸馏生成，未使用 verifier self-refine loop，内容级防泄露依赖 prompt 设计而非形式验证，存在潜在的 chain quality 波动。
- **冷启动 with 新领域**：方法在当前 MLLMU-Bench 虚构人物场景下验证充分，但在其他隐私场景（如科学图像中的实验数据、医疗影像中的患者信息）下的泛化性有待验证。
- **论文自述未来方向**：将熵驱动的解码时视角扩展至更广泛的安全目标（beyond privacy）。

## 研究启发与可借鉴点
- **熵动力学作为隐私泄露的可信信号**：将 per-token entropy 的动态变化用作干预触发和终止的元信号，是一种无需额外标注的内在监控机制，可迁移至其他需要精确时间窗口的推理时干预场景（如 safety filtering、factuality correction）。
- **Latent decoding 代替 hard masking**：用 weighted embedding interpolation 替代 hard token mask，保持了梯度流和分布连续性，避免了粗暴覆盖导致的 fluency 崩溃，这一思路可用于任何需要"软覆盖"生成轨迹的场景。
- **视觉锚点 re-grounding 策略**：将偏离的推理轨迹重新锚定到输入图像本身而非文本知识，这一"回到证据"的思路在多模态可信推理中具有普遍价值。
- **自适应退出阈值 κ·H̄_t**：相对于固定阈值的优越性在此充分验证，这一"相对熵基准"的设计模式可应用于其他需要动态窗口检测的任务。
- **蒸馏 pipeline 的设计细节**：三原则（Attributability、Conservativeness、Consistency）+ anti-leakage attribution context + format guard 的组合，为构建高质量推理链训练数据提供了可复用的框架。

## 关键术语表
- **MLRM（Multimodal Large Reasoning Model）**：通过 RL 训练获得原生推理能力的多模态大模型，如 R1-Onevision、Vision-R1，能生成显式的 ⟨think⟩…⟨/think⟩ 推理链。
- **Token-level entropy signature**：模型在回忆记忆化敏感内容时在 token 级概率分布上呈现的两阶段熵模式（犹豫期高熵 → 承诺期低熵），是 RL 训练模型的独特特征。
- **ForgetRequest / Forget set**：希望从模型中移除的目标隐私知识集合，由图像-问答对组成。
- **Constrained latent feedback**：在敏感模式下移除 forbidden token 概率质量并重归一化后，计算剩余分布的期望 embedding 作为反馈，保持可微性和候选竞争。
- **Visual anchor**：预训练视觉专用 token 的平均 embedding，用于将偏离的推理轨迹重新 grounding 到输入图像的视觉模态。
- **Safe-answer anchor**：固定拒绝模板（如"I'm not sure"）的 embedding 平均，用于引导输出朝向 benign 表达。
- **Entropy-modulated injection strength γ_t**：按当前步熵值动态调节的注入强度，高熵时强 steering，低熵时弱 steering，避免对已确定文本的过度干扰。
- **Subject-level Reasoning Leakage (SRL)**：衡量 reasoning trace 中是否泄露目标 subject 的任意隐私属性的指标，是本文引入的关键评测维度。

## 可复现要素
- **数据集**：基于 MLLMU-Bench 重建，原始基准来自 Liu et al. (2025a)；推理链由 Qwen3.5-35B-A3B 蒸馏生成；论文未明确说明重建后的数据集是否开源。
- **代码/权重**：论文未明确声明代码是否开源；使用了 R1-Onevision-7B、Vision-R1-7B、OpenVLThinker-7B 等公开模型权重。
- **关键超参**：ρ、ρ_lo（forgotten mass 阈值）、τ（参考熵）、γ_max（最大注入强度）、κ（自适应退出阈值系数）、W_max（最大相长度）、C（冷却窗口步数）、η（EMA 衰减率）、β（visual/safe 锚点混合比）——论文正文未给出具体数值，应在附录或补充材料中。
