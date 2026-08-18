---
title: "Share First, Route What Remains:"
source: https://arxiv.org/pdf/2608.10392v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 04:51:49"
field: "高效 MoE 架构设计"
keywords: ["mixture of experts", "shared expert", "token-adaptive computation", "dynamic routing", "sparse upcycling", "domain generalization"]
innovations: ["提出\"先共享、再路由剩余\"的统一原则，将共享建模、细粒度计算与动态路由耦合为有序流程", "设计 UniF-MoE 框架，通过单一 token 依赖预算协调共享宽度、共享内容与残差 expert 数量", "引入 Gram 正交正则化塑造 router embedding 几何，促进专家多样性与稀疏重叠"]
benchmarks: ["DomainBed", "GLUE"]
---

# 论文速读：Share First, Route What Remains: A Unified Framework for Token-Adaptive MoE Computation

## 一句话总结
论文提出 UniF-MoE 框架，揭示共享专家、细粒度计算与动态路由之间存在内在的顺序依赖，并通过"先共享可重用块、再路由残差 expert"的统一预算机制，实现 token 自适应的 MoE 计算，在视觉与语言任务上同时提升精度与效率。

## 研究问题与动机
- 传统 top-k MoE 将所有专家视为不可分割的整体，对简单和复杂 token 使用相同 expert 数量，导致计算冗余或容量不足。
- 已有方法（共享专家、模内细粒度计算、动态路由）各自独立发展，未考虑"提取可重用计算后，剩余内容、最佳 expert 选择及所需容量均会变化"这一依赖关系。
- 论文通过在 TerraIncognita 训练的一个标准 top-2 MoE 第 10 层上诊断发现：共激活 expert 对在约 80% 的 value 位置对齐，这些位置构成可重用响应；移除后仅 5.7% 的 token 保持相同 top-2 集，且共享比例与残差需求呈强负相关（r = −0.673）。
- 因此，共享建模、细粒度计算与动态路由应视为同一分配问题的三个阶段，而非三个并行旋钮。

## 核心贡献（创新点）
- **揭示路由条件依赖并提炼有序原则**：证明可重用计算与 token 特定计算之间存在依赖，把共享建模、细粒度计算、动态路由转化为单一有序原则"share first, route what remains"，与以往并行独立设计形成本质区别。
- **提出 UniF-MoE 统一框架**：将共享宽度、共享内容与残差 expert 数量耦合到单个 token 依赖预算，一个 router 协调"共享多少、共享什么、剩余需要多少 expert"三步决策，避免重复路由已计算响应。
- **Gram 正则化塑造路由几何**：对 router embedding 施加 Gram 正交约束，使路由方向相互分离并保持归一化，促进 expert 多样性与稀疏重叠，区别于以往直接对 expert 输出施加正交化的做法。
- **实证支持统一设计优势**：在 DomainBed 和 GLUE 上，相比静态与动态 MoE 基线取得更强 accuracy–efficiency 折衷，并显著降低推理延迟与显存占用。

## 方法详解
- **Blockwise 共享-残差划分**：每层含 1 个共享 expert $E_{\mathrm{shr}}$ 与 K 个残差 expert $\{E_1, \ldots, E_K\}$，每个 expert 的中间宽度 H 划分为 B 个对齐 block，block b 的输出为 $E_j^{(b)}(\mathbf{x}) = \sum_{h=(b-1)M+1}^{bM} \mathrm{GeLU}(\mathbf{x}\mathbf{K}_j[:,h])\mathbf{V}_j[h,:]$，$M=H/B$。同一 token 下每个隐藏位置只属于共享或残差路径之一。
- **Token-Adaptive 共享建模**：在 router 中新增共享列 $\mathbf{W}_{\mathrm{shr}}$，共享需求 $\alpha(\mathbf{x}) = \tau + (1-2\tau)\sigma(\mathbf{x}\mathbf{W}_{\mathrm{shr}})$，其中 $\tau = (B-1)/B^2$，保证 $\alpha(\mathbf{x}) \in (\tau, 1-\tau)$，共享 block 数 $b(\mathbf{x}) = \mathrm{round}(B\alpha(\mathbf{x})) \in \{1,\ldots,B-1\}$。共享 block 通过各 block 上投影 key 均值 $\mu_b = \frac{1}{M}\sum_{h \in \mathcal{H}_b}\mathbf{K}_{\mathrm{shr}}[:,h]$ 的优先级 $u_b(\mathbf{x})=\mathbf{x}\mu_b$ 选取 Top-b(x) 作为 $\mathcal{I}_{\mathrm{shr}}(\mathbf{x})$，其余为残差集合 $\mathcal{I}_{\mathrm{res}}(\mathbf{x})$。
- **Cumulative 残差 Expert 路由**：残差路由得分 $\mathbf{s}(\mathbf{x})=\mathrm{softmax}(\mathbf{x}\mathbf{W}_g)$，按降序排列为 $p_1 \ge \cdots \ge p_K$。残差需求 $\beta(\mathbf{x})=1-\alpha(\mathbf{x})$，激活的最小前缀满足 $\sum_{i=1}^{k(\mathbf{x})}p_i(\mathbf{x}) \ge \beta(\mathbf{x})$，即 $k(\mathbf{x})=\min\{n:\sum_{i=1}^n p_i(\mathbf{x})\ge\beta(\mathbf{x})\}$，实现 expert 数对前置共享分配的依赖。
- **共享-残差输出合并**：最终输出 $\mathbf{y} = \alpha(\mathbf{x})E_{\mathrm{shr}}^S(\mathbf{x}) + \sum_{i=1}^{k(\mathbf{x})}p_i(\mathbf{x})E_{q_i(\mathbf{x})}^{\mathcal{R}}(\mathbf{x})$，其中 $E_{\mathrm{shr}}^S$ 为选中共享 block 之和，$E_j^{\mathcal{R}}$ 为残差 block 之和。总系数质量受控：$\alpha + P_{k(\mathbf{x})} \in [1, 1+p_{k(\mathbf{x})})$。
- **训练目标**：$\mathcal{L}=\mathcal{L}_{\mathrm{task}}+\lambda_{\mathrm{div}}\mathcal{L}_{\mathrm{div}}$，其中 $\mathcal{L}_{\mathrm{div}}=\|(W_g^\star)^\top W_g^\star - I_{K+1}\|_F$，$W_g^\star=[\mathbf{W}_{\mathrm{shr}}, \mathbf{W}_g]$。正交初始化从同构几何出发，Gram 正则分离路由方向，鼓励 expert 角色多样与重叠稀疏。
- **计算量**：每个 token 激活 $C_B(\mathbf{x})=b(\mathbf{x})+k(\mathbf{x})[B-b(\mathbf{x})]$ 个 block；当 $k(\mathbf{x})=1$ 时恰好等价于一个完整 FFN，额外残差 expert 仅追加剩余 $B-b(\mathbf{x})$ 个 block。

## 实验与结果
- **数据集与评估**：视觉 Backbone 为 ImageNet 预训练 DeiT-S/16，替换 Transformer 层 8、10，设 K=6、B=8；在 DomainBed 五个数据集（PACS、VLCS、OfficeHome、TerraIncognita、DomainNet）报告 out-of-domain 准确率。语言 Backbone 为 BERT-large，替换层 20、22，设 K=16、B=16；在 GLUE 五任务（CoLA、MRPC、QNLI、MNLI、RTE）报告 in-domain 分数。所有结果为三次运行均值。
- **视觉结果**：UniF-MoE 在 DomainBed 平均准确率 69.5%，领先 Next-best LFME（68.5%）1.0pt；在 PACS 达 89.6%（领先 PC-MoE 89.4%）、VLCS 达 81.7%（领先 MASS 81.1%）、DomainNet 达 49.4%；仅在 TerraIncognita 略低于 LFME（52.6% vs 53.4%），因其强依赖位置背景。
- **语言结果**：UniF-MoE 在全部五个 GLUE 任务上均取得最优，CoLA 66.83、MRPC 91.57、QNLI 93.10、MNLI 86.84、RTE 75.47，平均 82.76，超越固定 top-k oracle envelope（per-task best 平均 81.95）与 DynMoE（81.64）、MASS（82.19）。
- **成本对比（VLCS）**：相对 top-2 GMoE，UniF-MoE 激活参数少 9.1%，FLOPs 少 16.1%，推理时间少 45.2%，推理显存少 52.7%；全面优于 DynMoE。
- **消融**：固定 α、前缀 block 选择、固定 top-2 残差路由三者任一替换为非自适应版本均导致两侧基准精度下降与计算上升；固定 α 影响最大；$B=8$ 或 16 在多数数据集表现最佳；$\lambda_{\mathrm{div}}=0.01$ 为 GLUE 平均最优，能使 co-activation 下降 62.9%。

## 相关工作脉络
- **共享专家与 expert 专业化**：DeepSeekMoE 分割专家并添加共享 expert、Union-of-Experts 构建虚拟共享 expert；正交/方差目标与 MP-MoE 鼓励专业化。本文与之本质区别在于：用 token 依赖的共享分配显式定义专业化所应对的"剩余计算"，而非平行处理。
- **动态路由**：DynMoE、MASS、Alloc-MoE 通过累积置信度、剪枝/扩池或预算分配控制 expert 数。本文将 expert 数由共享后剩余需求决定，二者共享同一条预算而非独立估计。
- **细粒度计算**：Emergent MoE、MoSE、nested/slimmable experts 调整 expert 内宽度。本文在 channel/block 粒度分离共享与残差，并与动态路由串联为单一有序流程。
- **路由正则化**：PC-MoE 扰动 cosine router；本文通过 Gram 约束直接塑造 router embedding 的归一化正交几何，以简单方式实现 diverse routing directions。
- **稀疏上采样**：Sparse Upcycling 使专家共享相同初始化 FFN，使 channel 对齐比较可行，是本文 block 级分解的前提。

## 局限性与未来方向
- 诊断实验主要在单模型单层（TerraIncognita 训练的第 10 层）上进行，跨层与跨模型的普适性需进一步验证。
- TerraIncognita 上未超越 LFME，说明强位置/视角依赖场景可能需要更明确的 domain 级 specialization，而非仅靠 token 级 shared-residual 分配。
- Block 划分假设 expert 间同一 index 位置可比，依赖于 sparse upcycling 初始化；对任意初始化需额外对齐策略。
- 论文未讨论极端 B 值或不同 K 比例下的长尾行为，超参 B 的经验选择可能随模型规模变化。
- 未来可在更大 LLM 与更长序列上验证，并探索 block 粒度与 expert 数之间的自动搜索。

## 研究启发与可借鉴点
- **"共享-残差预算"思路可迁移**：将多阶段决策耦合为单一 token 依赖预算的设计范式，适用于其他需要权衡复用与专用的 sparse mixture 结构。
- **Gram 正交正则对 router embedding 的有效性**：简单正则即可分离路由方向，可在其他 router-based 模型中作为通用技巧复用。
- **channel/block 级诊断方法**：通过 value 对齐度 $c_{ijh}$ 与 co-activation 关联分析揭示可重用响应，可作为 MoE 内部结构诊断的标准化工具。
- **实验设计值得借鉴**：pair-balanced CDF、ridge-stabilized 重建误差与 fidelity tolerance $k_\varepsilon$ 的结合，为"需要多少残差 expert"提供了可量化的度量。
- **与本团队方向结合机会**：若团队关注低资源或长序列推理，可将此 token-adaptive 预算机制嵌入现有 MoE 推理系统，换取延迟与显存的进一步优化。

## 关键术语表
- **MoE (Mixture of Experts)**：通过 router 将 token 路由到少量 expert FFN 的大模型架构，以稀疏激活扩展参数容量。
- **Shared-expert / 共享专家**：多个 token 共同使用的 expert，用于捕获可重用计算。
- **Residual expert / 残差 expert**：处理共享之后剩余 token 特定计算的 expert。
- **Block / 块**：将 expert FFN 的中间维度划分为若干对齐子块，每个块对应一组 hidden channels。
- **Token-Adaptive / token 自适应**：根据每个 token 的需求动态调整共享宽度、block 选择与 expert 数量。
- **Cumulative Routing Mass / 累积路由质量**：按 expert 亲和度降序累加直到覆盖残差需求的机制。
- **Gram Regularizer / Gram 正则化**：对 router embedding 矩阵施加 $(W^\top W - I)_F$ 惩罚，实现归一化与方向分离。
- **Sparse Upcycling / 稀疏上采样**：从同一个预训练 dense FFN 初始化多个 expert，使相同 channel index 可比较。

## 可复现要素
- **数据集**：DomainBed（PACS、VLCS、OfficeHome、TerraIncognita、DomainNet）与 GLUE（CoLA、MRPC、QNLI、MNLI、RTE），均为公开基准。
- **代码**：论文声明代码开源，地址 https://github.com/existence0420/UniF-MoE。
- **权重**：未明确声明公开权重。
- **关键超参**：视觉 K=6、B=8；语言 K=16、B=16；$\lambda_{\mathrm{div}}=0.01$；替换层索引视觉为 8、10，语言为 20、22；learning rate $\in \{2,3,5\}\times10^{-5}$；batch size 32（训练）、64（评估）；dropout 0.1；stochastic depth 0.1。
