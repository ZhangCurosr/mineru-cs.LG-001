---
title: "Share First, Route What Remains:"
source: https://arxiv.org/pdf/2608.10392v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 03:39:06"
field: "稀疏 MoE 路由与共享-残差计算"
keywords: ["mixture of experts", "token-adaptive computation", "shared experts", "dynamic routing", "blockwise partitioning"]
innovations: ["提出共享-残差顺序化统一预算，实现共享宽度/内容/残差专家数的联动", "引入 key-prototype 驱动的共享 block 选择机制", "通过 Gram 正则约束 router 嵌入几何以促进多样路由与稀疏重叠"]
benchmarks: ["DomainBed", "GLUE"]
---

# 论文速读：Share First, Route What Remains:

## 一句话总结
论文提出 UniF-MoE 统一框架，基于“先共享、后路由剩余”原则，将 token 自适应的计算预算（共享宽度、共享内容、残差专家数量）整合为有序的单一决策流程，在视觉与语言基准上优于静态/动态 MoE 基线，同时降低激活计算量、推理延迟与显存。

## 研究问题与动机
- 传统 top-k MoE 把完整 FFN 专家作为计算单元，未剥离不同专家间可复用的部分，导致简单 token 出现冗余计算。
- 多数 token 被分配相同专家数，忽略语义需求差异，使复杂 token 容量不足而简单 token 过算。
- 共享专家、细粒度专家、动态路由等改进通常独立设计，未考虑顺序依赖：提取可复用计算会改变剩余需求与偏好专家分布。
- 现有方法缺乏对“可重用与 token 特异性计算”内部依赖的统一建模，难以协调 intra-expert 与 inter-expert 稀疏性。

## 核心贡献（创新点）
- 揭示稀疏 upcycled 专家中可重用计算与 token 特异性计算的路由条件依赖，提出“先共享后路由剩余”的统一原则。
- 构建 UniF-MoE，通过单 token 依赖预算联动共享宽度、共享内容与残差专家数量，实现 intra/expert 两级稀疏的协调分配。
- 引入 Gram 正则约束 router 嵌入的几何结构，促进路由方向多样化与专家重叠稀疏化，提升可训练性与泛化表现。

## 方法详解
- **Blockwise Shared-Residual Partitioning**：每层含 1 个共享专家 $E_{\mathrm{shr}}$ 与 K 个残差专家 $\{E_1,\dots,E_K\}$；每个专家按 $B$ 个对齐 block 划分，token 选定若干 block 走共享路径，其余 block 走残差路径，同一 hidden 位置不对同一 token 同时分配给两路。
- **Token-Adaptive Shared Modeling**：router 新增共享列 $\mathbf{W}_{\mathrm{shr}}$，得到 shared-demand score $\alpha(\mathbf{x}) = \tau + (1-2\tau)\sigma(\mathbf{x}\mathbf{W}_{\mathrm{shr}})$，其中 $\tau = (B-1)/B^2$；shared block 数 $b(\mathbf{x}) = \mathrm{round}(B\alpha(\mathbf{x}))$，保证 $b\in\{1,\dots,B-1\}$。
- **Shared Block Selection via Key Prototypes**：用每 block 的 up-projection key 均值 $\mu_b$ 作为原型，优先级 $u_b(\mathbf{x}) = \mathbf{x}\mu_b$，TopK 选出 $\mathcal{T}_{\mathrm{shr}}(\mathbf{x})$，剩余为 $\mathcal{T}_{\mathrm{res}}(\mathbf{x})$。
- **Cumulative Residual-Expert Routing**：残差 demand $\beta(\mathbf{x})=1-\alpha(\mathbf{x})$；对 expert affinity $p_i(\mathbf{x})$ 排序后，取最小前缀满足 $\sum_{i=1}^{k} p_i(\mathbf{x})\ge \beta(\mathbf{x})$ 得到 $k(\mathbf{x})$，使专家数由前置共享分配决定。
- **Output Merging**：最终输出 $\mathbf{y} = \alpha(\mathbf{x}) E_{\mathrm{shr}}^{S}(\mathbf{x}) + \sum_{i=1}^{k(\mathbf{x})} p_i(\mathbf{x}) E_{q_i(\mathbf{x})}^{\mathcal{R}}(\mathbf{x})$；残差路由质量受 $\alpha+P_{k}<1+p_k$ 控制，总系数质量有界。
- **Training Objective**：$\mathcal{L} = \mathcal{L}_{\mathrm{task}} + \lambda_{\mathrm{div}}\mathcal{L}_{\mathrm{div}}$，其中 $\mathcal{L}_{\mathrm{div}} = \|(\mathbf{W}_g^\star)^\top \mathbf{W}_g^\star - \mathbf{I}_{K+1}\|_F$；正交初始化 + Gram 正则共同维持归一化且分离的路由方向。
- **Compute Accounting**：每 token 激活 $C_B(\mathbf{x}) = b(\mathbf{x}) + k(\mathbf{x})[B-b(\mathbf{x})]$ 个 block；共享部分仅执行一次，重复成本仅留在残差路径。

## 实验与结果
- **数据集/任务**：DomainBed 五项（PACS/VLCS/OfficeHome/TerraIncognita/DomainNet），基于 DeiT-S/16；GLUE 五项（CoLA/MRPC/QNLI/MNLI/RTE），基于 BERT-large。
- **基线**：Vision 含 GMoE/EMoE/EMoE-L/LFME/DMDA/PC-MoE/DynMoE/MASS；Language 含固定 top-k（k=1,2,4,8）、DynMoE、MASS。
- **主要结果**：
  - DomainBed 平均准确率 UniF-MoE = 69.5%，领先多数对比；PACS 89.6、VLCS 81.7、OfficeHome 74.2、DomainNet 49.4；TerraInc. 52.6（低于 LFME 的 53.4）。
  - GLUE 平均 82.76，在全部五项均最优：CoLA 66.83、MRPC 91.57、QNLI 93.10、MNLI 86.84、RTE 75.47；超过固定 top-k 的 per-task oracle 上界。
- **成本对比（VLCS）**：相对 top-2 GMoE，激活参数减 9.1%、FLOPs 减 16.1%；推理时间降 45.2%、显存降 52.7%；各项计算/运行时指标均优于 DynMoE。
- **关键结论**：UniF-MoE 在精度-效率权衡上整体最强，并将可复用块与残差路由联合调度的收益显式体现。

## 相关工作脉络
- 与 DeepSeekMoE/Union-of-Experts 等共享建模工作相比：它们提升复用或专业化，但未用 token 级共享分配定义后续专业化应处理的内容；UniF-MoE 显式建模共享-残差依赖关系。
- 与 EMoE/Emergent MoE 等细粒度/模块化方法相比：细粒度通常独立调节宽度或结构，未按共享响应作为前置条件驱动后续专家数决策。
- 与 DynMoE/MASS/Alloc-MoE 等动态路由相比：动态方法各自独立估计 expert count 或分配预算，缺少由共享路径先行缩减残差需求的顺序约束。
- 与 orthogonality/variance 类 specialist 正则（Advancing Expert Specialization 等）相比：这些方法主要压制 overlap，但不由 token 按需决定可复用分量；UniF-MoE 通过 Gram 约束塑造路由几何，服务于共享-残差解耦后的路由多样性。
- 与 sparse upcycling 范式的关系：利用同初始 FFN 对齐的 key/value 通道进行位置级分解，使 block 粒度上的共享选择具备可比性与可训练性。
- 定位差异：本文把共享建模、细粒度计算、动态路由视为同一有序分配问题的三个阶段，而非并行可调旋钮。

## 局限性与未来方向
- 在 TerraIncognita 等强位置/视角依赖域上，LFME 等显式域专业化方法仍更强，说明纯共享-残差分解未必在所有分布偏移情形占优。
- 超参敏感性问题：$\lambda_{\mathrm{div}}$ 过弱导致路由方向相关、过强可能压倒任务损失；需更稳健的搜索/自调节策略。
- Block 粒度 $B$ 的选择需权衡：$B=4$ 过粗、$B=32$ 过细，当前以 $B=8$ 为稳态平衡，但在更大宽度或不同任务下未必最优。
- 实验主要在 DeiT-S/16 与 BERT-large 的中规模设置上验证，扩展到更大模型/更长序列/更多层转换的效果与训练稳定性未充分披露。
- 共享专家与残差专家的初始化、更新耦合关系可能对优化路径产生偏置，尚需更深入的理论或大规模消融分析。

## 研究启发与可借鉴点
- **共享-残差顺序化预算**：将原本并行的多个决策（共享多少/共享哪些/路由几个）按因果顺序重组，是提升多组件协同效率的有效思路。
- **Key-prototype block selection**：用 up-projection key 均值构造 block 级原型进行 token 自适应选择，实现了可微、可解释的内容级共享路由。
- **Gram 正则塑造路由几何**：通过 $(W^* )^\top W^* \approx I$ 同时保证单位尺度与方向分离，是一种兼顾多样性与稳定性的轻量正则方案。
- **累积路由质量控制**：用最小前缀满足 $\beta$ 预算的方式决定 expert 数，既保留 router 置信度形状又避免固定 k 的僵化。
- **可与本团队方向结合**：在低资源或延迟敏感场景下，将“先提取可复用通道、再按需残差路由”的结构迁移至更大 BERT/DeiT 或跨模态 MoE，有望进一步压缩推理成本。

## 关键术语表
- **Token-Adaptive Computation**：依据每个 token 的语义需求动态调节计算量与专家数量的机制。
- **Shared-Residual Partitioning**：将 FFN 的 hidden channels 划分为共享 block 与残差 block 的两路分解策略。
- **Shared-Demand Score**：由 learnable 列 $\mathbf{W}_{\mathrm{shr}}$ 产生的 sigmoid 得分，决定共享宽度与共享路径权重。
- **Key Prototype**：每个 block 内 up-projection key 的均值向量，用于 token 对 block 重要性的打分与 TopK 选择。
- **Cumulative Routing Mass**：按 expert affinity 排序后累加至满足残差预算的最小前缀，决定激活残差专家数。
- **Gram Regularizer**：约束 router 嵌入矩阵近似正交归一的损失项，用于促进路由方向多样性与稀疏重叠。
- **Sparse Upcycling**：从预训练稠密 FFN 复制初始化多专家并微调，以获得跨专家位置对齐的可比性。
- **Intra/Inter-Expert Sparsity Coordination**：同一 token 预算内同时协调专家内部 block 选择与跨专家激活数量。

## 可复现要素
- **数据集**：DomainBed（PACS/VLCS/OfficeHome/TerraIncognita/DomainNet）与 GLUE（CoLA/MRPC/QNLI/MNLI/RTE）均为公开数据集。
- **代码开源**：是，GitHub 地址 https://github.com/existence0420/UniF-MoE。
- **模型权重**：论文未明确提供下载链接，仅声明代码可用。
- **关键超参**：Vision: $K=6, B=8$，DeiT-S/16，$d=384, H=1536$，dropout=0.1，stochastic depth=0.1，Adam+batch 32；Language: $K=16, B=16$，BERT-large，序列长 128，batch 32，学习率候选 $\{2e-5, 3e-5, 5e-5\}$，最多 10 epoch，无 warmup，weight decay=0。
- **随机性与统计报告**：三次独立 seed 均值±标准差；Vision 使用 train-validation selection criterion。
