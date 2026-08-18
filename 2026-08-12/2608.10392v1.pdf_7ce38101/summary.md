---
title: "Share First, Route What Remains:"
source: https://arxiv.org/pdf/2608.10392v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 04:51:54"
field: "高效深度学习架构"
keywords: ["mixture of experts", "token-adaptive computation", "shared expert", "dynamic routing", "domain generalization", "sparse upcycling"]
innovations: ["揭示共享建模、细粒度计算与动态路由间的顺序依赖，提出'先共享再路由剩余'统一原则", "UniF-MoE通过单一token依赖预算联合控制共享宽度和内容、残差专家数三个决策", "Gram正则化路由嵌入促进专家多样性与稀疏重叠，简化路由几何"]
benchmarks: ["DomainBed", "GLUE"]
---

# 论文速读：Share First, Route What Remains: A Unified Framework for Token-Adaptive MoE Computation

## 一句话总结
论文提出了 UniF-MoE，一个统一框架，将 MoE 中的共享专家建模、细粒度通道选择和动态路由整合为"先共享，再路由剩余"的有序过程，实现了 token 自适应的计算分配。在 DomainBed 和 GLUE 上均优于现有静态与动态 MoE 方法，同时降低激活计算量、推理延迟和内存占用。

## 研究问题与动机
- 传统 top-k 路由将完整专家作为计算单元，忽略专家间可能存在可复用响应（对齐的值位置），导致简单 token 冗余计算、困难 token 容量不足。
- 现有的共享专家设计（DeepSeekMoE、Union-of-Experts）、细粒度计算（EMoE、MoSE）和动态路由（DynMoE、MASS、Alloc-MoE）通常是独立机制，缺乏相互间的条件依赖关系建模。
- 论文发现：共激活专家在部分 value 位置对齐；移除这些对齐位置后，最优路由结果发生显著变化（仅 5.7% token 保持相同 top-2 集合）；共享覆盖度越高，所需残差专家数越低（相关系数 −0.673）。这表明三个决策是同一预算分配的连续阶段，而非并行旋钮。

## 核心贡献（创新点）
- **揭示路由条件下的共享-残差依赖关系**：首次系统证明提取可复用计算会改变专家偏好和残差需求分布，将共享建模、细粒度计算、动态路由统一为"先共享，再路由剩余"的有序原则。与已有工作将三者视为独立机制的本质区别在于显式建模了三者间的顺序依赖。
- **提出 UniF-MoE 统一框架**：通过单个 token 依赖预算协调共享宽度、共享内容和残差专家数量三个决策，实现内专家级（intra-expert）与外专家级（inter-expert）稀疏性的联合控制。与已有工作的区别在于单路由器完成三个有序决策，而非多个独立控制器。
- **Gram 正则化促进路由方向多样性**：通过对 router 嵌入施加 Gram 约束（正交初始化 + 离对角惩罚），使路由几何简化，增强专家角色多样性并减少重叠。与 Orthogonal/Variance 类方法的区别在于直接作用于路由嵌入空间而非专家输出空间。

## 方法详解
**Blockwise Shared-Residual Partitioning**：每个 FFN 专家被划分为 B 个对齐块（block），每块宽度 M=H/B。每层含 1 个共享专家 E_shr 和 K 个残差专家，所有专家从同一预训练 FFN 初始化，保证块边界对齐。对每个 token，共享专家执行一块子集，残差专家执行互补子集，无隐藏位置被重复分配。

**Token-Adaptive Shared Modeling**：
- **共享需求分数**：在 router 中新增 learnable 列 W_shr，计算 α(x) = τ + (1−2τ)σ(xW_shr)，其中 τ=(B−1)/B²。α(x) 同时控制共享块数量和共享路径权重，且满足 τ < α(x) < 1−τ，保证每个 token 至少保留 1 个共享块和 1 个残差块。
- **共享块选择**：用各块 up-projection key 的均值 μ_b 作为 prototype，通过 xμ_b 计算优先级，Top-K 选出 b(x)=round(B·α(x)) 个块作为共享内容。

**Cumulative Residual-Expert Routing**：残差需求 β(x)=1−α(x)，与共享需求共享总预算。将路由得分排序为 p_1 ≥ p_2 ≥ ... ≥ p_K，取最小前缀 k(x) 使得累计亲和力 ≥ β(x)。残差专家数因此依赖于前置共享分配。

**Shared-Residual Output Merging**：最终输出 y = α(x)·E_shr^S(x) + Σ_{i=1}^{k(x)} p_i(x)·E_{q_i(x)}^R(x)，总系数质量受控且有界过冲。

**Training Objective**：L = L_task + λ_div·||(W_g*)^T W_g* − I_{K+1}||_F，其中 W_g*=[W_shr, W_g]。正交初始化 + Gram 正则，抑制共激活（实验中降低 62.9%），同时保留有用协作。

**计算复杂度**：token x 激活 C_B(x)=b(x)+k(x)[B−b(x)] 个块。当 k(x)=1 时等于一个完整 FFN，额外残差专家只需增加 B−b(x) 个块。

## 实验与结果
**数据集与基线**：
- 视觉：DomainBed（PACS、VLCS、OfficeHome、TerraIncognita、DomainNet），骨干 DeiT-S/16，K=6，B=8。基线包括 dense DeiT-S/16、GMoE、EMoE、EMoE-L、LFME、DMDA、PC-MoE、DynMoE、MASS。
- 语言：GLUE（CoLA、MRPC、QNLI、MNLI、RTE），骨干 BERT-large，K=16，B=16。基线包括 dense BERT-large、fixed top-k MoE（k∈{1,2,4,8}）、DynMoE、MASS。

**主要结果**：
- **DomainBed**：UniF-MoE 平均准确率 **69.5%**，领先于 GFMoE（67.9%）、LFME（68.5%）等所有基线；在 PACS（89.6%）、VLCS（81.7%）、DomainNet（49.4%）上获最优，与 GMoE 并列 OfficeHome（74.2%）。
- **GLUE**：UniF-MoE 平均 **82.76%**，优于所有固定 k 变体（even per-task oracle 81.95%）及 DynMoE（81.64%）、MASS（82.19%）；在全部五个子任务上均获最优。
- **效率**：在 VLCS 上，相对 top-2 GMoE，激活参数减少 9.1%，FLOPs 减少 16.1%，推理时间降低 45.2%，内存降低 52.7%，是所有 MoE 基线中实测推理性能最优者。

## 相关工作脉络
- **DeepSeekMoE / Union-of-Experts**：添加显式共享专家或在 routing-neuron 上构建虚拟共享专家，但未利用 token 级共享分配来定义 specialization 应处理的内容。UniF-MoE 通过 block-wise 共享-残差分解显式建模这一依赖。
- **Emergent MoE / MoSE / MoNE**：关注 FFN 内部的细粒度计算（key centroids、slimmable/nested experts），但共享与细粒度决策独立、无顺序关联。UniF-MoE 通过单一预算协调二者。
- **DynMoE / MASS / Alloc-MoE**：动态调整活跃专家数，但未考虑共享响应对残差需求的影响。UniF-MoE 使动态路由条件于前置共享分配。
- **Orthogonal/Variance / MP-MoE / PC-MoE**：通过正交性、方差或协方差鼓励专家专业化，但作用于专家输出空间。UniF-MoE 的 Gram 正则直接作用于路由嵌入空间，以简化路由几何。
- **Sparsely Upcycled MoE（Komatsuzaki et al.）**：提供初始通道对齐的基础，是本文诊断实验和分析的前提条件。

## 局限性与未来方向
- **TerraIncognita 上不及 LFME**：该数据集以位置依赖背景和相机视角为主，LFME 的显式域专业化更强，表明 UniF-MoE 在极端域偏移场景下仍有提升空间。
- **依赖稀疏 upcycle 初始化**：分析方法建立在专家从同一预训练 FFN 初始化的前提上，通用场景下 block 对齐假设可能减弱。
- **块数量 B 需调优**：消融显示 B=8 或 16 较优，过小则粒度粗糙，过大则 prototype 覆盖语义不足，未来可探索自适应 B 选择。
- **未扩展到 LLM 训练**：仅在 DeiT-S/16 和 BERT-large 上验证，对于更大规模模型的适用性有待检验。

## 研究启发与可借鉴点
- **"先共享再路由"的有序分解范式**可迁移至其他混合专家架构，为共享建模、细粒度计算和动态路由的联合优化提供统一视角，避免独立调参带来的次优。
- **Gram 正则化路由嵌入**是一种轻量高效的多样性促进手段，无需修改专家内部结构即可改善路由几何，可直接应用于现有 MoE 变体。
- **累积路由质量（cumulative routing mass）**决定专家数量的设计比简单 threshold 更稳定，且自然地与 token 难度挂钩，可作为动态路由的通用组件。
- **Block-wise 共享-残差分解**使 intra-expert 和 inter-expert 稀疏性可联合控制，为后续研究 MoE 内部结构的可解释性提供了分析工具。
- **诊断实验设计**（value alignment 检测、路由 rank transition、残差需求 CDF）为理解 MoE 内部行为提供了可复用的分析框架。

## 关键术语表
**UniF-MoE**：一种统一的 token 自适应 MoE 计算框架，通过"先共享、后路由剩余"的原则协调共享宽度、共享内容和残差专家数。
**Block-wise Shared-Residual Partitioning**：将每个 FFN 专家划分为 B 个对齐块，共享专家执行部分块，残差专家执行互补块。
**Shared-demand score α(x)**：由共享专家嵌入生成的 token 依赖分数，同时控制共享块数量和共享路径权重。
**Key prototype**：每个共享块的 up-projection key 均值，用于评估该块对当前 token 的优先级。
**Cumulative routing mass**：按亲和力降序累加路由得分，直至覆盖残差需求 β(x)，由此决定激活的残差专家数。
**Gram regularizer**：对 router 嵌入矩阵施加 ||W^T W − I||_F 惩罚，促进方向正交性和专家多样性。
**Sparsely upcycled MoE**：从同一预训练 FFN 复制初始化专家的 MoE，使得通道位置对齐、可比较。
**Residual demand k_ε(x)**：用 ridge-stabilized least squares 衡量分离共享响应后，按序累积残差专家输出以恢复原始输出所需的专家数量。

## 可复现要素
- **数据集**：DomainBed（PACS、VLCS、OfficeHome、TerraIncognita、DomainNet）公开；GLUE（CoLA、MRPC、QNLI、MNLI、RTE）公开。
- **代码**：已开源，https://github.com/existence0420/UniF-MoE
- **权重**：论文未提及单独权重发布
- **关键超参**：视觉 K=6, B=8；语言 K=16, B=16；λ_div=0.01（经消融确定最优）；学习率依基线设置（DomainBed 范围 1e-5~5e-5，GLUE 范围 2e-5~5e-5）；AdamW optimizer，FP16 mixed precision，dropout=0.1，stochastic depth=0.1。
