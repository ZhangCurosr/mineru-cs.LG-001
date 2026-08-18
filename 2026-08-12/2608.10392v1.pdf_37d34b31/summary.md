---
title: "Share First, Route What Remains:"
source: https://arxiv.org/pdf/2608.10392v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 03:39:42"
field: "高效深度学习架构"
keywords: ["Mixture-of-Experts", "token-adaptive computation", "shared experts", "dynamic routing", "sparse MoE", "domain generalization"]
innovations: ["揭示共享计算与残差路由的序贯依赖关系，提出'先共享后路由'原则", "UniF-MoE统一框架：通过单一Token预算协调共享宽度、共享内容和残差专家数量三步序贯决策"]
benchmarks: ["DomainBed", "GLUE"]
---

# 论文速读：Share First, Route What Remains: A Unified Framework for Token-Adaptive MoE Computation

## 一句话总结
本文提出 UniF-MoE，一个统一框架，将专家混合（MoE）中的共享计算、细粒度计算与动态路由整合为序贯决策："先提取可复用计算，再对剩余部分进行路由"。通过在 Token 级别协调共享块数量、共享内容选择与残差专家数量，该方法在视觉与语言基准上均实现了优于现有静态与动态 MoE 的精度-效率平衡。

## 研究问题与动机
- **固定路由忽视计算单元冗余**：传统 top-k MoE 将完整专家作为计算单元，当选中专家存在共同响应时产生冗余计算；同时所有 Token 获得相同专家数量，无法区分简单与复杂 Token 的需求差异。
- **三种优化方向相互孤立**：共享专家设计、内部细粒度方法和动态路由通常独立发展，但共享计算的选择本质上改变了剩余需求和专家偏好，三者之间存在不可忽略的依赖关系。
- **关键实验洞察**：
  - co-activated expert 对在约 80% 的 value 位置上存在高对齐（Pearson 相关 0.697），说明复用性集中在特定 expert 对中。
  - 移除对齐位置后重训路由，仅 5.7% 的 Token 保持原 top-2 集合，说明共享提取后路由决策需重新评估。
  - 共享覆盖率越高，恢复原始输出所需的残差专家数越少（相关系数 −0.673）。
- **缺乏统一建模**：现有方法缺乏对"先共享、后路由"这一依赖关系的显式建模。

## 核心贡献（创新点）
1. **揭示路由条件的依赖关系**：发现稀疏上采样 FFN 中可复用计算与 Token 专属计算存在路由条件依赖，将共享建模、细粒度计算和动态路由统一为序贯原则，而非三个独立参数。与已有工作本质区别：前作将三者作为并行机制开发，本文揭示其内在的因果依赖。
2. **UniF-MoE 统一框架**：通过单一 Token 依赖预算耦合共享宽度、共享内容和残差专家数三个决策，协调 intra-expert 与 inter-expert 稀疏性。与已有工作本质区别：前作独立控制这些量，本文确保共享决策先于残差路由发生。
3. **Gram 正则化实现简洁路由几何**：对路由器嵌入施加 Gram 约束，使路由方向正交归一化，促进专家角色多样性与稀疏重叠。与已有工作本质区别：不同于复杂的辅助损失设计，本文仅用一个 Frobenius 范数正则项即达到分离与归一化的双重效果。
4. **多基准验证效率-精度优势**：在 DomainBed 和 GLUE 上优于代表性静态 MoE 和动态 MoE，同时降低激活参数、FLOPs、推理时延和显存占用。

## 方法详解

### 整体架构
每个 UniF-MoE 层包含 1 个共享专家 $E_{\mathrm{shr}}$ 和 K 个残差专家 $\{E_1, \ldots, E_K\}$，所有专家中间层宽度均为 H，划分为 B 个对齐块（每块宽度 $M = H/B$），初始化均来自同一稠密 FFN。

### 第一步：共享需求估计
扩展路由器嵌入矩阵为 $\mathbf{W}_g^\star = [\mathbf{W}_{\mathrm{shr}}, \mathbf{W}_g]$，共享专家对应新增列 $\mathbf{W}_{\mathrm{shr}}$。共享需求分数：
$$\alpha(\mathbf{x}) = \tau + (1-2\tau)\sigma(\mathbf{x}\mathbf{W}_{\mathrm{shr}}), \quad \tau = \frac{B-1}{B^2}$$
其中 $\sigma(\cdot)$ 为 sigmoid。$\alpha(\mathbf{x})$ 同时控制共享块数量和共享通路权重，保证 $b(\mathbf{x}) \in \{1, \ldots, B-1\}$（每 Token 至少 1 个共享块、至少 1 个残差块）。

### 第二步：共享块选择
每个块 b 的优先级由 up-projection keys 的均值计算：
$$\mu_b = \frac{1}{M}\sum_{h \in \mathcal{H}_b} \mathbf{K}_{\mathrm{shr}}[:, h], \quad u_b(\mathbf{x}) = \mathbf{x}\mu_b$$
Top-b(x) 个优先级最高的块构成共享索引集 $\mathcal{T}_{\mathrm{shr}}(\mathbf{x})$，其余构成残差索引集 $\mathcal{T}_{\mathrm{res}}(\mathbf{x})$。两个 Token 可使用相同共享宽度但选择不同内容。

### 第三步：累积残差专家路由
残差嵌入产生亲和分布 $\mathbf{s}(\mathbf{x}) = \mathrm{softmax}(\mathbf{x}\mathbf{W}_g)$，排序后为 $p_1 \geq p_2 \geq \cdots \geq p_K$。残差需求 $\beta(\mathbf{x}) = 1 - \alpha(\mathbf{x})$，激活满足累积亲和覆盖残差需求的最小前缀：
$$k(\mathbf{x}) = \min\left\{n : \sum_{i=1}^n p_i(\mathbf{x}) \geq \beta(\mathbf{x})\right\}$$
残差专家 j 的输出只包含残差块：$E_j^{\mathcal{R}}(\mathbf{x}) = \sum_{b \in \mathcal{T}_{\mathrm{res}}(\mathbf{x})} E_j^{(b)}(\mathbf{x})$。

### 第四步：输出合并
$$\mathbf{y} = \alpha(\mathbf{x}) E_{\mathrm{shr}}^S(\mathbf{x}) + \sum_{i=1}^{k(\mathbf{x})} p_i(\mathbf{x}) E_{q_i(\mathbf{x})}^{\mathcal{R}}(\mathbf{x})$$
其中 $E_{\mathrm{shr}}^S$ 为选定的共享块之和。总系数质量受控：$\alpha + P_k < 1 + p_k$。

### 训练目标
$$\mathcal{L} = \mathcal{L}_{\mathrm{task}} + \lambda_{\mathrm{div}} \|(\mathbf{W}_g^\star)^\top \mathbf{W}_g^\star - \mathbf{I}_{K+1}\|_F$$
Gram 正则化使嵌入矩阵近似正交，初始化也采用正交归一，促进路由方向多样性和专家稀疏重叠。

### 计算量
Token x 激活的块数：$C_B(\mathbf{x}) = b(\mathbf{x}) + k(\mathbf{x})[B - b(\mathbf{x})]$，共享块仅执行一次，重复成本限制在残差通路上。

## 实验与结果

### 数据集
- **视觉（DomainBed）**：DeiT-S/16 骨干（ImageNet 预训练，d=384, H=1536），K=6, B=8，Transformer 层 8/10 替换为 UniF-MoE 层。评测 PACS, VLCS, OfficeHome, TerraIncognita, DomainNet。
- **语言（GLUE）**：BERT-large 骨干，K=16, B=16，Transformer 层 20/22 替换。评测 CoLA, MRPC, QNLI, MNLI, RTE。

### 主要结果
**Vision（DomainBed 平均）**：
| 方法 | Avg. Accuracy |
|---|---|
| DeiT-S/16 | 65.5% |
| LFME | 68.5% |
| DynMoE | 67.9% |
| **UniF-MoE** | **69.5%** |

- PACS: 89.6%（+0.2 over PC-MoE 89.4%）
- VLCS: **81.7%**（最优，+0.6 over MASS 81.1%）
- TerraIncognita: 52.6%（LFME 53.4% 领先，因该数据集强依赖位置背景）

**Language（GLUE 平均）**：
| 方法 | Avg. Score |
|---|---|
| BERT-large | 81.37% |
| Top-8 MoE | 81.45% |
| MASS | 82.19% |
| **UniF-MoE** | **82.76%**（全部 5 个子任务均最优） |

- CoLA: 66.83%, MRPC: 91.57%, QNLI: 93.10%, MNLI: 86.84%, RTE: 75.47%

**效率对比（VLCS 上）**：
- 相对 top-2 GMoE：激活参数减少 9.1%，FLOPs 减少 16.1%，推理时延减少 45.2%，显存减少 52.7%
- 推理时延 I-TPS: 0.17s（vs. DynMoE 1.06s, GMoE 0.31s）
- 推理显存 I-RTM: 0.26 GiB（最低）

### 消融结果
| 配置 | DomainBed Acc | GLUE Acc | 平均块数 |
|---|---|---|---|
| 固定 α=0.4 | 68.20% | 81.73% | 14.17 |
| Top-2 残差激活 | 68.90% | 82.07% | 13.41 |
| 前缀块选择 | 68.80% | 81.95% | 13.06 |
| **全自适应** | **69.50%** | **82.76%** | **11.41** |

- 固定 α 影响最大（准确性下降最多），说明共享-残差分割是核心决策
- 三阶段联合增益大于各阶段之和，证实协同效应

### 正则化分析
- λ_div = 0.01 为最优超参；L_div 将平均 co-activation 降低 62.9%

## 相关工作脉络
1. **共享专家机制（DeepSeekMoE, Union-of-Experts）**：学习独立的共享专家或虚拟共享路由神经元，但未利用 Token 级共享分配来定义后续专业化计算需要处理的内容——本文将其显式化为首个序贯步骤。
2. **专家稀疏化/正交化（G-Switcher, MP-MoE）**：通过正交性和方差目标减少重叠，增强专门化，但未建立与共享计算需求的依赖关系——本文通过共享先行的方式自然引导稀疏化。
3. **动态路由（DynMoE, MASS, Alloc-MoE）**：通过累积置信度或约束预算改变专家数量，但这些方法独立于共享计算做决策——本文的残差专家数由累积亲和与共享后剩余需求共同决定。
4. **细粒度计算（Emergent MoE, MoSE, Mixture of Nested Experts）**：在 FFN 内部变化计算量（宽度/模块选择），但未考虑共享响应对后续路由的影响——本文通过分块共享直接消除跨专家重复。
5. **稀疏上采样（Sparse Upcycling, Komatsuzaki et al.）**：从稠密 checkpoint 初始化 MoE 专家，使 channel 位置可对齐——本文利用这一特性进行 value 位置对齐分析，揭示共享-残差依赖。

## 局限性与未来方向
- **规模验证有限**：仅在 DeiT-S/16 和 BERT-large 上验证，大语言模型（如 7B+）场景下 block 粒度选择的可扩展性有待验证。
- **块划分方式的潜在局限**：当前按 fixed block index 划分，若最优复用模式跨越固定块边界，则可能无法精确捕获。
- **TerraIncognita 上的不足**：该方法在强位置依赖背景下不及 LFME 等显式域专门化方法，说明对某些任务类型可能需要额外域信息。
- **Gram 正则化强度敏感**：λ_div 过大会压制任务损失，过小则路由方向仍相关，需精细调参。
- **单次共享提取假设**：共享块选择基于单轮分析，可能存在次优共享内容导致残差需求被低估的情况。

## 研究启发与可借鉴点
1. **"序贯决策优于并行参数"的设计哲学**：将共享建模、细粒度选择和动态路由整合为有序三步，而非独立参数调节，这一思想可迁移到其他稀疏计算场景（如多模态 MoE、跨层资源共享）。
2. **通过内部结构分析揭示依赖关系**：将 FFN 分解为 key-value channels，通过 value 对齐度量化复用性，这一"解剖式"分析方法可用于探索其他网络组件的隐式依赖。
3. **Gram 正则化的简洁有效性**：用单一 Frobenius 范数同时实现嵌入归一化和正交化，比复杂的对比/互信息损失更简洁，可作为路由模块的通用正则化工具。
4. **Token 级自适应的粒度意义**：全 Token 级别的共享-残差预算分配揭示了计算需求的细粒度变化，提示可进一步探索 layer-level 或 position-level 的自适应。
5. **累积路由质量的控制**：累积亲和覆盖策略提供了自然的质量-计算权衡，类似思想可应用于 beam search 或其他资源受限的序列决策场景。

## 关键术语表
- **Mixture-of-Experts (MoE)**：通过稀疏路由将输入分配到多个专家网络进行计算的架构，仅在推理时激活少量专家以节省计算。
- **Sparse Upcycling**：从预训练稠密 FFN checkpoint 初始化各专家权重，保留 channel 级对齐性，使跨专家的可复用部分可比较。
- **Token-Adaptive Computation**：根据每个 Token 的语义需求动态调整计算量的机制，而非对所有 Token 使用固定预算。
- **Residual Expert Demand**：共享计算提取后剩余的专家容量需求，由 Token 的互补信息需求决定，而非全局固定值。
- **Gram Regularization**：对路由器嵌入矩阵施加 $(\mathbf{W}^\top\mathbf{W} - \mathbf{I})_F$ 约束，使其列向量近似正交归一。
- **Key-Value Channel Decomposition**：将 FFN 的输出分解为各 hidden channel 的加权和，其中 key 决定激活、value 决定输出方向。
- **Cumulative Routing Mass**：按专家亲和排序后累积的权重总和，用于判断激活多少专家即可覆盖剩余计算需求。
- **Shared-Demand Score**：由共享专家嵌入产生的标量分数，同时控制共享块数量和共享通路的全局权重。

## 可复现要素
- **数据集**：DomainBed（PACS, VLCS, OfficeHome, TerraIncognita, DomainNet）和 GLUE（CoLA, MRPC, QNLI, MNLI, RTE），均为公开数据集。
- **代码开源**：https://github.com/existence0420/UniF-MoE
- **关键超参**：
  - Vision: K=6, B=8, d=384, H=1536, λ_div=0.01, hidden dropout=0.1, stochastic-depth=0.1
  - Language: K=16, B=16, max_seq_len=128, batch_size=32, AdamW, lr∈{2e-5, 3e-5, 5e-5}, weight_decay=0, 无 warm-up
- **硬件环境**：AMD EPYC 75F3 CPU, 503 GiB RAM, NVIDIA RTX 3090 GPU, PyTorch 2.4.1, CUDA 12.1
