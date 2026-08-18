---
title: "Share First, Route What Remains:"
source: https://arxiv.org/pdf/2608.10392v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 04:52:14"
field: "高效稀疏MoE架构"
keywords: ["mixture of experts", "token-adaptive computation", "shared expert", "dynamic routing", "domain generalization", "MoE efficiency"]
innovations: ["揭示共享-残差依赖并提出'先共享后路由'有序原则", "UniF-MoE通过单token预算协调共享宽度/内容与残差专家数", "Gram正则化促进路由嵌入正交与专家多样性"]
benchmarks: ["DomainBed", "GLUE"]
---

# 论文速读：Share First, Route What Remains: A Unified Framework for Token-Adaptive MoE Computation

## 一句话总结
本文揭示了稀疏上采样MoE中可复用计算与残差专家路由之间的内在依赖关系，提出"先共享、后路由剩余"的有序原则，并将其实例化为UniF-MoE统一框架——通过一个token级预算协调共享宽度、共享内容选择与残差专家数量，在视觉和语言基准上均优于现有静态/动态MoE，同时降低激活计算量、推理延迟与显存。

## 研究问题与动机
- **现有MoE将三个决策独立处理**：共享专家设计、细粒度专家内计算变化、动态路由器决定激活专家数，通常各自独立开发，忽略了它们之间的依赖关系。
- **可复用计算的提取会改变剩余内容**：一旦从专家中提取了可复用部分，路由所需的内容和专家偏好都会发生变化，独立决策会导致重复计算或容量分配失衡。
- **统一表述缺失**：共激活专家的value通道存在对齐，去除对齐位置后路由决策显著改变，共享覆盖率与残差专家需求呈负相关（相关系数−0.673），说明三者应被统一建模而非并行调节。

## 核心贡献（创新点）
1. **揭示路由条件依赖性**：发现稀疏上采样FFN专家中可复用与token特异性计算之间的内在依赖，将共享建模、细粒度计算与动态路由转化为单一有序原则"先共享、后路由剩余"。
2. **提出UniF-MoE统一框架**：通过一个token依赖预算耦合共享宽度、共享内容与残差专家数量，协调专家内部与专家间稀疏性，而非独立决策。
3. **Gram正则化促进路由多样性**：通过对路由器嵌入施加Gram约束，使嵌入正交化，促进专家角色多样化和稀疏重叠，形成简洁的路由几何结构。
4. **跨领域性能验证**：在DomainBed（视觉域泛化）和GLUE（语言理解）上，UniF-MoE在准确率-效率权衡上全面优于代表性静态与动态MoE。

## 方法详解
**块级共享-残差划分**：每层含1个共享专家$E_{shr}$和K个残差专家$\{E_1,...,E_K\}$，每个专家中间宽度H被划分为B个对齐块，每块宽度$M=H/B$，所有专家从同一稠密FFN初始化以保证块边界对齐。

**Token自适应共享建模**：
- 共享需求分数：$\alpha(x) = \tau + (1-2\tau)\sigma(xW_{shr})$，其中$\tau=(B-1)/B^2$，控制共享块数量和通路权重，保证$b(x)\in\{1,...,B-1\}$。
- 共享块选择：以每块上投影key均值$\mu_b$为原型，按优先级$x\mu_b$选top-b个块作为共享内容。

**累积残差专家路由**：
- 残差需求$\beta(x)=1-\alpha(x)$，与共享需求共享同一预算。
- 按排序亲和度$p_i(x)$累积，激活最小前缀使$\sum_{i=1}^{k(x)}p_i(x)\geq\beta(x)$，专家数$k(x)$取决于共享分配后的剩余。

**输出合并**：$y=\alpha(x)E_{shr}^S(x)+\sum_{i=1}^{k(x)}p_i(x)E_{q_i(x)}^R(x)$，总系数质量受控。

**训练目标**：$\mathcal{L}=\mathcal{L}_{task}+\lambda_{div}\|\left(W_g^\star\right)^\top W_g^\star-I_{K+1}\|_F$，Gram正则保持嵌入正交。

**计算量**：$C_B(x)=b(x)+k(x)[B-b(x)]$，共享工作仅执行一次，重复专家成本限于残差通路。

## 实验与结果
**数据集与基线**：
- 视觉：DomainBed（PACS/VLCS/OfficeHome/TerraIncognita/DomainNet）， backbone为ImageNet预训练DeiT-S/16，K=6, B=8。
- 语言：GLUE（CoLA/MRPC/QNLI/MNLI/RTE），backbone为BERT-large，K=16, B=16。
- 基线包括GMoE、EMoE、LFME、DynMoE、MASS等。

**主要结果**：
- **DomainBed平均精度69.5%**，领先各基线（GMoE 67.9%、DynMoE 67.9%），PACS 89.6%（+1.5% vs top）、VLCS 81.7%（+1.4% vs top）。
- **GLUE平均82.76%**，全五项任务最优（CoLA 66.83%、MRPC 91.57%、QNLI 93.10%、MNLI 86.84%、RTE 75.47%）。
- **计算效率**：相对top-2 GMoE，激活参数少9.1%，FLOPs少16.1%，推理时间减少45.2%，显存减少52.7%。

**消融实验**：逐一移除三个自适应决策均导致性能下降与计算增加，验证三者协调的重要性。

## 相关工作脉络
1. **共享专家建模**：DeepSeekMoE（细粒度专家分割+共享专家）、Union-of-Experts（从路由神经元构建虚拟共享专家）——本文区别在于用token级共享分配定义专业化应处理的内容。
2. **动态路由**：DynMoE（自动调优）、MASS（最优语义专业化）、Alloc-MoE（预算感知分配）——本文将它们排序为有序决策链而非独立机制。
3. **细粒度计算**：Emergent MoE（键质心模块化）、嵌套/可收缩专家（变化执行宽度）——本文通过共享-残差预算将宽度与专家数联合决定。
4. **专家专用化**：正交/方差目标减少重叠（Guo et al. 2025）、MP-MoE（协方差选择多样专家集）——本文用Gram正则促进方向多样性，同时保留有用协作。

## 局限性与未来方向
- TerraIncognita上LFME仍更强（53.4% vs 52.6%），该数据集受位置背景和相机视角主导，显式域专业化可能更有效。
- 块粒度B需经验调优：B过小（如4）过于粗糙，B过大（如32）覆盖不足，当前取B=8为折中。
- 实验仅验证了Vision Transformer和BERT场景，对LLM的直接适用性尚待验证。
- 未探索与显存优化技术（如Tutel sharding）的深度融合。

## 研究启发与可借鉴点
1. **有序决策框架**：将多个自适应机制排序为因果链而非并行开关，是提升MoE效率的系统性思路，可迁移至其他稀疏架构设计。
2. **Gram正则化路由嵌入**：正交化路由器嵌入以获得清晰路由几何，是一种简洁有效的多样性正则手段。
3. **通道级可复用性分析**：通过value通道对齐识别可复用响应，为专家内计算分解提供了可操作的诊断工具。
4. **单预算分配**：共享与残差共享同一token级预算（$\alpha+\beta=1$），避免了多控制器间的冲突，设计简洁。
5. **与团队方向的结合机会**：该方法可与大语言模型MoE训练结合，探索token级共享-残差分解在LLM注意力层和FFN层的应用。

## 关键术语表
- **MoE（Mixture of Experts）**：通过路由器将token分配给少量专家网络，以稀疏方式激活大参数容量。
- **稀疏上采样（Sparse Upcycling）**：从预训练稠密FFN复制初始化多个专家，进行微调形成MoE。
- **共享专家（Shared Expert）**：每个token只执行一次的公共通路，处理可复用计算。
- **残差专家（Residual Expert）**：处理共享通路之后剩余token特异性计算的专家。
- **累积路由质量（Cumulative Routing Mass）**：按排序亲和度累加至满足残差需求的第一个专家前缀，决定激活专家数。
- **Gram正则化（Gram Regularizer）**：约束路由器嵌入的Gram矩阵接近单位矩阵，促进方向正交与多样性。
- **块级划分（Blockwise Partitioning）**：将专家FFN中间宽度等分为B个对齐块，实现共享与残差的细粒度分配。
- **域泛化（Domain Generalization）**：在未见过的目标域上保持良好性能，DomainBed是标准评测基准。

## 可复现要素
- **代码开源**：https://github.com/existence0420/UniF-MoE
- **数据集**：DomainBed（PACS/VLCS/OfficeHome/TerraIncognita/DomainNet）、GLUE（CoLA/MRPC/QNLI/MNLI/RTE）——均为公开基准。
- **关键超参**：K=6（视觉残差专家数）、K=16（语言）、B=8（视觉块数）、B=16（语言）、$\lambda_{div}=0.01$、学习率$1\sim5\times10^{-5}$、batch size=32。
- **模型维度**：DeiT-S/16中$d=384, H=1536$；BERT-large中依原配置。
- **训练环境**：NVIDIA RTX 3090，FP16混合精度，AdamW优化器。
