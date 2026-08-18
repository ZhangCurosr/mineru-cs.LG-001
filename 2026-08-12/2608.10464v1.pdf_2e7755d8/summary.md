---
title: "Quantum Incremental Learning with Mixed State Prototypes"
source: https://arxiv.org/pdf/2608.10464v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 04:56:22"
field: "量子机器学习/增量学习"
keywords: ["量子增量学习", "混合态原型", "量子神经网络", "灾难性遗忘", "Hilbert-Schmidt距离", "CCPS"]
innovations: ["提出基于CCPS混合态原型的量子增量学习框架，避免正交基态容量约束", "利用HS距离分解与SWAP test实现高效量子原型度量，计算复杂度低于传统距离度量", "设计两阶段解耦训练范式（辅助MLP头+冻结主干+独立原型拟合），有效缓解量子灾难性遗忘"]
benchmarks: ["CIFAR-100", "TinyImageNet"]
---

# 论文速读：Quantum Incremental Learning with Mixed State Prototypes

## 一句话总结
本文提出了一种基于可训练混合态原型的量子增量学习框架，通过在共享量子主干之外添加类原型而非增加电路宽度来解决NISQ时代量子分类器受限于正交基态数量、无法适应持续增长类别数的问题，同时利用混合态的表示能力和Hilbert–Schmidt距离的分解计算优势，有效缓解了灾难性遗忘。

## 研究问题与动机
1. **量子电路宽度受限**：NISQ时代硬件条件限制了量子电路的qubit数量，传统量子分类器依靠固定数量的正交基态概率进行分类，难以随类别增加而扩展。
2. **直接扩展量子电路会导致灾难性遗忘**：增加qubit会根本性地改变全局量子态和Hilbert空间，破坏已学习的纠缠关联，加剧灾难性遗忘。
3. **盲目扩展电路深度会引发barren plateau**：增加电路宽度或深度必然触发 barren plateau 现象，使量子网络无法训练。
4. **经典增量学习策略无法直接量子化**：经典增量学习依赖线性层扩展或参数正则化，而量子网络的结构差异使其策略不能直接移植，且数据存储受限。

## 核心贡献（创新点）
1. **可扩展模块的量子原型框架**：通过添加类原型而非增加共享量子主干电路宽度来适应新类别，突破了正交基态数量的容量约束；与经典方法直接在末端扩展线性层不同，本框架将分类从显式softmax回归转为隐式距离匹配。
2. **混合态原型 via CCPS ansatz**：用多个变换后纯态的凸组合表示单个类原型，避免引入ancilla qubit，同时具备类似PCA的噪声过滤特性；与纯态原型（Fidelity Classifier）相比，混合态能更好地刻画类内方差，且softmax参数化避免了权重约束带来的梯度不稳定问题。
3. **基于HS距离分解的高效度量**：利用HS距离的可分解性将量子重叠计算转化为经典权重与SWAP test结果的加权求和，相比trace distance/Bures距离所需的$O(d^3)$矩阵运算，HS距离仅需$O(d^2)$且更适合重复原型拟合。
4. **混合架构解耦特征学习与分类**：采用"共享量子QCNN主干 + 轻量可扩展经典MLP头 + 每类独立混合态原型电路"的三段式设计，特征学习阶段冻结主干、仅更新原型，避免了端到端训练对已学量子态的扰动。

## 方法详解
**整体架构分为两个阶段：**

**阶段1 — 特征提取主干训练**：
- 经典预处理模块：输入图像经ResNet-18提取512维特征，经线性降维至256维后用ReLU和Dropout处理，L2归一化后进行振幅编码（amplitude encoding）映射到8-qubit初始纯态$| \psi(\mathbf{x})\rangle$。
- 量子特征提取主干（QCNN）：采用金字塔结构，交替堆叠量子卷积层$U_c(\theta_c)$和量子池化层（含受控旋转$U_p(\theta_p)$+偏迹$\mathrm{Tr}_S$）。池化操作丢弃部分qubit后，保留的子系统自然进入混合态（密度矩阵$\rho(\mathbf{x}) \in \mathbb{C}^{16\times16}$）。
- 辅助MLP头：将4-qubit计算基概率分布$\mathbf{p}(\mathbf{x})$（16维）映射为类别logits，经两层MLP输出，使用交叉熵损失$\mathcal{L}_{rep}$优化主干参数$\Theta_{bb}$。此头训练完成后被丢弃。

**阶段2 — 混合态原型拟合与增量学习**：
- 每类原型电路：采用CCPS形式，由$K=7$个变换后的基态构成$\rho_c = \sum_{i=0}^{K-1} w_{c,i} U_c |i\rangle\langle i| U_c^\dagger$，其中权重$w_{c,i}$经softmax参数化以保证单纯形约束。
- 原型损失函数（推导自HS距离最小化）：
$$\mathcal{L}_{proto}^c(\theta_c, s_c) = \sum_{i=0}^{K-1} w_{c,i}^2 - \frac{2}{N}\sum_{n=1}^{N}\sum_{i=0}^{K-1} w_{c,i} \mathcal{F}_{c,i}^{(n)}$$
其中$\mathcal{F}_{c,i}^{(n)}$通过SWAP test测量，仅需1个ancilla qubit。
- 回放记忆管理：训练后，按HS距离选取各类最近样本存入回放缓冲区（每类均分预算，总量640样本）。
- 推理：对查询样本提取$\rho(\mathbf{x})$，通过加权SWAP test计算与各原型的重叠，以$\logit_c(\mathbf{x}) = \mathcal{O}_c(\mathbf{x}) - \frac{1}{2}\sum_i w_{c,i}^2$取最大者分类。

## 实验与结果
**数据集**：CIFAR-100（三个互斥32类子集Split A/B/C）和TinyImageNet（32类），每数据集从16类起始，分4步增量至32类（共5个评估阶段）。

**评估基线**：
- 非增量分类基线：Linear Softmax、Cosine Classifier、Nearest Centroid、RBF-SVM、HEA-VQC、QCNN、TTN-QNN、Fidelity Classifier。
- 增量学习基线：FOSTER、iCaRL、PODNet、BiC、SSRE、DER、LUCIR、FeCAM、FeTrIL、PASS、ABD、NCM、MUC-LwF、Eucl-NCM、Finetune、DeeSIL、LwF-MC、TOPIC+、SDC、EWC。

**主要结果**：
- **非增量分类**：本文方法在四个数据集上的平均准确率为**0.7827**，为所有量子基线最高；在Split A（**0.8323**）和TinyImageNet（**0.6997**）上超越所有对比方法；相较直接测量的QCNN提升约12.88~15.83个百分点。
- **增量学习（Last Acc / Avg Acc）**：
  - CIFAR-100 Split A：**0.5728 / 0.7178**（超过BiC、SSRE、DeeSIL、TOPIC+、SDC、LUCIR等）
  - Split B：**0.4947 / 0.6465**
  - Split C：**0.5481 / 0.7097**
  - TinyImageNet：**0.3838 / 0.5371**
- **消融实验**：移除混合态（纯态替代）导致Split A Avg Acc从0.7178降至0.6668；冻结主干降至0.6586；移除回放降至0.6735，均验证各组件必要性。
- **Rank K敏感性**：K=7为预设统一值， Sweep显示各数据集最佳K在7~12之间，中等秩实现了最优的保结构-去噪声权衡。

## 相关工作脉络
1. **iCaRL / FOSTER / PODNet（回放基线）**：经典回放方法依赖herding算法存储代表性样本；本文同样使用回放但将样本选择改为基于HS距离的原型最近邻，且量子主干仅8-qubit，参数量级为$O(n\log_2(\max(n,d)))$。
2. **FeTrIL / PASS / NCM / Eucl-NCM（原型基线）**：经典原型方法使用均值特征向量作为原型；本文扩展到量子密度矩阵空间，用混合态原型表征类内分布而非单点质心，表达能力更强。
3. **QCNN [45] / HEA-VQC [48]（量子分类器基线）**：传统量子分类器依赖基态概率测量，类别数受限于保留qubit数（$2^4=16$）；本文通过混合态原型匹配绕过此限制，且显著优于QCNN约15个百分点。
4. **Fidelity Classifier [50]（纯态原型）**：用单一纯态表示每类、以量子保真度度量；本文指出纯态无法刻画类内方差，混合态原型通过CCPS实现PCA-like降维，表达能力更强。
5. **CCPS ansatz [41]**：密度矩阵混合态编译理论来源；本文首次将其引入增量学习框架，利用其无需ancilla qubit和HS距离可分解的优势。
6. **EWC / LwF / MUC-LwF（正则化基线）**：通过惩罚关键权重或蒸馏旧模型输出来防止遗忘；本文无需参数正则化，通过距离度量+回放实现稳定增量学习。

## 局限性与未来方向
1. **仅基于仿真验证**：所有实验在经典模拟器上进行，尚未在真实量子硬件上测试噪声和误差的影响。
2. **相对经典SOTA仍有差距**：在部分数据集（如Split B）上性能落后于最强的经典方法（如FeCAM、FOSTER）。
3. **原型Rank K需预设为固定值**：当前使用统一K=7，虽经sweep验证中等秩表现良好，但未实现自适应选择。
4. **内存预算较小**：每类约20个样本的重放缓冲区可能不足以支持更多类别或更大数据集的增量学习。
5. **未来方向（论文自述）**：探索更硬件高效的量子原型电路、研究真实量子设备上的噪声感知训练方法、扩展到更大规模增量场景和多样化数据集。

## 研究启发与可借鉴点
1. **混合态建模类内分布的思路可迁移**：用凸组合纯态代替单一点表示，可推广至其他需要刻画分布多样性的量子学习任务（如量子聚类、生成模型）。
2. **辅助头+冻结主干的解耦训练策略**：先用辅助MLP头监督学习判别性量子表示，再冻结主干仅拟合原型——此两阶段范式可降低联合优化的难度，适合其他量子-经典混合架构。
3. **HS距离+SWAP test的分解计算技巧**：将量子距离度量拆解为可并行SWAP test的加权求和，大幅降低硬件开销，可复用于其他需要状态比较的量子算法。
4. **softmax参数化替代约束优化**：用softmax将概率单纯形约束转化为无约束优化，避免了 penalty function 导致的梯度不连续，对含概率权重的量子电路设计具有通用参考价值。
5. **与团队方向的结合机会**：混合态原型的PCA-like噪声过滤特性可与本团队的量子特征表示或鲁棒性研究方向结合；原型rank的可学习机制可作为进一步创新的切入点。

## 关键术语表
**Incremental Learning（增量学习）**：模型在连续数据流中 sequentially 学习新类别而尽量避免灾难性遗忘的学习范式。
**Mixed State / Density Matrix（混合态/密度矩阵）**：描述含经典概率不确定性的量子态，表示为$\rho = \sum_i \lambda_i |v_i\rangle\langle v_i|$，比纯态具有更强的信息承载能力。
**CCPS（Convex Combination of Pure States）**：无需ancilla qubit、通过变换基态的概率混合来表示低秩混合态的量子电路ansatz，兼具PCA-like噪声过滤能力。
**Hilbert–Schmidt (HS) Distance（希尔伯特–施密特距离）**：量子态间的一种度量，平方形式为$\|\rho-\sigma\|_F^2$，可通过SWAP test高效分解计算，复杂度低于trace distance。
**SWAP Test（交换测试）**：用量子电路估计两态重叠$\mathrm{Tr}[\rho\sigma]$的测量方法，仅需1个ancilla qubit，测量$|0\rangle$概率可得重叠值。
**Quantum Feature Extraction Backbone（量子特征提取主干）**：基于QCNN的混合量子-经典网络，将经典输入映射到量子密度矩阵空间的特征提取模块。
**Prototype-based Classification（原型分类）**：将样本分类为与其量子表示HS距离最近的类原型，而非依赖显式参数化分类头。
**Catastrophic Forgetting（灾难性遗忘）**：神经网络在学习新任务时，对已有知识的表征被覆盖或破坏的现象。

## 可复现要素
- **数据集**：CIFAR-100、TinyImageNet（公开）
- **代码/权重**：代码已开源，链接为 https://anonymous.4open.science/r/QPIL-3E43；论文未提及预训练权重下载
- **关键超参**：
  - 主干QNN：8-qubit振幅编码
  - 原型电路：4-qubit，每个原型180参数（3层VQC × 4个SU(4)块 × 15参数/块）
  - 原型Rank K：统一预设K=7
  - 回放缓冲区：总量640样本，均匀分配给各类
  - 优化器：SGD(momentum=0.9, weight decay=1e-4)用于主干/MLP，Adam用于原型
  - 学习率：初始任务0.01（warm-up cosine），增量微调0.01，原型拟合0.04
  - 训练轮数：初始任务240 epoch，原型60 epoch，增量微调36 epoch
  - 输入预处理：ResNet-18 + 256维MLP投影
