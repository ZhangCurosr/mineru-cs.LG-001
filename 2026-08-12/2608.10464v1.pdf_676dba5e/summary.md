---
title: "Quantum Incremental Learning with Mixed State Prototypes"
source: https://arxiv.org/pdf/2608.10464v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 04:56:24"
field: "量子持续/增量学习"
keywords: ["quantum incremental learning", "mixed-state prototype", "Hilbert-Schmidt distance", "CCPS ansatz", "catastrophic forgetting", "NISQ", "quantum metric learning", "QCNN"]
innovations: ["以可扩张混合态原型替代增宽量子电路，绕过正交基态容量瓶颈", "CCPS 混合态原型配合 softmax 权重参数化，实现类内方差的低秩密度矩阵表征并兼具 PCA-like 去噪能力", "基于 HS 距离与 SWAP 测试的高效量子度量分类，计算复杂度 O(d²) 适合反复原型拟合"]
benchmarks: ["CIFAR-100 (Split A/B/C)", "TinyImageNet"]
---

# 论文速读：Quantum Incremental Learning with Mixed State Prototypes

## 一句话总结
本文提出了一种基于可训练混合态原型（Mixed-State Prototype）的量子增量学习框架，通过 CCPS 架构在各分类共享固定宽度量子骨干网络的基础上独立构建类别原型，以 Hilbert-Schmidt 距离进行原型匹配分类，从而在 NISQ 设备受限条件下避免灾难性遗忘。

## 研究问题与动机
- **量子增量学习的容量瓶颈**：传统量子分类器依赖固定的正交基态概率或期望值构建类别分数，随着标签空间增长需扩展测量结构或增加正交基态数量；在基态编码方案中额外类别往往需要更多量子比特，而追加量子比特会改变全局量子态与希尔伯特空间，加剧灾难性遗忘并可能触发 barren plateau。
- **经典增量策略难以直接量子化**：经典深度学习中新增类别通常只需扩展最终线性层，但量子网络的结构性差异使直接迁移受阻。
- **纯态表征能力有限**：纯态仅将单个输入编码为希尔伯特空间中的单一向量，缺乏表示类内多样分布的统计容量；混合态通过正交态的凸组合可容纳数据随机性与类内方差。
- **NISQ 硬件约束**：当前量子设备电路宽度受限，需要在少量子比特条件下实现高维语义集中与可增量扩展的分类能力。

## 核心贡献（创新点）
- **提出可扩展的量子原型框架**：通过添加类别原型而非增宽共享量子骨干电路的宽度来纳入新类别，绕开正交基态数量对分类容量的限制。*本质区别在于将"扩大电路/测量空间"改为"在同一密度矩阵空间内增加独立原型"，避免扰动已学习的全局纠缠结构。*
- **引入混合态原型表征（CCPS ansatz）**：每类原型由多个经参数化变换的纯态分量的凸组合构成，借助 softmax 参数化权重整合物化概率单纯形约束；单纯态无法表达的类内多变分布可通过混合态低秩近似刻画，具有类似 PCA 的去噪特性。*与纯态原型（如 Fidelity Classifier）的本质区别在于表征层次从单一向量提升为低秩密度矩阵，能够捕获类内方差。*
- **基于 HS 距离与 SWAP 测试的高效度量**：Hilbert-Schmidt 距离可分解为经典概率碰撞项与多个单量子比特 SWAP 测试重叠项之和，计算复杂度为 O(d²)，显著低于 trace distance、Bures 距离等需要谱分解的 O(d³) 方法，适合反复的原型拟合与比较。*与前作使用 fidelity/trace distance 的方案本质区别在于训练与推理的工程可执行性与计算开销更低。*
- **原型导向的少样本回放策略**：依据样本到各类原型的 HS 距离选取最近邻代表样本存入记忆缓冲区，按类别均分内存预算（640 样本），实现无额外参数负担的旧知识保持。*与 iCaRL 等基于采样/择种策略的经典回放相比，本机制直接在量子密度矩阵空间衡量代表性，与原型分类任务天然对齐。*

## 方法详解

### 整体架构（三阶段解耦范式）
- **Stage 1 骨干网络表征学习**：ResNet-18（d_res=512）提取语义特征 → 256 维 MLP 降维 + L₂ 归一化 → 8 量子比特振幅编码 → QCNN 金字塔结构（量子卷积 + 量子池化），池化层部分求迹保留 4 量子比特子系，输出 16×16 密度矩阵 ρ(x)；附带一个两层辅助 MLP 头（隐藏层 32 维）将 16 维计算基概率向量映射至 C_vis 类 logits，以交叉熵损失 L_rep 联合优化骨干与头。
- **Stage 2 混合态原型拟合**：冻结骨干网络，每个类别 c 维护独立原型电路（4 量子比特、180 参数、3 层 VQC，每层 4 个 SU(4) 块）；原型以 CCPS 形式构造为 ρ_c = Σ_{i=0}^{K-1} w_{c,i} U_c|i⟩⟨i|U_c†，权重经 softmax 参数化；优化 L_proto^c = Σ_i w_{c,i}² − (2/N) Σ_n Σ_i w_{c,i} F_{c,i}^{(n)}，其中 F 通过 SWAP 测试估计。
- **Stage 3 记忆更新**：对所有样本计算其与各类原型的 HS 距离，各取最近邻 m = ⌊M/|V_vis|⌋ 个样本加入回放缓冲区 E。
- **Stage 4 全局推理**：query 经冻结骨干得到 ρ(x)，对可见类别 c 计算 logit_c(x) = O_c(x) − ½ Σ_i w_{c,i}²（O_c 为加权 SWAP 测试重叠和），取最大值对应类别。

### 关键公式与组件
- **CCPS 混合态原型**（Eq. 2/10/25）：$\rho_c = \sum_{i=0}^{K-1} w_{c,i} U_c|i\rangle\langle i|U_c^\dagger$，K 为可控原型秩（主实验固定 K=7）。
- **权重 softmax 化**（Eq. 11）：$w_{c,i} = \exp(s_{c,i}) / \sum_j \exp(s_{c,j})$，消除单纯形约束带来的梯度不稳定性。
- **HS 距离分解**（Eq. 7/8/27）：$\mathcal{L}_{opt}(\theta, \phi) = \sum_i p_\phi(i)^2 - 2\sum_i p_\phi(i)\mathcal{F}_i(\theta)$，仅需 SWAP 测试与经典概率计算。
- **SWAP 测试**（Eq. 6）：辅存量子比特测得 |0⟩ 概率 P(0) = ½ + ½ Tr[ρ|ψ⟩⟨ψ|]，故重叠 $\mathcal{F} = 2P(0)-1$。
- **分类 logit**（Eq. 13）：$\text{logit}_c(x) = \mathcal{O}_c(x) - \frac{1}{2}\sum_i w_{c,i}^2$，等价于 −½ D_c(x)（忽略常数查询纯度项）。
- **QCNN 骨干演化**：$\rho_{conv} = U_c(\theta_c)\rho_{in}U_c^\dagger$ → $\rho_{pool} = \text{Tr}_S[U_p\rho_{conv}U_p^\dagger]$ → $\rho(x) = U_{c2}\rho_{pool}U_{c2}^\dagger$，输出 16×16 密度矩阵。
- **表征损失**（Eq. 23）：标准交叉熵 L_rep = −Σ_c y_c log(softmax(ŷ))，仅在骨干训练阶段使用，原型拟合阶段丢弃。

## 实验与结果
- **数据集**：CIFAR-100（三个互不相交 32 类子集 Split A/B/C，共 96 类）与 TinyImageNet（32/200 类子集）；每数据集固定 16→20→24→28→32 类的五阶段增量流程。
- **基线**：经典基线包括 Linear Softmax、Cosine、Nearest Centroid、RBF-SVM（非增量）；FEA-VQC、QCNN、TTN-QNN、Fidelity Classifier（量子基线）；增量基线覆盖回放类（FOSTER、iCaRL、PODNet）、偏差校正类（BiC、SSRE、DER）、原型类（FeTrIL、PASS、NCM、Eucl-NCM、TOPIC+）、统计类（FeCAM、SDC、DeeSIL）、正则化类（ABD、MUC-LwF、LwF-MC、EWC）及 Finetune。
- **设置**：ResNet-18（512 维）+ 256 维 MLP 投影 + 8 量子比特 QCNN 骨干 + 4 量子比特原型；内存预算 640 样本（等分至各类）；K=7 统一设定；初始任务 240 epoch（BS=128，lr=0.01），原型拟合 60 epoch（lr=0.04，density BS=16），增量微调 36 epoch（lr=0.01）；SGD（momentum 0.9，wd=1e-4）+ Adam；四随机种子报告均值±标准差。
- **静态分类结果（表 I）**：混合态原型分类器在四数据集平均准确率达 0.7827，超越全部对比量子基线；较直接测量的 QCNN 平均提升约 14–16 个百分点；在 Split A（0.8323）和 TinyImageNet（0.6997）取得全部分类器最高分，在 Split B/C 略低于最优经典分类器 0.23–1.63 pp。
- **增量学习结果（表 II/III）**：
  - **Split A**：Last Acc = 0.5728，Avg Acc = 0.7178；优于 BiC、Eucl-NCM、NCM、MUC-LwF、SSRE、DeeSIL、TOPIC+、SDC、LwF-MC、EWC 等。
  - **Split B**：Last Acc = 0.4947，Avg Acc = 0.6465。
  - **Split C**：Last Acc = 0.5481，Avg Acc = 0.7097。
  - **TinyImageNet**：Last Acc = 0.3838，Avg Acc = 0.5371。
  - 性能随增量阶段呈平缓下降趋势，未出现骤降；同 rank 设定下跨数据集一致性良好。
- **消融（表 IV）**：去除混合态（w/o Mixed State）在 Split A 上 Avg Acc 从 0.7178 降至 0.6668；冻结骨干（Fixed Backbone）降至 0.6586；去除回放（w/o Replay）降至 0.6735，验证三类设计均必要。
- **秩 K 敏感性（表 V）**：最优 Avg Acc 分别在 Split A（K=12，0.7286）、Split B（K=7，0.6465）、Split C（K=15，0.7159）、TinyImageNet（K=7，0.5371）取得；统一 K=7 并非各数据集调优结果，属事前固定超参，验证了方法对 K 具有一定鲁棒性。

## 相关工作脉络
- **iCaRL / FeTrIL / FeCAM（经典原型/回放）**：这些工作以欧氏或余弦距离在经典嵌入空间做原型匹配或统计校准；本文定位差异在于将原型从"标量/向量中心"推广为"低秩密度矩阵"，并在量子 Hilbert 空间度量，直接利用量子纠缠与非线性映射能力。
- **FOSTER / PODNet / BiC / DER（回放/偏差校正）**：这类方法依赖显式样本回放或对旧类 logits 的偏差修正；本文的"混合态原型+SWAP 测试距离"提供了物理意义上的近邻度量，与基于 classical logit 重校准的路线不同。
- **Fidelity Classifier（量子纯态原型）**：同类量子原型思路，但每类仅用单一纯态；本文通过 CCPS 混合态突破纯态表征瓶颈并给出类内方差建模能力，且引入 PCA -like 低秩控制。
- **QCNN / HEA-VQC / TTN-QNN（量子分类基线）**：这些基线直接测量基态概率或期望值作为 logits；本文将"分类头"替换为"距离匹配"，从根本上绕开基态维度与类别数的耦合，更适合类别可扩展场景。
- **EWC / LwF / MUC-LwF（正则化类方法）**：通过惩罚关键参数偏离或蒸馏旧分布保持知识；本文依靠混合态原型的紧凑表示 + 少量回放维持，不依赖对庞大经典参数的精细正则。
- **Quantum State Compiling / CCPS（前作 [41]）**：CCPS 最初用于一般混合态制备近似；本文将其创造性迁移至"类别原型构造 + 增量分类"这一新任务设定，并与 HS 距离、SWAP 测试结合形成端到端框架。

## 局限性与未来方向
- **绝对精度仍逊于主流经典基线**：在 CIFAR-100/TinyImageNet 增量设定下 Last Acc 显著低于 FOSTER、iCaRL、FeCAM 等，主要受限于 8 量子比特宽度与 640 样本小内存。
- **固定 K 选择的泛化性待验证**：跨数据集最优 K 并不一致（7/12/15），统一 K=7 虽避免 overfit，但对不同数据分布的自适应能力有限。
- **仅仿真验证**：所有实验在经典模拟器上完成，未在真实 NISQ 设备上测试退相干、门误差对混合态原型稳定性的影响。
- **仅验证类增量（class-incremental）**：未涉及特征增量、任务增量或开放世界设定。
- **论文自述未来方向**：探索更硬件友好的原型电路设计；研究面向真实量子设备的噪声感知训练方法；扩展至更大规模增量场景与多样化数据集。

## 研究启发与可借鉴点
- **混合态原型 + 低秩控制可作为表征增强的通用思路**：CCPS 的 PCA-like 去噪性质启示我们，在需要"以少样本稳定刻画类分布"的任务（如 few-shot、异常检测）中，用低秩混合态替代点估计中心可获得更强鲁棒性。
- **HS 距离因 O(d²) 可分解性在重复度量场景极具价值**：对于需要频繁比较大量原型-样本对的系统（如大规模度量学习、检索），HS 距离配合 SWAP 测试是比 fidelity/trace distance 更可行的量子原生方案。
- **解耦"表征学习 + 原型匹配"的两阶段范式**：先用辅助头训练出判别性量子密度矩阵空间，再冻结骨干单独拟合原型，这一设计既保留端到端表征能力，又避免原型拟合阶段干扰骨干；可迁移至任何"共享编码器 + 多类别元对象"的任务。
- **softmax 参数化代替单纯形投影**：将概率权重转为无约束 logits 再经 softmax，避免了 penalty/truncation 引发的梯度断裂，这一技巧适用于任何带 simplex 约束的量子/经典混合优化。
- **与团队方向结合机会**：若团队关注低资源/边缘侧持续学习，可将此"轻量量子骨干 + 紧凑混合态原型"架构作为参考蓝本，或在经典对应中复用"低秩密度矩阵原型 + SWAP-like 核度量"的思路以改进传统度量学习 pipeline。

## 关键术语表
- **Incremental Learning（增量学习）**：模型在连续任务流中逐步学习新类别/新任务，同时尽可能保持对已学知识的记忆而不发生灾难性遗忘。
- **Catastrophic Forgetting（灾难性遗忘）**：神经网络在适应新任务时，原先存储的知识因权重大幅更新而被快速抹除的现象。
- **Mixed State / Density Matrix（混合态/密度矩阵）**：描述含经典不确定性或与环境纠缠的量子系统的算子 ρ，可对角化为多个正交纯态的加权混合。
- **CCPS（Convex Combination of Pure States）ansatz**：一种无需辅助量子比特的混合态制备架构，将目标混合态近似为 K 个经同一参数化酉变换作用的计算基态的加权叠加。
- **Hilbert-Schmidt (HS) Distance（希尔伯特-施密特距离）**：两密度矩阵的 Frobenius 范数平方距离，可分解为纯度项与交叉重叠项，量子计算开销为 O(d²)。
- **SWAP Test（交换测试）**：用一个辅存量子比特与受控 SWAP 门估计两量子态重叠 Tr[ρσ] 的标准电路，测量辅存 |0⟩ 概率可得 ½ + ½Tr[ρσ]。
- **QCNN（Quantum Convolutional Neural Network）**：将经典 CNN 的卷积/池化思想移植到量子线路的金字塔型架构，通过参数化酉操作与部分求迹逐步压缩量子信息。
- **Exemplar Replay（样本回放）**：从旧类中选择少量代表性样本保留在记忆缓冲区，在新任务训练中与之混合以稳固旧知识。

## 可复现要素
- **数据集**：CIFAR-100、TinyImageNet（公开数据集）；类别划分协议细节见代码库。
- **代码**：开源，地址 https://anonymous.4open.science/r/QPIL-3E43。
- **权重**：论文未提供预训练权重下载链接。
- **关键超参**：QCNN 骨干 8 量子比特；原型电路 4 量子比特、K=7（统一固定）、每类 180 参数；ResNet-18 输出 512 维 → 256 维 MLP → 振幅编码；内存预算 640 样本（等分至可见类）；初始任务 240 epoch（BS=128，lr=0.01）；原型拟合 60 epoch（lr=0.04，density BS=16）；增量微调 36 epoch（lr=0.01）；SGD（momentum=0.9，wd=1e-4）+ Adam；warmup cosine → cosine scheduler；四种子统计显著性实验。
- **运行环境**：PyTorch + PennyLane；NVIDIA GeForce RTX 3090（24GB）；x86 平台。
