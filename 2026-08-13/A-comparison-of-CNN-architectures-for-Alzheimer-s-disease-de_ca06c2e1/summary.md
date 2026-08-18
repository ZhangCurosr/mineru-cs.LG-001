---
title: "A-comparison-of-CNN-architectures-for-Alzheimer-s-disease-de"
source: https://arxiv.org/pdf/2608.11762v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 07:16:12"
field: "医学影像分类与深度学习基准测试"
keywords: ["Alzheimer's disease", "CNN benchmark", "transfer learning", "medical imaging", "MRI classification", "OASIS dataset", "multi-class classification"]
innovations: ["两阶段迁移学习+全量微调训练管道", "统一协议下10种CNN架构的系统性基准测试", "揭示深度与性能的负相关及Non-Demented/Very Mild Demented边界分类难题"]
benchmarks: ["OASIS单视图MRI数据集", "四类阿尔茨海默病分级分类"]
---

# 论文速读：A-comparison-of-CNN-architectures-for-Alzheimer's-disease-detection-in-single-view-MRI-scans

## 一句话总结
本文在统一的评估协议下对 10 种 CNN 架构进行了基准测试，用于阿尔茨海默病（AD）四分类检测；提出了两阶段迁移学习+全量微调的训练管道，发现 VGG16 性能最佳（测试准确率 95.33%），且同一架构族内加深网络反而降低性能。

## 研究问题与动机
- 阿尔茨海默病目前无有效治愈手段，早期检测对延缓病程、改善生活质量至关重要。
- 现有 AD 检测研究多聚焦单一模型，缺乏统一协议下多架构的系统性比较，难以客观评估不同设计的真实优劣。
- OASIS 数据集类别严重不均衡（Non-Demented 占 77.77%，Moderate Dementia 仅占 0.56%），如何在受限条件下公平比较是挑战。
- 在仅 3,200 张训练图像的有限数据下，大参数模型容易产生过拟合，需要探索合适的训练策略与架构选择。

## 核心贡献（创新点）
- 建立了统一的 CNN 基准测试协议：同一数据集划分、预处理、数据增强、采样器、批大小、训练轮数与优化器，消除实验条件不一致带来的比较偏差。
- 提出了两阶段迁移学习与全量微调管道：前 5 个 epoch 冻结骨干网络只训练分类头，后 20 个 epoch 一次性解冻全部参数进行全量微调，缓解小样本下大模型的过拟合风险。
- 揭示了"深度并非越多越好"的现象：同一 ResNet/DenseNet 族系中，增大深度与参数量均导致测试准确率单调下降。
- 发现所有架构一致性地难以区分 Non-Demented 与 Very Mild Demented 两类（该边界错误占比 51%–79%），归因于早期病理变化的结构相似性。
- 给出性能与计算成本的权衡分析：VGG16 虽最优（95.33%），但训练耗时 30 分钟；ResNet18 以 1/4 训练时间达到 98.6% 的精度，是更经济的替代方案。

## 方法详解
- **数据集与划分**：使用 OASIS 单视图 MRI 数据集（86,437 张），按类别均衡截断至 2,500 张原始图像（每类 800 张），其中 Moderate Dementia 因自然上限 488 张仅取 100 张。划分：训练 2,500 / 验证 800 / 测试 600（每类 150 张），测试集为 held-out，全程不参与训练或验证。
- **预处理**：图像统一转为单通道灰度， resize 至 224×224，归一化使用 mean=0.5、std=0.5（而非 ImageNet 的 RGB 三通道归一化）。
- **首层卷积权重重初始化**：由于 ImageNet 预训练权重期望 3 通道 RGB 输入，将第一层卷积核沿通道维度取平均适配单通道：
  $$
  \mathbf{W}_{\text{gray}}^{(1)} = \frac{1}{3} \sum_{c=1}^{3} \mathbf{W}_{c}^{(1)}
  $$
  保留 ImageNet 预训练特征，避免随机初始化丢失先验知识。
- **加权随机采样器（Weighted Random Sampler）**：每 epoch 固定抽取 3,200 个样本（每类 800 个），对样本不足的类别（如 Moderate Dementia）通过重采样与在线数据增强补足。增强操作包括随机仿射变换（旋转 ±10°）与色彩抖动（亮度 ±0.08、对比度 ±0.08），仅在训练集生成，验证/测试集不使用增强。
- **两阶段训练协议**：
  - Phase 1（迁移学习）：冻结骨干网络，仅训练分类头，Adam 优化器，学习率 $1 \times 10^{-4}$，5 epochs。
  - Phase 2（全量微调）：一次性解冻全部参数，Adam 优化器，学习率降至 $1 \times 10^{-5}$，20 epochs。
  - 损失函数：Cross-entropy；模型选择：验证集最高准确率的 checkpoint。
- **分类头设计**：8 个架构共用同一 head（Linear(d→256) → ReLU → Dropout(0.5) → Linear(256→4)）；VGG16 保留原始 4096×4096 分类器（仅替换输出层）；MobileNetV3-L 保留 1280 隐藏层的原始结构。
- **评估指标**：Accuracy、Macro/Weighted Precision、Recall、F1、Support（因各类样本数相等，Macro 与 Weighted 一致，故仅报告 Macro）。

## 实验与结果
- **数据集**：OASIS，4 类（Non-Demented、Very Mild Demented、Mild Dementia、Moderate Dementia），测试集 600 张（每类 150 张）。
- **最佳模型**：VGG16，验证准确率 0.9637，测试准确率 0.9533，Precision 0.9536，Recall 0.9533，F1 0.9531，错误数 28 个。
- **次优模型**：ResNet18 / ResNet34，测试准确率均为 0.9400；ResNet18 训练仅需 7 分钟（VGG16 需 30 分钟），性价比最优。
- **最差模型**：MobileNetV3-L，测试准确率 0.8817，错误数 71 个。
- **最大跨度**：最佳与最差相差 0.072（7.2 个百分点），差距并不悬殊，说明两阶段训练协议对各架构均有良好稳定性。
- **过拟合情况**：多数模型验证→测试差距 <0.02；例外为 DenseNet169 和 MobileNetV3-L（差距 >0.03），二者同时也是性能最差的模型。VGG16 尽管参数最多（134.3M），但 backbone 仅 14M，验证→测试差距仅 +1.04pp，未出现严重过拟合。
- **逐类性能（VGG16）**：Moderate Dementia 完美分类（Precision/Recall/F1 均为 1.0000）；Mild Dementia 达 0.9867 Recall；Non-Demented Recall 最低（0.8800），与 Very Mild Demented 混淆最多（22/28 错误落在该边界）。

## 相关工作脉络
- **VGG 系列（VGG16）**：经典平面堆叠卷积网络，结构简单但参数量大，本文证明其在有限数据场景下仍是最优选择。
- **ResNet 系列（ResNet18/34/50/101, ResNeXt）**：引入残差连接解决深层网络退化问题，但本文发现加深并未带来提升，反而因容量-数据不匹配导致过拟合。
- **DenseNet 系列（DenseNet121/169）**：密集连接促进特征复用，但在本任务中 Deep 版本（DenseNet169）性能劣于轻量版本（DenseNet121）。
- **EfficientNet-B0**：复合缩放方法平衡深度、宽度与分辨率，在小参数下表现尚可（0.9033），但未超越简单架构。
- **MobileNetV3-L**：面向移动端部署的 NAS 设计，参数最少（4.2M），但性能最低（0.8817），且训练时间与 ResNet18 相当，效率优势未体现。
- **OASIS 数据集**：开放阿尔茨海默病影像研究数据集，广泛用于 AD 分类基准，本文首次在该数据集上进行大规模统一协议 CNN 基准测试。
- **迁移学习与医学影像分类**：已有大量工作利用 ImageNet 预训练权重进行医学图像分类，本文的两阶段协议是对该范式的系统化验证与对比。

## 局限性与未来方向
- **切片级划分导致数据泄漏**：数据集按单张 2D 切片划分而非按受试者（subject-level）划分，同一受试者的切片可能同时出现在训练集和测试集，高估了真实泛化能力。
- **分类头设计不一致**：VGG16 保留了原始分类器，而其他模型使用随机初始化的统一 head，可能影响公平性比较。
- **单一数据集评估**：所有实验仅在 OASIS 上进行，未测试跨中心、跨扫描仪、跨采集协议的域偏移（domain shift）泛化能力。
- **未包含 Vision Transformer**：当前 ViT 在医学影像分类中表现优异，本文未将其纳入基准。
- **类别不均衡仅通过截断处理**：虽保证了比较公平性，但丢弃了大量 Moderate Dementia 样本（仅用 100 张），可能限制模型的判别能力。
- **未来方向**：采用受试者级划分、扩展至 ViT 架构、在外部测试集上评估域泛化、探索更好的类别不均衡处理策略。

## 研究启发与可借鉴点
- **两阶段训练策略**：冻结骨干+头部热身→全量微调，对小样本医学图像分类具有良好的稳定性，可直接迁移至其他医疗影像任务。
- **单通道权重重初始化方法**：通过通道平均适配 ImageNet 预训练权重至灰度输入，是一种简洁有效的迁移学习技巧，适用于任何单通道医学图像分类任务。
- **深度与性能的负相关警示**：在同一架构族内，盲目加深网络可能因容量-数据不匹配导致过拟合；在小样本场景下，应选择适中深度架构（如 ResNet18）而非最大深度版本。
- **严格统一的基准协议**：保持数据集划分、预处理、数据增强、超参数完全一致，是公平比较不同架构的前提，该原则可推广至其他领域的模型基准测试。
- **成本-性能权衡分析**：除准确率外，还应记录训练时间与模型参数量，为实际部署提供经济可行的架构选择（如 ResNet18 作为 VGG16 的替代）。

## 关键术语表
- **OASIS 数据集**：Open Access Series of Imaging Studies，开放阿尔茨海默病影像研究数据集，包含 416 名受试者的 86,437 张单视图 MRI 扫描图像。
- **迁移学习（Transfer Learning）**：利用在大规模数据集（如 ImageNet）上预训练的模型权重，通过微调适配到下游目标任务，减少对小样本数据的依赖。
- **全量微调（Full Fine-tuning）**：解冻预训练模型的全部参数（包括骨干网络和分类头），在目标数据集上继续训练，以充分适配领域分布。
- **类别不均衡（Class Imbalance）**：数据集中各类别样本数量差异显著，本文 Non-Demented 占 77.77%，Moderate Dementia 仅占 0.56%。
- **加权随机采样器（Weighted Random Sampler）**：根据各类别样本数量分配采样权重，确保每个训练批次中各类别样本数均衡，缓解类别不均衡问题。
- **Held-out test split**：在训练和验证阶段完全隔离的测试集，仅在最终评估时使用，防止数据泄漏与信息污染。
- **Macro-average**：对各类别的指标（Precision/Recall/F1）取算术平均，每个类别同等权重，不考虑样本数量差异。
- **域偏移（Domain Shift）**：模型在训练分布之外的数据上性能下降的现象，本文未测试跨扫描仪、跨中心的域泛化能力。

## 可复现要素
- **数据集**：OASIS（Open Access Series of Imaging Studies），公开可用，需申请访问权限。
- **代码**：论文未明确开源代码仓库，但提供了详细的训练配置与超参数。
- **权重**：使用 torchvision 提供的 IMAGENET1K V1 预训练权重。
- **关键超参**：输入分辨率 224×224（单通道），归一化 mean=0.5、std=0.5，批大小 4，训练轮数 25（Phase 1: 5 epochs @ lr=1e-4，Phase 2: 20 epochs @ lr=1e-5），优化器 Adam，损失函数 Cross-entropy，随机种子 42。
- **实验环境**：Python 3.12.10，PyTorch + torchvision，scikit-learn，NumPy，NVIDIA GeForce RTX 5060 GPU，Intel Core i9 CPU。
