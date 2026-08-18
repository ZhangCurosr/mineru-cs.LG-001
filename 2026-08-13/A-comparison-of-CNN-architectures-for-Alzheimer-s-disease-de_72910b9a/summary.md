---
title: "A-comparison-of-CNN-architectures-for-Alzheimer-s-disease-de"
source: https://arxiv.org/pdf/2608.11762v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 07:16:09"
field: "医学图像计算与阿尔茨海默病辅助诊断"
keywords: ["Alzheimer's disease", "CNN benchmark", "transfer learning", "MRI classification", "medical imaging", "OASIS dataset", "single-view MRI", "two-stage fine-tuning"]
innovations: ["统一协议的十架构 CNN 基准评测消除比较偏差", "两阶段冻结-解冻迁移学习流程适配小样本医学图像分类", "单通道 ImageNet 权重均值适配方法统一应用于多架构"]
benchmarks: ["OASIS single-view MRI 四分类", "VGG16 测试精度 0.9533", "ResNet18 测试精度 0.9400 (成本效益最优)"]
---

# 论文速读：A-comparison-of-CNN-architectures-for-Alzheimer-s-disease-detection-in-single-view-MRI-scans

## 一句话总结
本文在统一实验协议下对十种经典 CNN 架构（VGG16、ResNet 系列、DenseNet、EfficientNet-B0、MobileNetV3-L 等）进行了针对单视角 MRI 脑扫描的阿尔茨海默病（AD）四分类基准评测，提出了一种两阶段迁移学习+全微调训练流程，并揭示了"模型越深性能不升反降"以及"所有架构均难以区分非痴呆与极轻度痴呆"两项稳定发现。

## 研究问题与动机
1. **临床动机**：阿尔茨海默病是全球第七大致死原因，目前尚无治愈手段，早期诊断可延缓进展并改善患者生活质量，因此需要高效、可靠的自动化辅助诊断工具。
2. **方法动机**：现有深度学习在 AD 分类研究中多聚焦单一模型或对比协议不一致，难以公平比较不同架构的真实相对性能，也缺乏在统一预处理、划分、采样和训练条件下进行的系统性基准评测。
3. **技术动机**：MRI 为单通道灰度图像，而主流 ImageNet 预训练模型均为三通道 RGB 设计，直接迁移存在输入通道不匹配问题，需要适配第一层卷积并保留预训练特征；同时训练数据有限（每类仅 800 张），需设计防止过拟合的训练策略。
4. **评估动机**：希望建立可复现的基准，为后续研究提供可靠的架构选择依据，而非单纯追求最高准确率。

## 核心贡献（创新点）
1. **统一协议的十架构基准评测**：在相同划分、相同预处理、相同增强、相同采样器、相同 batch size 和 epoch 数下对比十种 CNN，消除了协议差异带来的性能噪音，使比较结果更具参考价值。
2. **两阶段迁移学习+全微调训练流程**：先冻结主干 5 个 epoch 仅训练分类头（lr=1e-4），再解冻全部参数 20 个 epoch（lr=1e-5），稳定了头部适配后再进行全量微调，有效缓解小样本下的过拟合风险。
3. **单通道灰度 MRI 的 ImageNet 权重迁移适配**：将预训练权重的三个 RGB 通道在通道维取均值，生成单通道卷积核（W_gray = 1/3 Σ W_c），避免了随机初始化导致预训练特征丢失的问题。
4. **两项稳定的跨架构发现**：①在同一架构族内，模型加深反而导致测试精度单调下降（如 ResNet18 > ResNet34 > ResNet50 > ResNet101）；②所有架构均难以区分 Non-Demented 与 Very Mild Demented 两类，错误占比达 51%–79%，揭示了疾病早期过渡阶段的分类瓶颈。
5. **成本—性能权衡分析**：VGG16 以 0.9533 精度领先但训练耗时 30 分钟，ResNet18 以 0.9400 精度（差距仅 0.0133）且仅需 7 分钟训练，确立了高成本效益替代方案。

## 方法详解
### 数据集与划分
- 使用 **OASIS** 单视角 MRI 数据集，共 86,437 张，四类：Non-Demented（67,222）、Very Mild Dementia（13,725）、Mild Dementia（5,002）、Moderate Dementia（488）。
- 为控制比较条件，采用**截断策略**而非 Class Weighting/SMOTE/RUS：每类截至 800 张训练、200 张验证、150 张 held-out 测试；Moderate Dementia 原始仅 488 张，因此取 100 张训练、200 张验证、150 张测试（验证/测试使用原始图像）。
- 训练集每 epoch 通过 **Weighted Random Sampler** 保证每类 800 张样本，实时在线增强（random affine ±10°，color jitter 亮度/对比度各 0.08），验证集和测试集**不做增强**。

### 预处理与通道适配
- 输入：单通道灰度， resize 至 224×224，归一化 mean=0.5、std=0.5（非 ImageNet 标准归一化）。
- 通道适配公式（Eq. 1）：$$W_{gray}^{(1)} = \frac{1}{3} \sum_{c=1}^{3} W_c^{(1)}$$ 将预训练三通道权重平均为单通道，保留 ImageNet 预训练特征。

### 分类头设计
- 八个架构共享同一分类头（表 4）：Linear(d→256) → ReLU → Dropout(p=0.5) → Linear(256→4)。
- **VGG16** 例外：保留原预训练分类器（两层 4096 神经元），仅替换输出层为 4 类，因其占模型 89% 参数。
- **MobileNetV3-L** 例外：保留原设计 1280 隐藏层。

### 两阶段训练协议（图 3）
- **Phase 1（迁移学习）**：冻结 backbone，训练分类头，5 epoch，Adam optimizer，lr=1e-4。
- **Phase 2（全微调）**：一次性解冻全部参数，20 epoch，Adam optimizer，lr=1e-5（更低的学习率降低过拟合风险）。
- 两者均使用 Cross-entropy 损失，最佳 checkpoint 由最高验证 accuracy 选择，random seed=42。

### 评估指标
- 全局 Accuracy、Macro/Precision/Recall/F1（因四类样本均衡 150 张，Macro=Weighted）、Support；报告混淆矩阵。

## 实验与结果
### 数据集
- **OASIS** single-view MRI，86,437 张 224×224 单通道图像，四类均衡划分后共 3,200 训练样本（每 epoch 重采样）、800 验证、600 测试。

### 基线与主要结果（表 6）
| 排名 | 架构 | 验证精度 | 测试精度 | 错误数 |
|------|------|----------|----------|--------|
| 1 | **VGG16** | **0.9637** | **0.9533** | 28 |
| 2 | ResNet18 | 0.9375 | 0.9400 | 36 |
| 3 | ResNet34 | 0.9525 | 0.9400 | 36 |
| 4 | ResNet50 | 0.9363 | 0.9333 | 40 |
| 5 | ResNeXt50-32x4d | 0.9388 | 0.9250 | 45 |
| 6 | ResNet101 | 0.9388 | 0.9217 | 47 |
| 7 | DenseNet121 | 0.9300 | 0.9167 | 50 |
| 8 | EfficientNet-B0 | 0.8963 | 0.9033 | 58 |
| 9 | DenseNet169 | 0.9350 | 0.9000 | 60 |
| 10 | MobileNetV3-L | 0.9163 | 0.8817 | 71 |

- 最强结果：**VGG16 测试精度 0.9533**（Precision=0.9536, Recall=0.9533, F1=0.9531），仅 28 个错误。
- 次优性价比：**ResNet18 测试精度 0.9400**，训练时间仅 7 分钟（VGG16 需 30 分钟），差距仅 0.0133 精度。
- 最大架构（VGG16 134.3M 参数）与最小架构（MobileNetV3-L 4.2M 参数）差距仅 0.072 个百分点。
- 过拟合情况：多数模型 val→test 差距 <0.02；DenseNet169 和 MobileNetV3-L 超 0.03，且为表现最差的两模型。

### 关键发现
1. **深度不提升精度**（表 7）：ResNet 族、DenseNet 族内加深均导致测试精度单调下降；ResNet18（11.3M）> ResNet101（43.0M）。
2. **非痴呆与极轻度痴呆难以区分**（图 4、5）：所有模型 51%–79% 错误集中于 ND↔VMD 边界；VGG16 中 22/28（78.6%）错误在此边界。
3. **模型偏好 AD 阳性分类**（表 8）：VGG16 在 Mild MD recall=0.9867、Moderate MD recall=1.0000，但 Non-Demented recall 仅 0.8800，有利于临床谨慎策略。
4. **两阶段训练有效性**（图 7）：Phase 1 验证 loss 不稳定，Phase 2 立即稳定；最佳 checkpoint 出现在 epoch 22（val acc=0.9637），此时 train acc 已达 1.0000，说明模型仍在泛化。

## 相关工作脉络
1. **OASIS 数据集与早期 AD 影像研究**（Marcus et al., 2007，引用 [16]）：OASIS 是开放 Access Series of Imaging Studies，本文在其基础上构建了统一的单视角 MRI 分类基准，区别于以往在 OASIS 上使用 3D 体素或全脑分割的做法，聚焦 2D 切片级别的高效分类。
2. **VGG/ResNet/DenseNet/EfficientNet/MobileNet 系列**（引用 [20]-[25]）：本文系统性比较了 Plain deep stack、Residual、Grouped residual、Densely connected、Compound scaled、Mobile/NAS 六大设计家族，填补了这些架构在 AD 单视角 MRI 分类任务上的横向对比空白。
3. **深度学习 AD 分类综述**（引用 [13]-[15]）：Shaikh et al.（2025）、Nagarajan & Lakshmi Priya（2025）、Raj et al.（2025）综述了 AD 检测中的深度学习方法，本文为其提供了具体的架构级实证基准，补充了"哪种架构最优"的实证答案。
4. **迁移学习在医学影像中的应用**：本文提出的两阶段 freeze-then-unfreeze 策略与医学影像中常见的渐进式微调思想一致，但本文的创新在于将该策略以统一协议应用于十种架构，验证其跨架构有效性。
5. **通道适配方法**：将 ImageNet 三通道权重平均为单通道的方法（Eq. 1）在医学图像处理中已有类似实践，但本文将其作为统一预处理步骤对所有十种架构一致应用，保证了公平比较。
6. **数据不平衡处理方法对比**：本文选择截断策略而非 Class Weighting/SMOTE/RUS，理由是目标是公平比较架构而非最大化精度，这一设计决策明确了基准评测与性能优化的不同定位。

## 局限性与未来方向
1. **切片级划分导致数据泄露**：同一受试者的不同切片可能同时出现在训练集和测试集中，违反了独立的 subject-level 假设；未来需改为受试者级划分。
2. **分类头不统一**：VGG16 保留了预训练分类器而其余架构使用随机初始化共享头，引入了一定的比较偏差。
3. **单队列泛化性未知**：所有图像来自单一 OASIS 数据集，未测试不同扫描仪和采集协议的域偏移（domain shift）泛化能力。
4. **仅使用 2D 单视角切片**：缺少 3D 体积信息，早期 AD 的结构变化在 2D 切片上可能更为隐晦。
5. **未覆盖 Vision Transformer 架构**：未来可扩展到 ViT 等最新架构的比较。
6. **训练时间测量依赖特定 GPU**：RTX 5060 上的时间测量结果在其他实验环境下可能不同。

## 研究启发与可借鉴点
1. **两阶段迁移学习策略可直接迁移**：5 epoch 冻结主干+20 epoch 全微调的流程适用于小样本医学图像分类任务，当训练数据有限（每类数百张）且模型参数较大时，可有效缓解过拟合。
2. **单通道权重适配方法通用**：ImageNet 三通道权重取均值生成单通道卷积核的方法（Eq. 1）可推广到任何单通道医学图像分类任务（CT、X 光、超声等），避免从头训练。
3. **截断策略 vs. 重采样策略的实验设计启示**：当目标是架构公平比较而非追求最高精度时，截断策略（cap-based balancing）比 SMOTE/Class Weighting 更能保证比较的纯粹性，这一设计原则值得在其他基准研究中借鉴。
4. **"深度不等于性能"的实证警示**：在数据量有限的情况下，盲目加深模型可能导致性能下降而非提升；对于资源受限场景（如移动端 AD 筛查），小型架构（ResNet18、DenseNet121）可能是更优选择。
5. **Early-stage AD 分类瓶颈可作为创新切入点**：ND↔VMD 边界是所有架构的共同弱点，未来可在特征表示学习、对比学习、或引入时序/多视角信息等方面尝试突破这一瓶颈，作为潜在的创新点。

## 关键术语表
**Alzheimer's Disease (AD)**：一种进行性神经退行性疾病，以记忆衰退和认知功能丧失为主要特征，是目前老年人痴呆最常见的原因。
**OASIS (Open Access Series of Imaging Studies)**：公开的脑影像学数据集，包含多名受试者的 MRI 扫描数据，按痴呆程度分为四类标签，广泛用于 AD 研究。
**Single-view MRI**：单视角磁共振成像切片，本文使用 2D 轴状面切片而非 3D 体数据，降低了计算成本但可能丢失空间上下文信息。
**Two-stage transfer learning**：两阶段迁移学习，第一阶段冻结预训练主干仅训练分类头，第二阶段解冻全部参数进行全量微调，以在小样本下稳定训练。
**Fully fine-tuning**：全微调，解冻模型所有参数并继续训练，使 backbone 适配目标域数据分布。
**Weighted Random Sampler**：加权随机采样器，根据类别频率的倒数设置采样权重，确保每个 epoch 中各类别样本数均衡。
**Macro-average**：宏平均，对各类别的指标（Precision/Recall/F1）简单算术平均，不受类别样本数不均衡影响。
**Held-out test split**：预留测试集，训练和验证过程中完全不可见，仅用于最终报告性能，避免数据泄露。

## 可复现要素
- **数据集**：OASIS single-view MRI，公开可用（https://www.oasis-brains.org/），但原始 86,437 张图像需申请访问。
- **代码**：论文未明确声明开源代码仓库，仅提及使用 Python 3.12、PyTorch、torchvision、scikit-learn、NumPy。
- **权重**：使用 torchvision 提供的 IMAGENET1K V1 预训练权重（`pretrained=True`），公开可得。
- **关键超参**：输入分辨率 224×224，单通道；归一化 mean=0.5, std=0.5；batch size=4；Phase 1: 5 epoch, lr=1e-4；Phase 2: 20 epoch, lr=1e-5；Optimizer: Adam；Loss: Cross-entropy；Random seed: 42。
- **增强策略**：random affine（degrees=10）、color jitter（brightness=0.08, contrast=0.08），仅应用于训练集。
