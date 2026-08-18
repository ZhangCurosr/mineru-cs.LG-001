---
title: "Confidence-Calibration-of-Deep-Learning-Systems"
source: https://arxiv.org/pdf/2608.12100v1.pdf
model: agnes-2.5-flash
chunks: 5
summarized_at: "2026-08-18 17:34:35"
field: "机器学习可信度与校准"
keywords: ["置信度校准", "温度缩放", "共形预测", "噪声标签", "差分隐私", "无监督域适应", "深度学习"]
innovations: ["提出NACP实现噪声标签下与类别数k无关的共形预测覆盖界", "提出LDP-CP框架支持标签隐私与分数隐私两种差分隐私CP方案", "提出UTDC通过源/目标准确率比率实现无监督域适应校准"]
benchmarks: ["CIFAR-10/100/100N", "ImageNet", "ChestX-ray14-bal", "Office-home", "VisDA-2017", "DomainNet", "TissueMNIST"]
---

# 论文速读：Confidence-Calibration-of-Deep-Learning-Systems

## 一句话总结
该博士论文系统研究了深度学习模型置信度校准在多种现实挑战（标签噪声、差分隐私约束、无监督域偏移）下的鲁棒性问题，提出了NACP、LDP-CP、UTDC等方法，在多类医学影像与自然图像任务上实现了逼近干净数据校准水平的性能。

## 研究问题与动机
- 深度学习模型（尤其医学影像）的**预测置信度可靠性与预测准确性同等重要**；过度自信的错误预测在临床场景中可能导致灾难性后果。
- 现有校准方法**假设验证集标签干净**，在医学影像等真实场景中不现实——标签噪声和域偏移普遍存在，且网络校准比网络训练本身对标签噪声更敏感。
- 标准 Temperature Scaling (TS) 在有噪声的验证集上甚至可能比原始模型校准效果更差；Conformal Prediction (CP) 领域仅建议忽略噪声直接应用标准 CP，但导致预测集过大。
- 无监督域适应中，源域训练的模型迁移至目标域后过度自信，传统校准方法依赖源域标签无法处理域偏移。

## 核心贡献（创新点）
- **Noisy Temperature Scaling (NTS)**：将 adaECE 推广至噪声场景定义 Noisy-adaECE，通过估计每个 bin 内的平均标签噪声还原真实准确率，无需逐样本判断标签是否被污染，显著降低任务难度。
- **Noise-Robust Conformal Prediction (NACP)**：提出基于可逆噪声矩阵修正的 CP 框架，推导了不依赖类别数 k 的有限样本覆盖界（Δ 与 k 无关），在高维分类任务中避免了其他方法预测集膨胀至全类别的失效问题。
- **Local Differential Privacy CP (LDP-CP-L/S)**：分别支持标签隐私和分数层面隐私的差分隐私 CP 方案，并引入 shuffle model 扩展实现有效隐私 ε^eff ≈ ε/√n，在极低隐私预算下仍可逼近非隐私基线。
- **Unsupervised Target Domain Calibration (UTDC)**：在校准阶段直接利用无标签目标域数据，通过源/目标准确率比率重缩放各 bin 准确率，无需目标域标注即可实现接近 Oracle 的校准效果。

## 方法详解
### Noisy Temperature Scaling (NTS)
- 核心洞察：在 adaECE 中，**置信度项 {C_i} 不受标签噪声影响**，仅准确率项 {A_i} 受影响。
- 均匀噪声模型：以概率 ε 将正确标签替换为其余 (k−1) 类中的随机类。
- 噪声准确率到真实准确率的还原公式（大数定律）：
  $$\hat{A}_i = \frac{\tilde{A}_i - \epsilon/(k-1)}{(1-\epsilon) - \epsilon/(k-1)}$$
- 定义 **Noisy-adaECE**，将还原后的 $\hat{A}_i$ 代入 adaECE 计算，优化目标：
  $$\hat{T} = \arg\min_T \text{Noisy-adaECE}(T)$$
- 对估计值做 clip 到 [0,1] 防止越界。

### Noise-Robust Conformal Prediction (NACP)
- 对每个候选阈值 $q$ 计算矩阵 $\hat{M}_q(\ell,i) = \frac{1}{n}\sum_j \mathbb{1}_{\{\tilde{y}_j=i,\, \ell \in C_q(x_j)\}}$
- 经噪声矩阵修正后：$\hat{F}^c(q) = \mathrm{Tr}(\hat{M}_q P^{-1})$
- 均匀噪声特例（Sherman-Morrison 公式）：
  $$\hat{F}^c(q) = \frac{\hat{F}^n(q) - \epsilon \hat{F}^r(q)}{1-\epsilon}$$
- 网格搜索求解 $\hat{F}^c(q) = 1-\alpha$，测试样本预测集为 $C_q(x) = \{y \mid S(x,y) < q\}$
- **理论保证**（Theorem 4.3.5）：若取 $q$ 使 $\hat{F}^c(q) = 1-\alpha+\Delta$，则概率至少 $1-\delta$ 满足 $1-\alpha \leq p(y\in C_q(x)) \leq 1-\alpha+2\Delta$，其中 $\Delta = \sqrt{\frac{\log(4/\delta)}{2nh^2}}$，$h=\frac{1-\epsilon}{1+\epsilon}$

### LDP-CP（局部差分隐私共形预测）
- **LDP-CP-L**：采用 k-ary Randomized Response (k-RR) 扰动标签，样本量 $n = O\left(\frac{\log(1/\delta)}{\Delta^2 h^2}\right)$，隐私对象仅为标签 y，特征 x 暴露。
- **LDP-CP-S**：在分数层面进行本地隐私保护，样本量 $n = O\left(\frac{T}{\Delta^2}\left(\frac{e^\epsilon+1}{e^\epsilon-1}\right)^2\log(T/\delta)\right)$，x 和 y 均不暴露。
- **Shuffle Model 扩展**：有效隐私 $\epsilon^{\mathrm{eff}}\approx\epsilon/\sqrt{n}$，可将强隐私要求从 $\epsilon\geq 3$ 降至 $\epsilon^{\mathrm{eff}}\in[0, 0.2]$。

### Unsupervised Target Domain Calibration (UTDC)
- 关键思想：直接利用无标签目标域数据，通过**源/目标准确率比率**重缩放各 bin 准确率：
  $$\tilde{A}_{\mathrm{target},m} = A_{\mathrm{source},m}\cdot\frac{\tilde{A}_{\mathrm{target}}}{A_{\mathrm{source}}}$$
- 目标准确率 $\tilde{A}_{\mathrm{target}}$ 通过无标签估计获得（Meta、ATC、PN 等方法）。
- 优化目标：$\mathrm{UDA\text{-}adaECE}(T)=\frac{1}{M}\sum_m|\tilde{A}_{\mathrm{target},m}-C_{\mathrm{target},m}(T)|$，网格搜索求最优温度 T。
- 可扩展至 TS、VS、MS、Mix-n-Match、Weight Scaling 等多种校准方法。

## 实验与结果
### 噪声标签校准（NTS）
- **数据集**：ChestX-ray14-bal、CIFAR-10/100/100N、ImageNet
- **噪声水平**：ε = 0.2, ε = 0.4
- **结论**：NTS 在多种医学影像数据集、网络架构和噪声水平下，校准效果可与干净验证集获得的校准结果相当。

### 噪声标签下的 Conformal Prediction（NACP）
- **设置**：目标覆盖 1−α = 0.9，噪声水平 ε = 0.2，1000 次随机划分，δ = 0.001
- **核心结果**（Table 4.2 摘录）：

| 数据集 | 方法 | APS size | APS coverage | HPS size | HPS coverage |
|---|---|---|---|---|---|
| CIFAR-10 | NACP | **1.3±0.04** | **94.4±0.62** | **1.1±0.01** | **95.9±0.18** |
| CIFAR-100 | NACP | **9.0±0.46** | **93.0±0.49** | **2.5±0.09** | **93.0±0.52** |
| TinyImageNet | NACP | **22.6±1.87** | **93.7±0.71** | **7.0±0.87** | **93.6±0.72** |
| ImageNet | NACP | **20.9±0.72** | **91.9±0.32** | **4.8±0.23** | **91.9±0.36** |

- **关键对比**：ACNL/CRCP 在 CIFAR-100 及以上数据集完全失效（size=100 或 200 或 1000，coverage=100%）；NACP 在所有任务上保持合理预测集大小与目标覆盖率。

### LDP-CP
- **数据集**：OCTMNIST、TissueMNIST、OrganSMNIST、OrganAMNIST、OrganCMNIST（医学影像分类）
- **设置**：ε = 4, α = 0.1，100 次种子实验
- **核心结论**：
  - n ≥ 10^5 时 Δ_S ≤ Δ_L；k 增大（如 100/1000）时 LDP-CP-S 显著优于 LDP-CP-L
  - LDP-CP-L 在 ε ≥ 1 时已与 Not-Private-CP 持平；LDP-CP-S 在 ε^eff ≥ 0.03（ε ≥ 4）时接近非隐私基线
  - Shuffle Model 使 ε^eff 达到 0.026–0.081 范围，远优于无 Shuffle 时的 ε ≥ 3

### 无监督域适应校准（UTDC）
- **基准**：Office-home（4域）、Office-31（3域）、VisDA-2017（12类仿真→真实）、DomainNet（6域）
- **UDA 方法**：DANN、DANN+E、CDAN+E
- **核心结果**：

| 设置 | UTDC avg | Source-TS avg | Target-TS (Oracle) avg |
|---|---|---|---|
| Office-home CDAN+E | **7.93** | 16.73 | 5.22 |
| VisDA S→R CDAN+E | **7.84** | 34.29 | 2.02 |

- **关键发现**：
  - UTDC 在所有基准和 UDA 方法上，adaECE 均显著低于基线，接近 Oracle Target-TS
  - 准确率差距是主因：即使经过 UDA，源域准确率仍高于目标域，且域偏移越大差距越显著
  - Importance Weighting 假设不成立：按目标相似度分组的源域样本，其准确率在各 bin 几乎相同
  - UTDC 对目标准确率估计误差鲁棒：在合理比率范围内仍有效

## 相关工作脉络
- **温度缩放/后校准**：Platt scaling、Isotonic regression、Temperature Scaling——本文 NTS 突破其"干净标签"假设，适用于噪声场景。
- **Conformal Prediction 分数**：HPS、APS、RAPS、Rand-APS——本文在此基础上引入噪声鲁棒修正，提出 NACP。
- **噪声标签学习**：噪声矩阵估计、重加权、置信样本选择、鲁棒损失、伪标签等——本文聚焦校准阶段而非训练阶段，将噪声建模视为后处理修正问题。
- **CP + 噪声标签**：ACNL、ACNL⁺、CRCP——这些方法的有限样本项随 k 增大而增长（O(log k) 或 O(√k)），多类任务下预测集膨胀至全类别失效；NACP 的 Δ 与 k 无关。
- **域适应校准**：CPCS、TransCal（Importance Weighting-based）——本文指出 IW 假设在 UDA 场景下不成立，提出基于准确率比率的重缩放方法 UTDC。
- **局部差分隐私**：k-ary Randomized Response、Google RAPPOR——本文将其引入 CP 框架，提出 LDP-CP-L/S。

## 局限性与未来方向
- NTS 假设均匀噪声，对更复杂的噪声结构（如类别不平衡噪声）泛化能力未充分验证。
- NACP 要求已知或可估计噪声矩阵 P，对于未知噪声模式或对抗性噪声标签的鲁棒性待研究。
- LDP-CP-S 在极低隐私预算下仍需较大样本量（n ≥ 10^5）才能达到可接受的覆盖精度。
- UTDC 依赖无标签目标域准确率的估计质量，当估计误差较大时校准效果可能下降。
- 论文主要关注分类任务，回归任务的置信度校准扩展尚未涉及。

## 研究启发与可借鉴点
- **噪声建模的 bin 级别抽象**：NTS 将逐样本噪声判断转化为 bin 内平均噪声估计，这一降维思路可迁移至其他校准指标的噪声鲁棒化改造。
- **理论界的 k 无关性设计**：NACP 的 Δ 与类别数 k 无关这一性质，为高维分类的 CP 方法设计提供了理论范式参考。
- **Shuffle Model 的实用价值**：ε^eff ≈ ε/√n 的有效隐私提升机制，可在联邦学习或多机构协作场景下显著降低隐私预算要求。
- **源/目标准确率比率的校准修正**：UTDC 的比率重缩放思想简洁有效，可与其他校准方法（如 Mix-n-Match、Weight Scaling）结合扩展。
- **实验设计的完备性**：1000 次随机划分报告均值±std 的统计方式、多数据集/多模型/多噪声水平的交叉验证，为后续研究的实验设计提供了标杆。

## 关键术语表
- **Temperature Scaling (TS)**：通过单一温度参数 T 对模型 logits 进行缩放后取 softmax，是最简单的后校准方法。
- **adaECE (Adaptive Binning Expected Calibration Error)**：自适应分桶的校准误差度量，根据预测置信度分布动态划分 bin。
- **Conformal Prediction (CP)**：基于共形预测理论构造预测集的方法，提供覆盖率保证而不依赖分布假设。
- **Noise-Robust Conformal Prediction (NACP)**：通过噪声矩阵修正估计经验 CDF，实现噪声标签下的共形预测。
- **Local Differential Privacy (LDP)**：用户本地对数据施加随机扰动以满足差分隐私，无需可信中心化服务器。
- **Shuffle Model**：在 LDP 之上引入匿名shuffle层，有效隐私提升至 ε/√n，显著增强实用性。
- **Unsupervised Target Domain Calibration (UTDC)**：利用无标签目标域数据通过源/目标准确率比率校正各 bin 准确率的域适应校准方法。
- **k-ary Randomized Response (k-RR)**：LDP 下的经典扰动机制，用户以概率 p 报告真实类别、以概率 (1−p)/(k−1) 报告其他类别。

## 可复现要素
- **数据集**：CIFAR-10/100/100N、ImageNet、Tiny-ImageNet、ChestX-ray14-bal、Office-home、Office-31、VisDA-2017、DomainNet、TissueMNIST、OCTMNIST、OrganSMNIST/AMNIST/CMNIST（公开数据集，论文未提及自定义代码）
- **代码/权重**：论文未明确声明开源状态
- **关键超参**：噪声水平 ε = 0.2/0.4；覆盖水平 1−α = 0.9；隐私预算 ε = 2/4/8；δ = 0.001；分桶数 m = 15；迭代次数 1000 次随机划分
