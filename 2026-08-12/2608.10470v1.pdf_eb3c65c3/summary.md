---
title: "A JOINT-DISTRIBUTION ROUTE TO FAIR REPRESENTATIONS WITH CONTINUOUS SENSITIVE ATTRIBUTES<sup>∗</sup>"
source: https://arxiv.org/pdf/2608.10470v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 04:57:51"
---

# 论文速读：A JOINT-DISTRIBUTION ROUTE TO FAIR REPRESENTATIONS WITH CONTINUOUS SENSITIVE ATTRIBUTES<sup>∗</sup>

## 一句话总结
本文提出一种基于联合分布差异的公平表示学习方法 FRHSIC，通过闭式 HSIC 统计量直接度量表示 $Z$ 与连续敏感属性 $S$ 的依赖性，避免了传统方法对条件分布 $P_{Z|S=s}$ 的非参数平滑估计；该方法在理论上获得 $O(n^{-1/2})$ 的一致收敛速率，在五个真实数据集上实现与传统基线相当的公平-准确权衡，且单 epoch 训练速度比最优条件路线基线 FREM 快约 36 倍。

## 研究问题与动机
- **核心问题**：连续型敏感属性 $S$（如年龄、收入、风险评分）下的公平表示学习，目标是最小化表示 $Z=h(X)$ 与 $S$ 的统计依赖，使任意下游预测头满足广义人口统计均等（GDP）等公平性指标。
- **现有方法不足1**：GDP、EIPM/FREM、互信息等方法均采用“条件积分”路线，即对每个 $S=s$ 估计条件分布 $P_{Z|S=s}$ 与边缘分布 $P_Z$ 的差异，再对 $S$ 求平均；有限样本下必须依赖 Nadaraya–Watson 核平滑或留一法估计，引入带宽超参且数值不稳定。
- **现有方法不足2**：基于条件估计的收敛率为非参数 $O(n^{-2/5})$，在样本量有限或敏感属性高维时效率低下，且每批次需重复计算局部条件对象，计算开销随 $n$ 呈 $O(n^3)$ 增长。
- **动机**：将目标从“条件平均差异”转为“单一联合分布差异” $d(P_{Z,S}, P_Z \otimes P_S)$，利用闭合形式的依赖统计量直接优化，既能保持相同的公平性目标（理论可证明等价），又能获得更快的 $O(n^{-1/2})$ 收敛速率与更低的时间复杂度。

## 核心贡献（创新点）
1. **联合差异统一框架**：证明 GDP、EIPM、互信息等现有连续敏感属性公平性准则均可视为单一联合 IPM 在可分解见证类（decomposable witness class）上的条件积分解读，建立联合视图与条件视图的严格等价桥梁。
2. **HSIC 与条件 MMD 的谱等价性**：推导 HSIC 与条件 MMD 积分互为上下界，误差由敏感属性核的显式谱尾 $\rho_m^2$ 控制，从而闭合形式的 HSIC 可直接约束逐值条件偏差与下游 GDP 差距。
3. **编码器类的一致收敛界**：在固定核与有界 Lipschitz 条件下，证明经验 HSIC 在受控复杂度的编码器类上一致收敛于总体值，收敛率为 $\tilde{O}_p(n^{-1/2})$，为数据驱动的公平表示提供 train-to-population 理论保证。
4. **高效训练算法 FRHSIC**：将 HSIC 作为 mini-batch 正则项嵌入端到端训练，每批仅需 $O(m^2)$ 闭式矩阵迹运算；实验表明在 $n=20,000$ 时训练速度比 FREM 快约 36 倍，且
