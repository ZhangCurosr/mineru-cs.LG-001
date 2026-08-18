---
title: "A JOINT-DISTRIBUTION ROUTE TO FAIR REPRESENTATIONS WITH CONTINUOUS SENSITIVE ATTRIBUTES<sup>∗</sup>"
source: https://arxiv.org/pdf/2608.10470v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 04:57:58"
---

# 论文速读：A JOINT-DISTRIBUTION ROUTE TO FAIR REPRESENTATIONS WITH CONTINUOUS SENSITIVE ATTRIBUTES<sup>∗</sup>

## 一句话总结
本文提出联合分布视角的公平表示学习方法 **FRHSIC**，以 Hilbert–Schmidt 独立性准则（HSIC）直接度量表征 $Z$ 与连续敏感属性 $S$ 的联合依赖，避开传统方法对条件分布 $P_{Z|S=s}$ 的非参数核平滑估计；理论证明其与现有条件积分准则在零水平等价，且估计收敛率从 $O(n^{-2/5})$ 提升至 $O(n^{-1/2})$，在五个真实数据集上保持相当的公平‑准确率权衡的同时，将单 epoch 训练速度提升约 36 倍。

## 研究问题与动机
1. **连续敏感属性的公平表示学习缺乏高效统一的度量框架**：当 $S$ 为连续变量（如年龄、收入、风险评分）时，目标为 $Z \perp S$（即 $P_{Z|S=s}=P_Z$ a.e.），但现有准则均需对无重复观测的连续 $S$ 构造局部条件对象（条件分布或条件期望），统计困难且估计方差大。
2. **条件路由（conditional route）存在收敛慢与计算贵的问题**：GDP 采用 Nadaraya–Watson 核平滑，EIPM/FREM 采用留一核加权条件经验测度，二者均依赖局部带宽，收敛率仅为非参数率 $O(n^{-2/5})$；FREM 的锚点子采样 MMD 计算每 batch 达 $O(n^3)$ 量级，难以扩展至中大样本。
3. **公平‑准确率权衡与跨下游头稳定性有待加强**：现有连续方法多针对单一训练头优化，学到的表示往往仅在特定 head 下公平，更换下游预测器后公平性迅速退化；同时部分方法（如 Reg‑GDP）为压低 $\Delta_{GDP}$ 会牺牲大量预测性能。
4. **理论桥接缺失**：虽有 HSIC 被用作公平正则的先例，但缺乏将联合分布差异与条件积分准则严格统一、并给出训练到总体（train‑to‑population）均匀收敛保证的理论框架，导致实践缺乏细粒度误差控制。

## 核心贡献（创新点）
1. **联合差异框架统一连续 $S$ 公平准则**：证明特征 witness 类下的联合 IPM $\mathrm{IPM}_{\mathcal{F}}(P_{Z,S}, P_Z\otimes P_S)$ 在零水平上等价于 $Z\perp S$，且通过 disintegration identity 与 GDP/EIPM 共享同一条件积分目标，避免了显式构造条件族。
2. **HSIC 与条件 MMD 积分的谱等价性**：定理 4.5 证明 HSIC 与 $\mathbb{E}_S[\mathrm{MMD}^2_{k_Z}(P_{Z|S},P_Z)]$ 相互控制，误差由敏感核积分算子的显式谱尾 $\rho_m^2$ 界定，揭示闭式统计量为何能替代非参数条件估计。
3. **$O(n^{-1/2})$ 均匀集中与有限样本 GDP 控制**：定理 5.1 证明经验 HSIC 在受控复杂度的编码器类上一致收敛于总体值，速率 $O(n^{-1/2})$；定理 4.7 给出有限样本下 $\widehat{\Delta}_{\mathrm{GDP}}(f)^2 \leq \frac{\|f\|^2}{\hat{\lambda}_m}\widehat{\mathrm{HSIC}}_n$，使公平惩罚具备可解释的谱控界。
4. **高效可训练的 FRHSIC 算法**：将偏差 HSIC 估计量作为 mini‑batch 正则项，闭式 $O(m^2)$ 计算避免核平滑与留一修正；在五个真实数据集上取得与最强连续基线相当的 nAUP，且在 $n=20,000$ 时单 epoch 提速约 36×。

## 方法详解
1. **联合差异度量定义**：不估计 $P_{Z|S=s}$，转而度量联合分布与边缘乘积分布之间的积分概率度量：
   $\mathrm{IPM}_{\mathcal{F}}(P_{Z,S}, P_Z\otimes P_S) = \sup_{f\in\mathcal{F
