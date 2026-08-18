---
title: "Direct-Acceleration-of-Stochastic-Root-Finding-Without-Varia"
source: https://arxiv.org/pdf/2608.12043v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:03:42"
---

# 论文速读：Direct-Acceleration-of-Stochastic-Root-Finding-Without-Varia

## 一句话总结
本文提出无需方差缩减与双重循环正则化的随机对偶锚点加速方法 S-Dual-OHM，在期望余强单调（cocoercivity in expectation）假设下以固定步长与固定 mini-batch 直接实现 $\mathcal{O}(\epsilon^{-3})$ oracle 复杂度；若算子额外满足强单调性，通过提前停止可进一步达到 $\widetilde{\mathcal{O}}(\epsilon^{-2})$ 近最优复杂度。

## 研究问题与动机
- **核心问题**：随机根求解（或不动点）问题中，如何在摒弃方差缩减与 double-loop 正则化的前提下，直接继承确定性加速算法的最优收敛率。
- **现有方法不足**：传统锚点加速（如 Halpern 迭代及其极小极大变体）在随机设定下会因 oracle 误差累积而破坏快收敛，必须依赖递减方差、PAGE 类方差缩减或递归正则化才能稳定，导致算法复杂、超参多、内存开销大。
- **关键观察**：确定性加速的“最优表示”并非唯一，除 anchor-based 外还存在 H-dual 对偶族算法，它们在 worst-case 复杂度上等价，但在随机扰动下的行为存在本质差异。
- **动机**：打破“确定性加速在随机噪声下必然脆弱”的固有认知，证明通过合理选择加速机制的数学表示形式，即可实现干净直接的随机扩展，为简化 stochastic monotone inclusion 算法设计提供新范式。

## 核心贡献（创新点）
1. **提出 S-Dual-OHM 单循环算法**：首次将对偶锚点加速直接拓展至随机算子场景，全程无需方差缩减、无需递增 batch、无需 outer-inner 正则化结构。
2. **建立无 VR 情形下的最优复杂度上界**：在期望余强单调假设下，以固定步长 $\alpha \in (0, 2/L]$ 与固定批量 $B=\mathcal{O}(\epsilon^{-2})$ 实现 $N=\mathcal{O}(\epsilon^{-1})$ 迭代与 $\mathcal{O}(\epsilon^{-3})$ oracle 复杂度，与需 VR 的 S-HALPERN-PAGE 持平但形式更简洁。
3. **强单调情形下的近最优早停加速**：当算子同时强单调时，推导得到随机轨迹与确定性轨迹的偏差界，并结合 Dual-OHM 的线性收敛性证明可在 $k=\widetilde{\mathcal{O}}(L/\mu \log \epsilon^{-1})$ 步提前停止，达到 $\widetilde{\mathcal{O}}(\epsilon^{-2})$ 复杂度，近乎匹配 $\Omega(\epsilon^{-2})$ 下界。
4. **揭示加速机制的噪声敏感性差异**：从理论层面阐明 Dual-OHM 的误差累积系数为 $\mathcal{O}(N)$，而传统 OHM 为 $\Theta(N^2)$，从根本上解释了为何相同确定性速率的两种加速在随机设定下命运迥异。

## 方法详解
- **问题与假设**：求解 $\mathbb{F}(x)=0$，其中 $\mathbb{F}(x)=\mathbb{E}[\widehat{\mathbb{F}}(x;\xi)]$ 满足无偏性与有界方差（Assumption 3.1）；核心假设为期望余强单调性（Assumption 3.2）：$\langle \mathbb{F}(x)-\mathbb{F}(y), x-y \rangle \geq \mathbb{E}[\frac{1}{L}\|\widehat{\mathbb{F}}(x;\xi)-\widehat{\mathbb{F}}(y;\xi)\|^2]$。
- **确定性基础**：令 $\mathbb{T}=I-\alpha\mathbb{F}$，当 $0<\alpha\leq 2/L$ 时 $\mathbb{T}$ 为非扩张算子。传统 Halpern 迭代为 $y_{k+1}=\frac{1}{k+2}y_0+\frac{k+1}{k+2}\mathbb{T}y_k$，对偶锚点迭代为 $y_{k+1}=y_k+\frac{N-k-1}{N-k}(\mathbb{T}y_k-\mathbb{T}y_{k-1})$，两者均满足 $\|y_{N-1}-\mathbb{T}y_{N-1}\|^2\leq \frac{4\|y_0-y_\star\|^2}{N^2}$。
- **随机扩展（S-Dual-OHM）**：用 mini-batch 估计 $\mathbb{T}_{\mathcal{B}_k}$ 替换确定性算子，更新规则为 $x_{k+1}=x_k+\frac{N-k-1}{N-k}(\mathbb{T}_{\mathcal{B}_k}(x_k)-\mathbb{T}_{\mathcal{B}_{k-1}}(x_{k-1}))$，全程使用固定步长与固定批量。
- **核心证明思路**：构造精确恒等式将末项残差表示为历史噪声项 $\mathcal{Q}_{N,j}$ 的加权和；利用**留一法稳定性（leave-one-out stability）**证明每步误差 $e_{j-1}$ 传播至末次迭代的内积贡献被一致界定为 $\frac{\alpha\sigma^2}{B}$，求和后总噪声项仅随 $\sum\lambda_{N,j}=\mathcal{O}(N)$ 增长，从而避免误差累积爆炸，导出定理 4.1。
- **强单调早停分析**：强单调性使 $\mathbb{T}$ 成为 $\gamma$-压缩算子。随机轨迹偏离确定性轨迹的方差被 $\frac{\alpha^2\sigma^2}{B(1-\gamma^2)}$ 控制，而确定性 Dual-OHM 线性收敛至末项，故在 $k=\widetilde{\mathcal{O}}(L/\mu\log\epsilon^{-1})$ 时末项残差已足够小，实现定理 4.5 的早停加速。

## 实验与结果
- **实验设置**：三组人工构造算子——(1) 最坏情况非扩张算子（低维仿射下界构造）；(2) 有限和余强单调算子；(3) 带 Huber 正则化的强凸-强凹（SCSC）极小极大问题。基线包括 SGDA、SEG、S-OHM（常数批量）、Halpern-PAGE、RAIN、Halpern-VR-Finite。所有算法在
