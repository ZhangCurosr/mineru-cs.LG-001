---
title: "TESLA-Taylor-Expansion-of-Sinusoidal-Learnable-Activations"
source: https://arxiv.org/pdf/2608.11970v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:39:39"
---

# 论文速读：TESLA-Taylor-Expansion-of-Sinusoidal-Learnable-Activations

## 一句话总结
本文提出 TESLA，一种由可学习正余弦项线性组合构成的激活函数，通过在激活层直接控制有效多项式阶数与频谱成分，有效缓解标准激活的谱偏差问题。该方法在奇偶校验（Parity）、Forrelation 等全局高阶交互任务以及 ImageNet-100 视觉任务上均展现出强泛化能力与训练稳定性。

## 研究问题与动机
- 标准神经网络（ReLU、GeLU 等）存在强烈的谱偏差，优先拟合低频结构，难以高效表征依赖全局交互或高阶组合的任务。
- 奇偶校验等任务要求模型捕获整个输入向量的全局相位信息，传统 MLP 通常需极大的宽度、深度或极端权重范数才能近似。
- 现有傅里叶特征映射与周期激活（如 SIREN）主要在输入侧扩展频谱，未能在激活函数层面提供显式的多项式阶数控制与自适应调节机制。
- 缺乏对可学习周期激活的严格泛化界与训练动力学分析，导致 $K$（谐波数量）等关键超参难以理论指导选取。

## 核心贡献（创新点）
- 提出 TESLA 激活函数，以有限阶正余弦基底的可学习系数实现激活级多项式阶数控制，与仅在输入侧做频率映射的方法本质不同。
- 推导 TESLA 的 Lipschitz 常数界与基于 Rademacher 复杂度的泛化误差界，证明系数预算 $A_K$ 可同时约束梯度幅度与模型复杂度。
- 从 NTK 视角分析模式级学习动力学，揭示 TESLA 核特征值按 $m^{-2}$ 衰减，频谱比随机特征近似更平坦，能加速中高频率结构的收敛。
- 给出理论指导的 $K$ 选取准则，平衡逼近误差与估计误差，并在布尔全局交互、连续域 PDE/图像重建及大规模视觉基准上验证有效性与稳定性。

## 方法详解
- **激活定义**：$\phi_K(z) = \sum_{k=1}^K \left( \frac{a_k}{k} \sin(kz) + \frac{b_k}{k} \cos(kz) \right)$，$K$ 控制最大谐波频率，$\{a_k, b_k\}$ 为可学习标量系数。
- **系数预算**：定义 $A_K = \sum_{k=1}^K (|a_k| + |b_k|)$，作为 $l_1$ 型正则项，直接控制激活的 Lipschitz 常数与网络整体复杂度。
- **泰勒展开解释**：对 $\phi_K$ 做 Maclaurin 展开后，各多项式阶次的系数是 $\{a_k, b_k\}$ 的显式线性泛函，从而将三角基底系数与有效多项式阶数建立映射，但前向计算仍保留三角形式。
- **稳定性与泛化界**：证明 $\|\phi_K'\|_\infty \leq A_K$，$L$ 层网络满足 $\text{Lip}(f) \leq (\prod \|W_\ell\|_{\text{op}}) A_K^{L-1}$；Rademacher 复杂度界为 $O(AWR/\sqrt{N})$，泛化 gap 与 $A_K$ 呈线性比例。
- **NTK 模式动力学**：在傅里叶基下 NTK 特征值为 $\lambda_m^{(\phi)} = 1/(2m^2)$，误差模态以 $\exp(-\eta \lambda_m \tau)$ 衰减；相比标准激活，TESLA 在中高频段分配更大特征值，促进全局结构恢复。
- **$K$ 的理论选取**：综合逼近项 $B(K/m_{\text{eff}})$ 与估计项 $C A_K \sqrt{\log(2K)}/\sqrt{N}$，得 $K^* = \Theta(m_{\text{eff}}) \cdot \kappa(N, A_K)$；Parity 任务实践中推荐 $K \approx d/4$。

## 实验与结果
- **Parity 任务**：在 $d=32$、仅用 100K 样本（约占 $2^{32}$ 的 0.002%）时达到强泛化；面对 30% 标签噪声仍保持高精度，而 SIREN/SNAKE/Fourier-Emb 均退化至接近随机猜测（≈50%）。
- **Forrelation 任务**：两层 MLP 在 $d=12$ 上，TESLA 在隐藏宽 128/512/1024 时均收敛更快且最终准确率最高，周期基线性能接近随机。
- **连续域任务**：
  - PINNs（1D Burgers 方程，20000 epoch）：总损失降至 $1.21 \times 10^{-3}$，显著优于 SIREN（$1.04 \times 10^{-2}$）、Tanh 与 ReLU。
  - INRs（Kodak24/DIV2K）：PSNR 与 SSIM 最优，达到 25 dB 所需 epoch 少于 SIREN。
  - 混合频率回归：唯一 Test MSE 低于 0.011 的方法。
- **ImageNet-100**：在 ViT-T/16、MLP-Mixer、ResNet-18、MobileNetV3-S 上 Top-1 准确率与主流激活持平；FLOPs 与吞吐量仅增加约 1%；相同设置下 SIREN 出现严重优化崩溃（ViT 仅 4.11%），TESLA 训练稳定。
- **最强结果**：32-bit Parity 无噪声准确率达 95.87%，30% 噪声下仍达 68.78%；Burgers' PINN 损失降低近一个数量级。

## 相关工作脉络
- **参数化点激活（PReLU、APL）**：仅引入少量可学习点状非线性，无法控制
