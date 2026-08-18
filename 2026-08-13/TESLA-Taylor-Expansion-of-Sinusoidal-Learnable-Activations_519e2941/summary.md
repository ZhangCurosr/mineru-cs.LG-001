---
title: "TESLA-Taylor-Expansion-of-Sinusoidal-Learnable-Activations"
source: https://arxiv.org/pdf/2608.11970v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:38:44"
field: "神经网络激活函数设计与谱分析"
keywords: ["activations", "spectral bias", "parity problem", "NTK", "Lipschitz bound", "Fourier activation", "global interactions"]
innovations: ["提出基于有限正弦/余弦基的可学习激活 TESLA，通过系数预算 A_K 在激活层直接控制有效多项式阶次", "推导基于 A_K 的 Lipschitz/Rademacher 泛化界并分析 NTK 模态动力学 λ_m∝m^{-2}", "在 parity/Forrelation/PINN/INR/ImageNet-100 上验证，32-bit parity 100K 样本达 95.87% 且在 30% 标签噪声下显著优于 SIREN 等基线"]
benchmarks: ["Parity (d=16-32)", "Forrelation (n=12)", "LPN", "1D Burgers' equation PINN", "Kodak24 / DIV2K INR", "ImageNet-100"]
---

# 论文速读：TESLA-Taylor-Expansion-of-Sinusoidal-Learnable-Activations

## 一句话总结
论文提出了 **TESLA**（Taylor Expansion of Sinusoidal Learnable Activations），一种基于有限正弦/余弦基的可学习激活函数，通过在激活层面直接控制有效多项式阶次，有效解决了标准神经网络在全局高阶交互任务（如奇偶校验）上的光谱偏差问题，同时在 ImageNet-100 等真实视觉任务中保持可比性能与稳定训练。

## 研究问题与动机
- 标准激活（ReLU、GeLU 等）存在**光谱偏差**（spectral bias），倾向于先学习低频分量，难以高效表示依赖全局交互或高阶多项式结构的任务（如奇偶校验）。
- 传统方式需大幅增加深度、宽度或权重范数才能逼近振荡型/全局型函数，资源代价高昂。
- 已有周期/傅里叶方法（SIREN、Fourier features）在输入侧提升高频表示，但未提供**激活层面的显式多项式阶次控制**，且 SIREN 等在 ImageNet 训练时出现严重优化不稳定性。
- 奇偶校验与 Forrelation 等任务需要识别全比特的高阶交互，对本地 Piecewise-linear 激活构成强烈对抗。

## 核心贡献（创新点）
1. **提出 TESLA 激活函数**：以有限正弦/余弦基的可学习线性组合定义激活，直接控制有效多项式阶次；与 SIREN 等周期激活的本质区别在于：TESLA 的系数预算 $A_K$ 提供 Lipschitz/Rademacher 可控性，并在激活层面而非仅输入侧实现谱控制。
2. **理论保证与学习动力学刻画**：推导了基于系数预算 $A_K$ 的 Lipschitz 界与 Rademacher 复杂度泛化界，并从 NTK 角度证明 TESLA 的模态特征值 $\lambda_m \propto m^{-2}$，解释其更易恢复中高频率结构；与纯输入 Fourier features 相比，TESLA 将"频率选择权"内化到可学习激活系数。
3. **近似-估计权衡指导 K 的选择**：给出理论化的 $K^\star = \Theta(m_{\mathrm{eff}}) \cdot \kappa(N, A_K)$，明确谐波数应随任务有效阶数 $m_{\mathrm{eff}}$ 与样本量缩放；与 KAN/样条等方法相比，TESLA 在低数据 regime 下因系数预算正则而更稳定，不易过拟合。
4. **跨任务实证有效性**：在奇偶校验（n=32，100K 样本）、Forrelation、PINNs/INRs、混合频率回归与 ImageNet-100 上全面验证；与 SIREN 在 ImageNet 对比中，SIREN 出现严重崩溃（如 ViT-T/16 仅 4.11%），而 TESLA 稳定达到标准水平，体现实质性的训练稳定性优势。

## 方法详解
- **激活函数定义**：$\phi_K(z) = \sum_{k=1}^{K} \left( \frac{a_k}{k} \sin(kz) + \frac{b_k}{k} \cos(kz) \right)$，其中 $K$ 控制最大频率，$\{a_k, b_k\}$ 为层内共享的可学习标量系数；对神经元 $f(x)=\phi_K(v^\top x)$。
- **系数预算**：$A_K = \sum_{k=1}^K (|a_k|+|b_k|)$，用于控制 Lipschitz 常数和泛化界。
- **泰勒展开解释**：将 $\phi_K$ 作 Maclaurin 展开后，各多项式阶次系数是 $\{a_k, b_k\}$ 的显式线性泛函，从而将"频率系数"与"有效多项式阶次"建立映射；前向传播实际计算式 (1)，式 (3) 仅作解释性桥梁。
- **Lipschitz 与复杂度界**：$\|\phi_K'\|_\infty \le A_K$，进而网络级 $\mathrm{Lip}(f) \le (\prod_\ell \|W_\ell\|_{\mathrm{op}}) A_K^{L-1}$；Rademacher 复杂度 $\hat{\mathfrak{R}}_N(\mathcal{F}_K) \le \frac{AWR + B_0(A)}{\sqrt{N}}$，泛化隙为 $O(L_{\mathrm{loss}}(AWR+B_0(A))/\sqrt{N})$。
- **NTK 模态动力学**：在圆域 $\mathbb{T}$ 上，TESLA 诱导的 NTK 特征值 $\lambda_m^{(\phi)} = \frac{1}{2m^2}$，对应模态误差指数衰减 $e_{m,a}(\tau) = e_{m,a}(0) \exp(-\eta \lambda_m^{(\phi)} \tau)$；相比 RF-ReLU/GeLU 代理，TESLA 谱更平坦，中高模态得到相对更多训练动力。
- **K 的选择**：综合逼近误差 $B(K/m_{\mathrm{eff}})$ 与估计误差 $C A_K \sqrt{\log(2K)}/\sqrt{N}$，给出 $K^\star = \Theta(m_{\mathrm{eff}})\cdot \kappa(N,A_K)$；对 d 位奇偶校验 $m_{\mathrm{eff}}=d$，实证取 $K \approx d/4$ 稳定。

## 实验与结果
- **奇偶校验（Parity）**：用 100K 样本（约占 $2^{32}$ 的 0.002%）在 n=32 任务上取得 95.87% 准确率；在 30% 标签噪声下仍保持 ~70%（远超基线 ~50% 随机水平）；24-bit 30% 噪声下 TESLA 65.53% vs. SIREN 49.47%。
- **LPN（学习奇偶加噪声）**：TESLA 可持续至 27-bit 维持高准确率，SNAKE 与 Fourier-Emb 在所有设置下均接近随机。
- **Forrelation**：n=12 双隐藏层 MLP，TESLA 在所有宽度和 K 设置下收敛更快、最终精度更高；SIREN/SNAKE/Fourier-Emb 基本停留在机会水平。
- **PINNs（1D Burgers' equation）**：TESLA 总损失 $1.21 \times 10^{-3}$，显著优于 SIREN ($1.04 \times 10^{-2}$)、Tanh ($1.19 \times 10^{-2}$) 与 ReLU ($7.13 \times 10^{-1}$)。
- **INRs（Kodak24/DIV2K）**：Kodak24 上 TESLA PSNR=27.72 / SSIM=0.9671，超 SIREN（25.33/0.9404）；DIV2K 上 TESLA 24.73/0.9522 超 SIREN（22.77/0.9301）；到达 25dB 所需 epoch 更少。
- **混合频率回归（3.29–79.90 Hz）**：TESLA 测试 MSE=0.0100，是唯一低于 0.011 的方法；SIREN 为 0.0248，ReLU/GeLU/SiLU 均 >0.068。
- **ImageNet-100**：ResNet-18 上 TESLA Top-1=74.01%（微超 SiLU 73.99%）；MLP-Mixer 上 58.79% 为最佳；ViT-T/16 与 MobileNetV3-S 与 GeLU/SiLU 相当；吞吐量开销约 1%。SIREN 在同等设置下 ViT 仅 4.11%、ResNet-18 仅 37.41%，说明 TESLA 在保持周期性的同时避免了 SIREN 的数值不稳定。

## 相关工作脉络
- **PReLU / APL / KAN / 样条激活**：通过少量可学习参数增强非线性表达能力；但缺乏对多项式或频谱阶次的显式控制，低数据下易过拟合，TESLA 以系数预算 $A_K$ 缓解该问题。
- **SIREN（周期性激活）**：引入高频 sin 激活改善隐式表示；但 TESLA 的系数预算提供了额外 Lipschitz/Rademacher 控制，且在 ImageNet 上 SIREN 严重崩溃而 TESLA 稳定。
- **SNAKE / Fourier-feature embeddings**：前者为逐单元正弦激活，后者在输入侧做频率映射；TESLA 的关键差异是将频率控制内化到激活本身并可学习，减少对手工频率映射的依赖。
- **Barron 界 / 深度分离定理 / Yarotsky 下界**：为Piecewise-linear 近似振荡函数所需资源给出理论下界；本文据此论证 TESLA 在表示高阶/振荡结构上的资源效率优势。
- **NTK / 光谱偏差理论（Rahaman, Xu 等）**：解释标准网络优先学习低频；本文通过 NTK 对角化给出 TESLA 模态级学习速率 $\lambda_m \propto m^{-2}$，提供对"为何能更好恢复中高频率"的机制性解释。
- **PINNs / INRs / 混合频率回归**：连续域任务中 Sinusoidal 类激活已被证明有效；本文扩展其在更广泛任务与规模（ImageNet）上的适用性，并给出稳定性对比。

## 局限性与未来方向
- 理论目前主要在连续傅里叶/NTK 框架下推导，**离散布尔域**（如 parity 的 Walsh 展开）上的严格逼近界尚未完整建立。
- 仅通过实验给出 $K$ 的选择启发式（$K \approx d/4$），对更复杂任务的自动化 $K$ 寻优仍需探索。
- 在大规模 Transformer / LLM 中的实际部署（如 FFN 替换、位置编码增强）尚未验证，仅停留在概念与小型基准。
- 系数预算 $A_K$ 的 $\ell_1$ 惩罚需手动调参（论文给出 $10^{-3}$ 到 $10^{-1}$ 不等），缺乏自适应机制。
- 未来方向包括：稀疏因子化、低比特量化、硬件感知实现以降低推理延迟；扩展到大规模视觉/语言模型评估长程推理与组合泛化。

## 研究启发与可借鉴点
1. **激活层面的显式谱控制**：将频率/阶次控制从输入侧（Fourier features）迁移到激活侧，以少量可学习标量实现灵活的光谱调节；该思路可迁移到任意基于 MLP/ViT 的架构，作为即插即用组件。
2. **系数预算正则化（$A_K$）**：用 $\ell_1$-style 预算同时控制 Lipschitz 常数与泛化界，兼顾表达力与训练稳定性；可作为周期性激活的通用稳定化策略。
3. **NTK 模态动力学分析**：通过对角化 NTK 来解释不同激活的频率选择性学习行为，这一分析范式可直接复用于评估新激活函数在光谱偏差上的表现。
4. **全局交互基准设计**：parity、Forrelation、LPN 构成对"局部/全局"能力的一体化测试集；可作为团队评估新结构在长程依赖任务上表现的通用协议。
5. **连续-离散双向验证**：同时在 PINNs/INRs（连续）与 parity/Forrelation（离散组合）上验证，形成对"光谱能力"的全面刻画；建议在后续工作中沿用这一"连续+组合"双层评测策略。

## 关键术语表
- **TESLA**：Taylor Expansion of Sinusoidal Learnable Activations，一种基于有限正弦/余弦基的可学习激活函数，系数预算控制多项式阶次与 Lipschitz 性质。
- **光谱偏差（Spectral bias）**：神经网络在训练中优先学习低频分量、抑制高频分量的现象，源于标准激活与 NTK 谱衰减。
- **NTK（Neural Tangent Kernel）**：描述无限宽神经网络训练动力学的核函数，其对角特征值决定各频率模态的学习速率。
- **系数预算 $A_K$**：TESLA 各谐波系数的 $\ell_1$ 范数和，用于约束激活的 Lipschitz 常数与泛化复杂度。
- **奇偶校验（Parity）**：判断输入二进制向量中 1 的个数奇偶性，需全局高阶交互，传统 ReLU MLP 难以有效学习。
- **Forrelation**：衡量两个布尔函数经 Hadamard 变换后相关性的全局决策问题，用于分离量子与经典计算复杂性。
- **有效交互阶 $m_{\mathrm{eff}}$**：Walsh 展开中累积能量达到 $1-\varepsilon$ 所需的最低交互阶数，用于指导 K 的理论选择。
- **LPN（Learning Parity with Noise）**：给定带噪声奇偶标签的样本，隐式学习目标子集的组合推理任务，比无噪声 parity 更困难。

## 可复现要素
- **数据集**：Parity/Forrelation/LPN 为合成数据（论文未公开独立数据集文件，但给出采样与划分协议：固定哈希三分割，训练 100K、验证 20K）；ImageNet-100 为公开子集；Kodak24、DIV2K 公开；Burgers' equation PINN 为标准 PDE 基准。
- **代码/权重**：代码已开源于 https://github.com/KAU-QuantumAILab/TESLA（论文声明）；权重未明确提及。
- **关键超参**：Parity 任务 Hidden=128，K=d/4，LR=1e-3，Batch=1024，Epoch=30；ImageNet-100 使用 AdamW、50 epochs、Cosine LR、Warmup 5 epochs/0.85；$A_K$ 在 ViT/ResNet/MobileNet 上设为 1e-3，MLP-Mixer 上为 1e-1；硬件 NVIDIA A100 80GB × 1。
