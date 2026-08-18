---
title: "Direct-Acceleration-of-Stochastic-Root-Finding-Without-Varia"
source: https://arxiv.org/pdf/2608.12043v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:02:38"
field: "随机优化与单调包含"
keywords: ["随机算子求根", "对偶锚加速", "cocoercive算子", "方差缩减", "Halpern迭代", "双循环正则化", "oracle复杂度", "minimax优化"]
innovations: ["揭示对偶锚机制（Dual-OHM）对随机噪声的内生鲁棒性，首次无需方差缩减实现O(epsilon^{-3})随机根查找", "在强单调+cocoercive假设下通过早停达到tilde{O}(epsilon^{-2})近最优oracle复杂度", "建立基于留一稳定性的随机误差传播上界技术，证明误差累积权重为O(N)而非OHM的O(N^2)"]
benchmarks: ["Worst-case非扩张不动点算子（d=2001）", "有限和cocoercive算子（n=200, d=200）", "SCSC Huber正则化minimax问题（d=50）"]
---

# 论文速读：Direct-Acceleration-of-Stochastic-Root-Finding-Without-Varia

## 一句话总结
本文证明了确定性加速方法中的**对偶锚机制（Dual-OHM）**可以直接推广到随机算子求解场景，无需方差缩减或双循环正则化，即可在期望cocoercive条件下达到 $\mathcal{O}(\epsilon^{-3})$ 的oracle复杂度；进一步在强单调假设下通过早停实现 $\widetilde{\mathcal{O}}(\epsilon^{-2})$，近最优地匹配 $\epsilon$ 依赖下界。

---

## 研究问题与动机

1. **核心问题**：确定性根查找（root-finding）问题的加速算法（如Halpern迭代/OHM）在随机设定下为何失效？
2. **现有方法不足**：随机Halpern迭代及其minimax变体（S-OHM）在噪声积累下快速发散；要稳定其收敛必须依赖（1）递增batch size以衰减方差、（2）方差缩减技术（PAGE等）、或（3）更强的问题假设——这些均增加了实现复杂度与超参调优难度。
3. **认知盲区**：文献长期持有的" folklore 观点"——最优确定性加速对随机噪声本质脆弱、必然不可直接移植——是否有反例？
4. **已有工作的局限**：RAIN [12] 虽达到近最优 $\widetilde{\mathcal{O}}(\epsilon^{-2})$，但依赖更复杂的递归锚定（正则化）结构；Halpern-PAGE [9] 与 Halpern-VR-Finite [10] 需方差缩减模块，理论与实现复杂度更高。

---

## 核心贡献（创新点）

1. **首次揭示对偶锚机制对随机噪声的内生鲁棒性**：S-Dual-OHM 仅需恒定步长 $\alpha \in (0, 2/L]$ 与恒定 batch size $B = \mathcal{O}(\epsilon^{-2})$，即干净地达到 $\mathbb{E}[\|\mathbb{F}(x_{N-1})\|] \leq \epsilon$，无需任何方差缩减或双循环结构——这与同类工作的本质区别在于**不试图压制噪声，而是换一种加速框架避开噪声累积**。

2. **给出简洁的 $\mathcal{O}(\epsilon^{-3})$ oracle 复杂度上界**（定理4.1）：迭代步数 $N = \mathcal{O}(\epsilon^{-1})$，每步 oracle 调用 $B = \mathcal{O}(\epsilon^{-2})$，总复杂度 $\mathcal{O}(\epsilon^{-3})$；此前同类工作若无方差缩减只能达到 $\mathcal{O}(\epsilon^{-4})$ [20]，本文以显著更简洁的算法设计实现更优的理论保证。

3. **强单调情形下的 $\widetilde{\mathcal{O}}(\epsilon^{-2})$ 早停结果**（定理4.5）：当 $\mathbb{F}$ 同时满足 $\mu$-强单调与 $1/L$-cocoercive，S-Dual-OHM 可在 $k = \widetilde{\mathcal{O}}(L/\mu \cdot \log \epsilon^{-1})$ 步早停即达到精度 $\epsilon$，oracle 复杂度 $\widetilde{\mathcal{O}}(\epsilon^{-2})$ 近-optimal 匹配 [12] 的下界——关键差异是不依赖正则化模块，理论证明更轻量。

4. **建立误差传播的新分析框架**（引理4.3–4.4 + 附录A.2留一稳定性）：通过对 S-Dual-OHM 引入"留一稳定性"（leave-one-out stability）工具，证明每次迭代的随机误差 $e_{j-1}$ 传播至终点的贡献被一致有界为 $\mathcal{O}(\sigma^2/B)$，与 Halpern 迭代中误差权重 $\sum \nu_{j+1,j} = \Theta(N^2)$ 形成鲜明对比。

5. **概念性洞见**：打破"确定性最优加速=随机稳定"的默认假设，提出**加速表示的选择本身决定随机稳健性**，为随机优化设计提供了新视角。

---

## 方法详解

### 问题设定与算子性质

考虑算子 $\mathbb{F}: \mathbb{R}^d \to \mathbb{R}^d$ 的根查找问题 $\mathbb{F}(x) = 0$。相关算子性质定义如下：

- **单调**：$\langle \mathbb{F}(x) - \mathbb{F}(y), x-y \rangle \geq 0$
- **$\mu$-强单调**：$\langle \mathbb{F}(x) - \mathbb{F}(y), x-y \rangle \geq \mu \|x-y\|^2$
- **$M$-Lipschitz**：$\|\mathbb{F}(x) - \mathbb{F}(y)\| \leq M \|x-y\|$
- **$1/L$-cocoercive**：$\langle \mathbb{F}(x) - \mathbb{F}(y), x-y \rangle \geq \frac{1}{L}\|\mathbb{F}(x) - \mathbb{F}(y)\|^2$

cocoercivity 蕴含单调性与 Lipschitz 性；若 $\mathbb{F}$ 同时 $\mu$-强单调且 $M$-Lipschitz，则它是 $1/L$-cocoercive 的，其中 $L = M^2/\mu$。

当 $\mathbb{F}$ 为 $1/L$-cocoercive 时，可通过 $\mathbb{T} = \mathrm{I} - \alpha \mathbb{F}$（$0 < \alpha \leq 2/L$）将根查找转化为不动点问题，此时 $\mathbb{T}$ 为非扩张算子（1-Lipschitz）。

### 随机 oracle 模型（假设 3.1）

对任意 $x$，可查询无偏随机 oracle $\widehat{\mathbb{F}}(x;\xi)$，满足：
$$\mathbb{E}[\widehat{\mathbb{F}}(x;\xi)] = \mathbb{F}(x), \qquad \mathbb{E}[\|\widehat{\mathbb{F}}(x;\xi) - \mathbb{F}(x)\|^2] \leq \sigma^2$$

### 期望 cocoercivity 假设（假设 3.2）

$$\langle \mathbb{F}(x) - \mathbb{F}(y), x-y \rangle \geq \mathbb{E}\left[\frac{1}{L}\|\widehat{\mathbb{F}}(x;\xi) - \widehat{\mathbb{F}}(y;\xi)\|^2\right]$$

典型成立场景：（i）加法噪声 $\widehat{\mathbb{F}}(x;\xi) = \mathbb{F}(x) + \zeta(\xi)$，噪声与 $x$ 独立；（ii）各样本算子各自 cocoercive 的有限和/期望模型；（iii）各样本算子各自 $\mu$-强单调且 $M$-Lipschitz。

### S-Dual-OHM 算法

给定固定总迭代数 $N$ 与恒定 mini-batch 大小 $B$，算法更新为：
$$x_{k+1} = x_k + \frac{N-k-1}{N-k}\Big(\mathbb{T}_{\mathcal{B}_k}(x_k) - \mathbb{T}_{\mathcal{B}_{k-1}}(x_{k-1})\Big), \quad k=0,\ldots,N-2$$
其中 $\mathbb{T}_{\mathcal{B}_k} = \mathrm{I} - \alpha \mathbb{F}_{\mathcal{B}_k}$，$\mathbb{F}_{\mathcal{B}_k}(x) = \frac{1}{B}\sum_{b=1}^{B} \widehat{\mathbb{F}}(x;\xi_b)$，约定 $\mathbb{T}_{\mathcal{B}_{-1}}(x_{-1}) = x_0$。

### 关键误差恒等式（引理 4.2）

S-Dual-OHM 满足精确代数恒等式：
$$0 = \frac{(N-1)\alpha^2}{4}\|\mathbb{F}(x_{N-1})\|^2 + \frac{\alpha}{2}\langle \mathbb{F}(x_{N-1}), x_{N-1}-x_0\rangle + \sum_{j=1}^{N-1}\lambda_{N,j}\,\mathcal{Q}_{N,j}$$
其中 $\lambda_{N,j} = \frac{N\alpha}{2(N-j)(N-j+1)}$，$\mathcal{Q}_{N,j}$ 为涉及单调性项的结构量。在确定性情形下由 cocoercivity 保证 $\mathcal{Q}_{N,j} \geq 0$，从而导出 $\|\mathbb{F}(x_{N-1})\|^2 \leq 4\|x_0-x_\star\|^2 / (\alpha^2 N^2)$。

### 主要上界（定理 4.1）

$$\mathbb{E}\big[\|\mathbb{F}(x_{N-1})\|^2\big] \leq \frac{4\|x_0 - x_\star\|^2}{\alpha^2 N^2} + \frac{6\sigma^2}{B}$$
取 $N = \Theta(1/\epsilon)$、$B = \Theta(1/\epsilon^2)$ 即得 $\mathbb{E}[\|\mathbb{F}(x_{N-1})\|] \leq \epsilon$，总 oracle 调用 $\mathcal{O}(\epsilon^{-3})$。

### 强单调下的早停加速（定理 4.5）

设 $\mathbb{F}$ 同时 $\mu$-强单调与 $1/L$-cocoercive，令 $\gamma = \sqrt{1-\alpha\mu(2-\alpha L)} \in [0,1)$，则对任意 $k < N$：
$$\mathbb{E}\big[\|\mathbb{F}(x_k)\|^2\big] \leq \frac{8L^2 D^2}{N^2} + \frac{64L^4}{\mu^2} e^{-k\mu/L} D^2 + \frac{L}{\mu}\cdot\frac{4\sigma^2}{B}$$
取 $N \geq 4LD/\epsilon$、$k \geq \frac{L}{\mu}\log\frac{256L^4D^2}{\mu^2\epsilon^2}$、$B \geq \frac{16L\sigma^2}{\mu\epsilon^2}$ 即可早停在 $k = \widetilde{\mathcal{O}}(L/\mu \cdot \log \epsilon^{-1})$ 步获得 $\epsilon$-精度，oracle 总调用 $\widetilde{\mathcal{O}}(\epsilon^{-2})$。

### S-Dual-OHM 在噪声下稳定的直觉（§4.1.1）

- OHM 是 **anytime 算法**：必须在每一步都保持 $\|\mathbb{F}(x_{k-1})\|^2 \leq 4D^2/(\alpha^2 k^2)$，迫使其在每个迭代充分利用 cocoercivity 几何，因此对噪声极为敏感；其误差权重之和 $\sum \nu_{j+1,j} = \Theta(N^2)$。
- Dual-OHM **仅针对预定终点 $N$ 优化**，不要求中间迭代保持最优率，误差权重之和仅为 $\sum \lambda_{N,j} = \alpha(N-1)/2 = \mathcal{O}(N)$，噪声累积可控。

### 关键证明工具

- **留一稳定性（Leave-one-out stability，引理 A.2）**：替换单个 mini-batch $\mathcal{B}_s$ 为独立副本，终态迭代偏差 $\mathbb{E}[\|x_{N-1}-x_{N-1}^{(s)}\|^2] \leq \alpha^2\sigma^2/B$，用于控制 Lemma 4.4 中误差传播项 $\mathbb{E}[\langle e_{j-1}, \mathbb{T}x_{N-1}\rangle] \leq \alpha\sigma^2/B$。

---

## 实验与结果

### 实验设置

所有实验在 MacBook Air M3 / 24GB 内存上完成；每个实验运行10次独立随机种子，报告均值与 5th–95th 百分位带；以总样本预算 $Q = NB$ 为调参统一标准。

### 对比基线

SGDA、SEG、S-OHM（恒定B）、Halpern-PAGE [9]、RAIN（单循环简化版）[12]、Halpern-VR-Finite [10]。

### 实验1：worst-case 非扩张不动点算子（图1左）

- 构造：基于 Park & Ryu [40] 的最坏情况仿射下界，$d=2001$，高斯噪声 $\sigma=0.1$，预算2000。
- **S-Dual-OHM 最终残差最小**；S-OHM（同B）发散；RAIN/Halpern-PAGE 通过方差缩减/递归锚定稳定但复杂度更高；SGDA/SEG 稳定但收敛慢。

### 实验2：有限和 cocoercive 算子（图1右）

- $n=200$，$d=200$，$r=199$，预算2000；S-Dual-OHM **显著优于 S-OHM**；Halpern-VR-Finite 因预算限制进展极小。

### 实验3：SCSC Huber 正则化 minimax 问题（图2）

- 低噪声（$\sigma=0.05$）：最佳 $B^*=1$，mini-batching 无增益，S-Dual-OHM 前期快但后期停滞；RAIN/SGDA/SEG 更稳。
- 高噪声（$\sigma=1.5$）：最佳 $B^*=10$，**S-Dual-OHM 与 RAIN 表现相当**，显著优于 S-OHM。

### 最强结果与提升

| 条件 | S-Dual-OHM oracle 复杂度 | 对应基线 |
|---|---|---|
| 一般 cocoercive | $\mathcal{O}(\epsilon^{-3})$（无需方差缩减） | SEG/SGDA $\mathcal{O}(\epsilon^{-4})$ |
| 强单调 + cocoercive | $\widetilde{\mathcal{O}}(\epsilon^{-2})$（早停） | 近优，接近 RAIN 理论界 |

---

## 相关工作脉络

1. **Halpern 迭代与确定性加速** [21, 29, 22, 40, 45, 51, 56]：Halpern 迭代 $\mathcal{O}(1/k^2)$ 平方残差率为最优，Dual-OHM [55] 展示了同阶但更新规则不同的"对偶"加速族。
2. **锚加速在 minimax/单调包含中的扩展** [27, 53, 50, 54, 11, 1, 5]：Fast Extra Anchored Gradient 等移除对数间隙，但随机化均需方差缩减或递减 oracle 误差。
3. **随机 Halpern-PAGE** [9]：首次将 PAGE 方差缩减与 S-OHM 结合达到 $\mathcal{O}(\epsilon^{-3})$，但需额外方差估计模块与概率混合策略。
4. **随机 Halpern-VR-Finite** [10]：有限和设定下进一步优化至 $\widetilde{\mathcal{O}}(n + \sqrt{n}\epsilon^{-1})$，限于有限和结构。
5. **RAIN** [12]：单循环近似最优算法，递归锚定/正则化结构复杂；$\widetilde{\mathcal{O}}(\epsilon^{-2})$ 理论但超参多（$\eta,\lambda,\gamma,N_s,K_s$）。
6. **随机 Krasnosel'skiȋ–Mann/Halpern 迭代** [7, 41, 8]：在一般赋范空间建立 $\Omega(\epsilon^{-3})$ 下界，本文在无方差缩减前提下行达该下界。

本文定位：**用更简洁的算法（无方差缩减、无双循环）达到与 RAIN/Halpern-PAGE 相当甚至更优的理论复杂度**，同时揭示了加速表示选择这一被忽视的设计自由度。

---

## 局限性与未来方向

1. **仅适用于期望 cocoercive 算子**：假设3.2 不能覆盖一般 Lipschitz-单调样本算子，也不能处理噪声与 $x$ 相关的场景（§3.2 自陈）。
2. **条件数依赖次优**：强单调情形 oracle 复杂度对条件数 $L/\mu$ 的依赖仍非最优（$\mathcal{O}(L^2/\mu^2)$ vs 理论下界），能否通过 Dual-OHM 改进为开放问题（§6）。
3. **未扩展到一般单调 Lipschitz 算子**：Dual-OHM 的确定性加速在单调 Lipschitz 设定下亦存在 [55]，但本文方法依赖 $\mathbb{I}-\alpha\mathbb{F}$ 的非扩张性，推广需新工具。
4. **常数因子的实际效率**：理论保证中的常数（如6、64等）偏大，实际 mini-batch 选择依赖 $\sigma^2$ 先验知识；低噪声时 $B^*=1$ 退化为单次采样，失去批处理优势。

---

## 研究启发与可借鉴点

1. **"加速表示的选择即设计自由度"**：同一确定性速率可通过不同更新规则实现，其随机稳健性截然不同——这一观点可直接迁移至其他确定性加速算法（Nesterov momentum、 Katyusha 等）的随机化设计，提示研究者在构造随机版本前先检验其对噪声的敏感度。
2. **留一稳定性作为误差控制工具**（附录A.2）：通过替换单个 mini-batch 分析终态偏差，为随机算法的非局部误差传播提供了通用且简洁的上界技术，可复用于其他随机迭代方法的收敛分析。
3. **恒定 batch size 的实用价值**：相比需按迭代调整 batch size 的算法，S-Dual-OHM 使用恒定 $B$ 实现与递减 batch 方法同等的最终精度，工程实现更简单，适合在线/流式场景。
4. **S-Dual-OHM 可与 RAIN 等方差缩减模块组合**：作者指出可将 S-Dual-OHM 作为基线与其他随机优化技术结合，有望进一步突破条件数依赖——为混合设计提供思路。
5. **实验设计的对照策略**：与 S-OHM 共享相同 batch size 以隔离"更新规则"差异，而非单独最优调参——这种控制变量法能清晰验证算法本质贡献，值得借鉴。

---

## 关键术语表

**Root-finding（根查找）**：求解算子方程 $\mathbb{F}(x)=0$，广义涵盖优化、minimax、均衡搜索与不动点问题。

**Cocoercivity（余单调性 / $1/L$-cocoercive）**：$\langle \mathbb{F}(x)-\mathbb{F}(y), x-y\rangle \geq \frac{1}{L}\|\mathbb{F}(x)-\mathbb{F}(y)\|^2$，强于单调性，弱于强单调；梯度算子在凸光滑函数情形自动满足。

**Halpern 迭代 / OHM**：$x_{k+1} = \frac{1}{k+2}x_0 + \frac{k+1}{k+2}\mathbb{T}x_k$，任意时刻（anytime）最优加速，但对随机噪声极度敏感。

**Dual-OHM / 对偶锚机制**：$x_{k+1} = x_k + \frac{N-k-1}{N-k}(\mathbb{T}x_k - \mathbb{T}x_{k-1})$，仅针对预定终点 $N$ 优化，随机稳健性显著优于 OHM。

**S-Dual-OHM**：Dual-OHM 的随机版本，使用恒定 mini-batch 估计 $\mathbb{T}$，无需方差缩减或双循环。

**Variance reduction（方差缩减）**：通过累积历史梯度信息降低 oracle 噪声的技术（如 PAGE、SARAH），本文方法完全避免使用。

**Strong monotonicity（强单调）**：$\langle \mathbb{F}(x)-\mathbb{F}(y), x-y\rangle \geq \mu\|x-y\|^2$，使 $\mathbb{T}=\mathrm{I}-\alpha\mathbb{F}$ 成为压缩映射（$\gamma$-contractive），支持早停加速。

**Oracle complexity（Oracle 复杂度）**：达到精度 $\epsilon$ 所需随机算子查询的总次数，本文核心优化目标，从 $\mathcal{O}(\epsilon^{-4})$ 降至 $\mathcal{O}(\epsilon^{-3})$，强单调下近 $\mathcal{O}(\epsilon^{-2})$。

---

## 可复现要素

| 要素 | 状态 |
|---|---|
| 数据集 | 合成实验：worst-case 非扩张算子、有限和 cocoercive 算子、SCSC Huber minimax；均基于参数构造，无真实数据集 |
| 代码开源 | 论文未声明代码/权重开源仓库链接 |
| 关键超参 | 步长 $\alpha \in (0, 2/L]$；batch size $B \in \{1,10,20,50,100\}$ 网格搜索；N 由预算 $Q=NB$ 确定 |
| 重复要求 | 10次独立随机种子，报告均值与 5th–95th 百分位带 |
| 实验环境 | MacBook Air M3 / 24GB 内存 |

---
