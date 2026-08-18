---
title: "Clustered-Randomized-Smoothing-for-Stochastic-Prediction-Fun"
source: https://arxiv.org/pdf/2608.12037v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 07:35:03"
field: "鲁棒机器学习/安全关键预测"
keywords: ["randomized smoothing", "stochastic prediction", "robustness certification", "multi-modal regression", "clustered smoothing", "adversarial robustness"]
innovations: ["提出clustered α-smoothing框架，通过逐簇局部α-trimmed平滑+混合组合避免多模态mode collapse", "建立首个针对多模态随机预测器的概率鲁棒性下界定理，支持任意聚类算法", "通过anchor points将非凸优化转化为可计算的线性规划下界"]
benchmarks: ["L-GAP driving simulator trajectory prediction", "quadrotor navigation with obstacles (Badings et al. 2023)"]
---

# 论文速读：Clustered-Randomized-Smoothing-for-Stochastic-Prediction-Fun

## 一句话总结
论文提出**clustered α-smoothing**，通过在输出空间对噪声样本进行聚类，在每个簇内独立应用α-trimmed平滑后再以混合分布形式合并，从而避免传统randomized smoothing在随机多模态预测中的mode collapse问题，同时提供概率鲁棒性保证。

## 研究问题与动机
1. **多模态预测的鲁棒性需求**：随机预测器（如VAE、NF、BNN）可建模丰富的多模态分布，但在安全关键领域（自动驾驶、医疗、机器人）需要保证预测的鲁棒性，而随机预测对对抗性扰动高度脆弱。
2. **现有随机平滑在回归场景下的mode collapse**：标准α-smoothing（Rekavandi et al., 2024）通过对含噪输出取平均来平滑预测，但会抹平多模态结构，产生无意义的单峰均值预测（图1和图6中的示例）。
3. **已有聚类平滑工作的局限**：先前在LLM鲁棒性验证中使用的聚类方法（Su et al., 2025; Wang et al., 2026）倾向于直接剪枝小簇，而安全关键场景需要对所有模式（包括小概率模式）进行鲁棒性保障，不能简单丢弃。
4. **现有确定性鲁棒性验证方法计算代价过高**：MIP、抽象解释、SMT求解等方法在复杂模型上难以扩展，需要可扩展的近似替代方案。

## 核心贡献（创新点）
1. **提出clustered α-smoothing框架**：通过任意聚类算法将噪声预测样本分区，在每个簇内独立应用α-trimmed平滑，再以混合分布形式组合——本质区别在于将平滑从全局平均改为逐簇局部平均再混合，从根本上避免了mode collapse。
2. **建立概率鲁棒性下界定理（Theorem 3.1）**：基于Neyman-Pearson引理和union bound，证明smoothed预测以高概率落入由聚簇覆盖区域并集构成的有界区域内，且给出了显式的概率下界公式——这是首个针对多模态随机预测器随机平滑的严格认证理论。
3. **提供可计算的线性规划下界（Proposition 3.2）**：将Theorem 3.1中的非凸优化问题通过anchor points逼近为线性规划，误差项随anchor点数量K增大而趋于零——解决了多簇情形下的可计算性问题。
4. **双基准实证验证**：在L-GAP驾驶模拟器轨迹预测上Wasserstein距离降低27%（3.80 vs 5.18），在四旋翼机器人控制上将碰撞率降低81%，显著优于α-smoothing和RS-Reg。

## 方法详解

**核心思想**：给定输入x和N个i.i.d.噪声样本{ε₁,...,ε_N}，方法三步走：
1. **聚类**：对预测样本{h_w(x+εᵢ)}使用任意聚类算法（如DBSCAN）将其划分为M个簇ν₁,...,ν_M
2. **局部α-trimmed平滑**：在每个簇ν_m内，仅保留落入该簇的样本索引集I_m，然后在其上应用α-trimmed平均（按元素取第⌊α|I_m|⌋到第|I_m|-⌊α|I_m|⌋阶统计量的平均）
3. **混合组合**：以各簇样本占比π⁽ᵐ⁾ = |I_m|/N为权重，随机选择某一簇的平滑输出作为最终预测

**关键公式**：
- 定义2（α-smoothing）：$\tilde{\mathcal{H}}_{N,\alpha}(x) = \frac{1}{N-2\lfloor\alpha N\rfloor}\sum_{i=1+\lfloor\alpha N\rfloor}^{N-\lfloor\alpha N\rfloor}\mathcal{H}(x)_{(i)}$，其中$(\cdot)_{(i)}$为逐元素第i阶统计量
- 定义3（Clustered α-smoothing）：$\tilde{\mathcal{H}}_{N,\alpha,\mathcal{V}}(x) = \sum_{m=1}^{M}\mathbf{1}_{z=m}\tilde{\mathcal{H}}_{N,\alpha,\mathcal{V}_m}(x)$，z~Categorical(π⁽¹⁾,...,π⁽ᴹ⁾)

**鲁棒性保证（Theorem 3.1）**：给定输出空间划分ν={ν₁,...,ν_M}和置信覆盖集R_l⊂ν_l，对于任意扰动‖δ‖₂≤r，smoothed预测落在$\tilde{\mathcal{R}}=\cup_{l\in\mathcal{L}}\mathcal{R}_l$中的概率满足：

$$\mathbb{P}(\tilde{\mathcal{H}}_{N,\alpha,\mathcal{V}}(x+\delta)\in\tilde{\mathcal{R}}) \geq \inf_{p_{\mathcal{V}_m}}\sum_{l\in\mathcal{L}}\sum_{s=1}^{N}\frac{s}{N}\sum_{j=s-\lfloor\alpha s\rfloor}^{s}f_{s,j}\!\left(\frac{p_{\mathcal{R}_l}}{p_{\mathcal{V}_l}}\right)f_{N,s}(p_{\mathcal{V}_l})$$

其中概率界限通过高斯平移修正：$\underline{p}_{\mathcal{V}_m}=\Phi(\Phi^{-1}(\check{p}_{\mathcal{V}_m})-r/\sigma)$等。

**概率估计（Proposition 4.1）**：基于Clopper-Pearson区间和union bound，通过额外$\bar{N}$个采样以置信度1-β估计每个簇的包含概率上下界。

**算法流程**：
- Algorithm 1：CONSTRUCTCOVERAGESETS——采样、聚类（DBSCAN等）、选取每个簇内 fraction p 的样本构建覆盖集R_m、通过generalized Voronoi partition得到划分ν
- Algorithm 2：ROBUSTNESSCERTIFY——调用Algorithm 1、计算概率界限、输出平滑预测及鲁棒性证书

## 实验与结果

**数据集**：
- L-GAP驾驶模拟器数据集（47个场景，人类左转决策数据），配合TrajFlow归一化流模型
- 四旋翼导航基准（Badings et al., 2023），3D空间中穿障碍物到达目标区

**评估基线**：
- α-smoothing（Rekavandi et al., 2024）：标准逐元素α-trimmed平均
- RS-Reg（Rekavandi et al., 2025）：等价于α=0的无trimming平滑
- 原始随机预测器$h_{\mathbf{w}}$

**主要结果**：

| 方法 | Wasserstein W₂（轨迹，R⁴⁰） | W₂（最后一步，R²） |
|------|---------------------------|-------------------|
| Ours | **3.80 ± 1.59** | **1.49 ± 0.63** |
| RS-Reg | 4.70 ± 2.09 | 1.84 ± 0.84 |
| α-smoothing | 5.18 ± 2.19 | 2.04 ± 0.84 |

- **轨迹预测**：相比α-smoothing，W₂距离平均降低**27%**；安全分析中风险率仅**2.5%**（vs α-smoothing的16.5%，RS-Reg的7.5%）
- **四旋翼控制**：原始策略100次模拟中撞毁28次；α-smoothed策略撞毁48次；clustered α-smoothed策略仅撞毁**9次**，碰撞率相对α-smoothing降低约**81%**
- **参数分析**：增大α可提升认证下界（过滤更多离群值），减小鲁棒半径r同样提升下界（图5）

## 相关工作脉络
1. **Randomized Smoothing for Classification**（Cohen et al., 2019; Lecuyer et al., 2019）：将随机平滑引入分类任务的鲁棒性认证，本文将其推广至连续回归/多模态预测场景。
2. **α-Smoothing for Regression**（Rekavandi et al., 2024）：提出对回归预测器进行逐元素α-trimmed平均以去除异常值，是本文方法的直接前身；本文的核心改进在于引入聚类分解以避免mode collapse。
3. **RS-Reg**（Rekavandi et al., 2025）：无trimming的随机平滑回归方法，在实验中作为对比基线，其表现介于原始预测器和α-smoothing之间。
4. **Clustering for LLM Robustness**（Su et al., 2025; Wang et al., 2026）：先在LLM防御中应用聚类+平滑思路，但其目的是剪枝对抗样本所在的小邻域簇；本文强调保留所有簇（尤其是小概率安全关键模式），立意不同。
5. **Input-dependent Randomized Smoothing**（Sukenik et al., 2021; Alfarra et al., 2022）：允许噪声分布依赖于输入x，论文将此方向列为未来工作，暗示当前方法是input-agnostic的简化形式。
6. **形式化验证方法**（Reluplex, Neuralsat, Marabou等）：精确但计算昂贵的验证工具，本文作为其可规模化替代方案定位。

## 局限性与未来方向
1. **重复推理开销**：随机平滑依赖多次前向推理，在大型神经网络或计算成本高的基预测器上可能不实用。
2. **聚类算法敏感性**：框架对聚类算法agnostic既是优点也是缺点，聚类质量直接影响结果，需针对具体场景选择合适的算法和超参。
3. **凸集合划分假设**：Theorem 3.1要求输出空间划分为不相交凸集，对于无法被凸集干净分离的场景（如经典的half-moon toy example）受限。
4. **认证针对的是smoothed预测器本身**：若需要与原始基回归器保持等价性（如物理/动力系统仿真），则无法直接应用本方法。
5. **下界保守性**：多簇情形下的概率下界随簇数增加而变得更保守，可能需要更大的噪声水平或更小的认证半径来维持鲁棒性。
6. **未来方向**：扩展到input-dependent的平滑分布和输入相关的划分ν(x)。

## 研究启发与可借鉴点
1. **"先聚类后局部处理再混合"的范式可迁移**：该思路不仅适用于随机平滑，也可推广至其他需要在多模态输出上保持模式结构的操作（如不确定性传播、分位数估计），值得在本团队相关方向中探索。
2. **α-trimmed平均作为鲁棒统计量**：在每个簇内使用α-trimmed平均而非简单均值，天然抑制簇内离群值，这一设计可与任何基于期望平滑的框架结合。
3. **Generalized Voronoi Partition的使用**：Algorithm 1通过点到集距离而非点到点距离构建划分，使覆盖集R_m严格包含于对应的ν_m，这一技巧对保证理论定理的条件适用性至关重要，值得在其他鲁棒性认证工作中复用。
4. **Wasserstein距离作为多模态保真度量**：用W₂距离量化 smoothed预测与原始预测器分布之间的偏差，为多模态平滑方法提供了直观的量化评估指标。
5. **与RL策略的结合方式**：将平滑应用于DRL输出层（action distribution）而非策略网络本身，是一种无需重新训练的即插即用鲁棒化方案，适用于各种基于随机输出的控制器。

## 关键术语表
**Randomized Smoothing**：通过在输入上注入噪声并取预测期望来构造平滑预测器，从而获得对抗扰动的概率鲁棒性保证。
**α-Smoothing**：在随机平滑的基础上对每个输出维度进行α比例trimming（丢弃极端α分位样本后取平均），以降低离群值对平滑结果的影响。
**Mode Collapse**：多模态分布在平均操作中退化为单峰的现象，是标准随机平滑在多模态预测中的核心失效模式。
**Clustered α-Smoothing**：本文提出的方法，先将预测样本按输出值聚类，在各簇内独立执行α-trimmed平滑，再以簇大小加权混合输出。
**Coverage Region (R_m)**：算法构造的输出空间子集，包含该簇内约fraction p的预测样本，用于定义鲁棒性认证的安全区域。
**Generalized Voronoi Partition**：基于点到集合的距离（而非点到点距离）将空间划分为若干cell，使每个cell包含距离对应集合最近的点。
**Neyman-Pearson Lemma**（在本文中的应用）：用于导出高斯噪声下事件概率的上下界，是推导Theorem 3.1中概率保证的核心工具。
**Clopper-Pearson Interval**：基于二项分布的精确置信区间，本文用于从有限样本中以高置信度估计各簇的包含概率上下界。

## 可复现要素
- **数据集**：L-GAP（CC-BY许可证，来源：Zgonnikov et al., 2024）；nuScenes（CC BY-NC-SA）；四旋翼基准来自Badings et al. (2023)
- **代码**：论文声明代码开源，路径为 `https://github.com/DAI-Lab-HEReALD/General-Framework/blob/main/Framework/simulations_randomized_smoothing.py`
- **关键超参（轨迹预测）**：DBSCAN聚类，ε_db=0.45, min_samples=50, M_max=3, p=0.85（覆盖fraction）, σ=0.05, N=30, α=0.4, β=10⁻², N̄=4000
- **关键超参（四旋翼）**：σ=0.05, N=30, α=0.4, M_max=3, p=0.9, DBSCAN ε_db=0.45, min_samples=50, N̄=4000, β=10⁻²

---
