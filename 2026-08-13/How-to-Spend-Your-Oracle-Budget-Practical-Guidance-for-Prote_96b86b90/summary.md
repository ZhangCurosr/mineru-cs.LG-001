---
title: "How-to-Spend-Your-Oracle-Budget-Practical-Guidance-for-Prote"
source: https://arxiv.org/pdf/2608.12192v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:26:18"
field: "蛋白质结构与药物设计中的AI生成模型"
keywords: ["蛋白质结构预测", "Oracle引导", "贝叶斯优化", "潜空间优化", "DPO", "FK-steering", "扩散模型"]
innovations: ["首次将O3优化框架应用于蛋白质结构预测模型，在Oracle预算受限条件下实现高效引导", "系统比较四类引导方法在不同Oracle预算下的性能，揭示O3低预算优势与FK/DPO高预算优势的定性权衡", "对比在线与离线DPO变体，证明在线自适应重采样是DPO在预算约束下有效的关键因素"]
benchmarks: ["PDB: 1CLL (Calmodulin, TM-score Oracle)", "PDB: 9EEH (E. coli ATCase, MolProbity Oracle)"]
---

# 论文速读：How to Spend Your Oracle Budget: Practical Guidance for Protein Structure Prediction Models

## 一句话总结
系统比较 O3、FK-steering、DPO 和 Best K-of-N 四种引导方法在不同 Oracle 预算下的效果，发现 **O3 在低预算下最优**，而 FK-steering 和 DPO 随预算增大才显现优势，为实际科研场景中合理分配昂贵生物 Oracle 调用次数提供实用参考。

## 研究问题与动机
1. 蛋白质结构预测基础模型（如 AlphaFold3、Boltz-2）在某些目标上不可靠（坍塌至单一构象、几何不合理、大复合物组装错误等），需借助外部 Oracle 进行筛选与纠正。
2. 生物 Oracle（如分子动力学模拟、湿实验）代价极高——单次模拟需数天 GPU 计算，Oracle 预算是核心约束。
3. 现有引导方法（FK-steering、DPO、Best K-of-N）在"如何花费预算"上差异显著，但缺乏系统性比较来指导方法选择。
4. 推断时引导与微调两类主流方法的相对有效性尚不明确，本文旨在填补这一空白。

## 核心贡献（创新点）
1. **首次将 O3（Optimisation Over Outputs）框架应用于蛋白质结构预测**：将贝叶斯优化引入 Boltz-2 潜空间，在 Oracle 受限条件下实现高效引导。
2. **第一个系统比较四类引导方法在不同 Oracle 预算下的性能**：揭示"O3 胜低预算、FK-steering/DPO 胜高预算"的定性权衡，为实践者提供决策依据。
3. **对比在线与离线 DPO 变体**：发现在线 DPO 随预算持续增长，离线 DPO 几乎无感，明确了预算分配方式对微调效果的关键影响。
4. **在双目标（1CLL、9EEH）和双 Oracle（TM-score、MolProbity）下交叉验证**：揭示了方法在不同 Oracle 类型和蛋白质复杂度上的泛化能力差异。

## 方法详解

### O3（Bayesian Optimisation via O3）
- **潜空间子空间构建**：从生成模型 $g$ 采样 $M$ 个初始结构并用 Oracle 评分，选取得分最高的 $d$ 个结构对应的潜变量作为种子 $Z = [z_1, \ldots, z_d]^\top$，构建一个 $d-1$ 维子空间 $\mathcal{U} \subset [0,1]^{d-1}$。
- **映射机制**：通过 Knothe–Rosenblatt 变换 $\phi: \mathcal{U} \to \mathbb{S}_+^{d-1}$ 将 $u$ 映射到单纯形上的权重向量 $w$，再经 LOL 投影 $\ell(w, Z) = w^\top Z$ 将权重投影回各向同性高斯先验 $p_0$ 的支持集，得到潜变量 $z$，最终解码为结构 $x = g(z)$。
- **贝叶斯优化**：用高斯过程（GP，RBF 核）拟合子空间内的 Oracle 响应，以 Log Expected Improvement 为采集函数迭代选择新的查询点，直至耗完 $N - M$ 次查询预算。

### FK-steering（Feynman-Kac 引导）
- 在扩散去噪过程中维护 $M$ 个相互作用的粒子，每次重采样时根据当前去噪步 $t$ 的 Oracle 得分计算权重：
$$w(x_t^i) = \frac{g(x_t^i | x_{t_\text{prev}}^i)}{g_\text{proposal}(x_t^i | x_{t_\text{prev}}^i)} \lambda \exp(r(x_t^i) - r(x_{t_\text{prev}}^i))$$
- 重采样仅在中间去噪步（Boltz-2 注入正噪声的阶段）执行，Oracle 在起始步、结束步及前 3/4 轨迹中均匀分配调用，偏好"奖励递增"的粒子。

### DPO（Direct Preference Optimisation）
- 构造偏好对 $(x_w, x_l)$：分别来自 Oracle 得分高于和低于中位数的样本。
- 通过近似 log-likelihood ratio 计算 DPO 损失：
$$\log\frac{p_\theta(x)}{p_\text{ref}(x)} \approx \mathcal{L}_\text{DSM}^\text{ref}(x) - \mathcal{L}_\text{DSM}^\theta(x)$$
- **在线 DPO**：每轮用当前模型采样 $N/E$ 个结构、评分、配对、更新，参考模型每轮重置，KL 正则化逐步软化。
- **离线 DPO**：一次性采样全部 $N$ 个结构并固定训练，参考模型始终保持为预训练权重。

### Best K-of-N
- 直接采样 $N$ 个结构，取 Oracle 得分最高的前 $K$ 个返回，无任何优化过程，作为性能下界基准。

## 实验与结果

**数据集与模型**：使用 Boltz-2 基础模型，在两个蛋白质目标上测试：
- **1CLL**（Calmodulin，144 残基、1184 原子），Oracle 为 TM-score
- **9EEH**（E. coli aspartate transcarbamoylase，7232 原子复合物），Oracle 为 MolProbity（无参考结构）

**预算设置**：$(N, K) \in \{(20,2), (50,5), (100,10), (200,20), (500,50), (1000,100)\}$，覆盖低到中等预算范围。

**主要结果（1CLL，mean-of-K TM-score）**：
| 方法 | N=20 | N=100 | N=1000 |
|---|---|---|---|
| **O3** | ~0.60+ | ~0.81（最高，平台期） | ~0.81 |
| FK-steering | ~0.55 | ~0.65 | ~0.73 |
| Online DPO | ~0.55 | ~0.60 | ~0.71 |
| Offline DPO | ~0.55（恒定） | ~0.55（恒定） | ~0.55（恒定） |
| Best K-of-N | ~0.60 | ~0.60 | ~0.60 |

- **O3 在 N≤1000 的所有预算下均是最优方法**，且是唯一在低预算（N≤100）下显著优于 Best K-of-N 的方法。
- **FK-steering 和 DPO 随预算单调改善**，但起始效率低，直到 N≥200 才能与 O3 竞争。
- **Offline DPO 基本无效**，所有预算下均无提升（≈0.55），证明在线自适应重采样是关键。

**9EEH 结果（MolProbity Oracle）**：
- O3 在非最小预算下仍保持领先；FK-steering 表现明显弱于 1CLL 设置（因 MolProbity 对中间去噪步骤的噪声敏感，导致奖励信号质量差）。
- Max-of-K 场景下 Best K-of-N 在大预算时反超 O3，可能与确定性概率流 ODE 采样相比随机采样多样性更低有关。

## 相关工作脉络
1. **AlphaFold3 / Chai-1 / Boltz-2**：结构预测基础模型，本文均以 Boltz-2 为底层生成器，与前述工作聚焦模型本身架构改进不同，本文关注 Oracle 预算约束下的推理/微调引导策略。
2. **Latent space BO（VAE-BO、LOL-BO、COWBOYS）**：先前的潜空间贝叶斯优化方法依赖 VAE 编码，O3 的创新在于完全由示例种子定义潜空间，无需预训练编码器，首次适配至蛋白质结构预测。
3. **Classifier-free / Classifier guidance**：传统扩散模型引导方法，依赖可微 Oracle 或额外分类器；本文选用 FK-steering 作为梯度自由的推理引导代表，避免了这些限制。
4. **DDPO / DRaFT**：强化学习与可微奖励的微调方法；本文选用 DPO 作为不可微 Oracle 场景下的微调基线，因其仅需排序对而无需显式奖励建模或梯度。
5. **Feynman-Kac steering（Singhal et al., 2025）**：通用扩散引导框架，本文将其适配到 Boltz-2 的特定噪声调度（后半段停噪），并系统分析 $\lambda$、粒子数、重采样步数等超参数。

## 局限性与未来方向
1. **仅评估两个蛋白质目标**：1CLL 为已知 OOD 内结构，9EEH 为训练截止后新 deposited 的复合物，覆盖范围有限，难以推广到更广泛场景。
2. **使用廉价代理 Oracle**：TM-score 和 MolProbity 计算快速，但与真实昂贵的生物 Oracle（分子动力学、湿实验）存在差距，结论的外推需谨慎。
3. **未探索大预算场景**：DPO 在 N>1000 时是否进一步超越 O3 未验证，实际应用中微调方法的预算上限未被触及。
4. **批量多样性考量不足**：DPO 微调后采样批量多样性更优，而推理时方法每次重采Oracle值均需重新评估，未系统量化这一工程差异。
5. **未来方向**：扩展至更多蛋白质目标和昂贵 Oracle（MD 模拟、实验测量）；系统研究 O3 子空间维度 $d$ 与预算 $N$ 的自动匹配策略；探索多目标 Oracle 下的引导方法组合。

## 研究启发与可借鉴点
1. **O3 子空间构建范式**：利用种子示例+Knothe–Rosenblatt 变换+LOL 投影构建低维可优化子空间的方法，可迁移至其他生成模型（分子生成、材料设计）的 Oracle 受限优化场景。
2. **在线 vs 离线偏好优化对比**：在线 DPO 持续用当前策略重采样是关键增益来源，离线固定数据集几乎无效——这一发现对大模型微调实验设计有普适参考价值。
3. **方法选择决策树**：论文提炼的实用建议可抽象为通用框架——**低预算选 O3，中等预算选 FK-steering，高预算选 DPO**，适用于任何"生成模型+昂贵不可微 Oracle"的应用管线。
4. **Oracle 对引导方法的影响差异**：MolProbity 这种局部几何 Oracle 不适合 FK-steering 的中间步评分，而全局折叠 Oracle（TM-score）更适合——提示未来研究应根据 Oracle 性质选择引导策略。
5. **超参数扫描策略**：O3 的 $d$（子空间维度）需随 $N$ 增大而增大（N=20 时 d=6 最优，N=200 时 d=10 最优），这一动态匹配原则可推广到其他基于示例的子空间优化方法。

## 关键术语表
- **Oracle**：外部评分函数，对生成结构打分（如 TM-score、MolProbity、分子动力学稳定性）；通常昂贵且不可微。
- **O3（Optimisation Over Outputs）**：通过若干种子示例构建生成模型潜空间的低维子空间，并在该子空间上直接应用标准优化器的框架。
- **FK-steering（Feynman-Kac steering）**：在扩散去噪轨迹中通过 Sequential Monte Carlo 维护多个粒子，根据 Oracle 奖励加权重采样以引导生成方向。
- **DPO（Direct Preference Optimisation）**：基于偏好对（高分/低分样本）直接微调生成模型参数，无需显式奖励模型，适用于不可微 Oracle。
- **TM-score**：衡量预测结构与参考结构折叠相似度的分数，范围 (0,1]，>0.5 表示相同折叠。
- **MolProbity**：无参考的全原子结构验证分数，综合评估立体冲突、骨架二面角、共价几何等物理合理性指标。
- **Best K-of-N**：采样 $N$ 个候选并直接返回得分最高的 $K$ 个，作为最简无优化基线。
- **Latent space Bayesian Optimisation**：在高维生成潜空间中构建低维代理子空间，再用贝叶斯优化高效搜索的方法类别。

## 可复现要素
- **数据集**：PDB 公开结构 1CLL 和 9EEH（已公开）。
- **代码**：论文未提及代码开源，O3 框架引用 Willis et al. (2025)。
- **模型**：Boltz-2（开源，Passaro et al., 2025）。
- **关键超参**：
  - 总预算 $N$：20 / 50 / 100 / 200 / 500 / 1000
  - 返回批次 $K$：2 / 5 / 10 / 20 / 50 / 100
  - O3 子空间维度 $d$：随 $N$ 变化（20→5/6，100→7，200→10，500→30，1000→50）
  - FK-steering $\lambda$：5 / 20 / 50（论文扫过）
  - DPO $\beta$：KL 正则化强度（论文未详述具体值，仅声明 $\beta > 0$）
  - 随机种子：多数实验 5 次，部分 O3 实验 3 次，FK-steering 10 次
