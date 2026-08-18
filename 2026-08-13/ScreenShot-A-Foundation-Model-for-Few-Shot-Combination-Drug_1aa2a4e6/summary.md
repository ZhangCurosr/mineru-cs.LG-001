---
title: "ScreenShot-A-Foundation-Model-for-Few-Shot-Combination-Drug"
source: https://arxiv.org/pdf/2608.12219v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:37:48"
field: "计算肿瘤学 / 药物筛选"
keywords: ["组合药物筛选", "few-shot prediction", "in-context learning", "foundation model", "active learning", "hierarchical transformer", "drug response prediction"]
innovations: ["层级Transformer基础模型实现无需微调的few-shot组合药物响应预测", "基于治疗指数加权的k-means++主动学习策略以1/3预算达到均匀筛选相同命中召回率", "UNK token机制支持未见药物的上下文推断"]
benchmarks: ["BATCHIE", "NCI-ALMANAC", "GDSC-SQ", "PDO-Breast"]
---

# 论文速读：ScreenShot: A Foundation Model for Few-Shot Combination Drug Screening

## 一句话总结
ScreenShot 是一个层级 Transformer 基础模型，在 40 个泛癌药物筛选数据集（约 3,700 种药物、6,000 个生物样本）上预训练，仅依赖少量（药物-剂量-活力值）观测即可通过上下文学习（in-context learning）预测新患者样本对组合疗法的响应，无需微调、无需分子特征分析，且在多个基准上显著优于 XGBoost、TabPFN 和微调 MLP。

## 研究问题与动机
1. **组合药物筛选的组合爆炸问题**：穷举所有药物组合的成本与时间不可承受，现有最大规模组合筛选仅覆盖极小子集。
2. **现有方法依赖分子特征（omics），临床场景难以满足**：多数药物响应预测模型需基因表达或转录组数据，而患者来源样本（ex vivo）的分子谱系往往不可得或不一致。
3. **小样本条件下的样本效率不足**：精准医疗中可获取的细胞数极少，亟需能从少量观测中高效外推的方法，而非每例重新微调。
4. **实验设计（active learning）缺乏高效策略**：即使有了预测模型，如何选择最少数量的初始实验以最大化命中（hit）识别效率仍缺少系统性方案。

## 核心贡献（创新点）
1. **首个面向功能药物响应的上下文学习基础模型**：ScreenShot 在泛癌筛选数据上预训练，推理时仅需（药物-剂量-活力值）三元组，完全不依赖 omics 特征；与 TabPFN 等通用表格基础模型不同，其架构直接对齐药物筛选数据的嵌套层次结构。
2. **三层层级 Transformer 架构（DCE + RE + SE）**：将药物组合编码、响应条件化、样本聚合分三层处理，推理复杂度从平铺注意力的 $O(n^2)$ 降至 $O(p^2 + n)$（$p$ 为独特扰动数），并输出可复用的多粒度嵌入。
3. **嵌入驱动的多轮主动学习策略**：冷启动阶段用 k-medoids 在 Level-1 药物-剂量嵌入上做覆盖选择；自适应轮次用带 delta 加权的 k-means++ 在 Level-3 上下文条件嵌入上选择，以治疗指数（therapeutic index）为目标函数驱动探索-利用权衡。
4. **经 4 个 held-out 数据集验证的 SOTA 性能**：在 PDO-Breast、BATCHIE、NCI-ALMANAC、GDSC-SQ 上均取得最高 Pearson r 和 Top-hit recall；以 1/3 预算达到均匀筛选相同的命中召回率。

## 方法详解
**整体目标**：给定新样本 $s$ 的少量上下文 $\mathcal{D}_s = \{(\mathbf{d}_j, \delta_j, y_j)\}_{j=1}^n$，预测未测试查询 $(\mathbf{d}', \delta')$ 的活力值 $\hat{y}$，不进行梯度更新。

**三级层级架构（Figure 1/4，约 7M 参数）**：

- **Level 1 – Drug Combination Encoder (DCE)**：每个药物 $d_i$ 查嵌入表得 $\mathbf{e}_{d_i} \in \mathbb{R}^{256}$，剂量经可学习 Fourier 特征 $\phi(\delta)$ 映射后与药物嵌入相加，由自注意力捕捉药物间交互效应，再经 masked average pooling 得到组合嵌入 $\mathbf{c} \in \mathbb{R}^{256}$；通过注意力 mask 原生支持单药/双药/多药组合。
- **Level 2 – Response Encoder (RE)**：将组合嵌入与观测活力相加 $\tilde{\mathbf{c}}_j = \mathbf{c}_j + \phi_y(y_j)$，按独特扰动分组后，查询嵌入 $\mathbf{c}'$ 对每组做 cross-attention，产出每个扰动的条件化表示 $\mathbf{r}'_g$。
- **Level 3 – Sample Encoder (SE)**：对 $\{\mathbf{r}'_g\}$ 做自注意力跨扰动聚合，经线性头 + sigmoid 输出 $\hat{y} \in [0,1]$。

**预训练目标**：随机采样样本 $s$，均匀抽取上下文 $\mathcal{D}$ 和查询集 $\mathcal{Q}$，最小化平均绝对误差（MAE，对离群值鲁棒）：
$$\mathcal{L}(\theta) = \mathbb{E}_{s,\mathcal{D},\mathcal{Q}}\left[\frac{1}{|\mathcal{Q}|}\sum_{(\mathbf{d},\delta,y)\in\mathcal{Q}} |f_\theta((\mathbf{d},\delta),\mathcal{D}) - y|\right]$$
共 $\sim 400\text{K}$ 迭代，Adam $lr=10^{-5}$，单张 H100。

**UNK 药物处理**：预训练时随机替换最多 20% 药物身份为 100 个 UNK 嵌入之一，使模型学会从上下文推断未知药物行为；实验显示 pretrained UNK 优于随机映射到已知药物（BATCHIE budget=300 时 MAE +0.017）。

**主动学习算法（Algorithm 1）**：
- 冷启动（33% 预算）：对所有候选计算 Level-1 嵌入，k-medoids 选择初始批。
- 自适应轮次：用当前上下文预测各候选差异活力 $\Delta_t = \hat{y}_t^{\text{target}} - \hat{y}_t^{\text{control}}$，权重 $w_t = \max(-\Delta_t, 0)$，在 Level-3 嵌入上执行加权 k-means++，已选处理作为冻结中心保证多样性。

## 实验与结果
**数据集**：预训练 40 个数据集（约 30M 观测，3,700 药物，6,000 样本，含 CL 和 ex vivo）；hold-out 4 个：NCI-ALMANAC（~5,000 组合）、GDSC-SQ（~1,300 组合）、BATCHIE（~19,000 组合）、PDO-Breast（纯单药 ex vivo，16 样本）。

**评估指标**：Pearson r（预测精度）；Top-hit recall（top-10% 真实命中中 top-20% 预测命中的召回率）。

**主要结果（Table 1）**：
- **PDO-Breast**：n=10 时 ScreenShot r=0.78 vs MLP-FT 0.73；n=300 时 r=0.85 vs 0.82；hit recall n=300 时 96.0% vs 93.3%。
- **BATCHIE**：n=10 时 hit recall 63.9% vs MLP-FT 45.9%（+18pp）；n=300 时 76.0% vs 68.7%。
- **NCI-ALMANAC**：n=300 时 r=0.78 vs MLP-FT 0.76；hit recall 60.6% vs 66.0%（后者略优，因 MLP-FT 在小样本时过拟合）。
- **速度**：ScreenShot 推理 6–68 秒/样本，比 MLP/MLP-FT 快 5–50×。
- **主动学习（Table 6/7）**：BATCHIE budget=300 时，Adaptive(2 rounds) hit recall=84.8% vs Random 75.8%（+9pp），达到随机 300 预算的相同性能仅需约 100–133 样本（约 1/3 预算）。

## 相关工作脉络
1. **DrugCell / DeepCDR / HiDRA / GraphDRP**：基于基因组/转录组特征的单药响应预测，需分子谱系，不适用于 ex vivo 患者样本。
2. **Deepsynergy / DeepDDS / MatchMaker / MARSY 等组合协同预测**：输出标量协同分数而非完整剂量-响应曲线，且多数依赖 omics。
3. **TCRP / TRANSACT / ScreenDL（迁移学习）**：需源域与目标域共享分子特征；ScreenShot 完全 omics-free。
4. **TabPFN / CARTE / TabDPT（表格基础模型）**：通用架构，未针对药物筛选数据的嵌套结构（组合→扰动→样本）定制；ScreenShot 的层级设计更贴合数据生成过程且推理更快。
5. **BAITSAO / CancerGPT / SynerGPT（LLM-based 药物模型）**：依赖文献知识，对无先验文献的患者样本无效，且多数仅预测二元协同标签。
6. **Pichotta et al. (PPC-FM)**：仅支持单药场景， ScreenShot 同时覆盖单药与组合。

## 局限性与未来方向
1. **药物覆盖率有限**：预训练库覆盖约 89–100% 评估集药物，部分新型化合物需用 UNK token，预测精度下降（BATCHIE 上 UNK 药物 MAE 0.152 vs 已知 0.090，budget=300）。
2. **Level-3 自注意力的计算开销**：虽远优于平铺注意力，但对超大规模上下文仍有优化空间；消融表明移除 Level-3 仅小幅降低 hit recall，未来可探索更轻量的样本级聚合。
3. **全局精度与命中检测不完全对齐**：Uncertainty weighting（MC dropout）获得最优 MAE/Pearson 但 hit recall 低 6pp，说明探索性主动学习策略的目标函数需直接对齐下游任务。
4. **仅验证于 4 个 held-out 数据集**：未涉及真实临床试验或体内验证场景。
5. **UNO 药物扩展性**：随着公共筛选数据增长，预训练库可自然扩展，但当前 UNK 机制对 novel compound 的推断能力仍需进一步验证。

## 研究启发与可借鉴点
1. **层级结构对齐数据生成过程**：将 NLP 中"词→句→文档"的层级注意力迁移至"药物→组合→扰动→样本"的药物筛选数据，是一种值得复用的架构归纳偏置设计思路，可推广至其他嵌套结构化预测任务。
2. **上下文学习替代逐样本微调**：ScreenShot 用 cross-attention 将 adaptation 摊销为单次前向传播，避免了 fine-tuning 的超参敏感性和过拟合风险；对于同样面临每样本数据量少的生物学预测任务（如单细胞注释、CRISPR 筛选），该范式可直接借鉴。
3. **目标函数驱动的主动学习加权**：用 therapeutic index（治疗指数）而非模型不确定性作为选择权重，证明"选择策略的优化目标应直接对齐最终评估指标"，这一原则可迁移至实验设计自动化领域。
4. **UNK token 机制处理未见实体**：用专门的 UNK 嵌入而非随机初始化或映射到近邻，使模型学会"从上下文推断未见化合物行为"；可用于其他需要处理 out-of-vocabulary 实体的序列/表格建模任务。
5. **Fourier feature 编码连续变量**：用可学习 Fourier 特征（而非简单线性投影）编码剂量和活力值，是提升连续变量表示能力的轻量技巧，可复用到任何含剂量/浓度连续输入的模型。

## 关键术语表
- **In-context learning**：大模型范式，通过在输入 prompt 中提供少量示例，使模型在不更新权重的情况下完成目标任务；ScreenShot 将其应用于表格型药物筛选数据。
- **Therapeutic Index (TI)**：治疗指数，定义为靶样本与对照样本间差分活力的最小值，负值越大表示选择性越强；用于量化组合药物的选择性有效程度。
- **Perturbation**：单次药物-剂量处理，即一个（药物组合，剂量）对及其对应的活力观测值。
- **k-medoids / k-means++**：两种聚类初始化策略；k-medoids 以实际数据点作为中心，平衡代表性与多样性；k-means++ 以概率正比于加权平方距离选择中心，更强调分散性。
- **Morgan fingerprint**：一种分子指纹，将化学结构编码为固定长度二进制向量；本文基线 XGBoost/TabPFN 使用 512 维 radius-2 Morgan fingerprint。
- **Ex vivo screen**：体外筛选，使用患者来源的组织/类器官直接在实验室中测试药物反应，区别于细胞系（cell line, CL）筛选。

## 可复现要素
- **数据集**：预训练使用 40 个公开/授权筛选数据集（见 Table 2）；ex vivo 数据向 Pichotta et al. 申请获取；4 个 hold-out 数据集均为公开数据集。代码声明开源。
- **代码**：开源，GitHub https://github.com/tansey-lab/screenshot
- **预训练权重**：论文声明开源（同 GitHub 链接）
- **关键超参**：embedding dim $D=256$，Fourier frequencies $F=64$，transformer 5 layers × 8 heads（head dim 32），Adam $lr=10^{-5}$，$\sim 400\text{K}$ 迭代，单张 H100；冷启动预算占比 33%，adaptive rounds=1 或 2。
- **UNO 药物**：预训练时随机替换比例 $\alpha \sim \text{Uniform}[0, 0.2]$，共 100 个 UNK token。
