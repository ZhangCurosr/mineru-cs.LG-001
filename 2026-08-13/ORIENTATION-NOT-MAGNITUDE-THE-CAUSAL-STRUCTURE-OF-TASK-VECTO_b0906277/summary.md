---
title: "ORIENTATION-NOT-MAGNITUDE-THE-CAUSAL-STRUCTURE-OF-TASK-VECTO"
source: https://arxiv.org/pdf/2608.11797v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:33:13"
field: "大语言模型合并与干扰分析"
keywords: ["model merging", "task arithmetic", "causal intervention", "activation analysis", "cross-term", "expression gating"]
innovations: ["首次对task-vector cross-term实施激活级别因果干预并证明方向特异性", "揭示指令包装的expression gating机制——内部放大干扰但抑制表达", "证明bf16激活差异分析中75-90%均匀性为量化噪声"]
benchmarks: ["Qwen2.5-1.5B", "Llama-3.2-1B", "Qwen2.5-7B"]
---

# 论文速读：ORIENTATION-NOT-MAGNITUDE-THE-CAUSAL-STRUCTURE-OF-TASK-VECTO

## 一句话总结
论文通过精确分解merged LLM中layerwise cross-term并实施首个激活级别因果干预，证明模型合并的干扰由跨项方向（orientation）驱动而非幅度，且指令包装会内部放大cross-term但抑制其表达——现有所有magnitude指标均无法捕捉这一机制。

## 研究问题与动机
1. **现有诊断的盲区**：模型合并失败时，领域用层间表示偏差、跨任务线性度偏离、参数子空间重叠等**幅度指标**定位干扰，但从未验证"干扰大的地方就是干扰产生的地方"这一核心假设。
2. **缺乏因果干预**：已有工作全是观测性度量，没有一个研究对cross-term本身实施因果干预以确认其是否为causally load-bearing。
3. **格式效应的机制缺失**：合并模型在不同prompt format下表现差异巨大（raw vs. wrapped可达20×），但现有magnitude指标无法解释这一现象，甚至指向相反方向。
4. **bf16分析的可靠性存疑**：混合精度下的激活差异分析可能被量化噪声主导，但领域内缺乏系统性精度审计。

## 核心贡献（创新点）
1. **Transport主导的层间通量分解**：首次将cross-term的层间通量精确分解为传输（transport）与生成（generation）两部分，发现~70%通量来自已有项的传输，late-network每块增益达1.08；与CTL仅报告残差大小不同，本文揭示了residual如何被传播和维持。
2. **Cross-term erasure的因果干预框架**：提出首个针对task-vector cross-term的激活级别因果干预方法，通过四种norm-matched结构控制证明方向特异性——与CTL/SurgeryV2等纯度量方法不同，本文建立了"方向=因果承载"的因果链。
3. **Expression gating的发现**：揭示指令包装内部放大cross-term生成（wrapper下生成量更大）但同时抑制其表达（去除13×更少干扰），解释了格式敏感性的机制；与RAIN-Merging仅观察行为层面的差异不同，本文定位了隐藏的行为-干扰解耦。
4. **bf16量化噪声的方法论警示**：证明naive bf16生成的局部generation均匀性（±15%）实为75–90%的weight-rounding roughness，为整个activation-difference分析领域提供了精度审计范式。

## 方法详解
- **四角因子账本（Factorial Ledger）**：对每个prompt构建四种权重状态（base、A-only、B-only、merge），在每层block输入处记录残差状态 $h_\ell^{00}, h_\ell^{A0}, h_\ell^{0B}, h_\ell^{AB}$，定义累积cross-term $I_\ell = h_\ell^{AB} - h_\ell^{A0} - h_\ell^{0B} + h_\ell^{00}$。
- **传输-生成精确分解**：在共同参考状态 $\bar{h}_\ell = h_\ell^{A0} + h_\ell^{0B} - h_\ell^{00}$ 上评估所有四个block函数，得到新生成项 $G_\ell$，然后利用恒等式 $I_{\ell+1} = G_\ell + T_\ell + M_\ell$ 将剩余部分分解为传输项 $T_\ell$（已有cross-term的传递与放大）和边际失配项 $M_\ell$。
- **Cross-term擦除干预**：在冻结边界 $\ell$ 处将merged模型的残差状态替换为 $h_\ell^{AB} - \lambda I_\ell$（$\lambda=1$ 为完全擦除），通过dose ladder $\lambda \in \{0, 0.5, 1, 1.5, 2\}$ 探测剂量响应。
- **四种norm-matched控制**：wrong-pair（借用其他任务对的第二轴）、coefficient-mismatch（同仿射空间内错误系数组合）、cross-prompt（用捐赠prompt的additive state）、distance-matched random（高斯方向匹配范数），每种都保留 $\|I_\ell\|$ 但破坏一个结构属性。
- **输出评估量**：expressed interference $R = \frac{\text{mean JSD}(p^{\alpha\beta}, p^{\text{add}})}{\text{mean JSD}(p^{\alpha\beta}, p^{00})}$，干预效果 $D = R_{\text{original}} - R_{\text{patched}}$，分母冻结为unpatched主效应以避免混淆。

## 实验与结果
- **模型与设置**：主实验使用Qwen2.5-1.5B（28 blocks），复现实验使用Llama-3.2-1B（16 blocks, GQA）和Qwen2.5-7B（28 blocks）；6个任务（math, code, instruction, safety, summarization, translation）通过rank-16 LoRA微调获得task vector；pre-registered 1,296个cell（2 pairs × 6 strata × 3 seeds × 36 composition grid points）。
- **Transport主导**：transport占比~69%（fp32，3/3 seeds），late-network transport gain = 1.08/block；cumulative cross-term达0.974，远超最相干累积上界0.369的2.6倍。
- **Causal erasure效果**：λ=1时去除约19%的表达干扰（$D \approx +0.026$），top-1 agreement保持0.94–0.99；沿正确方向擦除效果饱和，而norm-matched错误方向控制反而恶化干扰（monotonic backfire）。
- **方向特异性**：正确方向vs. wrong-pair控制的差异在3/3 seeds上均为正（+0.049至+0.059）；效应大小与位移在擦除方向的投影成正比。
- **Regrowth实验**：在depth 3或8擦除后，网络在输出端重建cross-term至99%范数、方向余弦0.99；仅在靠近输出处（depth 20+）重建不充分，可去除干扰恰与未重建份额（$1-\rho$）一致。
- **Expression gating**：相同擦除在Alpaca wrapper下仅去除raw条件的1/13（$D_{\text{wrapped}}$ 近零），wrapper将主效应JSD放大13×同时interaction JSD不缩小。
- **Magnitudefallacy量化**：float32下两个任务对的local generation相差仅1.9×，但causally removable interference相差14×–337×。
- **bf16精度警示**：bf16的$\pm$15%均匀性经fp32审计后确认为75–90%的weight-rounding roughness；fp32下band跨度达3.4×。
- **尺度扩展**：Qwen2.5-7B上expression gating完全关闭（$D_{\text{wrapped}}$ 各seed均不显著异于零），pair contrast达6.9–22.9×。

## 相关工作脉络
1. **Cross-Task Linearity (CTL, Zhou et al., 2024)**：报告层间feature近似为 endpoint models'feature的线性组合；本文的$I_\ell$正是CTL残差，但CTL仅报告残差小，本文证明该小残差的**方向**（而非大小）是causally load-bearing的干扰来源。
2. **SurgeryV2 (Yang et al., 2024)**：通过深层表示修复合并模型；本文将其表示偏差重新定位为cumulative magnitude信号，在格式维度上与其行为脱钩。
3. **Task Singular Vectors (TSV, Gargiulo et al., 2025)**：在参数空间通过奇异向量对齐减少干扰；本文将"方向"区分为参数空间、激活空间和因果三个层次，指出TSV解决的参数方向与本文发现的活动方向不在同一层面。
4. **RAIN-Merging (Huang et al., 2026)**：在行为层面记录合并破坏指令格式处理；本文的expression gating机制是其机制级对应，两者互补——RAIN-Merging修复行为，本文定位行为所掩盖的干扰源。
5. **Activated Parameter Locating (APL, Kong et al., 2024)**：在参数级别施加因果干预；本文的干预发生在激活级别并直接针对cross-term本身，而非参数重要性。
6. **TIES-Merging (Yadav et al., 2023) 及后续**：通过sign agreement、layer selection等解决参数空间冲突；本文结果表明这些magnitude-level方法无法看到格式gate效应。

## 局限性与未来方向
1. **模型覆盖有限**：仅三个模型点（Qwen2.5-1.5B full / 7B reduced / Llama-3.2-1B reduced），一种adapter参数化（rank-16 LoRA），全量finetune和其他模型族仍未检验。
2. **统计单位受限**：以3 seeds为外层复制单位，尽管per-prompt bootstrap提供了27/28 CI排除零的支持，但seed间变异性仍是瓶颈。
3. **格式泛化有限**：expression gating证据基于单个wrapper模板（Alpaca）和两个核心任务对，多wrapper实验扩展了结论但未覆盖全部格式。
4. **Direction特征预测失败**：预注册的三个direction feature在held-out pairs上全部失败，cheap directional predictors仍待开发。
5. **行为恢复未证实**：连续NLL端点验证了干预的有效性但阴性结果——erasure控制表达的非加性而非专家相似性，对实际行为质量的影响仍需研究。
6. **Safety行为不做声明**：论文明确不对安全行为做因果推断。

## 研究启发与可借鉴点
1. **精度审计范式**：activation-difference分析必须进行 propagated-ULP noise floor 和 finite-difference convergence check，否则可能被量化噪声主导而误读——这是本团队做类似分析时可复用的方法论规范。
2. **因果干预优于相关性度量**：norm-matched结构化控制+剂量响应曲线能严格区分"方向因果承载"与"幅度粗相关"，这一实验设计可迁移至其他模型解释场景。
3. **评估格式的机制意识**：format-wrapped评估可能读取的是gate而非干扰本身，团队在做合并模型评测时应同时报告raw和wrapped条件下的机制指标。
4. **Regrowth/Attractor分析的实验设计**：擦除后追踪cross-term重建的动态过程，揭示了缓解方法必须作用于late layers——这对设计合并干预策略有直接指导价值。
5. **Pre-registration as methodological asset**：46条预注册预测中14条主干+32条复现，全部预先冻结假设和通过标准， falsification也如实报告——这种透明度的实践值得借鉴以提升研究可信度。

## 关键术语表
**Cross-term ($I_\ell$)**：合并模型相对于两个单任务模型additive prediction的non-additive残差，定义为四角差 $h_\ell^{AB} - h_\ell^{A0} - h_\ell^{0B} + h_\ell^{00}$。
**Transport ($T_\ell$)**：已有cross-term通过forward pass的传递与放大，在层间通量分解中占比约70%，late-network每块增益>1。
**Expression gating**：指令包装内部放大cross-term生成但同时抑制其输出的现象，导致wrapped条件下expressed interference缩小13–20×。
**Erasure干预**：在激活层面将merged状态替换为 $h_\ell^{AB} - \lambda I_\ell$，沿cross-term方向移除干扰。
**Magnitude fallacy**：假设干扰的大小（而非方向）决定其功能影响的认知偏差，本文证明所有幅度指标跨格式指向错误方向。
**Recovery ratio ($\rho$)**：擦除后下游重建的cross-term范数与原范数之比，接近1.0表明propagation将cross-term视为attractor。
**Factorial ledger**：基于四种权重状态的因子分解账本，精确分离每层新生成与传输贡献。
**Norm-matched control**：保持位移范数不变但破坏特定结构属性的对照干预，用于验证方向特异性而非扰动大小。

## 可复现要素
- **代码**：已开源（随reproducibility statement发布全部脚本）
- **权重**：base model commit pinned，LoRA adapters使用公开recipe训练
- **数据集**：6个公开数据集（MathQA/Codeswarm等），训练样本数、epochs、LoRA rank=16
- **关键超参**：composition grid $(\alpha, \beta) \in (0, 1.2]^2$ 36点，冻结边界 $\{3, 8, 14, 20, 25\}$，dose ladder $\lambda \in \{0, 0.5, 1, 1.5, 2\}$，5,000 paired bootstrap draws
- **数值精度**：所有因果干预在float32（TF32关闭）下进行；bf16仅用于描述性测量并经过audit
- **硬件**：主实验单卡A100，7B复现在单卡A800-80GB
- **预注册**：六版本freeze log，全部46条预测预先冻结，1,296+90+54条immutable per-cell结果随代码发布
