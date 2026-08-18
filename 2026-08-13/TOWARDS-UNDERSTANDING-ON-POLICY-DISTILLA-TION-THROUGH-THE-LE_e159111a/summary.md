---
title: "TOWARDS-UNDERSTANDING-ON-POLICY-DISTILLA-TION-THROUGH-THE-LE"
source: https://arxiv.org/pdf/2608.11829v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:00:36"
field: "大语言模型训练与蒸馏"
keywords: ["on-policy distillation", "test-time scaling", "pass@K", "capability boundary", "sampling efficiency", "illusory distillation"]
innovations: ["通过test-time scaling解耦OPD的sampling efficiency与capability boundary效应", "揭示OPD存在illusory distillation现象，主要表现为sampling efficiency提升而非能力扩展", "对比验证off-policy distillation可真正扩展能力边界而OPD不能"]
benchmarks: ["AMC2023", "AIME2024", "AIME2025", "AIME2026"]
---

# 论文速读：TOWARDS-UNDERSTANDING-ON-POLICY-DISTILLA-TION-THROUGH-THE-LENS-OF-TEST-TIME-SCALING

## 一句话总结
本文通过测试时扩展（test-time scaling）视角系统分析在线策略蒸馏（OPD），发现OPD主要提升采样效率而非真正扩展学生模型的推理能力边界，呈现出"幻觉蒸馏"（illusory distillation）效应。

## 研究问题与动机
- **核心问题**：OPD是否真正将更强教师的推理能力转移给学生、扩展其能力边界，还是仅改善学生对已有能力的访问效率？
- **现有认知局限**：业界普遍将OPD视为知识蒸馏的一种形式，认为学生可从此获得教师的新推理能力；但近期研究（Li et al., 2026; Zhu et al., 2026）已报告OPD训练后学生可能不如pre-OPD基座模型的表现。
- **评估缺口**：缺乏从sampling efficiency与capability boundary双重视角对OPD进行系统分析的框架，难以区分"学到新能力"与"更高效地访问已有能力"。
- **方法论动机**：引入test-time scaling（通过pass@K/avg@K曲线）作为诊断工具，可更精细地刻画模型在不同采样预算下的能力表现。

## 核心贡献（创新点）
- **首次通过test-time scaling系统解构OPD效应**：提出以pass@K随K增大趋势和avg@K一致性为判据，将OPD行为分解为"sampling efficiency改进"与"capability boundary变化"两个正交维度。
- **揭示"幻觉蒸馏"现象**：发现OPD在small K下优势显著、在large K下被基座模型反超，且遗忘可解题数量多于新学数量，表明其本质是重加权而非能力扩展。
- **对比分析OPD与off-policy distillation的差异机制**：证明off-policy distillation可真正扩展能力边界，而OPD仅引导学生在自身能力空间内更密集地搜索正确路径。
- **验证多种OPD变体的共性行为**：EOPD、ExOPD、Direct-OPD及纯forward-KL变体均呈现相同trade-off模式，表明该现象是OPD范式固有属性而非特定实现缺陷。
- **提供训练动态可视化证据**：通过pass@K随训练步数变化曲线，揭示large-K能力边界退化在训练早期（约80步）即已发生，且波动不稳定。

## 方法详解
- **评估框架**：采用test-time scaling范式，将采样预算K从1变化到1024，分别计算pass@K（至少一个正确样本的概率）和avg@K（所有样本平均正确率）。
- **核心指标定义**：
  - pass@K = E[∃i∈{1,...,K}, y^(i)正确]，反映有限预算下的可达性（small K）与能力边界（large K）
  - avg@K = E[(1/K)Σ𝕀(y^(i)正确)]，反映采样分布的整体质量与效率
- **OPD目标函数**：最小化学生在学生轨迹上分布与教师分布之间的reverse KL散度：L_OPD(θ) = E[Σ_t D_KL(π_θ(·|x,y_<t) || π_T(·|x,y_<t)))]，实践中采用top-k token KL实现。
- **问题级可解性分类**：以pass@1024为可解性判据，将问题分为retained（前后均可解）、learned（仅OPD后可解）、forgotten（仅base后可解），量化OPD对能力边界的净影响。
- **困惑度分析**：比较Base/OPD/Teacher模型生成轨迹在Base和Teacher分布下的perplexity，验证OPD轨迹是否偏向教师偏好但仍保留在Base支持范围内。
- **训练动态追踪**：记录OPD训练中不同checkpoint的pass@1与pass@1024，观察sampling efficiency与capability boundary的演化trade-off。

## 实验与结果
- **数据集**：DAPO-Math-17k用于训练；AMC2023、AIME2024/2025/2026用于评估。
- **三组学生-教师配置**：
  1. Qwen3：Base=Qwen3-1.7B-Base，Teacher=Qwen3-4B-Base-GRPO
  2. Skywork：Base=R1-Distill-Qwen-1.5B，Teacher=Skywork-OR1-Math-7B
  3. JustRL：Base=R1-Distill-Qwen-1.5B，Teacher=JustRL-DeepSeek-1.5B
- **关键数值结果**（Table 3-5）：
  - **Qwen3/AIME24, K=1**：Base 4.2% → OPD 12.0%（+7.8pp），但K=1024时Base 70.0% > OPD 53.3%
  - **Skywork/AIME24, K=1**：Base 29.9% → OPD 36.4%，但K=1024时Base 86.7% = OPD 86.7%
  - **JustRL/AIME24, K=1024**：Base 86.7% > OPD 83.3%
  - **avg@K优势**：所有设置下OPD均持续优于Base，如Qwen3/AIME24 avg@1024：Base 4.2% → OPD 12.0%
  - **可解性转移**：以pass@1024为判据，三个设置中forgotten比例均高于learned比例
  - **问题级精度提升**：retained问题中，Qwen3平均精度从13.5%→25.9%，Skywork从42.3%→48.7%，JustRL从42.0%→60.4%
- **最强结果**：JustRL设置下OPD对sampling efficiency提升最显著，K=1在AIME24达到49.4%（vs Base 29.9%，+19.5pp），但K=1024仍被Base反超。
- **OPD变体对比**：EOPD/ExOPD/Direct-OPD/FKL在small K均优于Base，但large K表现持平或低于Base，仅纯FKL在AIME26的pass@1024略超Base。
- **off-policy对比**：DeepSeek-R1-Distill-Qwen-1.5B/7B在所有K下均超越Base，证实off-policy可实现真正能力转移。
- **训练动态**：Step 80时pass@1024已在三个benchmark上低于Base（如AIME26下降10.0pp），而pass@1持续提升，表明能力边界退化早于收敛。

## 相关工作脉络
- **GKD与MiniLLM（Agarwal et al., 2024; Gu et al., 2024）**：首创通过student rollouts获取teacher反馈的on-policy蒸馏范式，本文在此基础上提出test-time scaling诊断框架，揭示其与RL-style分布重塑的本质相似性。
- **Rethinking OPD（Li et al., 2026）**：已观察到OPD在部分场景下不如base model的表现，但未系统区分sampling efficiency与capability boundary，本文补足这一理论缺口。
- **Test-time Scaling与pass@K（Brown et al., 2024; Yue et al., 2025）**：前者提出用重复采样提升推理性能的思路，后者用large-K pass@K分析RLVR对能力边界的影响；本文将其拓展至distillation场景并对比OPD与off-policy distillation的差异。
- **Direct-OPD与ExOPD（Feng et al., 2026; Yang et al., 2026a）**：分别通过RL-induced policy shift传递和reward extrapolation改进OPD；本文证明这些变体仍共享same test-time scaling trade-off，表明现象具有范式级普遍性。
- **EOPD与Trust Region OPD（Jin et al., 2026; Xing et al., 2026）**：引入entropy-aware forward KL或trust region保留多样性；本文实验显示此类正则化可缓解但不消除capability boundary退化。
- **Off-policy Distillation（Hsieh et al., 2023; Guo et al., 2025）**：依赖teacher生成轨迹的离线蒸馏；本文对比证实其可真正扩展能力边界，为OPD改进方向提供对照基准。

## 局限性与未来方向
- **局限性**：
  - 实验主要聚焦数学推理benchmark（AMC/AIME），结论在语言理解、代码生成等领域的泛化性待验证。
  - 仅分析small-to-medium规模模型（1.5B-7B），对更大规模模型的OPD行为推断需谨慎。
  - 训练数据相对有限（DAPO-Math-17k），可能不足以完全刻画OPD在大规模数据下的稳定行为。
  - 未深入分析teacher-student能力gap过大时的极端情况（如本文JustRL设置中teacher仅在small K有优势）。
- **未来方向**：
  - 探索混合策略：结合off-policy数据与on-policy guidance，兼顾能力扩展与采样效率。
  - 设计动态K调度：根据问题难度自适应分配采样预算，最大化有限compute下的期望收益。
  - 研究OPD早期停止策略：利用pass@1024退化信号判断最优训练步数，避免过度优化sampling efficiency而牺牲capability boundary。
  - 将test-time scaling分析推广至多模态、agent等其他LLM应用范式。

## 研究启发与可借鉴点
- **test-time scaling作为能力诊断工具**：pass@K曲线形态可清晰分离"sampling efficiency"与"capability boundary"两种不同维度的改进，建议在本团队的研究中引入该分析框架评估新方法的真实贡献。
- **问题级可解性转移矩阵**：retained/learned/forgotten分类及精度转移热力图（Figure 9）是量化模型能力变化的有效手段，可直接复用至RLVR、SFT等训练过程的效能评估。
- **early stopping信号设计**：pass@1024在训练早期即开始退化（约80步），可作为OPD训练的监控指标，避免盲目追求small K性能提升而导致长期能力损失。
- **off-policy作为能力扩展补充**：若目标确实是扩展学生能力边界（而非仅提升推理性价比），应优先考虑off-policy蒸馏或混合训练策略；本文的对比实验为该决策提供量化依据。
- **跨变体一致性验证**：本文对EOPD/ExOPD/Direct-OPD/FKL的统一分析范式值得借鉴——任何新提出的OPD改进都应通过相同test-time scaling框架检验，避免仅报告small K指标而隐藏capability boundary退化。

## 关键术语表
- **On-policy Distillation (OPD)**：学生模型用自己的轨迹采样进行蒸馏，教师仅作为token分布指导者，而非轨迹生成者。
- **Test-time Scaling**：通过增加推理时采样预算（如多次采样后取多数投票）提升模型性能的技术范式。
- **pass@K**：在K次独立采样中至少得到一次正确答案的概率，small K反映采样效率，large K反映能力边界。
- **avg@K**：K次采样答案的平均正确率，衡量模型整体采样分布的质量。
- **Illusory Distillation（幻觉蒸馏）**：本文 coined 的概念，指OPD表面上看似从教师学到了新知识，实则是将学生已有能力的访问概率重加权。
- **Reverse KL Divergence**：OPD中采用的目标函数方向，D_KL(π_student || π_teacher)，偏好student分布覆盖teacher高概率区域。
- **Capability Boundary**：模型在无限采样预算下能够解决的最难问题集合，由large-K pass@K逼近。
- **Sampling Efficiency**：模型在有限采样预算下找到正确解的概率，由small-K pass@K和avg@K共同刻画。

## 可复现要素
- **数据集**：DAPO-Math-17k（训练）、AMC2023、AIME2024/2025/2026（评估）；通过math-vault (Ge, 2026b)快照获取，公开可用。
- **代码/权重**：OPD各变体代码依赖各自开源仓库（DeepSeek-AI, Li et al. 2026, Feng et al. 2026, Jin et al. 2026, Yang et al. 2026a）；学生/教师基座模型（Qwen3-1.7B/4B、R1-Distill-Qwen-1.5B、Skywork-OR1-Math-7B、JustRL-DeepSeek-1.5B）均为公开模型。
- **关键超参**：评估temperature=0.7、top-p=0.95、seed=0、context=32768、prompt≤1024、output≤31744；训练batch=64/128/1024、LR=1e-6~1e-5、weight decay=0.01、cosine schedule、student/teacher top-k=16。
