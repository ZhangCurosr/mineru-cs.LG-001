---
title: "ORIENTATION-NOT-MAGNITUDE-THE-CAUSAL-STRUCTURE-OF-TASK-VECTO"
source: https://arxiv.org/pdf/2608.11797v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:33:00"
field: "语言模型合并与任务向量干扰机制"
keywords: ["model merging", "task arithmetic", "causal intervention", "activation patching", "interference diagnostics", "cross-term transport", "expression gating", "magnitude fallacy"]
innovations: ["跨项运输主导与方向吸引子机制", "激活级跨项因果消除与四重结构性对照协议", "指令包装器表达门控与幅度谬误证伪"]
benchmarks: ["Qwen2.5-1.5B/7B", "Llama-3.2-1B", "PKU-SafeRLHF", "CoT Math", "Stanford Alpaca", "Summarization", "Translation"]
---

# 论文速读：ORIENTATION-NOT-MAGNITUDE-THE-CAUSAL-STRUCTURE-OF-TASK-VECTO

## 一句话总结
本文通过因子账本（factorial ledger）与激活级因果干预，证明在任务向量合并的语言模型中，干扰并非由跨项（cross-term）的幅度决定，而是由其**方向在正向传播中被运输与放大**；沿该方向的精确消除可因果地移除干扰，而指令包装器会放大内部干扰却掩盖其表达。

## 研究问题与动机
- **核心问题**：任务算术合并 LLM 时，两个任务向量的功能干扰在计算中**因果性地 reside 于何处**——是网络产生的跨项“有多少”，还是网络对它的处理？
- **现有方法不足**：领域内主流诊断均依赖**幅度测量**（层间表示偏差、跨任务线性偏离、参数子空间重叠等），隐含假设“干扰表现大的地方就是干扰产生的地方”，但本文发现该假设在因果层面不成立。
- **评估误导性**：指令包装器（instruction wrapper）会使输出干扰大幅降低（20×），但内部跨项反而增大，导致基于格式包装的评估**读取的是门控而非干扰本身**。
- **精度陷阱**：naive bf16 估计的“全局均匀性”（±15%）实为量化噪声，误导幅度分析结论。

## 核心贡献（创新点）
1. **运输主导机制**：跨项层通量中约 65–70% 来自既有跨项的运输（而非新生成），晚期网络每块增益 >1；与 CTL/NeuroMerging 等仅描述幅度的工作本质不同，本文首次量化**传播动力学的结构性来源**。
2. **方向因果负载性**：沿跨项自身方向精确消除（λ=1）可剂量依赖性移除约 19% 的表达干扰且不损伤行为；而所有范数匹配的结构性对照失败或反效果——与参数级干预（APL）不同，本文在**激活空间**靶向跨项方向本身。
3. **表达门控机制**：指令包装器放大内部跨项生成（3/3 seeds），却因模板钉住输出分布使干扰表达下降 13–20×；该发现揭示了格式敏感性的**因果机制**，而非 RAIN-Merging 的行为级观测。
4. **幅度谬误证伪**：局部生成幅度最多排序任务对 1.9×，而因果可移除干扰差异达 14–337×；所有幅度指标（局部、累积、参数空间、表示偏差）在格式轴上均指向错误方向。
5. **严格预注册与因果验证框架**：46 项预测在数据前冻结，含对照构造、剂量阶梯、数值有效性门控；本文提供了**首个激活级跨项因果干预协议**，包含四重范数匹配结构性对照。

## 方法详解
- **因子账本（Factorial Ledger）**：对每个提示 x，记录四个权重配置下残差状态进入块 ℓ 的值：$h_\ell^{00}, h_\ell^{A0}, h_\ell^{0B}, h_\ell^{AB}$。
- **累积跨项定义**：
  $$
  I_\ell = h_\ell^{AB} - h_\ell^{A0} - h_\ell^{0B} + h_\ell^{00}
  $$
  表示进入块 ℓ 前积累的所有非加性残留。
- **精确分解（每块贡献分离）**：
  $$
  I_{\ell+1} = G_\ell + T_\ell + M_\ell
  $$
  其中 $G_\ell$ 是在加性参考状态 $\bar{h}_\ell = h_\ell^{A0} + h_\ell^{0B} - h_\ell^{00}$ 上由块权重产生的新跨项；$T_\ell$ 为既有跨项的运输部分；$M_\ell$ 为边缘路径状态不匹配项。
- **表达干扰估量**：采用交互比 $R$（JSD 之比），归一化主效应，避免 mean-of-ratios 偏差。
- **因果干预效应**：
  $$
  D = R_{\mathrm{original}} - R_{\mathrm{patched}}
  $$
  分母冻结为未修补主效应，确保 D 仅反映单一干预变化。
- **跨项消除与对照**：在冻结边界用 $h_\ell^{AB} - \lambda I_\leq$ 替换残差状态；四重范数匹配对照：wrong-pair、coefficient-mismatch、cross-prompt、distance-matched random，分别破坏特定结构属性。
- **数值有效性门控**：确定性空值（四位点 bitwise 一致）、传播 ULP 噪声 floor、有限差分收敛检查、块函数恒等式（$I_1=0, I_2=G_1$）。

## 实验与结果
- **数据集与模型**：Qwen2.5-1.5B（主实验）、Llama-3.2-1B（第二家族复现）、Qwen2.5-7B（规模点）；6 个任务（math, code, instruction, safety, summarization, translation）；rank-16 LoRA 微调。
- **评估基线**：CTL（跨任务线性）、SurgeryV2 表示偏差、TSV 参数方向余弦、NeuroMerging、APL、RAIN-Merging。
- **关键结果**：
  - **运输主导**：Qwen2.5-1.5B 上运输份额 69%，晚期每块增益 1.08；Llama-3.2-1B 上 65%、增益 1.14。
  - **幅度 vs 因果对比**：code+safety 与 code+math 在 float32 局部生成幅度最多相差 1.9×，但因果可移除干扰差异达 **14×–337×**。
  - **消除特异性**：λ=1 正确方向消除在 code prompts 上平均 $D = +0.028$（移除约 19% 表达干扰）；top-1 一致性保持 0.94–0.99。
  - **表达门控**：Alpaca 包装下内部生成增大约 1.7–1.8×，但表达干扰下降 13–20×；相同消除在包装条件下仅移除 13× 更少干扰。
  - **回归重建**：在深度 3/8 消除后，传播重建跨项至原始范数的 99–100%，方向余弦 0.99；仅在深度 20 之后重建衰减（ρ=0.84）。
  - **幅度指标失效**：所有幅度度量（cumulative norm, parameter cosine, SurgeryV2 bias）在格式轴上方向错误或仅粗略相关；bf16 ±15% 均匀性实为 75–90% 量化噪声。
- **最强结果**：跨项方向消除在 Qwen2.5-7B 上完全关闭表达干扰（$D_{\text{wrapped}}$ 不显著），同时保持方向特异性（15 个 bootstrap CI 均排除零）。

## 相关工作脉络
1. **CTL（Zhou et al., 2024）**：观测层特征近似线性组合；本文的 $I_\ell$ 是其残差，但 CTL 无干预、无格式因子、无传播分析，本文证明小残差的**方向**而非大小承载干扰。
2. **SurgeryV2（Yang et al., 2024）**：表征手术修复表示偏差；本文证明偏差仅粗略排序任务对，对表达门控盲。
3. **TSV/DC-Merge（Gargiulo et al., 2025; Zhang et al., 2026）**：参数空间方向一致性；本文表明参数方向余弦与因果干扰相关弱（ρ=0.25），且格式不可见。
4. **NeuroMerging（Fang et al., 2025）**：神经元级分解；仍属幅度/分解框架，未触及激活方向因果性。
5. **APL（Kong et al., 2024）**：参数级因果干预；本文在激活级靶向跨项方向，证明参数干预不捕获同一机制。
6. **RAIN-Merging（Huang et al., 2026）**：行为级格式敏感性观测；本文提供机制级对应——包装器内部放大干扰却钉住输出门控。

## 局限性与未来方向
- **模型覆盖有限**：仅三个模型点（Qwen2.5-1.5B/7B, Llama-3.2-1B），完整微调及其他家族待扩展。
- **统计强度**：外层复单元仍为 3 seeds；每提示重跑验证了数值稳定性，但 bootstrap CI 限于 seed 内。
- **干预范围**：λ>1 状态离流形，仅有序对照被解释；自然中介主张未 claim。
- **行为终点 null**：离散评分为 null instrument；连续 NLL 通过有效性门控但因果消除未改善专家相似性——干预控制**表达非加性**而非行为质量。
- **门控泛化**：表达门控证据基于一族包装器与两个核心任务对；长度匹配非指令前缀对照组显示长前缀亦影响分布但门控结构不同。
- **方向预测器失效**：预注册的三个方向特征在 held-out 对上均失败；廉价方向预测器仍开放。

## 研究启发与可借鉴点
1. **因果干预协议范式**：四重范数匹配结构性对照 + 剂量阶梯 + 数值有效性门控，为激活级机制研究提供可复用框架。
2. **精度审计必要性**：bf16 生成均匀性被量化噪声掩盖，证明 activation-difference 分析必须配套 propagated-ULP floor 与收敛检查，否则结论可能被精度 artifact 驱动。
3. **格式敏感性机制化**：表达门控概念可迁移至其他格式/模板比较研究，提示评估合并模型时需分离内部放大与表达出口。
4. **运输 vs 生成分解**：$I_{\ell+1}=G_\ell+T_\ell+M_\ell$ 分解可推广至其他组合系统（如 MoE 路由、插件合并），识别传播动力学主导环节。
5. **预注册与反证报告**：46 项预测冻结、两项 headline 预期被 falsified 并报告，为可重复性研究树立标准。

## 关键术语表
- **Cross-term ($I_\ell$)**：合并模型与边际专家模型在层 ℓ 的残差状态四角差，表示累积的非加性残留。
- **Transport dominance**：跨项层通量中约 70% 来自既有跨项方向的运输而非新生成，晚期网络每块增益 >1。
- **Expression gating**：指令包装器放大内部跨项生成却钉住输出分布，使表达干扰下降 13–20×。
- **Magnitude fallacy**：假设幅度测量能定位或解释干扰的错误前提；本文证明幅度仅粗略相关且跨格式方向错误。
- **Causal erasure**：沿跨项自身方向精确消除（λ=1）可剂量依赖性移除干扰，而范数匹配对照失败或反效果。
- **Regrowth attractor**：跨项方向被正向传播主动维持，消除后传播重建至 99% 范数与 0.99 余弦方向。
- **FP32 audit**：将 bf16 估计重算为 float32 揭示 75–90% 值为权重舍入噪声，幅度均匀性实为量化 artifact。
- **Factorial ledger**：追踪四个权重配置下每一层的残差状态，构建精确的跨项账本与分解。

## 可复现要素
- **数据集**：PKU-SafeRLHF、CoT Math、Stanford Alpaca、Summarization、OPUS、Translation；任务定义与采样协议与 companion study [Zhu, 2026] 相同。
- **代码/权重**：代码、冻结日志、每单元格不可变记录、bootstrap CI 报告、所有复现 preregistration 与 verdict 文档已随代码发布。
- **关键超参**：LoRA rank=16；响应式损失；任务向量范数匹配至全局核心中位数；组合网格 $(\alpha,\beta)\in(0,1.2]^2$；边界深度 {3,8,14,20,25}；剂量阶梯 λ∈{0,0.5,1,1.5,2}；float32 计算块；TF32 关闭；确定性 attention kernels；单 A100 执行。
- **基线实现**：CTL linearity、SurgeryV2 bias、TSV parameter cosine、NeuroMerging、APL、RAIN-Merging 均在相同条件重新测量或引用。
