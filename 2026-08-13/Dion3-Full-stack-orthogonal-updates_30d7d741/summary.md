---
title: "Dion3-Full-stack-orthogonal-updates"
source: https://arxiv.org/pdf/2608.11612v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:02:06"
---

# 论文速读：Dion3: Full-stack orthogonal updates

## 一句话总结
Dion3 是对 Muon 优化器的全栈改进方案，通过 Gram Newton-Schulz 代数重构、CuteDSL 对称矩阵内核、行子采样更新规则与 Megabatch 通信打包，系统性消除了正交化步骤的计算与分布式通信开销，使 Muon 在大规模 LLM 训练中的单步耗时降低最高达 6 倍，同时保持或提升了模型质量。

## 研究问题与动机
- Muon 优化器虽能以更少的训练步数收敛，但其核心的 Newton-Schulz 正交化步骤具有 $O(n^3)$ 的立方复杂度，随模型规模扩大开销急剧增长。
- 在分布式并行（如 FSDP）场景下，权重按分片存储，Muon 必须 Gather 完整矩阵才能计算极分解，导致频繁的 all-to-all 通信与带宽碎片化。
- 现有改进工作（NorMuon、Dion、Trion 等）多聚焦于更新规则的微调或低秩近似，但未改变 Newton-Schulz 依赖大量高成本矩形矩阵乘法的代数结构。
- Muon 已有成功缩放案例（如 Kimi K2）高度依赖特定架构（细粒度 MoE）与并行策略的巧合配合，缺乏作为通用优化器的灵活性，难以直接迁移至密集或混合架构。

## 核心贡献（创新点）
1. **Gram Newton-Schulz 算法**：将迭代变量从大型输入矩阵 $X$ 转移到小型对称 Gram 矩阵 $XX^\top$，将计算复杂度从 $O(T\alpha n^3)$ 降至 $O((T+\alpha)n^3)$，彻底减少昂贵矩形乘法的比例。
2. **CuteDSL 对称 GEMM 内核**：针对 Hopper/Blackwell 架构定制三角调度与转置写回机制，使对称矩阵乘法的浮点运算量减半，并与 Gram 变换深度协同放大加速比。
3. **Dion3 行子采样更新规则**：每步仅按 $\ell_1$ 范数选取动量矩阵的 $f$ 比例行进行正交化与参数更新，结合误差反馈（Error Feedback）机制，以更简单的实现替代 Dion 的幂迭代低秩近似。
4. **Megabatch 通信策略**：将同形状权重分片打包为单次 all-to-all 集合通信，把优化器步内的通信轮数从 $O(N/\text{world\_size})$ 降至 $O(1)$，显著缓解小消息带宽利用率低的问题。

## 方法详解
- **Gram Newton-Schulz 代数重构**：基于牛顿-舒尔茨多项式 $p_t(x) = x h_t(x^2)$ 的奇函数性质，将传统迭代 $X_{t+1} = a_t X_t + b_t X_t X_t^\top X_t + c_t (X_t X_t^\top)^2 X_t$ 等价转化为对 $R_t$ 与 $Q_t$ 的同步递推。算法仅需在首尾各做一次 $n \times m$ 的矩形乘法（初始化 $R_0 = XX^\top$、输出 $Q_T X$），中间 5 轮迭代全部在 $n \times n$ 对称矩阵上进行，理论 FLOPs 从 $T(3\alpha+1)n^3$ 降至 $(4T+3\alpha-3)n^3$。
- **半精度稳定性与重启策略**：bfloat16/float16 下 $XX^\top$ 会因舍入误差引入虚假负特征值，导致逆平方根迭代发散。作者在迭代第 2 次后执行“重启”（recompute $R_2 = X_2 X_2^\top$, reset $Q_2 = I$），切断误差累积；同时将常数项 $a_t I$ 隐式融合进后续乘法而非显式加法，并默认改用 float16 配合 1.05 安全因子，在零额外调参前提下保证训练稳定。
- **子采样更新规则**：输入分数 $f \in (0, 1]$，选取 $k = \lceil fn \rceil$ 个 $\ell_1$ 范数最大的行；对这 $k \times m$ 子矩阵执行 Gram Newton-Schulz 正交化；仅更新匹配的行权重，并对选中行的动量施加 $\mu$ 衰减（未选中行保持原动量不变，即误差反馈）。误差反馈项等效为 $(1-\mu)(M - \widehat{M})$ 的增量，使被跳过的行梯度在后续步中累积并最终被选中。
- **学习率转移规则**：由于每步仅更新 $f$ 比例的行，有效步长缩小，理论推导给出缩放关系 $\eta' = \eta / \sqrt{f}$，实验中验证最优 $\eta \sqrt{f} \approx 0.01$。
- **Megabatch 通信与系统实现**：在 FSDP2/DDP 后端，按权重矩阵形状分组，将各 rank 的对应分片合并为单个大 all-to-all 请求；归约结果直接进行批处理正交化后 scatter 回各设备。结合 CUDA Graph 捕获与自定义 Triton 更新内核，消除 Python/CPU 调度开销与频繁 upcasting/downcasting 导致的数值偏差。

## 实验与结果
- **数据集与基准**：预训练使用 ClimbMix（100B tokens 用于 1B 小规模消融，10B tokens 用于 3B-14B 主实验）；下游评估涵盖 ARC、MMLU、BoolQ、COPA、HellaSwag、LAMBADA、OpenBookQA、PIQA、RTE、TruthfulQA、WinoGrande 共 12 项标准基准。开源架构微测使用 FineWeb-Edu。
- **模型质量保持与提升**：1B 模型上 Dion3（$f=1/8$）最终验证损失低于调优后的 NorMuon 约 0.01；3B-14B 尺度上 Dion3（$f=1/4$）在所有模型尺寸均取得更低 CE（14B 降低 -0.027），下游宏观准确率提升最高 +0.7pp，且训练全程损失曲线稳定低于基线。
- **优化器单步加速**：在 7B 模型 4×GH200 设置中，Dion3 将 Muon 优化器单步耗时从 26×AdamW 降至 4×AdamW；组合全部贡献后，对 14B 模型可达 6.5 倍相对加速。Gram NS + 对称内核本身贡献 1.5-2× 加速，在高 aspect ratio（$\alpha=8$，如 Gemma MLP）架构上增益更为显著。
- **通信与系统开销**：Megabatching 在 1B/8-GPU 通信受限场景下减少 35% 步时间；CPU 端延迟经 CUDA Graph 后降至约 0.1ms，通信量与 $f$ 呈严格线性关系，验证了压缩更新在分布式环境中的通信削减效果
