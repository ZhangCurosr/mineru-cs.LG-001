---
title: "REOPD-Reliability-Adaptive-Reward-Extrapolation-for-On-Polic"
source: https://arxiv.org/pdf/2608.11698v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:35:12"
field: "大语言模型后训练与知识蒸馏"
keywords: ["on-policy distillation", "reward extrapolation", "adaptive supervision", "teacher-student alignment", "token-level weighting", "micro-batch budget"]
innovations: ["Token-level compatibility weight 动态调制 beyond-teacher residual 的可靠程度", "Micro-batch bounded extrapolation budget 通过 ρ/s 比值自适应控制全局外推强度", "无需额外 verifier 或 rollout，完全复用 white-box OPD 已有 log-probs"]
benchmarks: ["AIME 2024/2025", "HMMT February/November 2025", "HumanEval+", "MBPP+", "LiveCodeBench v6 test6"]
---

# 论文速读：REOPD-Reliability-Adaptive-Reward-Extrapolation-for-On-Polic

## 一句话总结
REOPD 提出了一种可靠性自适应的奖励外推机制，用于改进 On-Policy Distillation (OPD) 中的 teacher–reference 残差放大问题；通过 token-level 兼容权重与 micro-batch-level 自适应预算的结合，实现在不依赖额外 verifier 或 rollout 的前提下，动态调节外推强度以提升数学与多教师蒸馏的表现。

## 研究问题与动机
- **现有 ExOPD 方法**：在 OPD 框架下将 teacher–reference log-ratio 视为隐式 reward，并使用全局标量 λ 对所有 token 施加相同的放大倍数。
- **全局 λ 的缺陷**：
  - 容易导致 student 过度拟合 teacher–reference log-ratio 中的极端峰值（reward hacking），引发训练不稳定。
  - 最优 λ 因领域不同而差异较大，需要昂贵的 domain-specific 搜索，且仍可能失效。
- **Token 层面异质性**：并非所有 token 都适合同等强度的外推；某些 token 的 student–teacher 差异较大，直接放大残差可能破坏对齐信号。
- **缺乏细粒度控制**：现有方法未考虑 local student–teacher compatibility 对 residual signal 可靠性的影响。

## 核心贡献（创新点）
1. **提出 token-level 兼容权重 q_{b,i,t}**：基于 student–teacher log-prob 差异构造低方差 discrepancy proxy，动态调制每个 token 的外推强度。
2. **设计 micro-batch-level 自适应预算 γ_b**：通过聚合 batch 内 compatible residual 的比例 ρ_b 与尺度 s_b，生成 bounded 的全局外推预算。
3. **构建联合系数 λ_{b,i,t} = 1 + γ_b q_{b,i,t}**：将 token 级权重与 batch 级预算相乘，实现对 beyond-teacher residual 的细粒度控制，同时保留原始 teacher-alignment 项不变。
4. **无需额外模块**：完全复用 G-OPD 已有的 student/teacher/reference log-prob，不引入 verifier、reward model 或 value model。
5. **实验验证广泛适用性**：在单教师数学、代码及多教师混合场景下，REOPD 优于固定 λ=1.25 的 ExOPD，并在数学任务上超越 G-OPD。

## 方法详解
**1. 问题设定与基础公式**
- Student policy π_θ 生成 on-policy 响应 y，prefix state s_t = (x, y_{<t})
- Teacher alignment cost: a_t = log π_θ(y_t|s_t) - log π_T(y_t|s_t) （对应 reverse KL）
- Teacher–reference implicit reward: r_t = log π_T(y_t|s_t) - log π_ref(y_t|s_t)
- ExOPD cost: C_t^{ExOPD} = a_t - (λ - 1)r_t，其中 λ=1 退化为标准 OPD

**2. Token-Level Compatibility Weight**
- 构造低方差 discrepancy proxy：
  - x_{b,i,t} = log π_T(y_{i,t}|s_{i,t}) - log π_θ(y_{i,t}|s_{i,t})
  - δ̂_{b,i,t} = exp(x_{b,i,t}) - x_{b,i,t} - 1 ≥ 0
- 兼容权重：q_{b,i,t} = exp(-δ̂_{b,i,t}/τ)，τ > 0 为温度参数
- q ∈ (0,1]，小差异 → q≈1（保留完整残差），大差异 → q↓（衰减残差）
- 注意：q 测量的是 local compatibility，而非 task correctness；与计算图 detached

**3. Micro-Batch Reliable Residual Statistics**
- Compatibility-weighted residual proportion:
  - ρ_b = Σ_b m_{i,t}|r_{b,i,t}|q_{b,i,t} / (Σ_b m_{i,t}|r_{b,i,t}| + ε) ∈ [0,1]
- Reliable residual scale (RMS):
  - s_b = (Σ_b m_{i,t}(q_{b,i,t}r_{b,i,t})² / (Σ_b m_{i,t} + ε))^{1/2}
- 指数移动平均平滑：z̄_b = β z̄_{b-1} + (1-β)z_b，z∈{ρ,s}
- 所有统计量 across data-parallel ranks all-reduce 后同步

**4. Bounded Micro-Batch Extrapolation Budget**
- 目标预算：γ̃_b = clip(B₀ z̄_ρ_b / (z̄_s_b + ε), 0, γ_max)
  - ρ_b 大 → 允许更强外推；s_b 大 → 降低系数防主导更新
- 平滑后预算：γ_b = β_γ γ_{b-1} + (1-β_γ)γ̃_b
- 自动校准模式：前 K₀ 步用 alignment RMS 的缩放 EMA 初始化 B₀
- 可选 warm-up：初始步保持 γ_b 为预定值

**5. 最终目标与优化**
- Effective coefficient: λ_{b,i,t} = 1 + γ_b q_{b,i,t}，满足 1 ≤ λ_{b,i,t} ≤ 1 + γ_max
- Token cost: C_{b,i,t}^{REOPD} = a_{b,i,t} - γ_b q_{b,i,t} r_{b,i,t}
- PPO advantage: A^{REOPD} = -C^{REOPD}， masked 到 valid response tokens
- 所有控制信号（q, 统计量, EMA, γ_b）均 stop-gradient

## 实验与结果
**数据集与模型**
- 基础模型：Qwen3-4B（student & reference）
- Teacher：RL-trained Qwen3-4B non-thinking policies
  - 数学：57,046 examples from filtered DeepMath-103K
  - 代码：25,276 examples from Eurus code split
  - 多教师：各 25,276 examples，按 domain label 路由
- 评估基准：
  - 数学：AIME 2024/2025, HMMT Feb/Nov 2025（各 30 题，32 responses）
  - 代码：HumanEval+, MBPP+, LiveCodeBench v6 test6（pass@1）

**训练设置**
- AdamW, lr=10⁻⁵, weight decay=0.01, gradient clip=1.0, 1 PPO epoch
- 每 step 1024 prompts，max prompt=2048 tokens, max response=16384 tokens
- BF16 mixed precision, FSDP on 8×A100 80GB
- REOPD 超参：τ=0.007, γ_max=1, κ=0.5, β=0.95, β_γ=0.9，auto-calibrate B₀

**主要结果（Table 1）**

| Setting | Method | Math Avg (%) | Code Avg (%) |
|---------|--------|--------------|--------------|
| Single-teacher | OPD | 46.28 | 62.55 |
| | ExOPD (λ=1.25) | 47.47 | 61.72 |
| | **REOPD** | **47.66** | **63.45** |
| Multi-teacher | OPD | 46.43 | 61.99 |
| | ExOPD (λ=1.25) | 46.98 | 62.90 |
| | **REOPD** | **47.01** | **63.32** |

- **数学单教师**：REOPD 47.66% > ExOPD 47.47%（+0.19pp）> OPD 46.28%
- **代码单教师**：REOPD 63.45% > ExOPD 61.72%（+1.73pp）> OPD 62.55%
- **多教师数学**：REOPD 47.01% > ExOPD 46.98%（+0.03pp）> OPD 46.43%
- **多教师代码**：REOPD 63.32% > ExOPD 62.90%（+0.42pp）> OPD 61.99%

**与最优固定 λ 对比（Figure 2 + 文字）**
- 单教师数学最优 λ=1.25 → REOPD +0.19pp
- 单教师代码最优 λ=1.5 → REOPD -0.15pp（几乎持平）
- 多教师数学最优 λ=1.25/1.75 → REOPD +0.03pp
- 多教师代码最优 λ=1.25 → REOPD +0.42pp
- REOPD 在 4/4 设置中追平或超越 task-specific best fixed λ

**消融实验（Table 2，单教师数学）**
- Full REOPD：47.66%
- no_q（移除 token 兼容权重）：43.39%（↓4.27pp）→ **token-level 适配是核心**
- no_bound（移除 γ_b 上界）：47.16%（↓0.50pp）→ 显式 bound 提供 modest safeguard
- no_batch + fixed λ₀=1.25：47.66%（与 Full 持平）→ 自适应 batch 预算可避免手动调参

**Controller Dynamics（Figure 3）**
- 早期：γ 均值 ~0.60–0.96，q 均值 ~0.75–0.77，effective λ ~1.48–1.72
- 晚期：γ → 1.0（趋近 γ_max），q → 0.83–0.85，effective λ → 1.82–1.85
- 说明适应主要发生在训练前期，后期以 token-level 兼容性驱动差异

## 相关工作脉络
1. **G-OPD (Yang et al., 2026a)**：将 OPD 建模为 dense KL-regularized RL，teacher–reference log-ratio 作为隐式 reward；REOPD 在此基础上引入细粒度外推控制。
2. **ExOPD (Yang et al., 2026a)**：固定 λ>1 全局放大 residual；REOPD 指出其易受峰值影响且需 domain-specific 调参。
3. **ExPO (Zheng et al., 2025)**：在 weight space 进行 model extrapolation；REOPD 属于 objective-level 方法，作用于 token advantage。
4. **TIP (Xu et al., 2026b)**：选择 informative alignment tokens；REOPD 保留全部 token 的对齐项，仅调制 residual。
5. **Prune-OPD (Yang et al., 2026b)**：根据 local support drift 衰减 teacher 监督；REOPD 不涉及 rollout 截断。
6. **SCOPE/SG-OPD/Reward-gated OPD**：依赖 verifier 或 outcome labels；REOPD 完全 white-box，无需额外反馈信号。

## 局限性与未来方向
**自述局限性**
1. **兼容性权重非正确性估计**：q 仅反映 student–teacher 差异，无法识别双方均错但一致的轨迹。
2. **超参数仍需选择**：τ, γ_max, B₀ 校准等虽跨 setting 共享，但未完全消除调参负担。
3. **统计量依赖 micro-batch 组成**：batch 采样噪声可能影响控制信号稳定性。
4. **预算后期趋近上界**：训练后期 γ_b → γ_max，adaptivity 主要来自 token-level，未能持续调节全局强度。

**未来方向**
- 评估更多 model families、 scales 及 teacher configurations
- 比较 shared controller vs per-teacher controller 的泛化性
- 探索结合 outcome verifiers 的 hybrid 方案
- 研究更鲁棒的 discrepancy proxy 或引入 correctness estimation

## 研究启发与可借鉴点
1. **残差分离思想**：将 teacher-alignment 与 beyond-teacher residual 分开处理，只对后者施加适应性调制——这一分解可作为通用范式推广到其他 distillation 设定。
2. **兼容权重构造**：使用 exp(δ̂) - δ̂ - 1 作为 low-variance discrepancy proxy 是 elegant 的技巧，可迁移至其他需要 token-level weighting 的场景（如 adaptive masking, importance sampling）。
3. **Bounded adaptive budget**：通过 ρ/s 比值生成有界预算并平滑，避免了直接预测绝对值的困难；此设计可参考于 RLHF 中的 dynamic reward scaling 或 KL control。
4. **无需额外模块的 online adaptation**：完全复用已有 log-probs，不增加推理/训练开销——这对工业部署极具吸引力，可启发低成本 adaptive 方法设计。
5. **Controller dynamics 分析**：论文详细追踪了 γ, q, effective λ 的训练轨迹，这种可视化分析可作为方法诊断的标准实践。

## 关键术语表
- **On-Policy Distillation (OPD)**：学生在自己生成的轨迹上学习，teacher 在同一 prefix 上提供 token-level 监督的蒸馏范式。
- **Teacher–Reference Log-Ratio**：log π_T(y_t|s_t) - log π_ref(y_t|s_t)，作为 implicit token-level reward 驱动 beyond-teacher 外推。
- **Reward Extrapolation**：通过 λ>1 放大 teacher–reference residual，使 student 超越 teacher 直接模仿，探索更优策略。
- **Token-Level Compatibility Weight (q)**：基于 student–teacher log-prob 差异构造的权重，调制残差的可靠程度，值域 (0,1]。
- **Micro-Batch Extrapolation Budget (γ_b)**：由 batch 内 compatible residual 比例与尺度动态生成的全局外推上限，bounded 在 [0, γ_max]。
- **Effective Coefficient (λ_{b,i,t})**：λ = 1 + γ_b q_{b,i,t}，token-wise 的外推系数，保证 1 ≤ λ ≤ 1 + γ_max。
- **Discrepancy Proxy (δ̂)**：exp(x) - x - 1，其中 x 为 teacher–student log-prob 差，作为 low-variance 兼容估计器。
- **Stop-Gradient Control**：所有适应性信号（q, γ_b, 统计量）不参与梯度回传，仅作为 advantage 构造的控制参数。

## 可复现要素
- **数据集**：
  - DeepMath-103K（filtered level-6，57,046 examples）：需确认公开状态
  - Eurus code training split（25,276 examples）
  - AIME 2024/2025, HMMT Feb/Nov 2025, HumanEval+, MBPP+, LiveCodeBench v6 test6
- **代码/权重**：论文未明确声明开源，需查看 arXiv 附注或作者主页
- **关键超参**：
  - τ = 0.007, γ_max = 1, κ = 0.5（auto-calibrate B₀）
  - β = 0.95（统计量 EMA）, β_γ = 0.9（预算 EMA）
  - B₀ 前 K₀=10 步 auto-calibrate；数学 warm-up 5 步 γ=0.25
- **训练配置**：lr=10⁻⁵, weight decay=0.01, grad clip=1.0, 1 PPO epoch, BF16, FSDP 8×A100 80GB
