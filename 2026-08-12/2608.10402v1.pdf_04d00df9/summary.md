---
title: "TIDERL: Boosting Agentic RL Goodput with Readiness-Aware Scheduling"
source: https://arxiv.org/pdf/2608.10402v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 03:44:08"
field: "分布式强化学习系统"
keywords: ["Agentic RL", "KV Cache Scheduling", "Asynchronous Training", "Goodput Optimization", "Elastic Resource Scaling", "Multi-turn Inference"]
innovations: ["CTB 将调度粒度从 request 提升到 task 级以保留多轮 KV cache", "RA²P 基于 RAS/TPRM 就绪信号选择解耦流式或共置聚合执行模式", "ERS 复用 on-policy sync 边界实现零开销 rank 弹性迁移"]
benchmarks: ["WebShop", "AlfWorld", "OS-World", "ScienceBoard"]
---

# 论文速读：TIDERL: Boosting Agentic RL Goodput with Readiness-Aware Scheduling

## 一句话总结
论文针对多轮 agentic RL 工作负载中因任务交互延迟导致 GPU 利用率低的问题，提出了 TIDERL 系统，通过连续任务批处理（CTB）、就绪感知的 Ref-Actor 流水线（RA²P）和弹性资源扩展（ERS）三个协同组件，最大化训练 goodput，在文本和多模态任务上分别实现最高 5.6× 和 33% 的吞吐提升。

## 研究问题与动机
- **多轮交互导致 KV cache 灾难性丢失**：现有推理引擎基于 request 级连续批处理，多轮任务在等待外部反馈时会被预取，恢复时需重新 prefill 不断增长的上下文，导致命中率降至约 12%。
- **Trainer 侧 stall 与模型交换 thrashing 困境**：异步系统要么等待完整全局 batch 导致 GPU 空闲高达 81%，要么流式处理 micro-batch 但被迫频繁交换 Ref/Actor 模型，造成严重 PCIe 传输开销。
- **静态资源划分无法适应动态瓶颈切换**：agentic 任务复杂度差异大，瓶颈在 rollout 和 training 间频繁切换，固定 GPU 分配比例（如 0.25）无法同时优化两种就绪模式。
- **目标函数应为 goodput 而非 GPU 占用率**：多轮 agentic RL 中，GPU 等待和重复 prefill 是纯开销，应以 training throughput（进入 Trainer 并推进更新的 token 数）为核心指标。

## 核心贡献（创新点）
- **CTB 将调度粒度从 request 提升到 task 级别**：通过感知式优先级分层（评估任务 > GRPO 组完成度 > 活跃任务 > 上下文长度）实现任务级 admit/preempt/resume，保留有用 rollout 状态，使 KV cache 命中率提升 1.58×。
- **RA²P 提供两种就绪感知的 Ref-Actor 执行模式**：基于 RAS（启动时 ready backlog）和 TPRM（micro-batch 到达间隔）选择解耦流式模式或共置聚合模式，分别消除 model swap 开销或将其摊薄，使每步训练时间减少最高 44.3%。
- **ERS 实现零开销弹性 rank 迁移**：复用 on-policy RL 在权重同步时的天然 cache flush 边界，将 weight migration 与 sync_params broadcast 重叠，无需暂停 pipeline 或重建 NCCL 组，总等待时间减少最高 77.6%。

## 方法详解
**1. Continuous Task Batching (CTB)**
- 将每个 DP rank 视为 token 预算池，基于轻量 probe 估算任务初始 footprint，实时更新实际 token 使用量。
- 语义感知优先级分层（Table 1）：
  - Priority(t) = ω₁·I_eval + ω₂·G_completion + ω₃·I_active + ω₄·L_context
  - Eval 任务最高优先（需锚定权重进行全局同步）；GRPO 组完成度高者优先（避免 advantage 计算停滞）；活跃任务次之；长上下文最低。
- 恢复时强制 group affinity，在同一 rank 上 resume，保留共享 prefix 的 KV cache。

**2. RA²P (Resource-Aware Ref-Actor Pipelining)**
- **信号定义**：RAS = 训练步开始时已 ready 的 micro-batch 数量；TPRM = 后续 micro-batch 到达间隔。
- **解耦流式模式**（高 RAS 或短 TPRM）：Ref 和 Actor 分配到独立 GPU，重构计算图将 loss 计算从 Actor forward 移至 backward，实现 Ref forward (F₀, F₁, F₂) 与 Actor forward/backward 重叠，零 swap 开销。
- **共置聚合模式**（低 RAS 且长 TPRM）：Ref 和 Actor 共置同一 GPU，通过 ready-batch aggregation 一次加载 Ref 处理多个 micro-batch，再用 zero-copy shared memory 传递 ref_log_probs，摊薄 model swap 成本。
- **Micro-batch dispatching**：前期 micro-batch 累积至饱和所有 DP rank 计算能力；末期 micro-batch 刻意保持最小以缩短尾延迟。Colocated 模式用 zig-zag 负载均衡，Decoupled 模式使最后一个序列最短。

**3. Elastic Resource Scaling (ERS)**
- **就绪感知计划生成**：在 Trainer 完成 Actor update 后、新权重广播前立即生成扩展计划。
- 检测 generation deficit（全局 batch < B_ideal）→ 选择 colocated RA²P 并将 GPU 从 Ref 移至 Rollout，减少 TWT。
- 检测 consumption deficit（HoL latency > τ·T_step 或 RAS 超过 break-even）→ 选择 decoupled RA²P 并添加 Ref rank，减少 RWT。
- **Cache-free task migration**：利用 on-policy RL 在 post-update 时 invalidate KV cache 的天然时机，CTB 重新分配任务，新 rank 用同步后的权重 fresh prefill，零额外开销。
- **Latency-hiding role switching**：将 PCIe weight swap 与 sync_params broadcast 重叠：scale-down Ref rank 时，等 Actor backward 完成后 offload Ref 到 CPU，新 Rollout rank 加入 broadcast 加载 Actor；反之亦然。

## 实验与结果
- **测试环境**：4 节点集群，每节点 8× NVIDIA H100 GPU，NVLink 互联，共享 JuiceFS NFS。
- **工作负载**：文本类（WebShop, AlfWorld）；多模态类（OS-World, ScienceBoard）。
- **模型**：Qwen2.5-7B/14B（文本）、Qwen-3-VL 4B / Qwen-3.5 9B（多模态）。
- **基线**：VeRL（同步）、AReaL（异步）、StreamRL（SOTA 异步）。
- **主要结果**：
  - 文本任务：TIDERL 相比 VeRL 提升 5.6× 吞吐，相比 AReaL 提升 1.8×，相比 StreamRL 提升 7×+；BoN reward 差距仅 0.01，pass rate 差距 0.5%，收敛时间减少 51.1%。
  - 多模态任务：相比 VeRL 提升 6.02×，相比异步基线提升 33%+，训练时间减少 62.2%。
  - 组件消融（OSWorld + Qwen-3-VL 4B）：CTB 提升 4.8%，RA²P 提升 10.4%，ERS 提升 35.7%，累计 54.8%（首步）→ 最终 33.3k token/s。
  - CTB 使 KV cache 命中率从 0.9% 提升至 >6%（1.58×），生成吞吐提升 1.15×。
  - RA²P 在 high-RAS 场景下减少 per-step 训练时间最高 44.3%。
  - ERS 使总等待时间（RWT+TWT）减少 68.6%–77.6%。

## 相关工作脉络
- **VeRL [30]**：同步 RL 框架，强制 rollout-trainer barrier，作为主要同步基线；TIDERL 通过异步解耦打破其 long-tail stall。
- **AReaL [5]**：基础异步框架，使用 active partial rollout 管理 staleness；但 GPU 分配固定，无法联合决策 ref-actor 模式和 rollout/trainer 划分。
- **StreamRL [49]**：SOTA 异步框架，支持流式 micro-batch 到 Trainer；但在多 micro-batch 场景下触发频繁 model swap，TIDERL 通过 RA²P 消除该 thrashing。
- **Seer [23]**：同步框架 + suffix decoding 优化；论文表明其在多轮 agentic 任务中 SD 因低 acceptance rate (~40%) 而得不偿失。
- **Continuum [15]**：针对 agent serving 的 KV cache pinning 和会话感知调度；CTB 进一步纳入 GRPO 组和评估任务的算法依赖。
- **Laminar [29]**：异步 RL 后训练框架；TIDERL 的定位是提供运行时 readiness-aware 调度，而非静态异步架构。

## 局限性与未来方向
- 评估任务需锚定静态权重版本，当前强制同步会短暂打断异步 pipeline 节奏；未来可集成 relay weight sync（如 Laminar）实现多版本共存。
- 当前 ERS 的 scale-up 操作受限于每步最多迁移 1 个 rank，在多模态任务中可能导致 Trainer 成为长期瓶颈。
- CTB 与 hierarchical caching（如 SGLang HiCache）尚未整合，因 CPU RAM 已被 global buffer 和 offloaded Trainer 模型占满。
- Suffix Decoding 在当前模型规模下不适用，但未来更高效的 SD 机制可考虑集成。
- 论文未讨论 fault tolerance 的详细恢复策略，仅提及基于异步分布式 checkpointing 的基本容错。

## 研究启发与可借鉴点
- **就绪信号驱动的双模式选择范式**：用 RAS/TPRM 两个简单信号指导系统级决策（模式选择 + 资源分配），避免复杂 online optimization，可迁移到其他 producer-consumer pipeline 系统。
- **零开销弹性设计思路**：复用算法固有边界（on-policy sync 时的 cache flush）实现 rank 迁移，隐藏 PCIe swap 于 broadcast 窗口，为分布式系统弹性扩展提供新设计维度。
- **计算图重构解耦 dependency**：将 loss computation 从 forward 移至 backward，打破 Ref-Actor 串行依赖，实现零 swap streaming，可推广至其他需双模型计算的场景（如 DPO、KTO）。
- **Task-level 调度 vs Request-level 调度**：多轮交互任务中，将调度单元从 request 提升到 task 并追踪 context growth，对 agent serving、多轮对话系统等有参考价值。
- **端到端 goodput 指标设计**：将 throughput 定义为 "进入 Trainer 并推进更新的 token 数"，排除 stale/discarded 数据，为 RL 系统评测提供明确基准。

## 关键术语表
- **Goodput**：有效吞吐量，指进入 Trainer 并实际推进权重更新的 token 速率，排除等待和重复计算开销。
- **Agentic RL**：面向 autonomous agent 的强化学习，模型需多轮交互外部环境（浏览器、OS、GUI）生成轨迹。
- **GRPO**：Group Relative Policy Optimization，基于 group 内 rewards 标准化计算 advantage 的 on-policy 算法。
- **RAS (Ready-at-Start)**：训练步开始时已 ready 的 micro-batch 数量，反映初始数据堆积程度。
- **TPRM (Time Per Ready Micro-batch)**：训练步开始后 micro-batch 的平均到达间隔，反映数据供给速率。
- **KV cache preemption**：推理引擎为服务新请求而驱逐已缓存的历史 token 的 KV 存储，导致多轮任务恢复时 catastrophic cache miss。
- **Decoupled streaming mode**：RA²P 的一种执行模式，Ref 和 Actor 分置独立 GPU，通过计算图重构实现 streaming overlap。
- **Colocated aggregation mode**：RA²P 的另一种执行模式，Ref 和 Actor 共置同一 GPU，通过 ready-batch aggregation 摊薄 model swap 开销。

## 可复现要素
- **代码**：论文未提及开源仓库链接（摘要及正文无 GitHub 链接）。
- **数据集/工作负载**：WebShop [42]、AlfWorld [31]、OS-World [39]、ScienceBoard [33]，均为公开 benchmark。
- **模型权重**：使用 Qwen2.5-7B/14B、Qwen-3-VL 4B、Qwen-3.5 9B，均为开源模型。
- **后端框架**：vLLM（rollout）、Megatron Core / PyTorch FSDP（training）。
- **集群配置**：4 节点 × 8× NVIDIA H100 GPU，NVLink 互联，JuiceFS NFS 共享存储，1024 核 CPU 环境集群。
- **关键超参**：论文未详细列出学习率、batch size、staleness bound 等训练超参（仅在 Appendix 提及实现细节）。
