---
title: "TIDERL: Boosting Agentic RL Goodput with Readiness-Aware Scheduling"
source: https://arxiv.org/pdf/2608.10402v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 03:44:33"
---

# 论文速读：TIDERL: Boosting Agentic RL Goodput with Readiness-Aware Scheduling

## 一句话总结
针对多轮 Agent RL 中 rollout 任务长尾、上下文长度与交互时间高度可变导致的 GPU 利用率低下问题，TIDERL 提出了一套就绪感知（readiness-aware）弹性异步训练架构，通过 CTB、RA²P 与 ERS 三者协同，将 RL 训练 goodput（有效训练吞吐量）提升 1.8–7.0×（文本）和 33%+（多模态），同时保持与同步基线相当的模型性能。

## 研究问题与动机
- 多轮 Agentic RL 的工作负载具有极强的时空方差（交互轮数、上下文长度、环境网络延迟差异大），现有异步架构在应对此类负载时暴露出三大系统瓶颈。
- **KV Cache 频繁抢占**：传统推理引擎按 request 级 continuous batching 调度，多轮任务等待外部反馈时 KV cache 被 greedily evict，恢复时遭遇灾难性 cache miss，迫使昂贵且重复的 prefill 重计算。
- **Ref-Actor 执行冲突（Stall & Thrashing）**：Trainer 需在 Ref 与 Actor 之间共享 GPU 内存。等待全局大 batch 会导致 GPU 空转（最高达 81% 步时间），而流式处理 micro-batch 又引发频繁的模型 swap（PCIe 传输与 GPU context-switching）。
- **静态资源划分无法适应瓶颈漂移**：Agentic RL 是 producer-consumer 管道，瓶颈在 Rollout 与 Trainer 之间动态切换。固定 GPU 配比无法响应初始 ready backlog 或稳态到达间隔，结构性限制全局 goodput。
- 核心研究问题：如何构建统一的弹性训练架构，在多轮 Agent 执行下最大化以训练吞吐量为度量的 RL goodput？关键在于共享“就绪视图”并协同设计多轮内存管理、分布式执行管道与动态集群资源分配。

## 核心贡献（创新点）
1. **CTB（Continuous Task Batching）**：将 Rollout 调度粒度从 request 级提升至 task 级，基于语义优先级与 token 预算实现 admission、preemption 与 worker-affinity resumption。与现有 request-level 连续 batching 的本质区别在于以“任务生命周期”为单位管理内存，主动保留对 Trainer 有价值的历史上下文。
2. **RA²P（Resource-Aware Ref-Actor Pipelining）**：提出解耦流式与共址聚合两种 Ref-Actor 执行模式，由就绪信号（RAS/TPRM）动态选择。与仅支持单一静态执行模式的 Trainer 架构的本质区别在于将 Ref/Actor 的并行度与数据到达模式显式对齐，消除 stall 与 thrashing。
3. **ERS（Elastic Resource Scaling）**：基于同一就绪信号实时迁移 GPU rank，并利用 on-policy RL 权重同步边界实现零开销缓存刷新与角色切换。与传统需全局挂起、重建 NCCL 组或串行迁移权重的弹性框架的本质区别在于 rank 级弹性可无缝嵌入训练循环的固有同步窗口。
4. **端到端就绪感知协同闭环**：CTB 控制数据生产节奏，RA²P 暴露数据消费成本，ERS 根据统一信号调度 GPU 与执行模式，三者耦合形成反馈环，从根本上解决多轮 Agent RL 的 goodput 优化问题。

## 方法详解
- **CTB（Continuous Task Batching）**：每个 DP rank 维护 token budget。通过轻量 probing 估计任务初始上下文，每轮交互更新实际 token 占用。调度优先级分四层：① Evaluation tasks（最高，需锚定权重同步）；② GRPO Group completion（已完成比例高者优先）；③ Active execution tasks；④ Long context tasks（最低）。内存超限时按优先级 preempt，恢复时优先 resume 高优先级任务并维持 group affinity 以避免 prefix 重算。
- **RA²P（双模式 Ref-Actor 协作）**：
  - *Decoupled Streaming Mode*：Ref 与 Actor 分布在不同 GPU 上。重构计算图，将 Actor 的 loss 计算（含 KL penalty）推迟至 backward pass，实现 Ref forward → Actor forward → Actor backward 的重叠流水线，彻底消除 model swap 开销。
  - *Optimized Colocated Mode*：Ref 与 Actor 共址于同一 GPU。通过 ready-batch aggregation 批量加载 Ref 处理多个 micro-batch，再用 zero-copy shared memory 传输 ref_log_probs，摊销 swap 成本。
  - *模式选择*：根据 RAS（训练步开始时已就绪的 micro-batch 数量）和 TPRM（后续 micro-batch 的平均到达间隔）决定。高 RAS/短 TPRM 选用 Decoupled，低 RAS/长 TPRM 选用 Colocated。
- **ERS（Elastic Resource Scaling）**：
  - 监控全局 buffer 的 generation deficit（batch size < B_ideal）和 consumption deficit（HoL latency > τ×T_step）。
  - Trainer 饥饿时：切换 Colocated 模式，将 Ref GPU 迁移至 Rollout，加速数据生成以降低 TWT。
  - Rollout 数据堆积时：切换 Decoupled 模式，从 Rollout 迁移 GPU 至 Trainer 作为 Dedicated Ref，加速消费以降低 RWT。
  - *零开销弹性机制*：利用 on-policy 每步必须同步权重并 flush 历史 KV cache 的自然边界，rank 迁移时 CTB 自动重分配任务并在新 rank 上 fresh prefill；PCIe weight swap 与 sync_params broadcast 流水线重叠，无需暂停 pipeline 或重建通信组。

## 实验与结果
- **测试环境**：4 节点 × 8× NVIDIA H100 GPU（NVLink 互联），环境调度在 1024 CPU 核 Kubernetes 集群。
- **工作负载与模型**：文本类 WebShop、AlfWorld；多模态 OSWorld、ScienceBoard。模型：Qwen2.5-7B/14B、Qwen-3-VL 4B、Qwen-3.5 9B。
- **基线**：VeRL（同步）、AReaL（异步）、StreamRL（异步流式）。
- **主要结果**：
  - 文本任务：TIDERL 相对 VeRL 吞吐提升 5.6×，相对 AReaL 提升 1.8×，相对 StreamRL 提升 7×+；训练时间减少 51.1%，BoN reward 与 pass rate 仅差 0.01 / 0.5%。
  - 多模态任务：相对 VeRL 提升 6.02×，相对异步基线提升 33%+；训练时间减少 62.2%。
  - 组件消融（OSWorld + Qwen-3-VL）：RA²P 贡献 10.4%，ERS 贡献 35.7%，CTB 贡献 4.8%。
  - CTB 使 KV cache hit rate 提升 1.58×，生成吞吐提升 1.15×。
  - RA²P 在不同就绪模式下将单步训练时间最多减少 44.3%。
  - ERS 将总等待时间（RWT+TWT）最多减少 77.6%。

## 相关工作脉络
1. **VeRL / RLHFuse / SLIME / OpenRLHF**：主流同步或静态拆分异步框架，依赖固定相位屏障或固定 GPU 配比，无法响应多轮负载的动态瓶颈漂移。
2. **AReaL / APRIL**：异步 RL 基础架构，引入 active partial rollout 管理策略陈旧性，但仍停留在 request 级调度与静态资源划分。
3. **StreamRL / AsyncFlow**：支持流式 micro-batch 传输，但在高并发 burst 下触发频繁 model swapping thrashing，且缺乏 readiness-aware 的资源弹性调度。
4. **Seer / SGLang HiCache**：面向单轮/静态请求的 KV cache 优化与连续 batching，未考虑多轮 Agent 任务的生命周期语义与算法级约束（如 GRPO group 完整性、eval 锚定版本）。
5. **DistTrain / Pollux / Elasticflow**：通用弹性训练框架，但 scaling 操作通常需全局挂起、重建通信组或迁移大权重，不适用于步级瓶颈频繁切换的 agentic RL。
6. **Continuum**：面向 Agent serving 的 session-aware 调度，虽关注多轮 cache pinning，但未解决 RL 训练侧 Ref/Actor 协作
