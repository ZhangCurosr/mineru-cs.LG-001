---
title: "TIDERL: Boosting Agentic RL Goodput with Readiness-Aware Scheduling"
source: https://arxiv.org/pdf/2608.10402v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 03:43:25"
field: "分布式 RL 系统"
keywords: ["Agentic RL", "分布式训练系统", "弹性调度", "异步流水线", "KV Cache 管理", "RL goodput", "多模态 agent"]
innovations: ["任务级 CTB 调度替代请求级连续 batching 以保护多轮 KV cache", "RA²P 双模式 Ref-Actor 流水线根据就绪信号动态选择解耦/共置执行", "ERS 利用权重同步窗口实现零开销 rank 弹性迁移"]
benchmarks: ["WebShop", "AlfWorld", "OSWorld", "ScienceBoard"]
---

# 论文速读：TIDERL: Boosting Agentic RL Goodput with Readiness-Aware Scheduling

## 一句话总结
TIDERL 面向多轮 agentic RL 工作负载，提出一种就绪感知的弹性异步训练系统，通过任务级 CTB 调度、双模式 Ref-Actor 流水线（RA²P）和弹性资源重分配（ERS）三者的协同设计，将 RL 训练 goodput（训练吞吐）提升最多 5.6×（文本任务）/ 33%（多模态），同时达到与同步基线相近的任务性能。

## 研究问题与动机
- **现有异步架构处理多轮 agentic 工作负载时存在三类系统性低效**：异步系统虽解耦了 rollout 与训练，但多轮交互的上下文长度高度可变、任务完成时间差异极大，导致 GPU 空闲和重复计算成为主要 overhead。
- **C1（KV Cache 灾难性抢占）**：标准推理引擎按请求级连续 batching 调度，多轮任务在等待环境反馈时 KV cache 被驱逐，恢复时触发昂贵的 prefix 重计算（prefill），命中率高幅下降至 ~12%，吞吐量降至 ~650 tokens/s·instance。
- **C2（Ref 推理中的停滞与交换抖动）**：Trainer 内 Actor 与 Ref 模型共享 GPU 资源且存在严格阶段依赖；为避免显存溢出，系统要么等待完整 global batch（导致 GPU 空闲高达 step 时间的 81%），要么逐 micro-batch 交替加载两个模型（引发严重的 PCIe 传输与上下文切换开销）。
- **C3（静态资源分配无法适应动态瓶颈迁移）**：agentic RL 是 producer-consumer 流水线，任务复杂度差异导致瓶颈在 rollout 和训练之间频繁切换；固定 GPU 比例（如 0.125 或 0.5）在单步内就会经历 RWT（Rollout 等待时间）与 TWT（Training 等待时间）交替出现的现象，无法兼顾全局吞吐。
- **根本原因**：现有系统（如 AReaL、StreamRL）虽暴露了异步性，但未将 rollout-trainer 分割点、Ref-Actor 执行策略和 GPU 分配统一为基于就绪信号的联动决策；同时推理引擎的内存管理停留在请求级，未能上升至任务级以感知多轮生命周期。

## 核心贡献（创新点）
1. **Continuous Task Batching (CTB)**：将 Rollout 调度从请求级连续 batching 升级为任务级 admission/preemption/resumption，通过语义感知的优先级层次（评估任务 > GRPO 组完成度 > 活跃执行 > 上下文长度）保留有用的 rollout 状态，避免 KV cache 灾难性驱逐。与已有工作的本质区别在于：Seer/StreamRL 等仍以请求为中心调度，CTB 以任务为单位感知多轮生命周期的全局可见性。
2. **Resource-Aware Ref-Actor Pipelining (RA²P)**：提出两种互补的 Ref-Actor 执行模式——解耦流式模式（适合高 RAS 或短 TPRM）和共置聚合模式（适合低 RAS 长 TPRM）——由就绪信号动态选择。与已有工作的本质区别在于：VeRL/AReaL/StreamRL 均采用固定 Ref-Actor 执行策略，RA²P 首次根据微批到达节奏联合决定执行模式与资源占用。
3. **Elastic Resource Scaling (ERS)**：基于同一就绪信号（RAS/TPRM）在 rollout 与 trainer 之间动态迁移 rank，实现 pipeline 零暂停的弹性缩放。与已有工作的本质区别在于：传统弹性框架（如 Elasticflow/Pollux）需暂停作业、重建通信组，而 ERS 利用 on-policy RL 天然的权重同步边界，将 weight migration 叠加到 cache-flush 窗口中，实现零开销弹性。

## 方法详解
- **CTB（任务级连续批量调度）**：
  - Token-Aware Admission Control：对每个 GRPO group 发送轻量 probing task 估计首 turn token 数，按 DP rank 的 token 预算池 admission；每 turn 更新实际 token 使用量。同时受环境并发度、策略新鲜度（staleness bound）和任务多样性三项约束限制。
  - Semantic-Aware Pausing & Resuming：当 rank token 用量超预算时，按优先级降序 evict 低优先级任务（Evaluation > Group Completion > Active > Long Context）；恢复时优先 resume 同一 rank 上的暂停任务（enforce group affinity），减少 prefix recomputation。
  - 任务生命周期公式：$P(t) = \omega_1 \cdot \mathbb{I}_{\text{eval}} + \omega_2 \cdot G_{\text{completion}} + \omega_3 \cdot \mathbb{I}_{\text{active}} + \omega_4 \cdot L_{\text{context}}$。

- **RA²P（双模式 Ref-Actor 流水线）**：
  - 就绪信号定义：**RAS（Ready-at-Start）**= 训练 step 开始时已就绪的 micro-batch 数量；**TPRM（Time Per Ready Micro-batch）**= 后续 micro-batch 到达间隔。
  - **Decoupled Streaming Mode**：Ref 与 Actor 部署在独立 GPU 上，1 个 Ref rank 服务 M 个 Actor rank 形成 streaming pipeline。关键设计：将 Actor 的 loss computation（含 KL 散度惩罚）从 forward pass 推迟到 backward pass，打破 Ref forward → Actor forward 的串行依赖，实现 Ref forward、Actor forward、Actor backward 的零 swap 重叠。代价是 pipeline fill bubble：Actor i 需等待 $i \cdot t_{ref}$ 才能获得 ref_log_probs。
  - **Optimized Colocated Mode**：Ref 与 Actor 共置在同一 GPU 上。两个优化：（1）Ready-batch aggregation：监控全局 buffer，累积多个 micro-batch 后一次性加载 Ref（减少 swap 频次）；（2）Zero-copy 共享内存传输：同节点 Ref/Actor 间直接通过 GPU 共享内存传递数据，绕过网络序列化。
  - 调度分析：成本模型 $C_{dec} \approx \sum_{i=0}^{M-1}(i \cdot t_{ref}) + \sum_{i=1}^{N}\max(\delta_i, t_{act\_f}+t_{act\_b})$；Colocated 模式通过算法仿真而非闭式计算。
  - Micro-batch dispatching：早期 micro-batch 累积至满秩计算容量，末尾 micro-batch 刻意做小以缩短 pipeline flush bubble；Colocated 模式用 zig-zag（LPT-first）分配平衡各 rank 计算量；Decoupled 模式将最短序列排在管道末端最小化尾部延迟。

- **ERS（弹性资源重分配）**：
  - Readiness-Aware Plan Generation：ERS 在 Actor update 完成后、sync_params 广播前立即生成 scaling plan。监控 generation deficit（全局 batch 量 < $B_{ideal}$）和 consumption deficit（HoL latency > $\tau \times T_{step}$）两个队列指标来估计 RAS/TPRM 趋势，避免瞬时噪声导致的 reactive thrashing。
  - 策略选择：生成不足（高 TWT）→ 选择 Colocated RA²P + 扩容 Rollout rank；消费不足（高 RWT）→ 选择 Decoupled RA²P + 扩容 Trainer rank。
  - Cache-Free Task Migration：利用 on-policy RL 在权重同步后 KV cache 天然失效的窗口，将 Rollout rank 上的 active task 迁移到其他 rank，新 rank 上直接用新权重 fresh prefill，无需物理缓存迁移。
  - Latency-Hiding Role Switching：Scaling down Ref rank 时将 Ref offload 到 CPU，同时激活的 Rollout rank 参与 sync_params broadcast 加载 Actor；反之 Scaling down Rollout rank 时跳过 weight download，利用同步窗口加载 Ref 模型，所有操作与 sync_params 重叠，不阻塞关键路径。
  - 边界处理：冷启动/评估期预分配最大 Rollout 规模加速 buffer 填充；Scale-down 后采用 adaptive data retention，Trainer 侧自适应降低 fetch size 至 $B_{min}$ 以保持 micro-batch 储备。

## 实验与结果
- **实验设置**：4 节点集群，每节点 8× NVIDIA H100（NVLink 互联），32 GPU 总量；环境部署于 1024 核 CPU 的 K8s 集群；后端使用 vLLM（rollout）+ Megatron（training）。
- **模型**：文本任务用 Qwen-2.5-7B/14B；多模态任务用 Qwen-3-VL 4B / Qwen-3.5 9B。
- **任务**：文本（WebShop、AlfWorld）；多模态（OSWorld、ScienceBoard）。每 20 步做一次 evaluation。
- **基线**：VeRL（同步）、AReaL（异步）、StreamRL（SOTA 异步流式）。
- **文本任务端到端结果（100 steps）**：
  - TIDERL vs VeRL：**5.6×** 吞吐提升，训练时间减少 51.1%，BoN reward 差距仅 0.01、pass rate 差距 0.5%。
  - TIDERL vs AReaL：**1.8×** 吞吐提升。
  - TIDERL vs StreamRL：**>7×** 吞吐提升（StreamRL 14B 任务 6 小时内未完成）。
- **多模态任务端到端结果（40 steps）**：
  - TIDERL vs VeRL：**6.02×** 吞吐提升。
  - TIDERL vs 异步基线：**>33%** 吞吐提升，训练时间减少 62.2%。
- **组件消融（OSWorld + Qwen-3-VL 4B）**：
  - Baseline（StreamRL）：20.2 k token/s
  - +RA²P：23.1 k token/s（+10.4%）
  - +ERS：31.8 k token/s（+35.7%）
  - +CTB：33.3 k token/s（+4.8%）
- **CTB 组件**：KV cache 命中率提升 1.58×，生成吞吐提升 1.15×，step 时长减少 15.6%；vanilla 命中率低至 0.9%，CTB 保持在 >6.0%。
- **RA²P 组件**：在处理 8 个就绪 micro-batch 的设置下，相比 vanilla streaming（有/无 offloading），RA²P 减少高达 44.3% 的 per-step training time overhead。
- **ERS 组件**：在 8×H100 上 100 steps WebShop 任务中，ERS 将总 RWT+TWT 减少 68.6%–77.6%。

## 相关工作脉络
1. **VeRL（同步 RL）**：严格 phase barrier 耦合 rollout 与训练，强一致性好但 GPU 空闲严重；TIDERL 通过异步解耦和弹性调度克服其 stall 问题。
2. **AReaL（异步 RL，含 APRIL）**：首次物理解耦 rollout/trainer，用 active partial rollout 管理 staleness；但未将 ref-actor 执行模式和 GPU 分配与就绪信号联动。
3. **StreamRL（异步流式 RL）**：支持 streaming micro-batch 直接输送到 Trainer；但多微批 bursty 到达触发频繁 Ref-Actor 模型交换，且缺乏就绪感知调度。
4. **AsyncFlow（异步流式框架）**：类似 StreamRL 的思路，同样面临模型交换开销。
5. **Seer（同步上下文学习）**：针对单 turn reasoning 的 chunk-level 调度，对多轮 agentic 交互的中断特性不适应；其后缀解码在多轮场景中因 draft-verification 开销而适得其反。
6. **Continuum（多轮 agent 调度）**：引入 KV cache TTL 和 session-aware 调度改善 serving 吞吐，但未考虑 RL 算法层面的 group-relative advantage、staleness bound 等依赖。
7. **Elasticflow / Pollux（通用弹性训练）**：通过 co-adaptive batch size 和 throughput modeling 动态扩缩容，但需要暂停作业、迁移大状态或重建通信组，不适用于 agentic RL 的分钟级瓶颈切换。

## 局限性与未来方向
- **评估任务的版本锚定干扰异步节奏**：当前评估需要在静态模型版本上运行以保持一致性，强制版本同步会短暂打断异步流水线；未来可通过引入 relay weight synchronization 机制（如 Laminar）实现解耦的多版本 snapshot，但需在 host memory 中同时维护 reference model 和多版本轨迹中的大图片文件，内存管理有挑战。
- **Suffix Decoding 不兼容当前多轮场景**：由于多轮交互环境下 acceptance rate 极低（~40%）、acceptance length 很短（~2.8 tokens），draft-verification 开销超过投机收益；未来若出现更适合多轮场景的 SD 机制可集成。
- **ERS 保守缩放策略**：多模态任务中 Trainer scale-up 受限于每步最多 1 rank，长期可能成为瓶颈，导致系统自然趋于一步 stale 状态。
- **论文未提及**对极端长尾任务（如数万 turn 的 agent trajectory）的扩展评估。

## 研究启发与可借鉴点
1. **"就绪信号驱动的双层决策"设计范式**：RAS（启动 backlog）和 TPRM（到达间隔）作为统一的决策信号，同时驱动执行模式选择（RA²P）和资源重分配（ERS），避免了多个子系统各自为政的优化冲突；这一"信号 → 策略 → 资源"的联动框架可迁移到其它 producer-consumer pipeline 系统中。
2. **利用算法天然边界实现零开销弹性**：ERS 巧妙利用 on-policy RL 权重同步后 KV cache 必然失效这一事实，将 rank 迁移叠加到 cache-flush 窗口，避免了传统弹性框架的 pipeline 暂停和 NCCL group 重建开销；类似思路可应用于其它需要模型权重更新的迭代型训练系统。
3. **任务级调度优于请求级调度**：CTB 将调度单元从 request 提升到 task（多轮生命周期），并通过 semantic-aware priority 在 eviction/resume 时考虑 group completion、context length 等语义信息；这一思想可扩展到多轮对话服务、RAG pipeline 等场景。
4. **计算图重构打破跨模型依赖**：RA²P 将 Actor loss computation 从 forward pass 推迟到 backward pass，打破了 Ref forward → Actor forward 的串行瓶颈，使得 streaming pipeline 可以实现零 swap 重叠；这种将依赖关系从 forward 推向 backward 的重构技巧具有通用性。
5. **Micro-batch 尾部最小化策略**：刻意将最后一个 micro-batch 做小以减少 pipeline flush bubble，同时配合不同的 dispatch 策略（zig-zag vs smallest-last），是一种轻量高效的尾部延迟优化手段。

## 关键术语表
**Goodput**：衡量训练有效吞吐的指标，即每秒被 Trainer 消费并推进 update 的 rollout token 数，区别于单纯 GPU 利用率。
**RAS (Ready-at-Start)**：训练 step 开始时全局 buffer 中已就绪的 micro-batch 数量，反映训练初期的数据积压程度。
**TPRM (Time Per Ready Micro-batch)**：micro-batch 到达的时间间隔，反映训练过程中数据持续供给的速度。
**KV Cache Preemption**：推理引擎为服务新请求而驱逐已有请求的 KV cache（历史 token 的内存占用），在多轮交互暂停期间尤为有害。
**On-policy Staleness**：Rollout 使用旧权重生成的数据与当前训练权重的偏离程度，GRPO 等 on-policy 算法对 staleness 敏感，需通过 timely weight sync 约束。
**Group Relative Policy Optimization (GRPO)**：论文采用的 RL 算法， advantage 通过对同 group 内多条 trajectory 的 reward 做标准化计算得到：$A_i = (r_i - \text{mean}(\mathbf{r})) / \text{std}(\mathbf{r})$。
**Ref-Actor Thrashing**：Trainer 在同一 GPU 上频繁交替加载 Ref 和 Actor 模型权重导致的 PCIe 传输和 GPU context-switching 开销。
**Pipeline Fill Bubble**：Decoupled streaming 模式下，Ref model 串行计算 ref_log_probs 导致后续 Actor rank 等待的初始空泡期。

## 可复现要素
- **数据集/任务**：WebShop、AlfWorld（文本）；OSWorld、ScienceBoard（多模态）；均为公开 benchmark。
- **代码**：论文未明确声明开源，但提到实现约 16000 行 Python，附录 B 披露了实现细节（API 接口、调度 hook 等）。
- **模型权重**：使用开源 Qwen 系列模型（Qwen-2.5-7B/14B、Qwen-3-VL 4B、Qwen-3.5 9B）。
- **关键超参**：全局 batch 大小范围 $[B_{min}, B_{ideal}]$；HoL latency 阈值 $\tau \times T_{step}$；每步最大 scale-up rank 数为 1；文本任务 max turns=20、max context=8192 tokens；多模态任务 max turns=50，图片 sliding window 策略（保留最近 5 张）。
- **硬件**：4 节点 × 8× NVIDIA H100，NVLink 互联，JuiceFS NFS 共享存储。
- **后端**：Rollout 使用 vLLM，Training 使用 Megatron Core（PyTorch FSDP）。
