---
title: "TIDERL: Boosting Agentic RL Goodput with Readiness-Aware Scheduling"
source: https://arxiv.org/pdf/2608.10402v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 03:43:42"
field: "分布式强化学习系统"
keywords: ["Agentic RL", "distributed training", "KV cache management", "elastic scheduling", "throughput optimization"]
innovations: ["提出CTB+RA²P+ERS三组件协同的readiness-aware异步RL架构", "利用on-policy sync boundary实现zero-overhead rank迁移", "RAS/TPRM双信号驱动Ref-Actor双模式选择与动态资源分配"]
benchmarks: ["WebShop", "AlfWorld", "OSWorld", "ScienceBoard"]
---

# 论文速读：TIDERL: Boosting Agentic RL Goodput with Readiness-Aware Scheduling

## 一句话总结
论文针对多轮agentic RL中因任务长度、上下文增长和等待外部反馈导致的极端负载波动问题，提出了TIDERL系统——一个具备连续任务批处理(CTB)、资源感知Ref-Actor流水线(RA²P)和弹性资源扩缩(ERS)的异步弹性训练架构，通过共享的readiness信号协同调度，将RL训练goodput提升最高5.6×（text-only）和33%（多模态），同时保持与同步基线相当的任务性能。

## 研究问题与动机
- **KV Cache灾难性丢失**：标准推理引擎采用request-level continuous batching，在多轮agentic交互中，任务等待外部反馈时其KV cache会被频繁预empt，恢复时需对超长历史上下文进行昂贵的prefill重计算，导致cache hit rate降至~12%。
- **Reference inference stalls与model swapping thrashing的两难**：现有系统在同一GPU上顺序执行Ref和Actor模型，要么等待全局大batch造成高达81%的GPU idle时间，要么频繁切换模型引发严重的I/O thrashing。
- **静态资源分配无法适应动态瓶颈转移**：Agentic RL的producer-consumer pipeline中，复杂任务导致Trainer饥饿，简单任务又产生ready backlog，瓶颈在rollout和training之间频繁切换，固定GPU分区造成结构性throughput上限。
- **核心问题**：如何构建统一弹性训练架构，在多轮agentic执行下最大化以训练throughput度量的RL goodput？

## 核心贡献（创新点）
1. **Continuous Task Batching (CTB)**：将调度粒度从request级提升至task级，通过token-aware admission和semantic-aware priority hierarchy（eval > group completion > active > long context）保护有用rollout状态，避免KV cache预empt导致的重复prefill。
2. **Resource-Aware Ref-Actor Pipelining (RA²P)**：提出双模式Ref-Actor执行策略，由RAS（启动时ready backlog）和TPRM（micro-batch到达间隔）信号驱动选择decoupled streaming或colocated aggregation模式，消除model-swapping开销同时避免长尾stall。
3. **Elastic Resource Scaling (ERS)**：基于同一readiness信号动态迁移DP rank between rollout和training，利用on-policy RL天然权重同步边界实现零开销scale operation，无缝衔接RA²P模式切换。
4. **端到端系统集成**：模块化设计兼容主流训练(Megatron/FSDP)和推理(vLLM/SGLang)后端，解耦环境部署至外部CPU集群，在4节点32×H100测试床验证。

## 方法详解
**Continuous Task Batching (CTB)**
- 将每个Data Parallel rank视为token budget池，通过轻量probing task预估任务初始上下文大小，按实际token footprint而非request数量admit任务。
- 调度优先级函数：$P(t) = \omega_1 \cdot \mathbb{I}_{eval} + \omega_2 \cdot G_{completion} + \omega_3 \cdot \mathbb{I}_{active} + \omega_4 \cdot L_{context}$，确保evaluation任务(需锚定权重)、接近完成的GRPO组(需完整计算advantage)、活跃执行任务和长上下文任务获得保护。
- Resume时强制group affinity，将任务放回最可能保留共享prefix的rank，避免prefix recomputation。
- 引入/pause和/resume API防止环境timeout检测。

**Resource-Aware Ref-Actor Pipelining (RA²P)**
- 两个信号：RAS（训练步开始时已ready的micro-batch数量）和TPRM（新micro-batch到达间隔）。
- **Decoupled Streaming Mode**（高RAS或短TPRM）：Ref和Actor分配到不同GPU，重构计算图将loss计算（含KL penalty）从forward移至backward，实现Ref forward、Actor forward、Actor backward的零swap流水线重叠。
- **Optimized Colocated Mode**（低RAS长TPRM）：Ref和Actor同GPU，通过ready-batch aggregation一次load model处理多个micro-batch，利用zero-copy shared-memory跨GPU传递ref_log_probs，摊薄model交替开销。
- Micro-batch dispatching：early micro-batch累积至饱和compute capacity，last micro-batch刻意最小化以缩短pipeline flush bubble；colocated采用zig-zag(longest-processing-time-first)负载均衡，decoupled将最小序列放pipeline最后减少tail latency。

**Elastic Resource Scaling (ERS)**
- 通过generation deficit（global batch volume < B_ideal）和consumption deficit（HoL latency超过阈值）检测bottleneck shift。
- Trainer饥饿→colocated RA²P + 更多Rollout ranks；Rollout backlog堆积→decoupled RA²P + 更多Trainer capacity。
- **Cache-Free Task Migration**：利用on-policy RL同步边界天然invalid历史KV cache，CTB自动reassign任务到新rank后fresh prefill，无需物理cache迁移。
- **Latency-Hiding Role Switching**：将PCIe weight swapping与sync_params broadcast overlap，scaling down Ref rank时offload到CPU等待backward完成；scaling down Rollout rank时discard stale模型并用sync窗口load Ref模型。
- Adaptive data retention：scale-down后Trainer自适应降低fetch size至B_min，延长step duration给Rollout累积时间。

## 实验与结果
**测试环境**：4节点×8×NVIDIA H100 GPU集群，解耦环境部署在1024 CPU核心的Kubernetes集群。

**模型与任务**：
- Text-only：Qwen2.5-7B/14B，WebShop + AlfWorld
- Multi-modal：Qwen-3-VL 4B / Qwen-3.5 9B，OSWorld + ScienceBoard（处理高分辨率截图，压力prefill和context-carriage）

**基线**：
- VeRL（同步，严格phase barrier）
- AReaL（异步，request-level scheduling）
- StreamRL（异步streaming，当前SOTA）

**主要结果**：
| 场景 | TIDERL vs VeRL | TIDERL vs Async baselines | 性能保持 |
|------|----------------|--------------------------|----------|
| Text-only | 5.6× throughput提升，训练时间↓51.1% | 1.8×提升 | BoN reward deficit仅0.01，pass-rate deficit 0.5% |
| Multi-modal | 6.02×提升，训练时间↓62.2% | 33%+提升 | 40 steps内优于所有异步基线，wall-clock仅37.8% |

**组件贡献分解**（OSWorld + Qwen-3-VL 4B）：
- Baseline (StreamRL): 20.2k token/s
- +RA²P: 23.1k (+10.4%)
- +ERS: 31.8k (+35.7%)
- +CTB: 33.3k (+4.8%)

**微观指标**：
- CTB：KV cache hit rate提升1.58×，generation throughput提升1.15×，step duration↓15.6%
- RA²P：per-step training time↓up to 44.3%（消除PCIe transfers和GPU context-switching overhead）
- ERS：total RWT+TWT↓68.6–77.6%

## 相关工作脉络
1. **VeRL/RLHFuse**：同步RL框架，colocate rollout和training，继承严格phase barrier，TIDERL通过异步解耦和elastic scaling克服long-tail stall问题。
2. **AReaL/APRIL**：异步RL先驱，使用active partial rollout管理staleness，但固定partition无法响应readiness shift，TIDERL的ERS提供实时反应。
3. **StreamRL**：streaming micro-batch避免global batch等待，但bursty stream触发频繁model alternation导致thrashing（14B任务6小时未完成），TIDERL的RA²P双模式消除此开销。
4. **Seer/Suffix Decoding**：suffix decoding在single-turn高效但对agentic工作负载有害（acceptance rate ~40%，acceptance length仅~2.8 tokens），TIDERL明确排除SD。
5. **Continuum/KV-cache pinning**：serving系统的session-aware调度，但未考虑agentic RL的algorithmic dependencies（GRPO group completion、evaluation anchor、staleness bound），CTB将这些约束整合入task-level调度。
6. **Pollux/Elasticflow**：通用elastic训练框架通过co-adaptive batch size scaling，但scale操作需suspend job/migrate weights/rebuild comm groups，TIDERL利用on-policy sync boundary实现zero-overhead rank迁移。

## 局限性与未来方向
- **Evaluation tasks与异步pipeline的冲突**：评估需锚定静态模型版本，当前强制version synchronization会暂时打断异步流动；未来可集成relay weight synchronization机制（如Laminar）维持解耦frozen snapshot。
- **Suffix Decoding不适用**：多轮交互中acceptance rate低导致speculative gain被draft-verification overhead抵消，需未来SD机制适配agentic workload。
- **Evaluation的multi-version footprint优化**：参考模型、大图像文件和轨迹共存时host memory占用需谨慎管理。
- **One-step off-policy staleness**：多模态场景下Trainer成为长期bottleneck，系统自然趋向一步stale数据，存在algorithmic equivalence与throughput的trade-off。

## 研究启发与可借鉴点
1. **Readiness信号驱动的动态调度范式**：RAS+TPRM双信号可作为通用producer-consumer pipeline的资源分配依据，适用于任何异步RL或流式数据处理系统。
2. **利用算法天然边界实现零开销弹性**：on-policy RL的sync点天然invalid历史cache，为task migration提供安全边界，这一思路可扩展至其他需定期重同步的分布式训练场景。
3. **计算图重构消除流水线气泡**：将loss计算从forward移至backward以打破Ref-Actor同步依赖，是pipeline parallelism中经典的bubble reduction technique，可借鉴于其他多模型串行执行场景。
4. **Semantic-aware priority hierarchy替代启发式调度**：结合任务类型、完成比例、执行状态、上下文长度的多维度优先级，比单纯基于length或arrival time的调度更适配agentic workload的异质性。
5. **Zig-zag分配策略优化tail latency**：longest-processing-time-first负载均衡配合"最后序列最小化"设计，直接针对pipeline flush bubble做优化，对任何micro-batch streaming训练系统有参考价值。

## 关键术语表
**Goodput**：有效吞吐量，论文定义为最终被Trainer消费并推进update的rollout tokens/sec，排除stale或discarded tokens。
**RAS (Ready-at-Start)**：训练步开始时已ready的micro-batch数量，反映startup backlog压力。
**TPRM (Time Per Ready Micro-batch)**：训练步开始后新micro-batch到达的间隔时间，反映steady-state arrival pace。
**CTB (Continuous Task Batching)**：任务级连续批处理，替代request-level batching，支持task-level admission/preemption/resumption。
**RA²P (Resource-Aware Ref-Actor Pipelining)**：资源感知的Ref-Actor流水线，根据readiness信号选择decoupled streaming或colocated aggregation执行模式。
**ERS (Elastic Resource Scaling)**：弹性资源扩缩，动态迁移DP rank between rollout和training worker groups。
**GRPO (Group Relative Policy Optimization)**：论文采用的RL算法，advantage基于同组trajectories标准化：$A_i = (r_i - \text{mean}(\mathbf{r})) / \text{std}(\mathbf{r})$。
**Staleness bound**：异步RL中rollout使用的旧权重与最新权重之间的代差限制，超过会导致off-policy数据 degrade训练稳定性。
**HoL Latency (Head-of-Line Latency)**：全局buffer中最老micro-batch的等待时间，ERS用于检测consumption deficit。
**B_min / B_ideal**：有效训练步的最小batch size和GPU HBM/compute完全利用的理想batch size，ERS据此adaptive batching。

## 可复现要素
- **数据集**：WebShop、AlfWorld、OSWorld、ScienceBoard（公开benchmark）
- **代码开源**：论文未明确声明GitHub仓库（Appendix B提及implementation details，但无链接）
- **模型权重**：使用公开Qwen系列模型（Qwen2.5-7B/14B、Qwen-3-VL 4B、Qwen-3.5 9B）
- **关键超参**：
  - Text-only：max 20 interaction turns，max context 8192 tokens
  - Multi-modal：max 50 interaction turns，sliding window保留最近5张图（分辨率1280×720）
  - Evaluation frequency：每20 training steps一次
  - Best fixed partition ratio（AReaL baseline）：0.25 rollout GPU fraction
- **训练后端**：Megatron Core、PyTorch FSDP
- **推理后端**：vLLM、SGLang
