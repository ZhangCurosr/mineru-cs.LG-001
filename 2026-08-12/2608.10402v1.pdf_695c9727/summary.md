---
title: "TIDERL: Boosting Agentic RL Goodput with Readiness-Aware Scheduling"
source: https://arxiv.org/pdf/2608.10402v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 03:43:12"
field: "大模型分布式训练系统"
keywords: ["reinforcement learning", "agentic RL", "distributed systems", "KV cache", "elastic scheduling", "goodput optimization"]
innovations: ["任务级CTB调度解决多轮agentic KV cache预占", "基于RAS/TPRM的双模式RA²P消除Ref-Actor模型交替抖动", "利用on-policy sync边界实现零开销ERS弹性资源缩放"]
benchmarks: ["WebShop", "AlfWorld", "OSWorld", "ScienceBoard"]
---

# 论文速读：TIDERL: Boosting Agentic RL Goodput with Readiness-Aware Scheduling

## 一句话总结
论文提出TIDERL，一个面向多轮agentic RL的弹性异步系统，通过任务级调度(CTB)、就绪感知的Ref-Actor流水线(RA²P)和弹性资源缩放(ERS)三个协同组件，解决KV cache预占、模型交替抖动和静态资源分配失效三大瓶颈，在文本和多模态任务上分别实现最高5.6×和33%以上的训练goodput提升。

## 研究问题与动机
- **KV cache灾难性预占**：现有推理引擎按请求级别做continuous batching，多轮agentic任务在等待环境反馈期间KV cache被驱逐，恢复时触发大量冗余prefill重计算。
- **Trainer端stalls与thrashing困境**：现有异步系统要么等待完整全局batch导致GPU空闲最多81%，要么流式处理微批次引发频繁Ref-Actor模型交替，产生巨大PCIe交换开销。
- **静态资源划分失效**：agentic RL的生产-消费瓶颈在生成与训练间动态切换，固定GPU比例无法同时应对启动阶段的ready backlog和稳态微批次到达速率。
- **现有异步框架缺乏联合决策**：AReaL、StreamRL等虽暴露异步性，但仍将rollout-trainer划分视为固定配置，无法根据就绪信号联合选择Ref-Actor执行策略与GPU分配。

## 核心贡献（创新点）
- **CTB（Continuous Task Batching）**：将Rollout调度从请求级别提升到任务级别，按token足迹做准入控制、语义优先级暂停/恢复，保留有效rollout状态供给Trainer；区别于现有request-level continuous batching仅服务单次推理。
- **RA²P（Resource-Aware Ref-Actor Pipelining）**：基于RAS（启动就绪批次量）和TPRM（微批次到达间隔）两个就绪信号，在解耦流式模式（Ref/Actor分GPU、零swap）和共址聚合模式（同GPU、ready-batch聚合+零拷贝）间自动选择；区别于VeRL/AsyncFlow等固定执行模式的trainer。
- **ERS（Elastic Resource Scaling）**：利用on-policy RL的同步边界，将rank迁移与weight broadcast重叠实现零开销弹性，根据同一就绪信号协调CTB与RA²P模式；区别于Pollux等需暂停作业迁移权重的弹性框架。

## 方法详解
- **CTB核心设计**：
  - Token感知准入：轻量探测任务估算初始token使用，每轮交互更新实际token足迹；满足环境并发度、策略过时边界、任务多样性三约束。
  - 语义感知暂停/恢复：四级优先级——评估任务（最高，需锚定权重）、GRPO组完成度、执行状态（避免浪费已计算cycle）、上下文长度（防 catastropic prefilled）。
  - Worker affinity：恢复时优先调度至保留共享前缀的rank，避免prefix重算。

- **RA²P双模式**：
  - 解耦流式模式（高RAS或短TPRM）：Ref与Actor分置不同GPU，重构计算图将loss计算从Actor forward移至backward，实现Ref forward → Actor forward → Actor backward重叠，消除swap。
  - 共址聚合模式（低RAS且长TPRM）：Ref/Actor同GPU，监控全局buffer聚合多个ready micro-batch后统一执行Ref再执行Actor，一次swap处理多批次；利用同节点GPU共享内存零拷贝传递ref_log_probs。
  - 微批次分发：前期积累足够token饱和所有DP rank算力，末期刻意保留最小micro-batch减少pipeline flush bubble；共址用zig-zag负载均衡，解耦将最小序列置于末尾降低尾延迟。

- **ERS调度逻辑**：
  - 监测generation deficit（global batch < B_ideal）→ 选择共址RA²P并扩容Rollout；监测consumption deficit（HoL延迟 > τ×T_step）→ 选择解耦RA²P并扩容Trainer。
  - 冷启动/评估阶段预判性最大化Rollout规模。
  - 自适应数据保留：Rollout缩容后Trainer降低fetch size至B_min，延长step时长给Rollout积累时间。
  - 无缝迁移：利用on-policy sync边界天然invalidate KV cache，任务迁移无需物理cache传输；权重交换与sync_params broadcast重叠隐藏延迟。

## 实验与结果
- **测试环境**：4节点×8×NVIDIA H100 GPU集群，vLLM作为rollout后端，Megatron作为training后端，外部CPU集群运行环境仿真。
- **文本任务**（WebShop、AlfWorld + Qwen2.5-7B/14B）：TIDERL相较VeRL提升**5.6×** training throughput，相较AReaL提升**1.8×**，相较StreamRL提升**>7×**（14B模型StreamRL 6小时未完成）；训练时间减少51.1%，BoN reward deficit仅0.01、pass-rate deficit 0.5%。
- **多模态任务**（OSWorld、ScienceBoard + Qwen-3-VL 4B/Qwen-3.5 9B）：相较VeRL提升**6.02×**，相较异步基线提升**>33%**；训练时间减少62.2%。
- **组件贡献**（OSWorld + Qwen-3-VL 4B）：Baseline 20.2k token/s → +RA²P 23.1（+10.4%）→ +ERS 31.8（+35.7%）→ +CTB 33.3（+4.8%）。
- **微观指标**：KV cache hit rate提升1.58×，per-step training time减少最高44.3%，总waiting time减少最高77.6%。

## 相关工作脉络
- **VeRL [30]**：同步RL框架代表，强制rollout-trainer阶段屏障，是本文展示耦合架构idle overhead的主要对比基线。
- **AReaL [5]**：基础异步RL框架，引入active partial rollout管理off-policy staleness，但固定GPU划分无法响应就绪信号变化。
- **StreamRL [49]**：当前异步SOTA，支持stream generation无需等待全局batch，但微批次 burst触发频繁model swapping导致14B任务失败。
- **Seer [23]**：同步框架+ suffix decoding优化，在多轮agentic任务中因draft-verification开销反而降低rollout速度。
- **SLIME [54]/OpenRLHF [11]**：生产级框架，依赖固定执行角色，无基于就绪信号的动态调度能力。

## 局限性与未来方向
- **评估任务版本锚定冲突**：评估需静态模型版本，当前严格版本同步会暂时打断异步流水线；未来可集成relay weight sync机制（如Laminar [29]）维持frozen snapshot。
- **Suffix Decoding不适用**：多轮agentic任务中SD acceptance rate仅~40%、acceptance length~2.8 tokens，draft-verification开销超过投机增益。
- **弹性缩放粒度限制**：Trainer扩容受限于每step最多1个rank，长期pipeline bottleneck仍可能出现。
- **Host memory竞争**：HiCache等分层缓存因全局buffer和offloaded Trainer模型耗尽CPU RAM而未被集成。

## 研究启发与可借鉴点
- **就绪信号双维度设计**：RAS（启动 backlog）+ TPRM（稳态到达速率）比单一queue depth更能刻画异步生产-消费的不确定性，可迁移至其他streaming pipeline系统。
- **利用算法天然边界隐藏系统开销**：on-policy RL的sync boundary天然invalidate KV cache，ERS借此实现零开销rank迁移，提示可探索其他算法约束作为系统优化契机。
- **计算图重构解耦依赖**：将loss computation从forward移至backward实现Ref/Actor pipeline overlap，为其他多模型协作场景提供范式。
- **任务级调度超越请求级**：CTB将scheduling unit从request提升至task，在多轮交互式负载中具有普适价值。
- **代价模型指导模式选择**：RA²P的分析延迟模型（C_dec vs C_col）为类似双模式系统提供可量化的决策框架。

## 关键术语表
**Goodput**：有效训练吞吐，指被Trainer消费并推进update的rollout tokens速率，区别于仅反映GPU利用率的throughput。
**RAS (Ready-at-Start)**：训练步开始时全局buffer中已就绪的micro-batch数量，反映启动阶段数据积压程度。
**TPRM (Time Per Ready Micro-batch)**：训练开始后相邻micro-batch到达的时间间隔，反映稳态数据供给速率。
**GRPO (Group Relative Policy Optimization)**：本文采用的RL算法，组内trajectory rewards标准化计算advantage，要求同组完整才能valid update。
**KV Cache Preemption**：推理引擎为服务新请求而驱逐已有任务的KV cache，导致多轮任务恢复时触发灾难性cache miss。
**On-policy Staleness**：异步RL中Rollout使用旧权重生成的数据与当前Actor权重的偏差，GRPO等算法对此敏感需严格 bounds。
**Ref-Actor Model Swapping**：Trainer在相同GPU上交替加载Reference模型和Actor模型产生的PCIe传输与GPU context-switch开销。
**Zig-zag Dispatch**：按序列长度排序后交替分配给各DP rank的负载均衡策略，确保各rank计算量近似相等。

## 可复现要素
- **数据集**：WebShop、AlfWorld、OSWorld、ScienceBoard（均为公开benchmark）
- **代码**：论文未明确声明开源，但提供了~16000行Python实现细节附录
- **模型**：Qwen2.5-7B/14B、Qwen-3-VL 4B、Qwen-3.5 9B（open-weights）
- **关键超参**：全局batch [B_min, B_ideal]区间、staleness bound、τ×T_step阈值、最大GPU数
- **硬件**：4节点×8×H100 NVLink集群 + 1024核CPU环境集群 + JuiceFS NFS
