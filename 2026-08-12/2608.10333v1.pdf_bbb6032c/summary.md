---
title: "MERA: Model Evolution and Routing with Skill Adaptation for Agentic Systems at Scale"
source: https://arxiv.org/pdf/2608.10333v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 03:16:13"
field: "Agent 系统模型适配与路由"
keywords: ["LLM Agent", "Model Routing", "Small Language Model Adaptation", "SkillBook", "Verifier-backed Fallback", "Multi-cycle Evolution", "Code Generation"]
innovations: ["以单次调用为适应单元的多周期进化协议，同时更新 SkillBook/小模型适配器/路由器", "依赖感知的 Skill→LLM→Router 调度与联合 replay 准入机制", "Input-only 路由+verifier-backed fallback 架构实现 88.3% pass/60.8% 成本的 cost-quality 操作点"]
benchmarks: ["HumanEval+MBPP (582 held-out)", "TAU-2 (35-task fixed split)"]
---

# 论文速读：MERA: Model Evolution and Routing with Skill Adaptation for Agentic Systems at Scale

## 一句话总结
MERA 提出了一种以单次模型调用为适应单元的验证器驱动的循环进化协议，通过共享执行轨迹同时迭代更新 SkillBook、小模型适配器（LoRA）和路由策略，使小模型能力本身得到提升，而非仅靠路由绕过固定能力的 student；在 582 个 held-out HumanEval+MBPP 任务上，四周期适配将 Qwen2.5-Coder-1.5B 的 pass 从 28.7% 提升至 49.7%（SFT+GRPO），并在验证器回退机制下以 60.8% 的 Always-Luna 成本保留 88.3% 的 pass。

## 研究问题与动机
- **Agent 推理链的成本-质量张力**：单个 agent 任务包含异构的模型调用序列——部分需要深度推理，部分是格式化、工具参数构建等结构化步骤；每次调用都走最强模型可靠性高但昂贵，而仅在用户请求层做粗粒度路由无法捕捉同一工作流内难易步骤并存的现实。
- **现有路由方法的容量上限**：FrugalGPT、RouteLLM 等路由工作通过"易步骤→便宜模型、难步骤→强模型"的调度降低推理成本，但**不改进小模型本身的能力**，因此可节省的幅度被 student 已能解决的问题量所限制。
- **部署侧演化的安全性缺口**：在线 trace 天然可用于适配，但直接将原始轨迹转化为服务策略风险高——便宜模型可能静默失败、 Learned adapter 可能在稀有 case 上退化、技能模板只在狭窄 prompt 签名下安全；现有工作缺少一种将"观察"与"准入"解耦的系统层机制。
- **组件间缺乏共享监督与联合评估**：SkillBook 构建、invocation-level 路由、模型适配在现有系统中通常是独立工程决策，本文指出缺少一个以可执行 trace 为共同监督信号、并通过联合 replay 进行保守准入的协议。

## 核心贡献（创新点）
1. **以单次调用（single invocation）为适应单元的多周期进化协议**：将每条 agent 执行 trace 切片为步级单元，SkillBook、LLM 适配器和路由器共享同一批 trace 证据进行同步迭代更新，区别于以往仅路由整个用户请求的做法。
2. **Skill → LLM → Router 的依赖感知调度**：在同一周期内强制 SkillBook 先于 LLM 适配更新（适配时 prepend SkillBook procedure），Router 最后更新（使其标签反映当前周期的 student 输出），形成数据依赖链，避免并行训练导致适配器基于过时过程信息。
3. **验证器驱动的联合 replay 准入机制**：新路由/技能/适配器仅在联合 replay 中保持任务质量（cost-quality 操作点改善）时才被 admitted，路由准确性本身不构成准入理由，防止孤立组件指标虚假乐观。
4. **输入-仅（input-only）运行时路由 + verifier-backed fallback 架构**：路由决策只依赖当前调用的序列化 prompt，不耦合 agent 隐藏状态；可靠性由执行后 verifier 与 fallback 保障，使系统可保守起步、渐进扩展低成本执行区域。
5. **端到端成本-质量实证**：在 HumanEval+MBPP 上展示四周期 SFT+GRPO 将 1.5B 模型 pass 提升至 49.7%，并在 verifier fallback 下达到 88.3% pass / 60.8% 成本；TAU-2 上适配后的 Qwen3.5-2B 从 14/35 提升到 18/35 并匹配未适配的 4B 端点，加上金融 break-even 分析连接适配成本与部署收益。

## 方法详解
- **运行时路由与 Skill 层**：Router $R$ 仅观察当前调用的序列化 prompt $x_t$，从强模型、便宜模型（small student）、可选专用 student 中选择 $m_t$；Skill selector 可为格式化、提取、单工具参数构建等稳定模板分发 $k_t \in \mathcal{K}$；执行 $y_t \leftarrow m_t(x_t, k_t)$ 后由 Verifier $V$ 检查 schema 合法性、工具调用合法性、可执行测试或下游成功；若 $V(x_t, y_t)$ 失败则 fallback 至更强模型，并完整记录事件 $(x_t, y_t, m_t, k_t, V(x_t, y_t))$。
- **Trace 切片与三类证据流**：将完整 trace 规范化为 step slice，含 prompt、本地上下文、工具 schema、生成输出、verifier 结果、重试次数、fallback 元数据、skill 分配；同一 slice 同时支撑三条更新流：SkillBook 按重复 prompt 签名聚合成败统计；Router 以 prompt 文本为特征用 cheap/strong 标签训练（标签来自可执行小模型结果）；LLM 适配器以 cheap 模型失败或 router/SkillBook 分歧处的 hard example 为训练样本。
- **周期内调度（Algorithm 1）**：每个 update round 对三 track 使用共享 trace 更新，顺序为 Skill → LLM → Router；SkillBook 先更新以保证 LLM 训练时 prepend 的是当前周期的 procedure；Router 最后训练使其标签反映当前周期的 skill 状态和小模型输出；周期性迭代吸收残余 staleness。
- **SkillBook 设计**：外部过程性 prompt 记忆（非响应缓存或权重）；signature 函数按 humaneval / mbpp 两个粗粒度数据集键映射；每条 entry 结合静态任务格式指令 + 有界成功 exemplar + exemplar  grounding 的常见 pitfalls/patterns；新任务 prepended 时不返回相同 prompt 的存储答案。
- **Router 训练**：冻结的 Qwen3-Embedding-0.6B 提取 prompt 特征，接入 class-balanced logistic-regression 头；重复观测归入同一 cross-validation 组，阈值校准使用 disjoint shard，policy 结果报告于 held-out task identifiers；cheap-eligible 定义为当前小模型 rollout 通过 benchmark verifier，否则为 strong 标签。
- **Verifier**：HumanEval/MBPP 下在隔离 Python subprocess 中执行生成代码与 benchmark 提供的测试程序；TAU-2 下用官方 evaluator 检查环境状态、所需工具动作、自然语言断言；推理时 router 不调用 LLM judge。
- **联合准入（Joint Admission）**：SkillBook/router/verifier 证据共同定义 easy/hard/uncertain 区域；新组件仅在 replay 中保证 quality preserved 且 cost/fallback risk 改善时才被 promoted；replay 是 pre-deployment gate，生产部署仍需 shadow/canary 验证、drift 监控与 rollback。

## 实验与结果
- **数据集与设置**：HumanEval + MBPP 合并为 546 training tasks / 582 held-out evaluation tasks（disjoint task identifiers，零 exact prompt overlap）；小模型 Qwen2.5-Coder-1.5B-Instruct，大 teacher/fallback 模型 GPT-5.6 Luna；四周期 SkillBook+SFT 与 SkillBook+SFT+GRPO 比较，3 个独立训练 seed；teacher 输出缓存以保证 matched runs 共享同一 supervision；small:large 成本比 1:10。
- **Table 1（主适配结果，582 held-out tasks，3-seed mean）**：
  - Base SLM (Qwen2.5-Coder-1.5B)：Pass 28.7%，Cost 10.0%
  - Multi-cycle SFT：Pass 44.2%，Cost 10.0%
  - Multi-cycle SFT+GRPO：Pass 49.7%，Cost 10.0%
  - Always Luna：Pass 86.9%，Cost 100.0%
  - GRPO 相比 SFT 提升 5.5–6.7 点（95% paired-t 区间不含零）。
- **Table 2（带 fallback 部署策略对比）**：
  - Always small / exact-cache+small：Pass 49.7%，Cost 10.0%
  - Always Luna：Pass 86.9%，Cost 100.0%
  - RouteLLM-style + fallback：Pass 87.0%，Cost 97.4%
  - FrugalGPT-style response cascade：Pass 85.5%，Cost 106.7%
  - **MERA router + verifier fallback：Pass 88.3%，Cost 60.8%**（最优 cost-quality 操作点）。
- **Table 3（TAU-2 严格对比，35-task split，无 fallback）**：
  - Base Qwen3.5-2B：14/35 (40.0%)
  - Unadapted Qwen3.5-4B：17/35 (48.6%)
  - Trained Qwen3.5-2B + VERL GRPO：**18/35 (51.4%)**，匹配并略微超过未适配 4B 端点；McNemar 单边 p=0.1719，统计功效不足但支持可行性。
- **Table 4（金融 break-even 分析）**：500 行适配场景下名义 break-even 8.68 天，70% 保守 realization 下 12.47 天；300 行场景分别为 15.29 天 / 31.96 天。
- **核心结论**：最强证据是验证器回退部署下以 60.8% 成本达到 88.3% pass（超越 Always-Luna 的 86.9%）；预训练路由本身较弱，大部分质量保全是 verifier + fallback 的贡献；TAU-2 结果支持 adapter-only 改进的可行性但统计功效有限。

## 相关工作脉络
- **FrugalGPT / RouteLLM / BEST-route（Chen et al. 2024; Ong et al. 2024; Ding et al. 2025）**：这些工作聚焦 prompt-level 或 request-level 的路由决策，以模型置信度、偏好数据或测试时最优计算为信号；MERA 的路由单元是 agent 内部的单次调用，且路由准确性不是最终目标——质量保全依赖 verifier fallback。
- **Agent 优化与后训练（Zhou et al. 2024; Qin et al. 2024; Zhang et al. 2024; Spiess et al. 2025; Xi et al. 2026）**：如 Language Agent Tree Search、ToolLLM、GPTSwarms、AgentGym-RL 等改善 agent 策略/规划/工具行为，但通常将模型选择、路由和可复用过程视为固定工程组件；MERA 直接以在线 trace 更新模型混合与路由策略。
- **蒸馏与 Skill 构建（Hsieh et al. 2023; Yuan et al. 2025; Wang et al. 2026; SkillX）**：Distill-step-by-step、Agent-R、SkillX 等工作展示小模型可通过蒸馏或自我训练捕获强模型行为，但通常作为独立训练结果报告；MERA 将 skill 外部化为过程性 prompt 记忆并以 verifier 覆盖度约束 promotion 条件。
- **多轮路由与聚合（Na et al. 2025; Router-R1, Zhang et al. 2025）**：自动变速箱式 tier 调度与多轮路由+聚合通过 RL 训练；MERA 的路由是单步 input-only 分类器，训练信号来自可执行 outcome 而非 preference label。
- **Agent 评测基准（WebShop, Tau-bench, SWE-bench, OSWorld, ToolHop）**：这些基准提供 increasingly realistic 工具交互环境；MERA 利用其中部分（HumanEval+MBPP、TAU-2）证明 trace-centric 适配的可迁移性，同时指出当前 protocol 是 trace-centric 而非 fully online。

## 局限性与未来方向
- **Verifier 覆盖度决定上限**：若 verifier 无法捕获重要语义失败模式，replay 会高估 down-routing / skill promotion 的安全性；这是所有依赖 executable verification 的方法的共性问题。
- **简单步骤优先策略的偏差**：框架有意偏向 narrow、可检查的 step slices，使早期部署更安全，但也意味着困难长 horizon 推理任务将长期留在最强模型上，收益可能渐进释放。
- **Replay 非 fully online**：当前评估是 trace-centric replay，无法完美捕捉 policy 改变引起的 distribution shift；实际部署还需 shadow/canary、drift 监控和 rollback 机制。
- **Skill 泛化受限**：Skill 提升假设可识别并 canonicalize 重复 local subgraph，但高度异构的任务可能无法从中获益。
- **预训练路由较弱**：重建实验中大部分质量保全来自 verifier fallback 而非准确的路由预测，说明当前路由信号还不够强。
- **多 seed 最强证据限于代码生成**：TAU-2 检查干净但功效不足，跨 domain 通用性有待更大规模验证；非平稳 workload 或可复用 step 较少的场景中 specialization 收益有限。

## 研究启发与可借鉴点
- **Trace 切片作为多组件共享监督的统一接口**：将 agent 执行 trace 规范化为 step slice 并分流给 SkillBook/Router/Adapter 三条更新流，避免各组件各自维护训练数据；可迁移到任何具有可执行验证环境的 agent pipeline。
- **依赖感知的 Skill → LLM → Router 调度**：周期内按数据依赖顺序更新而非并行，避免 stale procedure 污染适配器训练；这一思想可推广到其他多组件在线学习系统中。
- **Input-only 路由 + 执行后置验证的架构分离**：将路由复杂度从在线路径卸载到离线 replay，在线路径只保留 verifier + fallback；对生产环境降低耦合度和故障面。
- **Joint replay admission 替代孤立组件指标**：用端到端 cost-quality 操作点作为更新准入标准，而非 router accuracy 或 adapter loss；这一原则可用于任何多模块协同的系统优化。
- **GRPO 在代码生成适配中的显著提升**：Matched SFT+GRPO 比纯 SFT 多获 5.5–6.7 点 pass；结合 verifier 作为 environment reward 的 GRPO 设计值得在其他 tool-use 或代码生成任务中复现。

## 关键术语表
- **MERA (Model Evolution and Routing with Skill Adaptation)**：一种以单次调用为适应单元、用共享 trace 同时迭代更新 SkillBook/小模型适配器/路由器的验证器驱动多周期进化协议。
- **SkillBook**：外部过程性 prompt 记忆，按重复 prompt 签名聚合成功 exemplar 和常见 pitfalls，以结构化形式 prepend 到新任务，不缓存回答。
- **Verifier-backed fallback**：运行时由可执行 verifier（代码测试/工具断言）检查小模型输出，失败则回退到强模型，以此保障部署质量。
- **Joint replay admission**：新路由/技能/适配器必须在 replay 中保持或改善端到端 cost-quality 操作点才能被 admitted，而非基于孤立组件指标。
- **Single invocation as unit of adaptation**：将 agent 执行链中的单次模型调用而非整个用户请求作为 trace 切片和训练样本的粒度单位。
- **Skill → LLM → Router schedule**：周期内按数据依赖顺序更新的固定调度——SkillBook 先于 LLM 适配（adapter 需 prepended procedure），Router 最后（标签反映当前周期 student 输出）。
- **Input-only router**：仅依赖当前调用序列化 prompt 的特征做路由决策，不耦合 agent 隐藏状态或实现细节的工具 trace。
- **Break-even analysis**：将一次性适配训练成本除以每日服务节省额，得到经济回收周期；论文在金融场景下设 70% 保守 realization 因子。

## 可复现要素
- **数据集**：HumanEval + MBPP（合并，546 train / 582 held-out，disjoint identifiers）；TAU-2（35-task fixed split）；金融场景为自定义 planning case 非公开基准。**代码与数据已开源**：https://github.com/yh-yao/MERA-Evolve
- **模型**：Small student = Qwen2.5-Coder-1.5B-Instruct；Teacher/Fallback = GPT-5.6 Luna；Router embedding = 冻结 Qwen3-Embedding-0.6B；TAU-2 实验用 Qwen3.5-2B / Qwen3.5-4B。
- **训练**：4 周期 SkillBook+SFT 与 SkillBook+SFT+GRPO；GRPO 使用 VERL 框架；teacher 输出缓存；3 个独立训练 seed。
- **Router**：Class-balanced logistic-regression head on frozen embeddings；threshold calibration 在 disjoint shard；cross-validation 中重复观测归组。
- **Cost 假设**：small:large 成本比 1:10；normalized cost 相对 always-Luna。
- **论文未提及**：具体 LoRA rank/alpha/learning rate 数值；GRPO 的 entropy coefficient、clip range、rollout 长度；SkillBook exemplar 上限的具体数字。
