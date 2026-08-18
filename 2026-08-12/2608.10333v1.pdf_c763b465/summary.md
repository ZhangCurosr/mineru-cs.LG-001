---
title: "MERA: Model Evolution and Routing with Skill Adaptation for Agentic Systems at Scale"
source: https://arxiv.org/pdf/2608.10333v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 03:37:50"
---

# 论文速读：MERA: Model Evolution and Routing with Skill Adaptation for Agentic Systems at Scale

## 一句话总结
MERA 提出了一种以单次模型调用为适配单元的 verifier-backed 多周期演化协议，通过回放失败轨迹、沉淀 SkillBook、微调小模型 LoRA 并结合输入仅依赖的路由与回退机制，在降低推理成本的同时实质性提升小模型自身能力；在 HumanEval+MBPP 上使 Qwen2.5-Coder-1.5B 的 pass 率从 28.7% 提升至 49.7%，部署策略以 60.8% 的成本保留 88.3% 的质量。

## 研究问题与动机
- LLM Agent 的单次任务内包含异构的模型调用链（复杂推理 vs 格式化/工具参数构建等结构化步骤），全用大模型可靠但昂贵，现有路由仅将简单调用分流至小模型，无法突破小模型固有能力上限，可节省成本存在硬边界。
- 生产环境直接基于原始在线轨迹变更 Serving 策略风险较高：廉价模型可能静默失败，所学适配器可能在稀有 case 上退化，可复用模板仅在窄 prompt 签名下安全，需要“观察”与“准入”解耦的保守机制。
- 现有方法的决策粒度多在用户请求或单轮 prompt 级别，而同一工作流中简单与困难步骤常并存，需要以单次调用为粒度进行适配、路由与技能提炼。
- 缺乏将执行痕迹同时用于 SkillBook 统计、调用级路由与小模型更新的系统级协议，导致路由、技能与模型适配往往孤立优化，难以保证端到端部署质量。

## 核心贡献（创新点）
- 提出以单次模型调用为基本适配单元的演化协议，首次将小模型能力进化置于核心位置，而非仅将小模型作为路由的被动承载。
- 设计依赖有序的 Skill → LLM → Router 调度更新流程，避免 stale supervision：SkillBook 先沉淀稳定过程，LLM 适配器在最新技能上下文下微调，路由器最后贴合当期小模型表现。
- 构建基于联合回放（joint replay）的保守准入机制，只有当组合策略（路由+技能+适配器）在回放中保持或提升任务质量时才 admit，实现观察与上线的严格分离。
- 提供算法收益与部署成本的联动评估框架，结合可执行 verifier 与财务盈亏平衡分析，实证表明演化后的小模型可在显著低于始终调用大模型的代价下维持高质量。

## 方法详解
- **运行时路由与验证链路**：路由器仅观察当前调用的序列化 prompt，在强模型、廉价模型与可选专业学生模型间选择；Skill selector 可分发稳定模板（格式化、抽取、单工具参数构建）；Verifier 检查 schema 合法性、工具调用合规性、可执行测试或下游成功；验证失败则 fallback 至更强模型并完整记录事件。
- **轨迹切片与多流证据**：将完整 trace 标准化为 step slice，包含 prompt、局部上下文、tool schema、生成输出、verifier 结果、重试计数、fallback 元数据与 skill 分配；同一 slice 可分化支持 SkillBook 统计、路由训练与 LLM 微调，不强制统一监督格式。
- **SkillBook（外部过程性记忆）**：非响应缓存或权重，按数据集级签名（humaneval/mbpp）聚合高频调用模式；每次渲染组合静态任务格式指令、有界成功示例与示例驱动的常见陷阱/模式，前置到新任务 prompt，永不返回相同 prompt 的存储答案。
- **LLM 适配器训练**：从廉价执行失败或路由-SkillBook 分歧的 slice 中挑选 hard examples，将当前 SkillBook 程序前置后对 Qwen 系列进行 SFT；可选叠加 GRPO 强化优化，教师输出缓存以确保对比公平。
- **路由训练与标签**：使用冻结的 Qwen3-Embedding-0.6B prompt 特征 + 类别平衡逻辑回归头；标签由可执行小模型结果严格生成（cheap-eligible 当且仅当当前小模型 rollout 通过 benchmark verifier）；重复观测同组 cross-validation，阈值在校验 shard 确定，策略评估在 held-out task identifiers 进行。
- **联合准入（Joint Admission）**：SkillBook/路由器/verifier 证据划分 easy/hard/uncertain 区域；新 router/skill/adapter 仅在 replay 证明联合质量不下降且成本或 fallback 风险优化时 admitted；replay 仅作预部署门控，生产上线仍需 shadow/canary 验证、漂移监控与回滚。

## 实验与结果
- **数据集与设置**：HumanEval + MBPP 合并为 546 train / 582 held-out tasks（task identifier 不重叠，零 exact prompt 重叠）；SLM 为 Qwen2.5-Coder-1.5B-Instruct，Teacher/Fallback 为 GPT-5.6 Luna；small:large cost ratio = 1:10。TAU-2 使用官方固定 35-task split。
- **多周期小模型适配**（Table 1，三轮均值）：Base SLM 28.7% → Multi-cycle SFT 44.2% → SFT+GRPO 49.7%；GRPO 较 SFT 提升 5.5–6.7 分（95% paired-t 区间不含零）。
- **Verifier-backed 部署策略**（Table 2）：MERA router + verifier fallback 达 88.3% pass，成本仅为 always-Luna 的 60.8%；对比 RouteLLM-style + fallback (87.0%, 97.4%) 与 FrugalGPT-style response cascade (85.5%, 106.7%)。最强结果为 SFT+GRPO 直接适配后的联合部署策略，相对始终调用大模型节省近 40% 成本且质量持平。
- **TAU-2 适配消融**（Table 3）：Qwen3.5-2B 从 14/35 提升至 18/35，达到未适配 Qwen3.5-4B (17/35) 的水平；McNemar 检验 p=0.1719，属功效不足但可行性验证。
- **财务盈亏平衡**（Table 4）：500 行适应条件下名义盈亏平衡 8.68 天，保守（70% 收益实现率）12.47 天。

## 相关工作脉络
- **LLM Routing**（FrugalGPT, Routellm, BEST
