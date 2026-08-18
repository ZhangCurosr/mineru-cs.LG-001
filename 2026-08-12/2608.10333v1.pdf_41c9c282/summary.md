---
title: "MERA: Model Evolution and Routing with Skill Adaptation for Agentic Systems at Scale"
source: https://arxiv.org/pdf/2608.10333v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 03:15:12"
field: "LLM Agent 系统与高效推理"
keywords: ["Agent 系统", "模型演进", "多周期适配", "验证器回退", "SkillBook", "LLM 路由"]
innovations: ["以单个模型调用为适配单元的多周期演进协议", "Skill→LLM→Router 依赖感知调度与联合回放准入机制"]
benchmarks: ["HumanEval+MBPP", "TAU-2"]
---

# 论文速读：MERA: Model Evolution and Routing with Skill Adaptation for Agentic Systems at Scale

## 一句话总结
MERA 提出了一种**验证器驱动的多周期演进协议**，以单个模型调用为适配单元，将运行时轨迹数据用于更新 SkillBook、训练小模型适配器并优化路由器，使小模型能力本身得到提升（而非仅靠路由规避）。在 HumanEval+MBPP 上，四周期适配将 Qwen2.5-Coder-1.5B 直接通过率从 28.7% 提升至 49.7%，结合验证器回退策略后，部署策略以 60.8% 的大模型成本实现了 88.3% 通过率。

## 研究问题与动机
1. **Agent 推理链的异质性未被充分利用**：单次任务包含大量结构化步骤（格式整理、工具调用构造等），这些步骤与困难推理并存，但现有路由方法仅在用户请求级别做粗粒度分发。
2. **现有方法无法提升小模型能力**：传统路由（如 FrugalGPT、RouteLLM）将简单调用分配给小模型，但小模型能力固定不变，可节省成本的上限受限于小模型本身已有的能力。
3. **缺乏将在线轨迹转化为安全演进的系统性机制**：生产系统中直接使用原始轨迹更新服务策略存在风险——便宜模型可能静默失败、适配器可能在罕见案例上退化、技能模板仅在特定提示签名下安全。
4. **缺失连接路由、技能与模型更新的系统层**：现有工作各自独立研究路由、技能库或后训练优化，但缺乏一个保守机制将这些组件在运行的 agent 内部连接起来，并通过回放验证共同效果。

## 核心贡献（创新点）
1. **以单个模型调用为适配单元的演进协议**：不同于只路由整个用户请求，MERA 将执行轨迹切分为 step slice，在每个周期内用失败调用获取教师演示并蒸馏可复用程序。
2. **Skill → LLM → Router 依赖感知的调度顺序**：适配器先用当前 SkillBook 程序进行训练，路由器最后拟合以反映当前周期的小模型结果，避免并行训练导致的监督信号过时。
3. **联合回放准入机制（Joint Admission）**：新路由器、技能状态或适配器只有在联合回放中保持任务质量时才被采纳，防止孤立组件指标误导部署决策。
4. **验证器背书的成本-质量权衡**：通过可执行验证 + 强模型回退，在 60.8% 的大模型成本下实现 88.3% 通过率，相比 RouteLLM/FrugalGPT 基线显著降低成本。
5. **财务盈亏分析框架**：将一次性适配成本与日常服务节省挂钩，给出保守场景下的盈亏平衡天数（500 行适配在 70% 节省实现率下约 12.47 天）。

## 方法详解
1. **运行时路由（Input-only Router）**：路由器仅观察当前调用的序列化 prompt，从强模型、便宜模型和可选专业学生模型中选择；技能层可分发稳定模板处理重复局部结构（格式化、提取等）。
2. **轨迹切片与证据流**：每个运行事件被规范化为 step slice，包含 prompt、本地上下文、工具 schema、生成输出、验证结果、重试次数、回退元数据。三种证据流分别用于：
   - SkillBook：记录重复 prompt 签名成功/失败统计
   - 路由器：用便宜/强标签训练，标签来自可执行小模型结果
   - LLM 适配器：从便宜执行失败或路由/SkillBook 分歧的困难样本训练
3. **调度依赖**：Skill → LLM → Router 顺序确保适配器训练时 SkillBook 是最新的，路由器标签反映当前周期结果。
4. **SkillBook 设计**：外部程序性 prompt 记忆（非响应缓存或权重），按 humaneval/mbpp 数据集签名组织，每个条目组合静态任务格式指令与有界成功示例及示例驱动的常见陷阱。
5. **路由器实现**：冻结 Qwen3-Embedding-0.6B 提取 prompt 特征，类平衡逻辑回归分类头；重复观察归入同交叉验证组，阈值校准使用独立划分。
6. **验证器**：HumanEval/MBPP 使用隔离 Python 子进程执行测试；TAU-2 使用官方 evaluator 检查环境状态、工具动作和自然语言断言。
7. **联合准入**：更新仅在回放中保留质量时才被提升；不安全区域保留给强模型回退。

## 实验与结果
1. **HumanEval+MBPP 主实验（582 个 held-out 任务，3 seeds）**：
   - Base SLM (Qwen2.5-Coder-1.5B)：28.7% pass，成本 10%
   - Multi-cycle SFT：44.2% pass
   - Multi-cycle SFT+GRPO：49.7% pass（比 SFT 提升 5.5-6.7 点）
   - Always Luna：86.9% pass，成本 100%
2. **部署策略对比（Table 2）**：
   - MERA router + verifier fallback：**88.3% pass at 60.8% cost**
   - RouteLLM-style + fallback：87.0% pass at 97.4% cost
   - FrugalGPT-style cascade：85.5% pass at 106.7% cost
3. **TAU-2 适配器消融（35-task split）**：
   - Base Qwen3.5-2B：14/35 (40.0%)
   - 训练后 Qwen3.5-2B + GRPO：18/35 (51.4%)
   - 匹配未适配 Qwen3.5-4B：17/35 (48.6%)
   - McNemar p=0.17，证据力不足但可行性成立
4. **财务盈亏分析**：
   - 300 行适配：盈亏平衡 15.29 天（保守 31.96 天）
   - 500 行适配：盈亏平衡 8.68 天（保守 12.47 天）

## 相关工作脉络
1. **FrugalGPT / RouteLLM**：基于提示特征或预测质量做单轮路由，决策单元是用户请求而非内部调用；MERA 将路由粒度细化到单调用并配合验证器回退。
2. **React / Toolformer /ToolLLM**：提升 agent 的工具使用和规划能力，但视模型选择、路由和技能为固定工程选择；MERA 直接用执行轨迹更新这些组件。
3. **AgentGym-RL / AutoPDL**：通过 RL 或提示优化改进 agent 策略，但不解决部署中如何安全演进模型组合和路由策略的问题。
4. **Distillation / SkillX**：知识蒸馏和技能库构建通常报告为独立训练结果；MERA 将技能作为可演进的 prompt 记忆，并通过联合回放验证其安全性。
5. **WebShop / TAU-bench / SWE-bench**：提供现实工具使用和代码生成评估环境；MERA 利用这些 benchmark 的可执行验证器作为准入 Gate。
6. **Router-R1 / BEST-route**：多轮路由和聚合策略；MERA 的路由器是输入仅依赖的简单分类器，可靠性由事后验证器保证而非事前预测。

## 局限性与未来方向
1. **验证器覆盖度限制**：若验证器遗漏语义失败模式，回放可能高估降路由或技能推广的安全性。
2. **长程推理覆盖不足**：当前设计偏向窄而可检查的步骤，困难长程推理可能长期保留给强模型。
3. **回放无法完美捕捉分布偏移**：改变运行时策略会引入分布偏移，回放仅是预部署门控，生产仍需 shadow/canary 验证和漂移监控。
4. **技能泛化受限**：技能推广假设重复局部子图可被识别，但高度异构的任务可能无法从中受益。
5. **预路由器效果弱**：大部分质量保留来自验证器回退而非准确的前置路由， Learned router 能力有限。
6. **跨域证据不足**：最强多 seed 证据在代码生成领域；TAU-2 比较证据力不足，非平稳工作负载下的增益待验证。

## 研究启发与可借鉴点
1. **轨迹切片与证据分离**：将运行轨迹切分为 step slice 并提取多种监督信号（SkillBook 统计、路由标签、适配样本），同一 slice 支撑不同更新轨道而无需共享监督格式。
2. **依赖感知调度**：Skill → LLM → Router 的顺序设计避免了监督信号过时问题，为多组件联合训练提供了可复用的调度范式。
3. **联合回放准入机制**：用回放验证代替孤立组件指标决定部署，防止"看起来好但组合失败"的情况，可作为 agent 系统安全演进的标准实践。
4. **财务盈亏分析框架**：将适配成本与服务节省量化挂钩，为团队决策是否投入 adaptation 资源提供可操作的评估标准。
5. **输入仅依赖路由器**：避免路由决策耦合隐藏 agent 状态，提升跨 harness 部署可行性；可靠性由事后验证器保证而非事前预测精度。

## 关键术语表
**MERA**：Model Evolution and Routing with Skill Adaptation，一种以验证器驱动的多周期演进协议，用于大规模 Agent 系统。
**SkillBook**：外部程序性 prompt 记忆，记录重复调用模式的成功/失败统计和可复用程序，作为适配器的输入上下文。
**Step Slice**：轨迹规范化后的基本单元，包含 prompt、上下文、工具 schema、输出、验证结果等元数据。
**Joint Admission**：联合准入机制，只有当路由器、技能状态和适配器的组合在回放中保持质量时才被部署。
**Verifier-backed Fallback**：验证器背书回退，当便宜模型输出未通过验证时自动升级到强模型。
**SFT+GRPO**：监督微调结合 Group Relative Policy Optimization，用于小模型适配器训练。
**MCQ (Multi-choice Question)**：此处指 TAU-2 benchmark 中的多工具交互任务，用于评估 agent 的 tool-use 能力。
**Break-even**：盈亏平衡点，一次性适配成本被日常服务节省回收所需的时间。

## 可复现要素
- **数据集**：HumanEval+MBPP（合并 546 train / 582 held-out）、TAU-2（35-task split）、Finance priority-data（内部，未公开）
- **代码开源**：https://github.com/yh-yao/MERA-Evolve
- **模型**：Qwen2.5-Coder-1.5B-Instruct（学生）、GPT-5.6 Luna（教师/回退）、Qwen3-Embedding-0.6B（路由特征提取）
- **关键超参**：4 周期适配、小:大模型成本比 1:10、SFT+GRPO 联合训练、Frozen embedding + logistic regression 路由器
- **评估协议**：3 seeds 均值、disjoint task identifiers、no exact prompt overlap、isolated Python subprocess 执行验证
