---
title: "Policy-as-Logic-for-Robust-Reasoning-over-Rules"
source: https://arxiv.org/pdf/2608.11905v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:33:45"
field: "神经符号推理"
keywords: ["Policy-as-Logic", "Answer Set Programming", "Neurosymbolic AI", "Robust Reasoning", "Policy Reasoning", "LLM Extraction"]
innovations: ["提出PaL框架，分离LLM事实提取与ASP符号推理以提升策略推理鲁棒性", "实证客观策略场景下PaL较policy-as-prompt准确率提升62pp且鲁棒性接近准确率", "揭示主观策略场景下形式逻辑推理价值有限的方法边界"]
benchmarks: ["RuleArena", "PolyGuard"]
---

# 论文速读：Policy-as-Logic-for-Robust-Reasoning-over-Rules

## 一句话总结
本文提出 **Policy-as-Logic (PaL)** 框架，将自然语言政策编码为 Answer Set Program (ASP) 形式逻辑程序，通过 LLM 提取事实、符号求解器执行确定性推理的两阶段架构，在航空行李费、税务计算、NBA交易规则等客观策略场景中实现高精度（最高 1.00）与高鲁棒性，同时 token 消耗降低约 10 倍。

## 研究问题与动机
- **LLM 端到端策略推理缺乏鲁棒性**：细微输入扰动（措辞变化、规则重排）即可改变决策结果，且生成式模型的随机性导致决策不可审计。
- **策略类型存在主客观光谱**：客观策略（如重量计费、薪资上限）依赖可验证知识，主观策略（如内容审查中的"有害意图"）依赖模型世界观判断；现有方法未针对此差异设计。
- **Policy-as-prompt 的上下文膨胀**：完整政策文本作为 prompt 上下文会导致 token 消耗巨大，且小模型表现急剧下降。
- **Policy-as-code 的实现成本高**：手动编写代码或让 LLM 生成可执行程序在复杂策略场景下准确率低且维护困难。

## 核心贡献（创新点）
1. **提出 Policy-as-Logic (PaL) 范式**：将策略表示为非单调逻辑程序（ASP），通过严格分离"LLM 事实提取"与"符号求解器推理"两个阶段，使决策过程可解释、可审计。
2. **实证结构化推理对客观策略的鲁棒性增益**：在 RuleArena 三个客观领域（Airline/Tax/NBA）上，PaL 较 policy-as-prompt 提升显著（Airline Acc: 0.38→1.00，Tax Rob: 0.00→0.29），且鲁棒性损失仅来自提取步骤。
3. **揭示主观策略的局限边界**：在 PolyGuard HR 主观分类任务中，PaL 无系统优势，证明形式逻辑对信念型策略的价值有限，为方法适用边界提供实证依据。
4. **展示 token 效率优势**：PaL 仅需 schema 而非完整政策文本，平均 token 消耗降低约一个数量级（如 Airline: 11,684→1,175）。

## 方法详解
PaL 框架包含五个核心模块，流水线如图 1 所示：

1. **Semantic Parsing（策略→ASP）**：使用 LLM（Claude Opus 4.7）将自然语言政策文档翻译为 Answer Set Program，编码策略中的谓词、条件、例外与结果。此步骤一次性完成，生成策略逻辑程序 P 及 fact extractor 所需的 schema。

2. **Extraction（查询→事实）**：推理时，LLM 仅根据 schema 提取用户查询 x 的结构化事实（JSON 格式），不直接看到完整政策文本。提取结果为一个事实列表。

3. **Grounding（事实→命题逻辑）**：将提取事实通过映射表转换为 ASP 原子，处理类型转换（如布尔值→命题原子、数值缩放如 cents），得到无变量的命题逻辑程序。

4. **Solver（稳定模型求解）**：使用 Clingo 求解器计算 grounded ASP 的所有 stable models（answer sets）。对于规则歧义场景（如多个行李哪个免费），可添加 optimization directives 选择最优解。

5. **Interpretation（决策映射）**：将输出原子映射回领域决策（数值缩放、类别标准化等）。

**关键设计点**：
- 采用**非单调逻辑**（default negation `not`），支持不完整输入下的默认推理。
- 求解器输出稳定模型集合 M，当 |M|>1 时按成本最小化等准则选择推荐决策。
- 误差分析表明，失败主要源于提取步骤：默认分类错误（Airline）、数字转录错误（Tax）、跨字段混淆（NBA）、关键词触发忽略立场（HR）。

## 实验与结果
**数据集与基线**：
- 四个领域：Airline（n=300）、Tax（n=300）、NBA（n=216，来自 RuleArena）；HR（n=300，来自 PolyGuard）。
- 基线：policy-as-prompt（0-shot/1-shot，GPT-OSS-120B/Qwen-2.5 72B/Llama-3.3 70B/Granite-4.1 8B）、policy-as-code（引用文献数值）。
- 评估指标：准确率（Acc）、鲁棒性（Rob，六种 LLM 扰动后的平均正确率）。

**主要结果**：

| 领域 | 最佳 Acc | 最佳 Rob | PaL vs 基线提升 |
|------|----------|----------|-----------------|
| Airline | 1.00 (GPT-OSS 120B) | 0.98 | vs 0.38 (+62pp) |
| Tax | 0.31 | 0.29 | vs 0.10 (+21pp)，基线 Rob 全为 0 |
| NBA | 0.50 (Qwen-2.5) | 0.48 | vs 0.25 (+25pp) |
| HR | 0.96 | 0.94 | 与基线无显著差异 |

**关键结论**：
- 客观策略（Airline/Tax/NBA）：PaL 大幅超越基线，且鲁棒性接近准确率，证明求解器提供确定性保障。
- Tax 领域基线鲁棒性全部跌至 0.00，PaL 保持 0.24-0.29。
- 主观策略（HR）：PaL 无系统优势，支持"逻辑推理对信念型策略价值有限"的论断。
- Token 效率：PaL 输入 token 从数千~两万降至 895-3,537，总消耗降低约 10x。
- Answer set 大小：多数领域 |M|=1，Airline 有 30% 案例 |M|=2-3（需优化指令消歧）。

## 相关工作脉络
1. **LINC (Olausson et al., 2023)**：从输入推导一阶逻辑并用定理证明器推理；本文聚焦政策场景且使用 ASP 而非定理证明。
2. **Yang et al. (2023) LLM-ASP pipeline**：首个结合 LLM 语义解析与 ASP 推理的工作，但程序为手工编写；本文自动从政策文本生成 ASP。
3. **Logic-LM (Pan et al., 2023)**：添加自纠正模块改进逻辑推理；专注纯逻辑任务，本文研究真实世界政策。
4. **OrLOG (Hoveyda et al., 2026)**：概率推理处理排除/否定等结构；本文使用确定性 ASP 适用于客观策略。
5. **Policy-as-prompt (Palla et al., 2025)**：主流范式，将政策文本作为上下文；本文通过 schema 压缩上下文，实现 10x token 节省。
6. **Policy-as-code (Dou et al., 2026a)**：策略翻译为可执行程序；本文用声明式逻辑替代命令式代码，更易表达策略语义。

## 局限性与未来方向
- **主观策略收益有限**：HR 内容审查等依赖信念的判断，提取误差直接传导至决策，求解器无法补偿。
- **提取步骤是误差瓶颈**：数字转录错误、跨字段混淆、立场理解缺失等提取失败会导致决策翻转。
- **ASP 覆盖不完全**：LLM 生成的策略程序可能存在已知 gap，无法覆盖所有 edge case。
- **未来方向**： disaggregate extraction calls、利用 llm-as-a-judge 缓解 positional bias 和 criteria phrasing 问题、开发从命题原子回溯原文的错误分析工具、探索 human-in-the-loop 补充缺失属性。

## 研究启发与可借鉴点
1. **提取-推理分离架构**：对于任何"规则明确+事实需从文本提取"的场景（合规审查、定价计算、资格判定），可复用此两阶段设计，用符号求解器保证推理确定性。
2. **Schema 驱动的 token 压缩**：用结构化 schema 替代完整政策文本作为 prompt，可实现数量级的 token 节省，适合部署到小模型或成本敏感场景。
3. **鲁棒性评估协议**：采用六种语言重述扰动 + llm-as-a-judge 语义保留验证，为策略推理任务提供标准化鲁棒性度量。
4. **非单调逻辑处理不完整信息**：ASP 的 default negation 支持部分事实下的默认推理，适合真实查询缺失属性的场景。
5. **适用边界意识**：主客观策略需区分处理；客观规则场景优先采用 PaL，主观判断场景可考虑混合架构或纯 LLM 方案。

## 关键术语表
- **Answer Set Programming (ASP)**：一种基于稳定模型语义的非单调逻辑编程范式，支持默认推理与规则歧义处理。
- **Policy-as-Logic (PaL)**：本文提出的框架，将策略编码为形式逻辑程序，分离事实提取与符号推理。
- **Grounding**：将含变量的逻辑程序实例化为无变量命题逻辑程序的过程。
- **Stable Model / Answer Set**：ASP 程序的语义解释，满足所有规则的最小原子集合。
- **Default Negation (not)**：ASP 中的否定形式，表示"无法证明为真则默认为假"。
- **Policy-as-prompt**：将完整政策文本作为上下文注入 LLM prompt 的主流方法。
- **Policy-as-code**：将策略翻译为可执行代码（如 Python）的方法。
- **RuleArena / PolyGuard**：本文使用的两个评测基准，分别侧重离散规则推理与多语言内容审查策略。

## 可复现要素
- **数据集**：RuleArena (Airline/Tax/NBA)、PolyGuard (HR)；论文未声明开源，但引用原始工作。
- **代码**：论文未提及代码开源。
- **权重/模型**：PaL 使用 Claude Opus 4.7（策略生成）+ 开源模型（GPT-OSS-120B/Qwen-2.5 72B/Llama-3.3 70B/Granite-4.1 8B）；Clingo v5.8.0 求解器。
- **关键超参**：未显式列出，提示模板见 Appendix 6.1（policy_text/output_description/domain_notes 三字段）。
- **Token 统计**：使用 o200k_base tokenizer 统一测量。
