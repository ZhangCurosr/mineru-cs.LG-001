---
title: "Policy-as-Logic-for-Robust-Reasoning-over-Rules"
source: https://arxiv.org/pdf/2608.11905v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:33:29"
field: "神经符号推理"
keywords: ["Policy-as-Logic", "Answer Set Programming", "Neurosymbolic AI", "Robust Reasoning", "Policy Compliance"]
innovations: ["提出PaL框架分离LLM事实提取与ASP符号推理", "非单调逻辑支持不完整输入下的默认推理", "在客观规则场景下实现~10倍token节省与显著提升的鲁棒性"]
benchmarks: ["RuleArena", "PolyGuard"]
---

# 论文速读：Policy-as-Logic-for-Robust-Reasoning-over-Rules

## 一句话总结
本文提出 Policy-as-Logic (PaL) 方法，将策略以形式逻辑（Answer Set Programs）编码，通过分离 LLM 事实提取与符号求解器的推理步骤，实现可解释、可审计且对输入扰动鲁棒的自动化决策。在航空、税务、NBA 等客观规则场景下，准确率显著超越 policy-as-prompt 和 policy-as-code 基线，并减少约 10 倍 token 消耗。

## 研究问题与动机
1. **LLM 策略推理可靠性不足**：端到端 LLM 方法对输入的微小扰动（如语气变化、规则重排序）敏感，输出不稳定，难以用于关键决策场景。
2. **提取与推理能力的天然分离**：LLM 擅长语境理解和表征编码，但在离散实体推理上不可靠；符号求解器擅长确定性逻辑推理但缺乏语义理解。
3. **策略类型差异**：策略可分为客观标准（如行李重量、税率计算）与主观信念（如内容审核中的"伤害意图"），前者更适合形式化表达。
4. **成本与效率需求**：现有 policy-as-prompt 方法需要将完整策略文本作为上下文，token 消耗高，难以规模化部署。

## 核心贡献（创新点）
1. **提出 PaL 框架**：将策略编码为 Answer Set Programs，利用 LLM 仅提取事实而由求解器完成推理，实现两阶段分离架构。
2. **非单调逻辑支持**：采用 ASP 的默认否定机制处理输入不完整的部分接地情况，使推理仍能输出最可能结果。
3. **显著降低 token 消耗**：推理时只需传入 schema 而非完整策略文本，token 使用量减少约 10 倍。
4. **系统化的鲁棒性评估**：通过 6 种语言重述扰动（verbose、paraphrase、distraction 等）量化评估决策一致性。

## 方法详解
**Pipeline 共五步：**

1. **Semantic Parsing**：使用 LLM（Claude Opus 4.7）将策略文档翻译为 ASP 程序，编码谓词、条件、异常和结果。此步骤一次性完成，生成 schema 及映射规则。
2. **Extraction**：推理时，LLM 仅根据 schema 从用户查询中提取结构化事实（JSON 格式），不接触完整策略文本。
3. **Grounding**：将提取的事实通过预生成映射转换为命题原子，处理类型转换（如布尔值到命题原子、数值缩放），得到无变量的接地 ASP。
4. **Solver**：使用 Clingo 求解器计算接地程序的所有稳定模型（answer sets），即兼容的决策集合。必要时加入优化指令消除歧义（如选择最小代价方案）。
5. **Interpretation**：将输出原子映射回领域决策（如单位换算、分类标准化）。

**关键设计**：使用非单调逻辑支持默认推理； grounding 后搜索空间与谓词数同量级，不引入额外延迟；答案集大小通常≤3。

## 实验与结果
**数据集与基准**：
- **RuleArena**：Airline（n=300）、Tax（n=300）、NBA（n=216）三个领域
- **PolyGuard**：HR 内容审核（n=300）

**基线方法**：
- policy-as-prompt：0-shot 和 1-shot，测试 GPT-OSS-120B、Qwen-2.5 72B、Llama-3.3 70B、Granite-4.1 8B
- policy-as-code：引用 Dou et al. 2026a 报告数据

**核心结果**：
| 领域 | PaL 准确率范围 | 最佳基线准确率 | 提升幅度 |
|------|---------------|--------------|---------|
| Airline | 0.94–1.00 | 0.38 (0-shot) | +56%~+62% |
| Tax | 0.31（所有模型一致） | 0.10 (GPT-OSS 1-shot) | +21% |
| NBA | 0.48–0.50 | 0.39 (Qwen-2.5 0-shot) | +9%~+11% |
| HR | 0.93–0.97 | 0.96 (多数基线) | 持平或略低 |

**鲁棒性结果**：
- Tax 领域所有基线鲁棒性降至 0.00，PaL 保持 0.24–0.29
- Airline 领域 PaL 鲁棒性 0.93–0.98，与准确率基本一致

**Token 效率**：PaL 方法 token 消耗仅为 policy-as-prompt 的 1/10 左右（如 Airline 领域从 11,684 降至 1,175）

## 相关工作脉络
1. **LINC [Olausson et al., 2023]**：从输入推导一阶逻辑并用定理证明器推理；本文聚焦真实策略而非纯逻辑推理任务。
2. **Yang et al. [2023]**：LLM 语义解析 + hand-crafted ASP 程序；本文策略由 LLM 自动生成，适应更广泛领域。
3. **Pan et al. [2023] (Logic-LM)**：添加自修正模块改进逻辑推理；本文聚焦现实世界策略推理而非抽象逻辑任务。
4. **Hoveyda et al. [2026] (OrLog)**：概率推理处理排除、否定等特殊结构；本文使用确定性 ASP 解决客观规则场景。
5. **policy-as-prompt [Palla et al., 2025]**：将策略文本作为上下文输入 LLM；本文通过逻辑分离克服 token 瓶颈和鲁棒性问题。
6. **policy-as-code [Dou et al., 2026a]**：将策略翻译为可执行代码；本文使用声明式逻辑编程，更具可解释性和审计性。

## 局限性与未来方向
1. **主观策略效果有限**：HR 内容审核等依赖信念的判断场景，PaL 无明显优势，因提取步骤误差主导结果。
2. **LLM 到 ASP 翻译不完全**：部分领域存在已知 gap，无法保证策略完全覆盖。
3. **数值提取错误敏感**：单 digit 错误（如 29314→29114）会传播为显著决策偏差。
4. **未来方向**：分步提取降低一次性失败风险；引入 llm-as-a-judge 缓解位置偏差；构建从原子到原文的错误追溯诊断。

## 研究启发与可借鉴点
1. **提取-推理分离架构**：将 LLM 的语义理解与符号求解器的确定性推理解耦，可迁移至任何涉及规则决策的场景。
2. **非单调逻辑处理不确定性**：ASP 的默认否定机制可用于处理输入不完整的部分观测场景。
3. **Schema 驱动的事实提取**：仅让 LLM 提取结构化事实而非直接推理，降低幻觉风险并节省 token。
4. **鲁棒性评估框架**：六种语言扰动 + llm-as-a-judge 语义守恒验证，可作为策略推理系统评估的标准流程。
5. **小模型可行性**：即使 Granite-4.1 8B 等中小模型配合 PaL，在 Airline 领域仍达 0.61 准确率，显著优于零样本基线的 0.01。

## 关键术语表
**Policy-as-Logic (PaL)**：将策略编码为形式逻辑程序，通过分离事实提取与符号推理实现稳健决策的方法。
**Answer Set Programming (ASP)**：一种非单调逻辑编程范式，支持默认推理和稳定性语义，由 Clingo 等求解器实现。
**Grounding**：将含变量的逻辑程序转换为无变量的命题形式，以便求解器处理。
**Stable Model / Answer Set**：接地程序的所有兼容原子集合，代表满足策略规则的有效决策。
**Default Negation**：ASP 中的"not"算子，表示某事实未被证明为真时默认为假。
**Policy-as-Prompt**：将完整策略文本作为上下文直接输入 LLM 的决策范式。
**Policy-as-Code**：将策略和输入翻译为可执行程序执行的决策范式。
**Robustness (Rob)**：对输入扰动保持决策一致性的能力，本文定义为 6 种扰动下正确回答的比例。

## 可复现要素
- **数据集**：RuleArena 和 PolyGuard，论文中提供了具体样例和规模（Airline n=300, Tax n=300, NBA n=216, HR n=300）
- **代码开源状态**：论文未提及代码开源
- **模型**：主实验使用 Claude Opus 4.7（策略解析）、GPT-OSS-120B/Qwen-2.5-72B/Llama-3.3-70B/Granite-4.1-8B（提取与基线）
- **求解器**：Clingo v5.8.0
- **关键超参**：ASP 程序生成 prompt 见 Appendix 6.1；6 种扰动类型（verbosity, paraphrase, distraction, misleading context, cheerful sentiment, frustrated sentiment）
- **评估指标**：Accuracy（精确匹配 ground truth）、Robustness（扰动一致性）、Token 消耗
