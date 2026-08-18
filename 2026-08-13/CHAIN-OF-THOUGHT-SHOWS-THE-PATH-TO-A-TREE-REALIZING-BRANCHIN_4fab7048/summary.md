---
title: "CHAIN-OF-THOUGHT-SHOWS-THE-PATH-TO-A-TREE-REALIZING-BRANCHIN"
source: https://arxiv.org/pdf/2608.11716v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 07:32:01"
field: "Transformer 可表达性与计算复杂性理论"
keywords: ["Chain-of-Thought", "Transformer Expressivity", "Graph Traversal", "Strahler Number", "Dyck Path", "Branching Complexity", "NC¹", "Hard Attention"]
innovations: ["首次给出 DFS 和 Dijkstra 在有界深度唯一定向注意力 Transformer 上的 CoT 显式构造", "以共用的遍历基底在 O(n) 步内计算任意 n 叉树的 Strahler number 和 tree width", "在树↔Dyck 路径双射两侧独立构造分支复杂度指标并揭示跨表示不可直接迁移"]
benchmarks: ["theoretical construct (no empirical benchmark)"]
---

# 论文速读：CHAIN-OF-THOUGHT SHOWS THE PATH TO A TREE: REALIZING BRANCHING COMPLEXITY

## 一句话总结
本文首次给出了基于唯一定向注意力（unique hard-attention）的有界深度 Transformer 解码器在 CoT 框架下对 DFS、Dijkstra 算法的显式构造，并以此作为共享计算基底，在无需层归一化和位置编码的极简架构下，分别用 4 层（DFS 基底）和 3 层（Dijkstra 基底）解码器在 O(n) 步内计算 n 叉树的 Strahler number 和 tree width，同时利用树–Dyck 路径的经典双射给出了两条独立路径上的等价实现。

## 研究问题与动机
- **CoT 表达能力缺乏具体构造**：Merrill & Sabharwal（2024）、Barceló 等（2025）已从理论上证明 CoT 能让有界深度 Transformer 突破单次前向传播的复杂度天花板（如线性步数对应 NC¹），但仅给出"存在性"结果，未给出显式的、深度受限的遍历算法构造。
- **图遍历在 CoT 设定下尚未被显式实现**：Giannou 等（2023）、De Luca & Fountoulakis（2024）用 looped Transformer 实现了 DFS/BFS/Dijkstra，但属于非自回归循环模型；Zhu 等（2026）只在经验上研究了连续 CoT 的可达性问题，均未在"步骤计数的自回归 CoT"范式下给出显式构造。
- **分支复杂度是 NC¹ 完全性的典型候选问题**：二叉树的 Strahler number 计算已被 Ganardi & Lohrey（2026）证明为 NC¹-complete，而 EH rank 与 Strahler number 在二叉树上满足相同递归更新（Dahiya & Mahajan, 2021），这正好落在线性步数 CoT 刻画的目标复杂度类，具有非平凡的验证价值。
- **树–Dyck 路径双射下的跨表示一致性问题未被研究**：给定由显式算法化实现的双射 φ/ψ，在两种表示下计算同一导出量时，CoT 实现是否可以跨表示迁移，这是一个此前未被提出的新问题。

## 核心贡献（创新点）
1. **首次给出 DFS 和 Dijkstra 的 CoT 显式构造**：DFS 用 2 层 2 头、O(|V|+|E|) 步实现；Dijkstra 用 2 层 1 头、|V|−1 步实现，且 Dijkstra 实现中以常数时间 attention 替代了传统搜索最小距离顶点的操作。
2. **以共用基底给出 Strahler number 和 tree width 的 CoT 实现**：复用 DFS 解码器在 2n−1 步、4 层内计算任意 n 叉树的 Strahler number（不依赖层归一化）；复用 Dijkstra 解码器在 n−1 步、3 层内计算树宽，是对 Merrill–Sabharwal 线性步数框架的一次非平凡具体实例化。
3. **利用树–Dyck 路径双射给出另一独立实现路线**：DFS 的遍历过程本身即编码了从树到 Dyck 路径的双射 φ；基于此，论文又分别在路径表示下独立构造了 Strahler number（4 层 1 头，n 步）和 width（2 层 1 头，n 步）的计算，并证明两套实现无法在不改变机制或层数的情况下直接跨双射迁移，揭示了表示选择对构造结构的影响。

## 方法详解
- **CoT 解码器形式化**：基于 Merrill–Sabharwal（2024）和 Barceló 等（2025）的框架，采用唯一定向硬注意力（unique hard attention, UHA），无 layer normalization、无位置编码，输入嵌入 x_i = WE(σ_i)。解码器在 t 步输出 y_t = L(x_0,...,x_{m-1}, y_0,...,y_{t-1})，形成自回归链式上下文扩展。
- **DFS 构造（Theorem 5）**：2 层 2 头 Transformer。第 1 层 FFN 通过双线性映射计算未访问邻居数 γ_t，以 ReLU 差值生成 traverse/backtrack 指示位 π_{flg₁}，并将当前顶点缓存入 buffer。第 2 层两个互斥头：Traverse 头通过 bilinear query (1−π_vis)⊙π_nbr 掩码出已访问节点，检索首个未访问邻居；Backtrack 头仅在 π_{flg₁}=0 时激活，检索当前顶点的父节点。引入时间戳标志 flg₄ 使 backtrack 头选择最近一次出现的父节点而非首次。
- **Dijkstra 构造（Theorem 6）**：2 层 1 头 Transformer。第 1 层 attention 定位当前顶点 token，FFN 对所有出边做松弛更新 Dist 向量。第 2 层通过 bilinear query (1−π_vis)⊙(λ−π_dis) 为所有未访问顶点赋分，UHA 以常数时间选出最小距离未访问顶点并更新 visited 指示和 Dist。步数恰为 |V|−1。
- **Strahler number on trees（Theorem 8）**：在 Theorem 5 的 2 层基础上追加两层 FFN（attention 为空），在每条 CoT token 中维护累加器对 (M, c)，分别记录当前顶点已完成子树的最大 Strahler 值和达到该最大值的子树数量。回溯时通过 tmp 寄存器块存储 D=st(v)−M_u 以及 c_u 相关的符号指示量（利用 ReLU 恒等式：𝟙[v=0]=ReLU(v+1)−2ReLU(v)+ReLU(v−1)），在第 4 层通过双线性门控合并 (D=0 ∧ c_u>0) 条件完成 M 和 c 的更新；第 4 层末尾的 bilinear gate (1−π_{flg₁}) 负责在 traverse 步骤清零累加器。最终 π_M(y_{2n−1}) = st(G_T)。
- **Strahler number on Dyck paths（Theorem 11）**：4 层 1 头，n 步。借助 Observation 3——路径上一步回退（q_t=−1）时需同时知道当前顶点 v 和其"右边界同高水平点"对应的祖先 u 的 (M,c) 对——通过带大权重 λ 的 key 构造找到该右边界点，再复用 Theorem 8 中 (M,c) 递归结构，第 4 层以 Lemma 10 的 ReLU gate 替换双线性门控完成 traverse 抑制。
- **Tree width via Dijkstra（Theorem 13）**：在 Dijkstra 构造基础追加第 3 层 FFN。扩展三元标量 (π_dif, π_twd, π_mwd)：dif 记录相邻出队节点的 BFS 深度差（∈{0,1}），twd 累计当前层节点数，mwd 全局最大宽度。第 3 层 FFN（同 Lemma 12 思路）在 dif=0 时 td 递增并更新 mwd=max(mwd,twd)，在 dif=1 时提交当前 twd 至 mwd 并重置 twd=1。
- **Width on Dyck paths（Theorem 16）**：2 层 1 头，n 步。Embedding 维度 d=3n/2+4，分 block 存放 one-hot 位置、步 q_t、累积高度 ht、计数数组 p_i（记录深度 i 处上步次数）、宽度标量 wd。第 1 层 attention 读入 q_t，FFN（引理 14）以离散脉冲函数在 ht 索引处对 p 数组增一；第 2 层 FFN（引理 15）扫描 p 数组并做全局 max-aggregation 更新 wd=max(p_i)。
- **Bilinear map 使用**：关键的非线性操作（如 query 构造、跨块门控）均通过 Rizvi-Martel 等（2024）建立的双线性投影实现，避免引入 LayerNorm 等辅助原语。

## 实验与结果
- 本文为纯理论构造论文，无数值实验，结果全部为定理级别的正确性证明，关键数值如下：

| 任务 | 输入表示 | 层数 | 头数 | 步数 |
|---|---|---|---|---|
| DFS 遍历 | 图 G | 2 | 2 | O(|V|+|E|) |
| Dijkstra 最短路径 | 图 G | 2 | 1 | |V|−1 |
| Strahler number | 树 G_T | 4 | 2 | 2n−1 |
| Strahler number | Dyck 路径 w | 4 | 1 | n=|w| |
| Tree width | 树 G_T | 3 | 1 | n−1 |
| Width | Dyck 路径 w | 2 | 1 | n=|w| |
| Algorithm 1（路径→树重建 ψ）| Dyck 路径 w | 1 | 2 | 2(m−1) |

- **最强结论**：Strahler number 计算在 n 叉树场景下于线性步数内完成，是对 NC¹-complete 问题在 CoT 线性步数 regime 下的首个非平凡显式见证（witness）；宽度计算仅用 2–3 层即可实现，优于组合式方案（2 层 DFS + 2 层 path-width = 4 层）。
- **提升幅度（与传统算法对比）**：传统 Dijkstra 需 O(n²) 时间（或 O(|E|+n log n) 优化版），本文以 n−1 个 CoT 步完成；DFS 传统 O(|V|+|E|) 同样映射到同量级 CoT 步，但关键差异在于每一步都是单层 Transformer 计算，而非外部循环。

## 相关工作脉络
1. **Merrill & Sabharwal（2024）**：证明线性 CoT 步数对应 NC¹ 表达能力；本文将其从抽象刻画落地为 Strahler/width 等具体问题的显式构造，填补"仅有存在性、缺乏具体实现"的空白。
2. **Barceló 等（2025）**：以 EH rank 精确刻画单步 CoT 最少步数；论文自承其 bound 通过穷举树遍历达成但未给出具体遍历构造；本文直接给出了该遍历的 Transformer 实现。
3. **Giannou 等（2023）、De Luca & Fountoulakis（2024）、Sanford 等（2024）**：用 looped Transformer 模拟图算法；属于非自回归循环模型，未计入 CoT 步数；本文明确区分于这一范式，在严格自回归 CoT 设定下给出同等功能的实现。
4. **Zhu 等（2026）**：经验研究连续 CoT 能否求解可达性；未提供显式遍历过程；本文的 DFS/Dijkstra 构造可视为对该经验的理论补全。
5. **Rizvi-Martel 等（2024）**：建立双线性映射模拟加权有限状态自动机的基础；本文沿用其 bilinear map 作为基本构件，并将之扩展到树/图结构的图算法实现。
6. **Ganardi & Lohrey（2026）**：证明二叉树 Strahler number 计算为 NC¹-complete；本文将该结果推广到任意 n 叉有序树，并给出线性步 CoT 的显式算法实现。

## 局限性与未来方向
- **构造非唯一**：多种满足相同理论的替代构造可能成立；例如 Theorem 8 最后一层使用了双线性门控抑制 traverse 步骤中的冗余更新，而 Theorem 11 的证明中 Lemma 10 表明双线性可被纯 FFN 替代。
- **Strahler 计算的"最后一步"略超出 CoT 迭代**：Theorem 11 最终输出的是 M 和 c 的值，还需一步外置运算 st=M+𝟙[c≥2] 才能得到最终 Strahler number。
- **未探索 Łukasiewicz 路径表示**：平树与 Łukasiewicz 路径之间存在经典对应，本文未给出其上的 CoT 实现，可能蕴含新的架构洞察。
- **层数、头数、嵌入维度的最优性未定**：论文明确提出识别最小嵌入维度、最小层深、最少注意力头数是一个重要的未来方向。
- **跨双射的封闭性未明**：Tree↔Dyck path 双射下的 CoT 可实现在两种表示间是否保持封闭，目前仅观察到两套实现在结构上缺乏直接迁移性。
- **正则性未检验**：φ/ψ 双射在连续极限下非 Lipschitz，其离散情形的 CoT 可实现性所隐含的度量正则性尚未系统分析。

## 研究启发与可借鉴点
1. **"时间戳 key 改造左向选择为右向选择"的技巧**（Theorem 8）：通过给 key 附加递增时间戳并赋予超大权重 λ，可将 UHA 的"选最左匹配"转化为"选最新匹配"，这一技巧可直接迁移到任何需要在回溯/回退时访问最近状态而非最早状态的场景。
2. **FFN 实现离散脉冲和符号指示器**（引理 9–15）：利用 ReLU 的"分段线性"性质，以 O(1) 隐藏维度实现精确的 if-then-else 门控、移位操作和最大值聚合；这套构造块可复用于其他需要在 Transformer 内部执行离散算法（如 stack、计数器）的工作。
3. **Bilinear gate 替代 LayerNorm 实现算术门控**：本文全程不使用层归一化，而是依靠双线性乘积实现条件门控（如 traverse/backtrack 之间的互斥更新），提示了在更受约束的架构下仍可实现复杂条件逻辑的可能路径。
4. **树–路径双射的算法化实现本身即可成为 CoT 构造组件**：DFS 既完成了图遍历，又同时生成了 Dyck 路径，这种"一次遍历同时产出两种表示"的设计可作为其他组合对象的 CoT 实现的模板。
5. **跨表示不可直接迁移的负结果启示**：Theorem 8（树）与 Theorem 11（路径）虽共用同一递归式（2），但实现机制不同（双线性 vs FFN 门控、2 头 vs 1 头），提示在设计多模态或多表征统一的 CoT 框架时不能假设可简单复用。

## 关键术语表
- **Chain of Thought (CoT)**：在自回归 Transformer 解码中，将中间推理步骤显式生成为新 token 并累积进上下文，从而让有界深度网络完成多步串行计算。
- **Unique Hard Attention (UHA)**：在多个 position 上注意力得分并列最大时，始终选择下标最小的那个 position 的 attention 机制。
- **Strahler number**：衡量树分支复杂度的经典递归指标；叶节点为 0，若子树最大 Strahler 值 M 有至少两个子树同时达到，则当前节点为 M+1，否则为 M。
- **Tree width**：树中同一深度处的最大节点数，即最大层宽。
- **Dyck path / Dyck word**：长度为 2(n−1) 的回溯路径，仅由 +1（U）和 −1（D）步进构成，全程非负且终点回到 0；与 n 顶点有序树一一对应。
- **EH rank（Ehrenfeucht–Haussler rank）**：一阶逻辑表达能力度量的树深度指标，与二叉树 Strahler number 满足相同递归式。
- **Bilinear map（双线性映射）**：形如 B(u,v)=uᵀWv 的运算，在理论 Transformer 文献中用于在单层内表达乘法交互。
- **NC¹**：可由常深对数空间并行电路判定的复杂度类；Merrill–Sabharwal 证明线性 CoT 步数恰好刻画该类。

## 可复现要素
- **数据集**：无经验数据集；构造针对任意 n 顶点有序树和 Dyck 路径的通用实例。
- **代码/权重是否开源**：论文未声明代码仓库或权重开源；全文为理论构造+证明，所有参数以存在性方式给出（未列具体数值矩阵）。
- **关键超参**：嵌入维度 d 随问题规模变化（DFS：6n+3；Dijkstra：5n+1；Algorithm 1：m²+6m；Strahler-tree：6n+14；Strahler-path：3n+15；Width-path：3n/2+4）；时间戳权重 λ ≥ 2n−1；无层归一化、无位置编码；使用标准 ReLU 激活。
