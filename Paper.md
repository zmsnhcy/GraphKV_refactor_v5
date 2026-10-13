# Graph-KV：通过向大语言模型注入结构偏差打破序列化

**作者：**
Haoyu Wang¹*，Peihao Wang²，Mufei Li¹，Shikun Liu¹，Siqi Miao¹，Zhangyang Wang²，Pan Li¹*

¹ 佐治亚理工学院（Georgia Institute of Technology）
² 德克萨斯大学奥斯汀分校（The University of Texas at Austin）

---

# 摘要

现代大语言模型（Large Language Models，LLMs）本质上是自回归的，这要求输入无论其内部具有怎样的结构依赖，都必须被序列化为扁平的序列。

这种序列化限制了模型利用**结构归纳偏置（structural inductive bias）**的能力，尤其是在检索增强生成（Retrieval-Augmented Generation，RAG）以及具有原生图结构的数据推理任务中，因为这些任务中的不同文本片段之间往往存在十分重要的依赖关系。

我们提出了 **Graph-KV**，旨在克服这一限制。

Graph-KV 将文本片段的 **KV Cache** 视为一种压缩的信息表示，并利用结构归纳偏置控制这些表示之间的交互。在这一框架中，一个“目标（target）”文本片段只对其被指定的“源（source）”文本片段的 KV Cache 进行选择性注意，而不是像传统序列化方式那样，对所有位于其之前的文本片段进行注意。

这种方法形成了一种**图结构化的 Block Mask**，从而使 Attention 得到稀疏化，并能够在 LLM 内部实现类似**消息传递（message passing）**的操作。

此外，我们对源文本片段和目标文本片段采用策略性的位置编码分配，从而降低位置偏差（positional bias），同时减少上下文窗口消耗。

我们在三类场景下对 Graph-KV 进行了评估：

1. 七个 RAG 基准测试，涵盖直接推理、多跳推理以及长文档理解；
2. **ARXIV-QA**，一个新的学术论文问答任务，其中完整的科学论文按照引用关系组织成引用自我中心图（citation ego-graphs）；
3. 引用网络中的论文主题分类任务。

实验表明，通过有效降低位置偏差并利用结构归纳偏置，Graph-KV 在各种设置下均能显著优于基线方法，包括计算开销较高的标准顺序编码方法。

代码和 ARXIV-QA 数据集公开于：

https://github.com/Graph-COM/GraphKV

---

# 1 引言

尽管现代大语言模型（LLMs）取得了显著成功，但它们从根本上来说仍然是**自回归模型**。

这种特性源于其训练方式，因此模型必须以序列化方式处理信息。结果是，无论输入本身具有怎样的结构，或者其中存在怎样复杂的依赖关系，例如：

* 顺序无关性；
* 时间关系；
* 逻辑关系；

都必须被强制转换为一个有序序列。

这种强制序列化可能并不是最优的，并且会引入一种**顺序偏置（sequential bias）**，从而限制 LLM 充分利用数据内部的真实关系。

例如，在检索增强生成（RAG）中，被检索出的文本片段可能本身没有线性顺序，或者其内部存在复杂的非线性交互关系，但这些片段仍然必须被人为地序列化。

这可能导致：

1. 多跳推理能力受到限制；
2. 引入明显的位置偏差。

类似的问题也存在于具有原生图结构的数据中，例如**论文引用网络**。

在引用网络中，引用关系实际上表达了知识依赖关系。然而，如果将引用论文简单地序列化，就会产生若干问题：

1. **位置偏差**：本来应该是并行存在的引用关系，却被人为地赋予了不同的序列位置；
2. **二次复杂度**：所有文档之间进行完整 Attention，计算复杂度较高；
3. **上下文窗口限制**：当引用论文数量增加时，位置索引会迅速增长。

因此，一个关键问题出现了：

> **我们能否让数据本身的结构归纳偏置，与自回归 LLM 的机制相匹配，从而摆脱简单的序列化方式？**

---

## 现有工作

已有研究主要尝试缓解 token 序列化带来的负面影响，其目标主要是消除位置偏差，并使 LLM 尽可能以排列不变（permutation-invariant）的方式处理文本片段。

例如，Wang 等人的工作提出在不使用位置编码的情况下，根据 Attention Score 对文档进行重新排序。

但是，这种方法需要进行排序以及完整 Attention 计算，因此计算开销较大。

另一类工作更加关注高效推理，它们通过将不同文档的 KV Cache 独立并行地进行 Prefill 来提升效率。

在这些方法中，文档 KV Cache 的编码通常依赖于：

* 共享位置编码；
* 或者根据外部检索器得到的语义相似度进行排序。

虽然这些并行编码方法提高了计算效率，但是它们存在一个共同的问题：

> **它们无法建模文本片段之间的交互依赖关系，更无法利用结构化数据本身所包含的归纳偏置。**

---

## 表 1：已有方法比较

| 方法                  | 长上下文友好 | 稀疏 Attention / 高效 | 消除位置偏差 | 结构归纳偏置 |
| ------------------- | -----: | ----------------: | -----: | -----: |
| Sequential Encoding |      ✗ |                 ✗ |      ✗ |      ✗ |
| PromptCache         |      ✗ |                 ✓ |      ✗ |      ✗ |
| PINE                |      ✗ |                 ✗ |      ✓ |      ✗ |
| PCW                 |      ✓ |                 ✓ |      ✓ |      ✗ |
| APE                 |      ✓ |                 ✓ |      ✓ |      ✗ |
| Block-Attention     |      ✗ |                 ✓ |      ✗ |      ✗ |
| **Graph-KV**        |  **✓** |             **✓** |  **✓** |  **✓** |

其中：

* **Long-Context Friendly**：表示输入文本片段数量增加时，不会迅速消耗上下文窗口；
* **Free from Positional Bias**：表示模型预测结果不依赖输入文本片段的放置顺序。

---

# Graph-KV 的核心思想

为了解决上述问题，我们提出 Graph-KV。

Graph-KV 的核心思想是：

> **将某一个文本片段的 KV Cache 看作该文本片段的一种压缩信息表示，并使用结构归纳偏置来控制这一表示的生成过程。**

具体来说：

首先，对所有文本片段进行独立的 KV Cache Prefill。

之后，一个“目标”文本片段的 KV Cache 不再像传统因果 Attention 那样关注序列中所有位于其之前的片段，而是：

> **只关注与其存在结构依赖关系的“源”文本片段。**

“源 → 目标”的关系由数据本身或特定任务中的结构归纳偏置决定。

换一种角度来看，Graph-KV 实际上引入了一个：

> **图结构化 Block Mask**

它可以对 KV Cache 生成过程中的 Attention 进行稀疏化，从而在 LLM 内部实现一次类似于：

> **“沿图进行消息传递”**

的操作。

同时，为了降低 LLM 原有的位置偏差，Graph-KV 对所有源文本片段共享位置编码范围，并让目标文本片段使用紧随源节点位置之后的位置编码。

这种设计可以在保持结构关系的同时，显著降低上下文窗口消耗。

---

# 2 相关工作

## 2.1 LLM 中的位置偏差

大语言模型存在明显的位置偏差，其性能会受到输入数据顺序的影响。

通常认为，这种现象来源于：

* 位置编码；
* 因果 Attention；

二者之间的共同作用。

一些研究发现，如果从 Transformer 中移除位置编码，可以改善模型在更长上下文上的泛化能力。

但是，即使没有显式位置编码，因果 Attention 本身仍可能隐式地产生位置偏差。

例如，在 RAG 中，模型经常更偏向于使用：

* 位于上下文开头的信息；
* 位于上下文结尾的信息。

在 In-Context Learning 中，示例的排序同样会显著影响模型结果。

本文研究的任务需要模型更加明确地理解结构依赖关系，而实验发现：

> 简单地进行输入序列化，会进一步放大这种位置偏差。

---

## 2.2 并行编码与 Block Attention

已有研究提出了多种方法，通过独立并行地生成检索文档的 KV Cache，从而避免 RAG 中的二次计算复杂度。

PCW 最早开启了这一研究方向。

然而，PCW 在很多情况下会由于新的 KV Cache 形式导致**分布偏移（distribution shift）**，从而造成性能明显下降。

APE 提出了一种无需微调的方法，通过重新调整 Attention 的幅度来缓解并行编码中的分布偏移。

进一步的研究训练一个小型语言模型作为评分器，对并行检索上下文进行进一步优化。

Block-Attention 则通过更加充分的后训练获得了进一步的性能提升。

然而，这些并行处理方法普遍存在一个限制：

> **无法建模文档之间的交互依赖。**

Graph-KV 在保持并行编码效率的同时，尝试解决这一问题。

---

## 2.3 使用 LLM 建模结构化数据

目前 LLM 处理结构化数据主要有两种方式。

### 方法一：将结构化数据序列化成自然语言

例如将图结构转化为自然语言，然后输入 LLM。

这种方法存在：

* 二次 Attention 复杂度；
* 很难准确用自然语言表达复杂结构依赖；
* 随规模增长难以扩展。

因此，即使是规模适中的文本属性图，例如几十个文档、超过 100k token，也可能难以进行有效推理。

### 方法二：Adapter

另一类方法使用 Adapter，将图结构投影到 LLM 的 Token Embedding 空间中。

但是，这些 Adapter 方法通常泛化能力有限，主要原因在于：

> 很难实现稳定、可靠的 Adapter 对齐。

Graph-KV 与上述方法不同。

它并不是额外训练一个 Adapter，而是更加基础地：

> **直接修改 LLM 的 Attention 机制，使 LLM 能够直接整合结构信息。**

---

## 2.4 有噪声多跳推理的挑战

多跳推理需要模型捕获不同信息之间任意的结构依赖关系。

在标准序列编码下，这一直非常困难。

尤其是在现实环境中：

* 信息可能分散；
* 信息可能不连续；
* 信息可能埋藏在很长的上下文中；
* 上下文中还可能存在大量无关噪声。

这些因素会导致 LLM 性能出现数量级下降。

实验结果表明：

> 如果显式注入结构归纳偏置，Graph-KV 可以显著增强模型的推理能力。

---

# 3 方法

本节介绍 Graph-KV。

我们假设文本片段之间存在的结构归纳偏置，可以用一张连接这些文本片段的图表示。

这些结构可能：

* 原生存在于数据中；
* 或根据具体任务人为定义。

不同任务中具体如何构造图结构，将在实验部分详细介绍。

---

## 3.1 预备知识

令：

* \(q\)：一个自然语言查询；
* \(G=(V,E)\)：表示输入结构化数据的有向图。

每一个节点：

$$
u\in V
$$

对应一个文本片段，其平均 token 长度记为 \(d\)。

每一条有向边：

$$
(u_1,u_2)\in E
$$

表示源文本片段 \(u_1\) 与目标文本片段 \(u_2\) 之间存在某种结构依赖。

我们的目标是让 LLM：

$$
f
$$

通过同时编码 \(q\) 和 \(G\)，产生答案：

$$
a=f(q,G)
$$

这要求 LLM：

1. 理解每一个节点的文本内容；
2. 正确利用图 \(G\) 中编码的结构归纳偏置进行推理。

---

# 3.2 Sequential Encoding

Sequential Encoding 是目前现代 LLM 默认采用的方法。

将所有文本片段任意线性化：

$$
[u_1,u_2,\ldots,u_n]
$$

每个 \(u_i\) 都由一系列 Token Embedding 构成。

然后将：

$$
[u_1,u_2,\ldots,u_n,q]
$$

作为模型输入。

现代 LLM 普遍采用因果 Attention，使得每一个 token 可以关注之前的 token。

其计算复杂度为：

$$
O(n^2L^2)
$$

其中：

* \(n\)：文本片段数量；
* \(L\)：最大文本片段长度。

Sequential Encoding 的问题包括：

1. 模型输出对输入顺序非常敏感；
2. 产生位置偏差；
3. 没有天然机制利用数据中的结构归纳偏置。

---

# 3.3 Parallel Text Encoding

另一类方法将文本片段视为一个无序集合：

$$
S=\{u_1,u_2,\ldots,u_n\}
$$

LLM 独立且并行地对每一个 \(u_i\) 进行编码。

通常所有文本片段共享相同的位置编码，以表达它们没有明确的顺序。

其计算复杂度：

$$
O(nL^2)
$$

因此相对于 Sequential Encoding，它对文本片段数量 \(n\) 呈线性增长。

但是，这种方法的代价是：

> 它完全丢弃了文本片段之间直接交互的信息。

因此，在多跳推理中非常重要的关系信息会被牺牲。

在回答查询时，查询 token 会同时关注：

$$
S
$$

中的所有文本片段以及查询 \(q\)。

---

# 3.4 Graph-KV

Graph-KV 使用两个主要机制注入结构归纳偏置：

1. **结构感知 Attention 机制**
2. **合理分配的共享位置编码**

---

## 3.4.1 结构感知 Attention

首先，Graph-KV 对每一个文本片段 \(u_i\) 进行离线并行编码，从而获得初始潜在表示：

$$
h_{u_i}^{(0)}
$$

然后利用这些表示构造初始 Key-Value 对：

$$
\{(k_{u_i}^{(0)},v_{u_i}^{(0)})\}_{u_i\in V}
$$

这些 KV 可以被加载进 KV Cache。

根据图结构：

$$
G=(V,E)
$$

对于目标节点 \(u_j\)，Graph-KV 通过与其源节点进行 Attention 来更新它的表示。

目标节点 \(u_j\) 的源节点集合为：

$$
N(j)=\{u_i\mid(u_i,u_j)\in E\}
$$

随后计算稀疏 Attention：

$$
\operatorname{softmax}
\left(
\frac{Q_jK_{N(j)}^T}{\sqrt{d_h}}
\right)
V_{N(j)}
$$

其中：

* \(Q_j\)：目标文本片段 \(u_j\) 的 Query；
* \(K_{N(j)}\)：所有源文本片段的 Key；
* \(V_{N(j)}\)：所有源文本片段的 Value；
* \(d_h\)：用于归一化的 \(QK\) 维度。

这里最关键的变化是：

> **目标节点不再对所有前序文本进行 Attention，只对图中与自己相连的源节点进行 Attention。**

因此 Attention 被稀疏化了。

---

## 多轮传播

上述更新过程理论上可以迭代多个轮次。

但是，在本文实验中：

$$
t=1
$$

即只进行了**一轮结构信息传播**。

作者将其视为一种 Proof-of-Concept，用于证明：

> 相比传统并行编码中完全独立的文本片段，Graph-KV 可以让不同文本片段进行结构性交互，同时保持较低的计算复杂度。

作者还发现：

> 在当前评测任务中，多轮迭代并没有产生显著的额外性能收益。

不过作者认为，在：

* 更复杂的任务；
* 或针对 Graph-KV 注意力结构进一步微调模型；

的情况下，多轮传播可能会获得更明显的收益。

最终，在查询阶段：

> 查询 token 以及后续生成的答案 token 都可以关注 source 和 target 两类文本片段的表示。

---

# 3.5 位置编码的分配

Graph-KV 对位置编码的设计具有两个目标：

1. 降低位置偏差；
2. 提高上下文利用效率。

首先考虑不存在显式有向边的情况。

此时，图结构实际上退化成若干相互独立的文本片段。

这种情况类似于已有的并行编码方法。

为了降低独立文本片段的位置偏差：

> 所有文本片段共享位置范围 \([0,L)\)。

也就是说，即使存在大量文本片段，它们并不会使用：

$$
[0,L),[L,2L),[2L,3L),\ldots
$$

这样不断增长的位置编号。

---

当存在结构依赖时：

目标节点使用：

$$
[L,2L)
$$

范围内的位置编码。

而查询 token 和生成 token 使用：

$$
2L
$$

之后的位置。

---

## 多轮结构传播

虽然本文没有实验验证“上一轮 target 继续作为下一轮 source”的情况，但方法允许进行自然扩展。

例如第二轮 target 可以使用：

$$
[2L,3L)
$$

第三轮则可以继续使用：

$$
[3L,4L)
$$

依次类推。

其最大优势是节省上下文窗口。

由于大量文本片段可以共享同一个位置范围，所以总体位置范围大约为：

$$
TL
$$

其中：

* \(T\)：传播迭代次数；
* \(L\)：单个文本片段最大长度。

作者指出，\(T\) 通常是一个较小常数，而 \(L\) 往往小于 10k。

---

## 一个直观解释

假设图中存在：

$$
Doc.2\rightarrow Doc.1
$$

以及：

$$
Doc.3\rightarrow Doc.1
$$

Graph-KV 可以理解成让 LLM 分别处理两个“有效文档”：

$$
Doc.2\rightarrow Doc.1
$$

和：

$$
Doc.3\rightarrow Doc.1
$$

然后将这两个过程中得到的 Doc.1 表示进行聚合。

因此最终：

> Doc.1 的表示同时包含了来自 Doc.2 和 Doc.3 的结构信息。

这也是 Graph-KV 与普通并行 KV Cache 最大的区别之一。

---

# 3.6 计算复杂度

第一轮所有文本片段的表示计算复杂度为：

$$
O(|V|L^2)
$$

与已有并行编码方法相同。

设目标节点集合为 \(T\)，那么更新这些目标节点的复杂度：

$$
O\left(
\sum_{u_j\in T}|N(j)|L^2
\right)
$$

由于：

$$
|E|=\sum_{u_j\in T}|N(j)|
$$

因此：

$$
O(|E|L^2)
$$

也就是说，结构传播的计算量取决于**图中边的数量**，而不是所有节点之间的全连接关系。

在 Query 阶段，对于每一个查询或生成 token，对全部 source 和 target 文本片段进行 Attention 的复杂度为：

$$
O(|V|L)
$$

与普通 Sequential Encoding 和 Parallel Encoding 在生成阶段的复杂度相近。

---

## Attention Sink

APE 在共享位置编码范围时，会避开前几个位置，以防止 Attention Sink 问题。

但作者发现：

> Graph-KV 即使让所有文本片段从第一个 token 开始共享位置范围，也没有受到明显影响。

作者认为，这是因为所使用的模型已经通过独立 Attention Block 进行了微调，因此能够适应该位置编码形式。

---

# 4 实验

作者设计了四类任务：

### Task 1：RAG

检验检索增强生成能力。

### Task 2：ARXIV-QA

新的多跳学术论文理解任务。

### Task 3：Paper Topic Classification

引用网络中的论文主题分类任务。

### Task 4：Scalability & Efficiency Stress Test

使用合成图测试可扩展性和效率。

---

# 4.1 Graph-KV 的 Backbone

Graph-KV 要求对不同文本片段进行独立编码，这会使标准 LLM Backbone 产生一定的分布偏移。

作者考虑了两种解决方法。

第一种：

> 使用 APE 等无需微调的方法，通过调整 Attention Temperature 和缩放 Attention 权重来降低分布偏移。

第二种：

> 对模型进行后训练，使不同文本片段采用相互独立的 Attention Block。

例如 Block-RAG。

实验发现：

> 后训练后的模型在 Graph-KV 下表现更加稳定。

因此，作者默认使用：

**Llama-3.1-8B-Block-FT**

作为 Graph-KV 的 Backbone。

该模型基于预训练的 Llama-3.1-8B，并使用：

* Tulu-3；
* 20k 个来自 2Wiki 和 TriviaQA 的 RAG 数据；

进行独立 Attention Block 的后训练。

由于计算资源有限，本文只研究 Llama-3.1-8B 家族。

作者没有进一步将这种训练扩展到其他 LLM。

同时：

> 作者也没有直接使用 Graph-KV 对模型进行进一步微调。

作者认为，如果进行这种专门的 Graph-KV 微调，很可能还能进一步提升性能。

---

# 4.2 Task 1：RAG

RAG 中作者研究了三种情况：

1. 直接推理；
2. 多跳推理；
3. 长文档理解。

使用了七个数据集：

* NarrativeQA
* TriviaQA
* HotpotQA
* 2Wiki
* Multihop-RAG
* MorehopQA
* LongBench-v2

对于每个问题：

> 提供 10 个文本片段。

主要指标为 Accuracy。

---

## Graph-KV 中的图结构

在 RAG 中，文本片段之间通常存在：

* 逻辑关系；
* 时间关系；

但这些依赖并不是显式提供的。

因此作者构造了一个**二部图（bipartite graph）**。

实验中：

* Top-1；
* Top-3；

相似度最高的文本片段作为 Source。

然后：

> 将这些 Source 的 KV Cache 拼接起来，由其余文本片段进行 Attention，从而生成目标文本片段的 KV。

此外还进行了：

### Graph-KV-Full

将所有检索文本片段同时视为：

* source；
* target。

因此相当于建模所有潜在的 pair-wise dependency。

---

# 4.3 RAG 实验结果

作者观察到一个明显趋势：

> **Graph-KV-Full 总体上优于其稀疏版本。**

同时：

$$
Graph\text{-}KV\text{-}Top\text{-}3
$$

通常优于：

$$
Graph\text{-}KV\text{-}Top\text{-}1
$$

而在大多数情况下：

> Graph-KV-Top-1 已经可以超过 Block-RAG。

Graph-KV-Top-3 则在所有测试设置下都超过 Block-RAG。

这说明：

> Graph-KV 注入的稀疏依赖关系，可以有效弥补普通并行编码完全缺乏文本间依赖的问题。

尤其是在：

* 多跳推理；
* 中等长度推理；
* 长文档推理；

任务中，Graph-KV-Top-3 的优势更加明显。

---

## 表 2：Multihop-RAG、MorehopQA、LongBench-v2

| Backbone | Attention      |     Infer |   Compare | Temporal |      Null |       Avg |
| -------- | -------------- | --------: | --------: | -------: | --------: | --------: |
| 8B-Block | Sequential     |     63.55 |     34.54 |    38.03 |     53.09 |     43.60 |
| 8B-Block | Block-RAG      |     72.73 |     34.16 |    38.03 |     40.73 |     43.32 |
| 8B-Block | Graph-KV Top-1 | **73.25** |     34.19 |    38.18 |     53.85 |     44.81 |
| 8B-Block | Graph-KV Top-3 |     73.12 |     34.43 |    38.69 | **63.64** | **45.79** |
| 8B-Block | Graph-KV Full  |     69.04 | **34.63** |    38.48 | **88.79** | **46.41** |

在 MoreHopQA 上：

| 方法             |     Hop-1 |     Hop-2 |     Hop-3 |     Hop-4 |     Hop-5 |       Avg |
| -------------- | --------: | --------: | --------: | --------: | --------: | --------: |
| Sequential     |     59.23 |     20.43 |     25.32 |     23.07 |     30.76 |     37.38 |
| Block-RAG      |     56.75 |     25.72 |     24.67 |     15.38 |     27.47 |     37.92 |
| Graph-KV Top-1 |     54.72 |     25.72 |     24.67 |     23.07 |     31.86 |     37.96 |
| Graph-KV Top-3 |     56.53 |     25.72 |     24.67 |     23.07 |     29.67 |     38.11 |
| Graph-KV Full  | **64.86** | **30.52** | **25.97** | **30.76** | **47.25** | **44.90** |

论文的核心观察是：

> Graph-KV-Top-3 在多跳推理任务上能够比 Sequential Encoding 高约 2%–10%。

不过，在一些简单的直接推理任务中，Graph-KV 并不一定超过 Sequential Encoding。

原因是：

> 如果问题本身只需要一个文本片段就能够回答，那么是否显式建模结构依赖并不会带来明显收益。

而在真正需要两跳或多跳关系的任务中，结构建模的价值就会明显体现。

---

# 4.4 Fine-tuning 的重要性

APE 虽然比 PCW 表现更好，但与经过后训练的 Block-RAG 相比依然存在明显差距。

因此论文认为：

> 如果一个 LLM 原本是在完整 causal attention 下预训练的，那么要让它适应独立文本片段编码与新的 KV Cache 结构，专门的 Fine-tuning / Post-training 仍然非常重要。

---

# 4.5 Task 2：ARXIV-QA

ARXIV-QA 是论文新提出的任务。

作者随机选择了：

> 100 篇学术论文。

经过数据清洗后：

> 最终得到 60 篇中心论文。

加上其引用论文后：

> 总共包含 472 篇具有完整正文的论文。

每篇中心论文构造一个技术问题。

---

## ARXIV-QA 的问题特点

回答一个问题需要：

1. 理解中心论文的主要内容；
2. 理解中心论文与引用论文之间的引用关系；
3. 理解引用论文中的具体内容。

平均每个问题需要处理约：

$$
88k
$$

个 token。

已经非常接近基础模型的：

$$
128k
$$

有效上下文限制。

---

## Distractor

为了进一步测试位置偏差，作者引入了 Distractor。

例如：

> 1 篇真正相关的论文 + 其引用论文
>
> * 2 篇无关论文 + 其引用论文。

三组论文加起来的平均输入长度：

$$
264.6k
$$

token。

因此已经远远超出：

$$
128k
$$

上下文窗口。

---

# 4.6 Graph-KV 在 ARXIV-QA 中的结构

Graph-KV 将：

> 中心论文 → 所有引用论文

构造成 target-source 关系。

当存在大量 Distractor，导致总文本长度超过 128k 时：

> 共享位置编码策略就非常关键。

因为它使得大量论文可以共享相同的位置编码范围，而不需要随着论文数量增长而不断增长 position index。

---

# 4.7 ARXIV-QA 结果

在没有 Distractor 时：

> Sequential Encoding 可以取得较好的结果。

因为此时上下文结构比较简单。

但：

* PCW；
* APE；
* Block-RAG；

由于不能显式建模论文之间的引用关系，所以表现明显下降。

---

当加入 Distractor 后：

> Sequential Encoding 出现严重的位置偏差。

如果真正相关的论文被放在序列末尾：

> Sequential Encoding 有时还能获得比较好的结果。

原因主要是预训练自回归模型具有明显的**近因偏差（recency bias）**。

即：

> 模型更加关注上下文中后面出现的文本。

但是，如果真正相关的信息位于序列开头：

> 随着 Distractor 和上下文长度增加，性能快速下降。

当加入两个 Distractor 时：

> 所有 Sequential Encoding 基线都无法正确回答问题。

Graph-KV 则由于：

1. PE Sharing；
2. Structural Inductive Bias；

能够避免这种位置偏差，并持续保持最佳性能。

---

# 4.8 Task 3：论文主题分类

作者进一步在两个引用图数据集上测试 Graph-KV：

* Cora；
* Pubmed。

任务是：

> 给定中心论文以及它引用的论文，判断中心论文属于哪个主题类别。

有些中心论文可能拥有：

> 数百个引用。

例如 Pubmed 中最多可达到 130 个引用。

因此这个任务非常适合测试：

> LLM 是否能够真正利用图结构中的邻居信息。

---

## 表 4：论文主题分类

| 方法                    |       Cora |     Pubmed |
| --------------------- | ---------: | ---------: |
| 8B-SFT Sequential     | 66.66±0.62 | 80.64±0.39 |
| 8B-SFT PCW            |      68.63 |      76.95 |
| 8B-SFT APE            |      66.92 |      77.01 |
| 8B-RAG Sequential     | 70.35±0.17 | 82.06±0.16 |
| 8B-RAG PCW            |      66.05 |      76.52 |
| 8B-RAG APE            |      68.46 |      76.49 |
| 8B-Block Query-Only   |      57.38 |      83.60 |
| 8B-Block Sequential   | 67.09±0.17 | 79.79±0.16 |
| 8B-Block Block-RAG    | 69.55±0.30 | 83.24±1.16 |
| **8B-Block Graph-KV** |  **71.03** |  **84.61** |

结果说明：

> Graph-KV 利用论文与其引用论文之间的依赖关系后，能够超过 Sequential Encoding 和普通 Parallel Encoding。

作者认为这是 Graph-KV 的一个重要优势。

因为已有工作通常是：

* 在文本 embedding latent space 中进行图节点分类；
* 或使用 Adapter 将参考论文映射到 LLM token space。

而 Graph-KV 则直接：

> **改变 LLM 本身的 Attention 机制。**

---

# 4.9 Task 4：可扩展性和效率测试

作者构造了合成 Star Graph。

硬件：

* NVIDIA RTX6000；
* 48GB GPU Memory；
* AMD EPYC 7763；
* 64 核 CPU。

每个节点分别设置：

* 500 words；
* 1000 words。

然后不断增加 Neighbor 数量。

实验结果：

> Graph-KV 能够编码的 Neighbor 数量超过 Sequential Encoding 的 3 倍。

---

## TTFT

作者进一步测试 Time-To-First-Token（TTFT）。

固定：

$$
10
$$

个 Neighbor。

然后改变每个节点的 token / word 数量。

结果表明：

> Graph-KV 的 TTFT 明显低于 Sequential Encoding。

原因在于：

> Graph-KV 可以提前对文本片段进行 KV Cache Prefill，并在 Prefill 阶段直接注入结构信息。

---

# 5 结论与未来工作

本文提出 Graph-KV，一种用于解决自回归 LLM 处理结构化数据问题的新方法。

Graph-KV 通过：

1. 直接将结构归纳偏置注入 Attention；
2. 策略性地分配位置编码；

从而：

* 降低位置偏差；
* 降低上下文窗口需求；
* 建模文本片段之间的结构依赖；
* 保持较好的计算效率。

在：

* RAG；
* ARXIV-QA；
* 论文主题分类；

等多个任务上的实验表明：

> Graph-KV 相比 Sequential Encoding 和 Parallel Encoding，都可以取得显著性能提升。

尤其是在：

* 多跳推理；
* 长上下文；
* 多文档结构化推理；

场景下优势最为明显。

不过，本文当前主要评估的是：

> **一跳结构依赖（one-hop structural dependencies）**

作者认为：

> 利用结构依赖关系来提升 LLM 对更加复杂拓扑结构的理解能力，仍然具有非常大的研究空间。

---

# 致谢

H. Wang、M. Li、S. Liu、S. Miao 和 P. Li 的部分研究工作获得以下项目支持：

* NSF IIS-2239565
* CCF-2402816
* IIS-2428777
* PHY-2117997
* DOE DE-FOA-0002785
* JPMC Faculty Awards
* OpenAI Research Credits
* Meta Research Award

作者感谢 Xinyu Yang 和 Beidi Chen 的宝贵讨论。

---

# A 附录

## A.1 额外实验结果

### A.1.1 对普通 LLM 的适用性

作者进一步将 Graph-KV 应用于标准的：

**Llama-3.1-8B-SFT**

并与各种 Parallel Encoding 方法比较。

实验结果表明：

> Graph-KV 持续优于其他 Parallel Encoding 基线。

### MoreHopQA / 2Wiki

| 方法                 |     Hop-1 |     Hop-2 |     Hop-3 | Hop-4 |     Hop-5 |       Avg |
| ------------------ | --------: | --------: | --------: | ----: | --------: | --------: |
| PCW                |     11.26 |     10.81 |     10.38 |  0.00 |     17.58 |     11.35 |
| APE                |     13.28 |     15.86 |      9.47 | 61.53 |     10.98 |     14.13 |
| Block-RAG          |     13.96 |     13.94 |      9.74 |  0.00 |     15.38 |     13.32 |
| **Graph-KV Top-3** | **15.54** | **16.10** | **12.98** |  0.00 | **18.68** | **15.47** |

在 2Wiki 上：

| 方法                 |   Compare |     Infer | Bridge |   Compose |       Avg |
| ------------------ | --------: | --------: | -----: | --------: | --------: |
| PCW                |     70.13 |     13.07 |  67.17 |     32.58 |     46.94 |
| APE                |     74.44 |     18.26 |  61.17 |     37.41 |     49.20 |
| Block-RAG          |     72.17 |     15.17 |  68.01 |     35.54 |     48.99 |
| **Graph-KV Top-3** | **76.41** | **17.43** |  64.15 | **47.19** | **54.30** |

---

# A.1.2 Multi-Hop Graph-KV

作者进一步尝试多层图结构，让信息能够进行多轮传播。

### 2-Hop

构造：

$$
Top\ 1 \rightarrow Top\ 2-4 \rightarrow Rest
$$

第一轮：

> Top-1 作为 Source，更新 Top-2 至 Top-4 的 KV。

第二轮：

> 更新后的 Top-2 至 Top-4 再更新其余节点。

---

### 3-Hop

构造：

$$
Top\ 1
\rightarrow
Top\ 2-3
\rightarrow
Top\ 4-5
\rightarrow
Rest
$$

进行三步传播。

---

## Multi-Hop 实验结果

### Multihop-RAG

| 方法                             | Hop 数 |     Infer |   Compare |  Temporal |      Null |       Avg |
| ------------------------------ | ----: | --------: | --------: | --------: | --------: | --------: |
| Graph-KV Top-1                 |     1 |     73.25 |     34.19 |     38.18 |     53.85 |     44.81 |
| Graph-KV Top-3                 |     1 |     73.12 |     34.43 |     38.69 |     63.64 |     45.79 |
| Graph-KV Full                  |     1 |     69.04 |     34.63 |     38.48 |     88.79 |     46.41 |
| Graph-KV Top-1-Top-3           |     2 |     73.20 |     34.54 |     38.78 |     70.82 |     46.15 |
| **Graph-KV Top-1-Top-3-Top-5** |     3 | **73.30** | **34.76** | **38.78** | **75.44** | **46.74** |

### HotpotQA

| 方法                             | Hop 数 |   Compare |    Bridge |       Avg |
| ------------------------------ | ----: | --------: | --------: | --------: |
| Graph-KV Top-1                 |     1 |     70.41 |     74.14 |     73.39 |
| Graph-KV Top-3                 |     1 |     71.01 |     76.24 |     75.19 |
| Graph-KV Full                  |     1 |     70.41 |     77.62 |     76.17 |
| Graph-KV Top-1-Top-3           |     2 |     71.95 |     77.25 |     76.19 |
| **Graph-KV Top-1-Top-3-Top-5** |     3 | **72.02** | **77.57** | **76.45** |

---

### MoreHopQA

| 方法                             | Hop 数 |     Hop-1 |     Hop-2 |     Hop-3 |     Hop-4 |     Hop-5 |       Avg |
| ------------------------------ | ----: | --------: | --------: | --------: | --------: | --------: | --------: |
| Graph-KV Top-1                 |     1 |     54.72 |     25.72 |     24.67 |     23.07 |     31.86 |     37.96 |
| Graph-KV Top-3                 |     1 |     56.63 |     25.72 |     24.67 |     23.07 |     29.67 |     38.11 |
| Graph-KV Full                  |     1 |     64.86 |     30.52 |     25.97 |     30.76 |     47.25 |     44.90 |
| Graph-KV Top-1-Top-3           |     2 |     67.11 |     32.69 |     25.32 |     30.76 |     37.36 |     45.70 |
| **Graph-KV Top-1-Top-3-Top-5** |     3 | **67.56** | **32.93** | **25.97** | **38.46** | **43.95** | **46.69** |

### 2Wiki

| 方法                             | Hop 数 |   Compare |     Infer |    Bridge |   Compose |       Avg |
| ------------------------------ | ----: | --------: | --------: | --------: | --------: | --------: |
| Graph-KV Top-1                 |     1 |     82.76 |     52.42 |     90.00 |     65.94 |     73.60 |
| Graph-KV Top-3                 |     1 |     83.25 |     56.74 |     90.47 |     67.83 |     75.15 |
| Graph-KV Full                  |     1 |     82.69 |     54.16 |     90.43 |     67.11 |     74.38 |
| Graph-KV Top-1-Top-3           |     2 |     83.45 |     57.04 |     90.77 |     68.97 |     75.76 |
| **Graph-KV Top-1-Top-3-Top-5** |     3 | **83.75** | **57.04** | **90.77** | **69.34** | **75.99** |

这个结果非常重要，因为它实际上初步说明：

> **将 Graph-KV 从“一跳结构传播”扩展到“多跳图消息传递”，确实可以继续获得收益。**

---

# A.2 实现细节

## 硬件与平台

所有实验均使用：

* PyTorch；
* HuggingFace Transformers；
* FlashAttention-2。

在 ARXIV-QA 中：

* Parallel Text Encoding、Block-RAG、PCW、APE 和 Graph-KV 使用 4 张 NVIDIA A100 Tensor Core GPU；
* Sequential Encoding 使用 8 张 NVIDIA A100 Tensor Core GPU，因为其显存需求更高。

其他任务使用：

> NVIDIA RTX 6000 Ada GPU。

---

## 模型权重

实验采用以下开源模型：

* llama-3.1-8B-sft
* llama-3.1-8B-TAG
* llama-3.1-8B-Block-FT

作者由于计算资源有限：

> 没有进一步针对 Graph-KV 对 Llama-3.1-8B 进行 Fine-tuning。

但作者认为：

> 进一步 Fine-tuning 很可能继续提高实验性能。

---

# A.2.1 RAG 实现细节

对于：

* 2Wiki；
* NarrativeQA；
* TriviaQA；
* MoreHopQA；
* HotpotQA；

数据处理流程严格遵循 Block-RAG。

对于 Multihop-RAG，则遵循原始实现。

所有基准：

> 根据相似度分数检索 Top-10 文本片段。

然后将它们按照相似度排序后输入模型。

评价指标为：

> Accuracy。

模型最多生成：

$$
256
$$

个 token。

---

## RAG Prompt

整个 Prompt 被划分为三部分：

### Prefix

> You are an intelligent AI assistant. Please answer questions based on the user’s instructions. Below are some reference documents that may help you in answering the user’s question.

中文：

> 你是一名智能 AI 助手。请根据用户的指令回答问题。下面提供了一些可能有助于回答用户问题的参考文档。

---

### Text Chunks

格式：

```text
-Title: {Title #1}. 
{Text #1}

-Title: {Title #2}.
{Text #2}

...

-Title: {Title #10}.
{Text #10}
```

---

### Question

2Wiki、HotpotQA、NarrativeQA 和 TriviaQA 使用：

> 请根据给出的检索文档回答问题，其中部分文档可能与问题无关。请写出高质量答案。

Multihop-RAG 使用：

> 请仅根据所提供的检索文档回答问题。答案应该是一个单词或实体。如果所提供的信息不足以回答问题，请回答“Insufficient Information”。最后请以“The answer is:”开头给出你的答案。

MoreHopQA 使用：

> 请仅根据给出的检索文档回答问题。如果答案是日期，请采用 YYYY-MM-DD（ISO 标准）格式。在逐步思考之后，请按照“Answer:”格式给出最终答案。

---

# A.2.2 ARXIV-QA 实现细节

## 数据集构建

作者最初从 OGBN-ARXIV 引用网络中随机选取：

> 100 篇中心论文及其引用论文。

然后通过 Arxiv API 下载 PDF，并使用：

> fitz

将 PDF 转换成全文。

数据清洗过程中要求：

1. 下载的论文必须正确；
2. 每篇中心论文至少包含 3 个有效引用。

对于每一篇引用论文，如果它在中心论文中以独立句子的形式出现，并且中心论文至少使用一个句子讨论或比较该论文，那么作者就人工设计一个问题。

最终：

> 选出了 60 篇中心论文以及对应的问题。

从而形成 ARXIV-QA 数据集。

---

## Distractor

添加 Distractor 时：

> 使用随机种子 42–44。

随机抽取其他论文的 citation ego-graph。

实验中没有观察到不同随机种子造成明显不同的结果。

---

## ARXIV-QA Prompt

输入分成：

1. Prefix；
2. Paper Texts；
3. Question。

Prefix：

> You are an intelligent AI assistant. You will first read the related works of a paper, then you will read the paper. Then answer the question.

中文：

> 你是一名智能 AI 助手。你将首先阅读一篇论文的相关工作，然后阅读论文正文，最后回答问题。

---

## Sequential Encoding

直接输入：

$$
Prefix+Paper\ Texts+Question
$$

存在 Distractor 时，将多个 citation ego-graph 依次拼接，同时保持每个 ego-graph 内部的结构完整。

---

## Parallel Encoding

分别独立编码：

* Prefix；
* 每篇引用论文全文；
* 中心论文全文；
* Question。

然后将它们的 KV Cache 拼接。

---

## Graph-KV

独立编码：

* Prefix；
* Question；

并按照 Section 4.2 的方法向 Paper Text 注入结构归纳偏置。

---

# ARXIV-QA 示例

论文给出了一个示例。

问题大意：

某篇论文引用了一篇关于利用显卡视频输出中的 FM 无线电信号，在物理隔离网络和附近手机之间建立隐蔽通信的恶意软件论文。请给出该论文的完整名称。

正确答案：

> **AirHopper: Bridging the Air-Gap between Isolated Networks and Mobile Phones using Radio Frequencies**

在没有 Distractor 时，各模型基本能够回答。

但加入 Distractor 后：

Sequential Encoding 会明显受到：

> 位置偏差

影响。

Graph-KV 仍然能够正确找到真正相关论文。

---

# A.2.3 论文主题分类实现细节

数据来自：

* Cora；
* Pubmed。

模型采用 [7] 中使用的测试集划分。

每篇论文的输入文本：

> 标题 + 摘要。

Prompt 分为：

1. Prefix；
2. Paper Title & Abstracts；
3. Question。

---

## Prefix

> You are an intelligent AI assistant. You will first read a list of titles or abstracts of papers cited by a central paper. Then, you will read the title or abstract of the central paper itself. Finally, you will answer a question related to the central paper.

中文：

> 你是一名智能 AI 助手。首先阅读中心论文所引用论文的标题或摘要列表，然后阅读中心论文本身的标题或摘要，最后回答与中心论文相关的问题。

---

## Paper Title & Abstracts

形式为：

```text
{Reference 1 的标题与摘要}
{Reference 2 的标题与摘要}
...
{Reference k 的标题与摘要}

它们均被下面的中心论文引用：

{Central Paper 的标题与摘要}
```

---

## Question

> 将中心论文分类到以下类别中的一个：{Classes}。请按照“Answer:”给出答案。

对于 Sequential Encoding 和 Block-RAG：

> 使用随机种子 42–44 对参考论文的排列顺序进行随机打乱，并报告平均结果。

---

# A.2.4 Stress Test 实现细节

硬件：

* NVIDIA RTX 6000；
* 48GB；
* AMD EPYC 7763；
* 64 核。

所有方法使用：

> FlashAttention-2。

文本来自：

> Cora 测试集的第一个样本。

为了满足实验输入长度要求：

* Memory Test：500 / 1000 words；
* Generation Latency：100 / 200 / 400 / 800 words。

作者通过重复或截断原文本实现。

---

## Memory Test

逐渐增加合成 Star Graph 的 Neighbor 数量。

同时使用：

```text
torch.cuda.reset_peak_memory_stats()
torch.cuda.max_memory_allocated()
```

记录 GPU 峰值显存使用情况。

---

## TTFT

Graph-KV 与其他 Parallel Encoding 方法一样，可以从：

> KV Cache Prefill

中获益。

因此 Graph-KV 的：

> Time-To-First-Token（TTFT）

明显低于 Sequential Encoding。

作者使用：

```text
time.time()
```

来测量实际耗时。

---

# 对这篇论文最核心的一句话概括

如果把论文压缩成一句话：

> **Graph-KV 并不是简单地“把 KV Cache 存成图”，而是把每个文本块的 KV Cache 当成图节点表示，再利用图结构规定哪些 KV 可以参与哪些目标节点 KV 的生成，从而把图上的消息传递机制注入 LLM 的 KV Cache 构建过程。**

这一点与你前面关注的：

> **“将 KV Cache 本身组织成图结构”**

实际上有非常直接的关系，但二者并不完全相同。

Graph-KV 当前做的是：

$$
\text{Text Chunk}
\rightarrow
\text{KV Cache Node}
\rightarrow
\text{Graph-guided KV Update}
$$

而不是：

$$
\text{KV Cache}
\rightarrow
\text{动态 KV 图存储}
\rightarrow
\text{图结构化 KV Memory}
$$

这也是这篇论文对你之前提出的 **Graph-structured Latent KV Memory** 思路最值得关注的地方。
