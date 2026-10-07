# GraphKV 审计与修复说明

## 结论

你的 v4 重构存在可复现的 bug；把逻辑 `position_ids` 和物理 `cache_position` 分开这一核心方向是正确的。两个 notebook 中的空输出，现有证据不足以归因于这些 bug：测试 prompt 不公平、缺少生成起始格式，以及诊断后重复使用已被修改的 KV cache，必须先排除。

本次依据：用户提供的论文 2506.07334v2，第 3.1 节、图 2、PDF 第 4–5 页；官方仓库固定提交 `ba01bbcc98fc9a352582dd1e75ab88fa6cac0dfb`；四个 v4 模块与两个 notebook 的源代码及已保存输出。文件中原有结论只作为待验证内容，未当作用户指令或已证事实。

## 1. 论文、官方代码与 v4 应如何区分

论文用全局最大 chunk 长度 L 分配 source 的 `[0,L)`、target 的 `[L,2L)`，query 从后一段开始。若保留长度 P 的独立 prefix，可整体平移成 source `[P,P+L)`、target `[P+L,P+2L)`、query 从 `P+2L` 开始。L 必须覆盖 target，不能只看 source。

物理拼接仍为 prefix + 各 source + 各 target，长度为 `P + sum(source_len) + sum(target_len)`。物理长度与逻辑跨度不同是预期设计，不是 bug。source/target 节点内保留 causal attention；target 可见其前驱的完整 KV；query 可见 prefix、选中图节点及自身历史。

独立 prefix 仅在 query 阶段可见，是沿用用户代码和官方路径的工程约定；不能宣称论文唯一规定了该 prefix 语义。

官方 RAG 的 `pcw.py::gapemp` / `gapemp_appr` 在 source flatten 后施加连续 RoPE，最终合并后又施加连续 RoPE。这与论文共享位置方案不一致。v5 选择论文方案，因此不会承诺逐 token 复现官方输出，也不能拿官方最终指标直接当作 v5 的已达指标。[固定提交的 pcw.py](https://github.com/Graph-COM/GraphKV/blob/ba01bbcc98fc9a352582dd1e75ab88fa6cac0dfb/pcw.py)

官方 citation 路径则已有共享 source PE，但把 `max(neighbor_len)` 用于 `cache_position`，未与实际 `sum(neighbor_len)` 区分。eager/SDPA 的物理 causal mask 会受到影响；FlashAttention 路径可能不通过相同的物理 mask 分支，不能概括为所有 backend 都以相同方式出错。[固定提交的 pcw_parallel.py](https://github.com/Graph-COM/GraphKV/blob/ba01bbcc98fc9a352582dd1e75ab88fa6cac0dfb/pcw_parallel.py)

## 2. v4 中已确认的问题

| 严重程度 | 位置/触发条件 | 后果与修复 |
|---|---|---|
| 高 | `GraphKVCache.source_span/target_position_ids/query_position_ids`，target 比最长 source 长 | query 可能落入 target 区间。v5 预先统计所有 source/target，统一 L；底层不允许超出规划的 target。 |
| 高 | `pcw_parallel.gapemp_graph_batch` 在循环内边加 source 边传播 target | 第一个子图冻结 L；后续 source 更长即抛异常，而且此前已写入状态。v5 先收集所有子图再执行。 |
| 中 | 同一函数在每个子图内调用 `set_edges` | 后一次清空先前边；由于已编码 KV 不会被回滚，不一定立即改变 query 输出，但保存的图不完整，后续检查/复用错误。v5 一次构建完整边集。 |
| 中 | `graph_kv_adapter.build_graphkv_cache` 及直接调用 `propagate_target` 时未关闭 autograd | 返回的 logits 携带计算图，8B 模型推理增加不必要的显存开销。v5 为推理路径统一 no_grad，并要求 eval。 |
| 中 | `pcw_parallel.block/block_batch` 的独立 cache 全从 0 编码，query 却从所有物理长度之和开始 | key 与 query 使用不一致的位置策略。v5 提供明确的共享位置独立块基线；说明其不同于官方 baseline。 |
| 中 | `gapemp_appr` 的 `top_k=0` 与负值 | Python 的 `[-0:]` 变成全量；负数也产生意外选择。v5 拒绝非法 k。 |
| 中 | 两个 pcw 文件的 RoPE helpers 强制转 bfloat16、移动模型自有 rotary module | float32/fp16 输入发生精度损失；Accelerate dispatch 下移动带 hook 的模块有设备风险。v5 按层设备创建局部固定频率 RoPE，保留原 dtype；核心直接按目标位置编码。 |
| 较低 | cache clone/slice/concat 直接赋值张量列表 | `_seen_tokens` 与实际长度不一致。4.50 Llama 显式 cache_position 路径主要用 `get_seq_length()`，不能声称这就是当前 EOS 根因。v5 用 cache update 重建计数。 |
| 较低 | prefix 长度校验前已改写对象；缺少重复节点、round 等验证 | 失败操作留下部分状态，重复节点可重复计入注意力。v5 增加原子 prefix 校验及约束。 |
| 较低 | citation 生成先 decode query 再重新 tokenize | 无法保证 token 序列不变。v5 直接传 query_input_ids。 |

已在真实随机 Llama 上运行原 v4 得到：source 长度 3、target 长度 6 时，target 的逻辑区间是 `[3,9)`，query 却从 6 开始；返回 logits 的 `requires_grad=True`；clone 后 `_seen_tokens=0` 而实际长度为 3；后加入长度 4 的 source 后传播抛出布局冻结异常。原始结果见 `v4_reproduction.json`。

注意：你提供的 Paris RAG 例子中，target 就是 source 文本的副本，没有超长 target，因此第一项确实是代码 bug，却不能解释该例的首 token EOS。

## 3. 两个 notebook 的问题及结果解释

### 单卡 diagnose_v4

1. GraphKV query 只有问题；顺序 baseline 添加了 `Evidence:`、`Question:`、`Answer:` 及不同分隔符。baseline 输出 Paris 而 GraphKV 输出 EOS，不是控制变量实验。
2. cell 19 的 `model(... past_key_values=...)` 会原地追加 query。cell 20 随后 `past = past_key_values`，再次输入完整 query，却沿用旧位置计数，造成重复 query 和物理长度不匹配。第一次 probe 的 EOS 发生在重复 query 之前，因此此 bug 影响后续 generation，不能倒过来解释首次 probe。
3. 新 notebook 每次 probe/generate 都从保存的节点重新拼接缓存，并验证重复 probe 一致、节点长度不变。

### position_ablation

1. 保存输出中 Sequential / Official / Paper 三者首 token 都是 EOS，概率约为 0.4457 / 0.3236 / 0.3738；paper 路径的 EOS 概率并不比 sequential 更高。它不支持“共享位置导致独有的空输出”这一结论。
2. 原仓库 `utils.py` 的 Tulu prompt 包含 `<|user|>\n` 和结尾 `<|assistant|>\n`，notebook 没有这些边界。模板不匹配是有证据支持的待检验原因；没有运行你的本地权重之前，不能保证加上标记必然解决所有问题。[固定提交的 utils.py](https://github.com/Graph-COM/GraphKV/blob/ba01bbcc98fc9a352582dd1e75ab88fa6cac0dfb/utils.py)
3. `F.kl_div(log_p_A, p_B)` 实际计算 KL(B||A)，标签却写 A||B。因此所有成对标签应交换。v5 直接计算 `sum(p_A*(log_p_A-log_p_B))`，另有非对称测试。
4. 顺序 baseline 额外插入换行并整体 tokenize；graph 则分别 tokenize。除了图结构和 PE，还混入 token 边界差异。新版 sequential 直接拼接同一批分块 IDs。
5. `official_result['cache']` 是已经被 query forward 修改的对象。所报 183 是 query 后长度；query 前为 `33 + 58 + 58 = 149`。论文方案 query 的逻辑起点是 69，物理起点 149，二者不同正确；不能把 183 当作原始已存文档 cache。
6. “Official 与 Paper 接近，所以差异主要来自 message passing”是过强的因果结论。旧实验同时改变 PE、KV 构建、浮点去旋转/再旋转、padding、prompt 分块等；单个 next-token KL 无法隔离主因。
7. `token_sequence_probs` 若答案包含多个 token，只是把多个词表概率都从同一个 next-token 分布中取出，不能解释为答案序列概率。此例 Paris 恰好是单 token，尚未触发该泛化错误。

新版 `GraphKV_validate_v5.ipynb` 同时提供 Tulu、plain、Answer cue 三个 prompt 条件，并在各条件内比较 sequential、independent、graph_full。它是修复后的诊断实验，不冒充逐字复现旧的 official-position 消融。

## 4. 验证及可信边界

18 项 CPU 测试通过。模型为随机初始化的三层小型 Llama，使用 GQA；运行环境 PyTorch 2.5.1+cpu、Transformers 4.50.0。关键检查：

- eager 与 SDPA × 稀疏图/全连接图/无边图：分阶段 cache 的每层 K/V 及 query logits 与一次性完整图掩码 forward 一致（atol=2e-6，rtol=2e-5）。
- 长 target、大于后续 source 的规划、多个子图边保存、source/target 排列变化、重复 probe 不改节点缓存。
- 逐 token greedy decode 与每一步完整重算的参考答案 token IDs 一致；多 EOS ID 停止逻辑。
- 默认及 Llama3 RoPE 重定位对照直接编码；兼容函数精度与 cache 计数；空 prefix、无 target、非法 round/k/重复节点；KL 方向。

真实 8B 权重、CUDA、FlashAttention2、跨 GPU device_map、原服务器端、论文全数据集指标未在当前机器上运行。这里的“验证通过”仅指以上测试范围，不能替代你的 Linux GPU 复测，也不证明论文效果已经复现。

v5 是以正确性为优先的参考重构：节点依次编码，尚未实现真正批量/分块并行；大量完整 KV 副本仍可能有显存压力。Cache API 固定到 4.50.0；不支持 static/sliding/quantized cache 或依赖序列长度动态变化的 RoPE。低层 API 允许显式跨轮构建，但本次主要验证一轮消息传递。

## 5. 建议运行顺序

先运行 `python -m pytest tests -q`。随后重启 notebook 内核，用新 notebook 运行同一个本地 Tulu3-Block-FT，先 SDPA 后 FlashAttention2，保存两个 JSON。先看正确 prompt 下的 raw token/EOS，再看 backend、排列和图结构差异。最后才进行固定数据集、多例与论文设置对齐的 EM/F1 等评估。
