# GraphKV refactor v5

这是对 Graph-COM/GraphKV 与用户 v4 重构的修订版，重点是可验证的 GraphKVCache 核心及 notebook 调用。完整问题分析见 [AUDIT_zh.md](AUDIT_zh.md)。原文件未被覆盖；输入副本在 `references/v4`，来源与校验值在 `SOURCE_MANIFEST.json`。

## 快速开始

在你已有的 Linux GraphKV 环境中，保留 PyTorch 2.5.1 / Transformers 4.50.0，安装 pytest，然后进入本目录：

```bash
python -m pytest tests -q
jupyter notebook GraphKV_validate_v5.ipynb
```

Notebook 中只需设置准确的 `MODEL_PATH`、`GPU_ID`、`BACKEND`。加载 8B 模型需要 accelerate；使用 flash_attention_2 还需要 Linux/CUDA 对应的 flash-attn。CPU 测试不下载模型权重，只生成小型随机 Llama，验证数学实现，不能测问答质量。

全新 CPU 环境可运行 `pip install -r requirements.txt`。Linux GPU 环境应按 CUDA 版本安装 PyTorch，而不是替换现有正确安装的 GPU 版本。

## 核心 API

```python
from graph_kv_adapter import build_graphkv_cache, greedy_generate_graphkv

model.eval()
gkv = build_graphkv_cache(
    model=model, tokenizer=tokenizer,
    source_texts={"s0": "France is in Europe.\n", "s1": "Paris is the capital of France.\n"},
    target_texts={"t0": "France is in Europe.\n", "t1": "Paris is the capital of France.\n"},
    edges=[("s0", "t0"), ("s1", "t0"), ("s0", "t1"), ("s1", "t1")],
    prefix="<|user|>\nAnswer using these documents.\n",
)
result = greedy_generate_graphkv(
    model=model, tokenizer=tokenizer, graph_cache=gkv,
    query="What is the capital of France?\n<|assistant|>\n",
    max_new_tokens=32, return_details=True,
)
print(result.text, result.token_ids, result.stopped_on_eos)
```

这里的 prompt 适用于仓库使用的 Tulu 模型；不要自动套用于其他 chat 模型。库不会修改调用者的 prompt。EOS 可以导致合法空字符串，`return_details=True` 会保留 token IDs 和停止原因。

底层 `GraphKVCache(source_position_start=P, chunk_span=L)` 中，`L` 必须覆盖所有 source 和 target 的长度。先注册所有 source，再传播 target；不能在布局冻结后添加 source。文本/ID builder 自动完成全图长度规划。低层 API 若省略 `chunk_span`，只以 source 最大长度为上界，超长 target 会明确报错。

## 文件与兼容范围

- `graph_kv_cache.py`：节点缓存、图边、位置规划、拼接、切片、传播与 query 准备。
- `graph_kv_adapter.py`：Llama 前向、tokenization、无副作用 probe、greedy generation。
- `pcw.py`：保留原 RAG `gapemp` / `gapemp_appr` / `vanilla` 调用签名。
- `pcw_parallel.py`：保留 citation graph / multi-graph 入口；batch 表示多个子图供同一个 query 使用，不是多个 query 的张量 batch。
- `kv_compat.py`：旧 notebook 所用的兼容函数；仍然原地修改参数。v5 核心不依赖手工去旋转/再旋转。
- 原仓库 inference/eval/server 文件按来源提交保留，未完成服务器端端到端评估；其说明保留为 `UPSTREAM_README.md`。

经过本地验证的范围：Transformers 4.50.0、PyTorch 2.5.1、Llama、普通完整 DynamicCache、batch size 1、无 padding 的单节点输入、固定频率 RoPE（default / llama3 / linear）。不声称支持新版本 Cache API、滑动窗口、量化/静态 cache、训练或动态缩放 RoPE。`model.eval()` 是必要条件。

默认路径按节点顺序执行，未实现真正的多节点 GPU 并行。逻辑位置跨度减少不代表物理 KV 显存减少；物理缓存仍包含所有选中 token。

## 有意改变的行为

1. GraphKV 使用论文共享位置设计，不再复现官方 RAG 的最终连续 re-RoPE 行为。
2. `pcw_parallel.block/block_batch` 是共享位置的独立块基线。它与 upstream 连续 re-RoPE 基线、以及独立的 `server/block_generate_server.py` 路径不同；不能混用其标签比较论文指标。
3. `gapemp_appr` 选择输入末尾 k 个 context，要求调用者已按相关性升序排列；没有排序分数就不能推断真实 top-k。`top_k<=0` 报错。
4. RAG 的非空 `middle` 被放到 query 前，不再静默丢弃。逻辑窗口预算按 `P+2L+Q+256` 计算，可能比旧版更早截断长段落。
5. 保留 `temperature/scale/mode` 参数是为了调用兼容，GraphKV 入口仍只进行 greedy 推理，不实现这些扩展模式。
6. 孤立 target 可在 target 区间独立编码；所有 citation 子图都没有邻居时，center 统一作为独立 source 块处理。

## 与原仓库集成

本目录已包含原仓库的调用脚本。如果合并到你现有 checkout，需要同时复制上述五个 Python 模块（包括新增 `kv_compat.py`），然后重启 notebook 内核；不要只替换一个文件。服务器额外依赖见原说明，服务器、数据下载与论文全量评估不属于本次已验证结果。

不需要移动或删除原有模型权重。新 notebook 会把诊断 JSON 写在当前目录，保留 raw token 和各 prompt 条件，便于后续定位真实模型问题。
