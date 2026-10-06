"""Drop-in replacement for Graph-COM/GraphKV ``pcw_parallel.py``.

``gapemp_graph`` and ``gapemp_graph_batch`` use GraphKVCache directly.  The
baseline ``block`` functions are kept as simple independent-KV baselines.
"""

from __future__ import annotations

import torch

from graph_kv_cache import GraphKVCache
from graph_kv_adapter import _encode_independent_chunk, _tokenize_one, greedy_generate_graphkv


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return ((k * cos) + (rotate_half(k) * sin)).to(dtype=torch.bfloat16)


def apply_pkv_rerotary_position_embeddings(pkv, emb, position_ids=None):
    device = pkv.key_cache[0].device
    emb.to(device=device)
    seq_len = pkv.key_cache[0].size(-2)
    if position_ids is None:
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
    position_ids = position_ids.unsqueeze(0).expand(pkv.key_cache[0].size(0), -1)
    cos, sin = emb(x=pkv.key_cache[0].to(dtype=torch.float32), position_ids=position_ids)
    for i in range(len(pkv.key_cache)):
        pkv.key_cache[i] = apply_rotary_pos_emb(
            pkv.key_cache[i].to(dtype=torch.float32), cos.to(pkv.key_cache[i].device),
            -sin.to(pkv.key_cache[i].device), position_ids.to(pkv.key_cache[i].device)
        ).to(dtype=pkv.key_cache[i].dtype)
    return pkv


def apply_pkv_rotary_position_embeddings(pkv, emb, position_ids=None):
    device = pkv.key_cache[0].device
    emb.to(device=device)
    seq_len = pkv.key_cache[0].size(-2)
    if position_ids is None:
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
    position_ids = position_ids.unsqueeze(0).expand(pkv.key_cache[0].size(0), -1)
    cos, sin = emb(x=pkv.key_cache[0].to(dtype=torch.float32), position_ids=position_ids)
    for i in range(len(pkv.key_cache)):
        pkv.key_cache[i] = apply_rotary_pos_emb(
            pkv.key_cache[i].to(dtype=torch.float32), cos.to(pkv.key_cache[i].device),
            sin.to(pkv.key_cache[i].device), position_ids.to(pkv.key_cache[i].device)
        ).to(dtype=pkv.key_cache[i].dtype)
    return pkv


def cut_pkv(pkv, positions):
    for layer_id in range(len(pkv.key_cache)):
        pkv.key_cache[layer_id] = pkv.key_cache[layer_id][:, :, positions, :]
        pkv.value_cache[layer_id] = pkv.value_cache[layer_id][:, :, positions, :]
    return pkv


def divide_pkv(pkv, split_id):
    after = type(pkv)()
    after.key_cache, after.value_cache = [], []
    for layer_id in range(len(pkv.key_cache)):
        after.key_cache.append(pkv.key_cache[layer_id][:, :, split_id:, :])
        after.value_cache.append(pkv.value_cache[layer_id][:, :, split_id:, :])
        pkv.key_cache[layer_id] = pkv.key_cache[layer_id][:, :, :split_id, :]
        pkv.value_cache[layer_id] = pkv.value_cache[layer_id][:, :, :split_id, :]
    return pkv, after


def flatten_pkv(pkv, mask):
    for layer_id in range(len(pkv.key_cache)):
        pkv.key_cache[layer_id] = (
            pkv.key_cache[layer_id].transpose(1, 2).flatten(0, 1)[mask]
            .unsqueeze(0).transpose(1, 2)
        )
        pkv.value_cache[layer_id] = (
            pkv.value_cache[layer_id].transpose(1, 2).flatten(0, 1)[mask]
            .unsqueeze(0).transpose(1, 2)
        )
    return pkv


def stack_pkv(pkv, bsz):
    for layer_id in range(len(pkv.key_cache)):
        pkv.key_cache[layer_id] = pkv.key_cache[layer_id].repeat(bsz, 1, 1, 1)
        pkv.value_cache[layer_id] = pkv.value_cache[layer_id].repeat(bsz, 1, 1, 1)
    return pkv


def concact_pkv(pkv1, pkv2):
    for layer_id in range(len(pkv2.key_cache)):
        pkv2.key_cache[layer_id] = torch.cat((pkv1.key_cache[layer_id], pkv2.key_cache[layer_id]), dim=2)
        pkv2.value_cache[layer_id] = torch.cat((pkv1.value_cache[layer_id], pkv2.value_cache[layer_id]), dim=2)
    return pkv2


def concact_pkv_before(pkv1, pkv2):
    for layer_id in range(len(pkv1.key_cache)):
        pkv1.key_cache[layer_id] = torch.cat((pkv1.key_cache[layer_id], pkv2.key_cache[layer_id]), dim=2)
        pkv1.value_cache[layer_id] = torch.cat((pkv1.value_cache[layer_id], pkv2.value_cache[layer_id]), dim=2)
    return pkv1


def topk_pkv(pkv, top_k):
    for layer_id in range(len(pkv.key_cache)):
        pkv.key_cache[layer_id] = pkv.key_cache[layer_id][-top_k:, :, :, :]
        pkv.value_cache[layer_id] = pkv.value_cache[layer_id][-top_k:, :, :, :]
    return pkv


def init_empty_pkv(pkv_example, total_length):
    new_pkv = type(pkv_example)()
    new_pkv.key_cache, new_pkv.value_cache = [], []
    for layer_id in range(len(pkv_example.key_cache)):
        B, H, _, D = pkv_example.key_cache[layer_id].shape
        new_pkv.key_cache.append(torch.empty(
            B, H, total_length, D,
            dtype=pkv_example.key_cache[layer_id].dtype,
            device=pkv_example.key_cache[layer_id].device,
        ))
        new_pkv.value_cache.append(torch.empty(
            B, H, total_length, D,
            dtype=pkv_example.value_cache[layer_id].dtype,
            device=pkv_example.value_cache[layer_id].device,
        ))
    return new_pkv


def _build_arxiv_graph_cache(tokenizer, model, emb, prefix, center_node, neighbor_nodes, query):
    device = model.device
    prefix_ids = _tokenize_one(tokenizer, prefix, device)
    query_ids = _tokenize_one(tokenizer, query, device)
    middle = '\nNow you will read the center paper and answer a related question: \n'
    center_text = middle + center_node
    center_ids = _tokenize_one(tokenizer, center_text, device)

    gkv = GraphKVCache(source_position_start=int(prefix_ids.shape[-1]))
    gkv.set_prefix_cache(_encode_independent_chunk(model, prefix_ids))

    source_ids = []
    for idx, text in enumerate(neighbor_nodes):
        sid = f"neighbor::{idx}"
        cache = _encode_independent_chunk(
            model,
            _tokenize_one(tokenizer, text, device),
            position_start=gkv.source_position_start,
        )
        gkv.add_source(sid, cache, assume_positioned=True)
        source_ids.append(sid)

    if not source_ids:
        raise ValueError("center_node must have at least one neighbor source")

    target_id = "center::0"
    gkv.set_edges((sid, target_id) for sid in source_ids)
    gkv.propagate_target(model, target_id, center_ids, source_ids=source_ids, round_idx=1)
    gkv.validate()
    return gkv, query_ids, source_ids, [target_id]


def gapemp_graph(tokenizer, model, emb, prefix, center_node, neighbor_nodes, query,
                 model_name, temperature, scale, mode):
    """ARXIV-QA Graph-KV: center paper attends only to its citation neighbors."""
    with torch.no_grad():
        gkv, query_ids, source_ids, target_ids = _build_arxiv_graph_cache(
            tokenizer, model, emb, prefix, center_node, neighbor_nodes, query
        )
        return greedy_generate_graphkv(
            model=model,
            tokenizer=tokenizer,
            graph_cache=gkv,
            query=tokenizer.decode(query_ids[0].tolist(), skip_special_tokens=False),
            max_new_tokens=256,
            source_ids=source_ids,
            target_ids=target_ids,
            skip_special_tokens=True,
        )


def gapemp_graph_batch(tokenizer, model, emb, prefix, center_node_list, neighbor_nodes_list,
                       query, model_name, temperature, scale, mode):
    """Multi-graph variant used by the original ARXIV distractor evaluation.

    All citation ego-graphs are represented in one GraphKVCache; each center
    target is connected only to its own neighbor sources.  Targets in the same
    propagation round share the same logical target range, matching the
    permutation-insensitive structural layout.
    """
    with torch.no_grad():
        device = model.device
        prefix_ids = _tokenize_one(tokenizer, prefix, device)
        query_ids = _tokenize_one(tokenizer, query, device)
        gkv = GraphKVCache(source_position_start=int(prefix_ids.shape[-1]))
        gkv.set_prefix_cache(_encode_independent_chunk(model, prefix_ids))

        center_target_ids = []
        all_source_ids = []
        for graph_idx, (center_node, neighbor_nodes) in enumerate(
            zip(center_node_list, neighbor_nodes_list)
        ):
            local_sources = []
            for n_idx, text in enumerate(neighbor_nodes):
                sid = f"graph{graph_idx}::neighbor{n_idx}"
                cache = _encode_independent_chunk(
                    model,
                    _tokenize_one(tokenizer, text, device),
                    position_start=gkv.source_position_start,
                )
                gkv.add_source(sid, cache, assume_positioned=True)
                local_sources.append(sid)
                all_source_ids.append(sid)
            if not local_sources:
                raise ValueError(f"graph {graph_idx} has no neighbors")
            target_id = f"graph{graph_idx}::center"
            gkv.set_edges((sid, target_id) for sid in local_sources)
            middle = '\nNow you will read the center paper and answer a related question: \n'
            center_ids = _tokenize_one(tokenizer, middle + center_node, device)
            gkv.propagate_target(model, target_id, center_ids, source_ids=local_sources, round_idx=1)
            center_target_ids.append(target_id)

        return greedy_generate_graphkv(
            model=model,
            tokenizer=tokenizer,
            graph_cache=gkv,
            query=tokenizer.decode(query_ids[0].tolist(), skip_special_tokens=False),
            max_new_tokens=256,
            source_ids=all_source_ids,
            target_ids=center_target_ids,
            skip_special_tokens=True,
        )


# -------------------------------------------------------------------------
# Baseline Block-RAG compatibility implementation
# -------------------------------------------------------------------------

def block(tokenizer, model, emb, prefix, center_node, neighbor_nodes, query,
          model_name, temperature, scale, mode):
    with torch.no_grad():
        device = model.device
        prefix_ids = _tokenize_one(tokenizer, prefix, device)
        query_ids = _tokenize_one(tokenizer, query, device)
        texts = list(neighbor_nodes) + ['\nNow you will read the center paper and answer a related question: \n', center_node]
        caches = [_encode_independent_chunk(model, _tokenize_one(tokenizer, t, device)) for t in texts]
        prefix_cache = _encode_independent_chunk(model, prefix_ids)
        merged = GraphKVCache.concat_caches(prefix_cache, *caches)
        # Use a tiny compatibility wrapper for generation from an externally assembled cache.
        class _Wrap:
            pass
        w = _Wrap()
        w.prefix_cache = None
        w.nodes = {'all': merged}
        w.meta = {}
        w.graph = {}
        w._max_round = 0
        # Direct generation keeps the baseline contiguous positions.
        generated = query_ids
        past = merged
        position_ids = torch.arange(merged.key_cache[0].shape[-2], merged.key_cache[0].shape[-2] + query_ids.shape[-1], device=device).unsqueeze(0)
        cache_position = torch.arange(merged.key_cache[0].shape[-2], merged.key_cache[0].shape[-2] + query_ids.shape[-1], device=device)
        answer = []
        for _ in range(256):
            out = model(generated, past_key_values=past, position_ids=position_ids, cache_position=cache_position, use_cache=True)
            past = out.past_key_values
            token = out.logits[:, -1].argmax(dim=-1, keepdim=True)
            if tokenizer.eos_token_id is not None and int(token.item()) == int(tokenizer.eos_token_id):
                break
            answer.append(int(token.item()))
            generated = token
            position_ids = position_ids[:, -1:] + 1
            cache_position = cache_position[-1:] + 1
        return tokenizer.decode(answer, skip_special_tokens=True)


def block_batch(tokenizer, model, emb, prefix, center_node_list, neighbor_nodes_list, query,
                model_name, temperature, scale, mode):
    # Preserve the original batch API by concatenating all independently encoded blocks.
    with torch.no_grad():
        device = model.device
        prefix_ids = _tokenize_one(tokenizer, prefix, device)
        query_ids = _tokenize_one(tokenizer, query, device)
        caches = [_encode_independent_chunk(model, prefix_ids)]
        for centers, neighbors in zip(center_node_list, neighbor_nodes_list):
            for text in list(neighbors) + ['\nNow you will read the center paper and answer a related question: \n', centers]:
                caches.append(_encode_independent_chunk(model, _tokenize_one(tokenizer, text, device)))
        merged = GraphKVCache.concat_caches(*caches)
        generated = query_ids
        past = merged
        pos = torch.arange(merged.key_cache[0].shape[-2], merged.key_cache[0].shape[-2] + query_ids.shape[-1], device=device).unsqueeze(0)
        cache_pos = torch.arange(merged.key_cache[0].shape[-2], merged.key_cache[0].shape[-2] + query_ids.shape[-1], device=device)
        answer = []
        for _ in range(256):
            out = model(generated, past_key_values=past, position_ids=pos, cache_position=cache_pos, use_cache=True)
            past = out.past_key_values
            token = out.logits[:, -1].argmax(dim=-1, keepdim=True)
            if tokenizer.eos_token_id is not None and int(token.item()) == int(tokenizer.eos_token_id):
                break
            answer.append(int(token.item()))
            generated = token
            pos = pos[:, -1:] + 1
            cache_pos = cache_pos[-1:] + 1
        return tokenizer.decode(answer, skip_special_tokens=True)


__all__ = [
    'gapemp_graph', 'gapemp_graph_batch', 'block', 'block_batch',
    'apply_pkv_rerotary_position_embeddings', 'apply_pkv_rotary_position_embeddings',
    'cut_pkv', 'divide_pkv', 'flatten_pkv', 'stack_pkv', 'concact_pkv',
    'concact_pkv_before', 'topk_pkv', 'init_empty_pkv'
]
