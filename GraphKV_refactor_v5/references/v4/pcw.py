"""Drop-in replacement for Graph-COM/GraphKV ``pcw.py``.

The Graph-KV paths are refactored to use ``GraphKVCache`` while keeping the
original public function signatures intact.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

try:
    from transformers.cache_utils import DynamicCache
except Exception:  # pragma: no cover - keeps helper module importable in toy tests
    DynamicCache = object

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
    """Compatibility helper: remove RoPE from the cache K tensors."""
    device = pkv.key_cache[0].device
    emb.to(device=device)
    seq_len = pkv.key_cache[0].size(-2)
    if position_ids is None:
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
    position_ids = position_ids.unsqueeze(0).expand(pkv.key_cache[0].size(0), -1)
    cos, sin = emb(x=pkv.key_cache[0].to(dtype=torch.float32), position_ids=position_ids)
    for i in range(len(pkv.key_cache)):
        new_device = pkv.key_cache[i].device
        cos_i, sin_i = cos.to(new_device), sin.to(new_device)
        pkv.key_cache[i] = apply_rotary_pos_emb(
            pkv.key_cache[i].to(dtype=torch.float32), cos_i, -sin_i, position_ids
        ).to(dtype=pkv.key_cache[i].dtype)
    return pkv


def apply_pkv_rotary_position_embeddings(pkv, emb, position_ids=None):
    """Compatibility helper: apply the supplied logical RoPE positions to K."""
    device = pkv.key_cache[0].device
    emb.to(device=device)
    seq_len = pkv.key_cache[0].size(-2)
    if position_ids is None:
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
    position_ids = position_ids.to(device).unsqueeze(0).expand(pkv.key_cache[0].size(0), -1)
    cos, sin = emb(x=pkv.key_cache[0].to(dtype=torch.float32), position_ids=position_ids)
    for i in range(len(pkv.key_cache)):
        new_device = pkv.key_cache[i].device
        pkv.key_cache[i] = apply_rotary_pos_emb(
            pkv.key_cache[i].to(dtype=torch.float32),
            cos.to(new_device),
            sin.to(new_device),
            position_ids.to(new_device),
        ).to(dtype=pkv.key_cache[i].dtype)
    return pkv


def cut_pkv(pkv, positions):
    for layer_id in range(len(pkv.key_cache)):
        pkv.key_cache[layer_id] = pkv.key_cache[layer_id][:, :, positions, :]
        pkv.value_cache[layer_id] = pkv.value_cache[layer_id][:, :, positions, :]
    return pkv


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
        pkv2.key_cache[layer_id] = torch.cat(
            (pkv1.key_cache[layer_id], pkv2.key_cache[layer_id]), dim=2
        )
        pkv2.value_cache[layer_id] = torch.cat(
            (pkv1.value_cache[layer_id], pkv2.value_cache[layer_id]), dim=2
        )
    return pkv2


def topk_pkv(pkv, top_k):
    for layer_id in range(len(pkv.key_cache)):
        pkv.key_cache[layer_id] = pkv.key_cache[layer_id][-top_k:, :, :, :]
        pkv.value_cache[layer_id] = pkv.value_cache[layer_id][-top_k:, :, :, :]
    return pkv


def vanilla(tokenizer, model, prompt, temperature=1, scale=1, mode=None):
    with torch.no_grad():
        input_ids = tokenizer(
            prompt, truncation=False, return_tensors="pt", add_special_tokens=False
        ).input_ids
        response = model.generate(
            input_ids=input_ids.to(model.device),
            use_cache=True,
            eos_token_id=[tokenizer.eos_token_id],
            tokenizer=tokenizer,
            max_new_tokens=512,
        )[0]
        return tokenizer.decode(
            response[input_ids.shape[-1]:].tolist(), skip_special_tokens=False
        )


def _normalize_contexts(contexts):
    if isinstance(contexts, str):
        return [contexts]
    return list(contexts)


def _build_rag_graph_cache(
    tokenizer,
    model,
    emb,
    prefix,
    query,
    contexts,
    *,
    top_k: Optional[int] = None,
    max_length: int,
):
    device = model.device
    contexts = _normalize_contexts(contexts)
    prefix_ids = _tokenize_one(tokenizer, prefix, device)
    query_ids = _tokenize_one(tokenizer, query, device)

    gkv = GraphKVCache(source_position_start=int(prefix_ids.shape[-1]))
    prefix_cache = _encode_independent_chunk(model, prefix_ids)
    gkv.set_prefix_cache(prefix_cache)

    source_ids = []
    for idx, text in enumerate(contexts):
        node_id = f"source::{idx}"
        ids = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids.to(device)
        cache = _encode_independent_chunk(
            model, ids, position_start=gkv.source_position_start
        )
        gkv.add_source(node_id, cache, assume_positioned=True)
        source_ids.append(node_id)

    if not source_ids:
        raise ValueError("contexts must contain at least one text chunk")

    chosen_source_ids = source_ids if top_k is None else source_ids[-min(top_k, len(source_ids)):]
    target_ids = []
    for idx, text in enumerate(contexts):
        target_id = f"target::{idx}"
        gkv.add_edge(chosen_source_ids[0], target_id)
        for sid in chosen_source_ids[1:]:
            gkv.add_edge(sid, target_id)
        target_text_ids = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids.to(device)
        gkv.propagate_target(
            model,
            target_id,
            target_text_ids,
            source_ids=chosen_source_ids,
            round_idx=1,
        )
        target_ids.append(target_id)

    gkv.validate()
    return gkv, query_ids, source_ids, target_ids


def _rag_generate(
    tokenizer,
    model,
    gkv,
    query_ids,
    source_ids,
    target_ids,
):
    # Use the adapter's exact physical/logical position handling.
    query_text = tokenizer.decode(query_ids[0].tolist(), skip_special_tokens=False)
    return greedy_generate_graphkv(
        model=model,
        tokenizer=tokenizer,
        graph_cache=gkv,
        query=query_text,
        query_input_ids=query_ids,
        max_new_tokens=256,
        source_ids=source_ids,
        target_ids=target_ids,
        skip_special_tokens=True,
    )


def gapemp(tokenizer, model, emb, prefix, middle, query, contexts,
           model_name, temperature, scale, mode):
    """Graph-KV Full: every target context reads every source context."""
    with torch.no_grad():
        prefix_len = int(_tokenize_one(tokenizer, prefix, model.device).shape[-1])
        middle_len = int(_tokenize_one(tokenizer, middle, model.device).shape[-1])
        query_len = int(_tokenize_one(tokenizer, query, model.device).shape[-1])
        max_length = max(1, 8192 - prefix_len - query_len - middle_len - 256)
        gkv, query_ids, source_ids, target_ids = _build_rag_graph_cache(
            tokenizer, model, emb, prefix, query, contexts,
            top_k=None, max_length=max_length,
        )
        return _rag_generate(
            tokenizer, model, gkv, query_ids, source_ids, target_ids
        )


def gapemp_appr(tokenizer, model, emb, prefix, middle, query, contexts,
                model_name, temperature, scale, top_k):
    """Graph-KV Top-k: every target reads the highest-scoring k chunks.

    The original repository orders retrieved chunks by ascending similarity;
    therefore the last ``top_k`` chunks correspond to the top-k retrievals.
    """
    with torch.no_grad():
        prefix_len = int(_tokenize_one(tokenizer, prefix, model.device).shape[-1])
        middle_len = int(_tokenize_one(tokenizer, middle, model.device).shape[-1])
        query_len = int(_tokenize_one(tokenizer, query, model.device).shape[-1])
        max_length = max(1, 8192 - prefix_len - query_len - middle_len - 256)
        gkv, query_ids, source_ids, target_ids = _build_rag_graph_cache(
            tokenizer, model, emb, prefix, query, contexts,
            top_k=int(top_k), max_length=max_length,
        )
        return _rag_generate(
            tokenizer, model, gkv, query_ids, source_ids, target_ids
        )


__all__ = [
    "vanilla", "gapemp", "gapemp_appr", "apply_pkv_rerotary_position_embeddings",
    "apply_pkv_rotary_position_embeddings", "cut_pkv", "flatten_pkv", "stack_pkv",
    "concact_pkv", "topk_pkv",
]
