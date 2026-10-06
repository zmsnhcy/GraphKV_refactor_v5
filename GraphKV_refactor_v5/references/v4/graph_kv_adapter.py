"""HuggingFace adapter for the refactored GraphKVCache.

This file is intentionally thin: GraphKVCache owns graph/KV state, while this
adapter owns tokenization, model prefill, and greedy decoding.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

from graph_kv_cache import GraphKVCache


def _tokenize_one(tokenizer, text: str, device: torch.device, *, add_special_tokens: bool = False):
    """Tokenize one text chunk and place input IDs on the model input device."""
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text)!r}")
    ids = tokenizer(
        text,
        return_tensors="pt",
        truncation=False,
        add_special_tokens=add_special_tokens,
    ).input_ids
    return ids.to(device)


def _encode_independent_chunk(
    model,
    input_ids: torch.Tensor,
    *,
    position_start: int = 0,
):
    """Encode one chunk independently with an optional absolute RoPE start.

    ``position_start`` controls logical RoPE only. The returned DynamicCache
    remains a fresh physical cache beginning at slot 0.
    """
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("input_ids must have shape [1, seq_len]")
    if position_start < 0:
        raise ValueError("position_start must be >= 0")

    position_ids = (
        position_start
        + torch.arange(input_ids.shape[-1], device=input_ids.device)
    ).unsqueeze(0)
    cache_position = torch.arange(
        input_ids.shape[-1], device=input_ids.device
    )

    with torch.no_grad():
        outputs = model(
            input_ids,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=True,
        )
    return GraphKVCache._clone_cache(outputs.past_key_values)


def build_graphkv_cache(
    *,
    model,
    tokenizer,
    source_texts: Mapping[str, str],
    edges: Iterable[Tuple[str, str]],
    target_texts: Mapping[str, str],
) -> GraphKVCache:
    """Build a one-hop Graph-KV cache from text chunks.

    The function mirrors the paper's one-hop design: source chunks are independently
    prefilling in parallelizable units; each target then attends only to the KV of
    its graph predecessors.
    """
    device = next(model.parameters()).device
    gkv = GraphKVCache()
    gkv.set_edges(edges)

    # Source nodes: each independently starts at RoPE position 0.
    for node_id, text in source_texts.items():
        ids = tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=False,
        ).input_ids.to(device)
        cache = _encode_independent_chunk(model, ids)
        gkv.add_source(node_id, cache)

    # Target propagation is intentionally explicit so each target gets only the
    # source KV specified by the graph.
    for node_id, text in target_texts.items():
        ids = tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=False,
        ).input_ids.to(device)
        gkv.propagate_target(model, node_id, ids)

    gkv.validate()
    return gkv


def greedy_generate_graphkv(
    *,
    model,
    tokenizer,
    graph_cache: GraphKVCache,
    query: str = "",
    query_input_ids: Optional[torch.Tensor] = None,
    max_new_tokens: int = 256,
    source_ids: Optional[Sequence[str]] = None,
    target_ids: Optional[Sequence[str]] = None,
    skip_special_tokens: bool = True,
) -> str:
    """Greedy decode from a prepared Graph-KV cache."""
    device = next(model.parameters()).device
    if query_input_ids is not None:
        if query_input_ids.ndim != 2 or query_input_ids.shape[0] != 1:
            raise ValueError("query_input_ids must have shape [1, seq_len]")
        query_ids = query_input_ids.to(device)
    else:
        query_ids = _tokenize_one(tokenizer, query, device)

    past_key_values, position_ids, cache_position = graph_cache.prepare_query(
        query_ids,
        source_ids=source_ids,
        target_ids=target_ids,
    )

    generated = query_ids.clone()
    current_input = query_ids
    eos_id = tokenizer.eos_token_id

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(
                current_input,
                past_key_values=past_key_values,
                position_ids=position_ids,
                cache_position=cache_position,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)

            if eos_id is not None and int(next_token.item()) == int(eos_id):
                break

            # After the initial query step, every new token has one logical RoPE
            # position and one physical cache slot. The two counters advance
            # independently.
            position_ids = position_ids[:, -1:] + 1
            cache_position = cache_position[-1:] + 1
            current_input = next_token

    answer_ids = generated[:, query_ids.shape[-1] :]
    return tokenizer.decode(
        answer_ids[0].tolist(),
        skip_special_tokens=skip_special_tokens,
    )


__all__ = ["_tokenize_one", "_encode_independent_chunk", "build_graphkv_cache", "greedy_generate_graphkv"]
