"""Inference adapter for Transformers 4.50.0 Llama with full DynamicCache.

No hidden prompt rewriting. Token IDs, logical positions and physical slots
are kept explicit. Source/prefix blocks are independent, as in the paper.
"""
from dataclasses import dataclass
from typing import Optional
import torch
from graph_kv_cache import GraphKVCache


def input_device(model):
    return model.get_input_embeddings().weight.device


def _check_model(model):
    import transformers
    if transformers.__version__ != "4.50.0":
        raise RuntimeError("This adapter is validated for transformers==4.50.0; use the pinned environment")
    if model.config.model_type != "llama":
        raise ValueError("Only Llama models are supported by this adapter")
    if model.training:
        raise ValueError("Call model.eval() before inference")
    rope = model.config.rope_scaling or {}
    if rope.get("rope_type", rope.get("type", "default")) not in ("default", "llama3", "linear"):
        raise ValueError("Sequence-length-dependent RoPE is unsupported for reusable independent caches")


def _tokenize_one(tokenizer, text, device, *, add_special_tokens=False):
    if not isinstance(text, str):
        raise TypeError("text must be str")
    return tokenizer(text, return_tensors="pt", truncation=False,
                     add_special_tokens=add_special_tokens).input_ids.to(device)


@torch.no_grad()
def _encode_independent_chunk(model, input_ids, *, position_start=0):
    _check_model(model)
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[-1] == 0:
        raise ValueError("input_ids must have shape [1, nonzero length]")
    if position_start < 0:
        raise ValueError("position_start must be nonnegative")
    slots = torch.arange(input_ids.shape[-1], device=input_ids.device)
    out = model(input_ids, position_ids=(slots + position_start).unsqueeze(0),
                cache_position=slots, attention_mask=torch.ones_like(input_ids),
                use_cache=True, return_dict=True)
    return GraphKVCache._clone_cache(out.past_key_values)


@torch.no_grad()
def build_graphkv_from_ids(*, model, source_inputs, target_inputs, edges,
                           prefix_input_ids=None, chunk_span=None):
    """Plan ALL lengths before encoding; edges are (source ID, target ID).

    One hop only here. An isolated target is independently encoded in the target
    position range. The low-level cache also supports explicitly planned rounds.
    """
    _check_model(model)
    if not source_inputs:
        raise ValueError("At least one source is required")
    if set(source_inputs) & set(target_inputs):
        raise ValueError("Source and target IDs must be disjoint; use s:: and t:: namespaces")
    device = input_device(model)
    chunks = {**source_inputs, **target_inputs}
    for ids in chunks.values():
        if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[-1] == 0:
            raise ValueError("Every node must contain unpadded [1, nonzero length] IDs")
    edges = list(edges)
    for source, target in edges:
        if source not in source_inputs or target not in target_inputs:
            raise ValueError(f"Invalid one-hop edge: {source!r} -> {target!r}")
    longest = max(ids.shape[-1] for ids in chunks.values())
    span = longest if chunk_span is None else chunk_span
    if span < longest:
        raise ValueError("chunk_span must include the longest source AND target")
    prefix_len = 0 if prefix_input_ids is None else prefix_input_ids.shape[-1]
    gkv = GraphKVCache(prefix_len, chunk_span=span)
    if prefix_len:
        gkv.set_prefix_cache(_encode_independent_chunk(model, prefix_input_ids.to(device)))
    gkv.set_edges(edges)
    for name, ids in source_inputs.items():
        cache = _encode_independent_chunk(model, ids.to(device), position_start=prefix_len)
        gkv.add_source(name, cache, assume_positioned=True)
    for name, ids in target_inputs.items():
        gkv.propagate_target(model, name, ids.to(device))
    gkv.validate()
    return gkv


def build_graphkv_cache(*, model, tokenizer, source_texts, edges, target_texts,
                        prefix="", chunk_span=None):
    device = input_device(model)
    tok = lambda text: _tokenize_one(tokenizer, text, device)
    return build_graphkv_from_ids(model=model,
        source_inputs={k: tok(v) for k, v in source_texts.items()},
        target_inputs={k: tok(v) for k, v in target_texts.items()}, edges=edges,
        prefix_input_ids=tok(prefix) if prefix else None, chunk_span=chunk_span)


@dataclass
class GenerationResult:
    text: str
    token_ids: list
    stopped_on_eos: bool


@torch.no_grad()
def graphkv_next_logits(*, model, graph_cache, query_input_ids,
                        source_ids=None, target_ids=None):
    """A probe always starts from a fresh assembled cache; reusable nodes stay intact."""
    _check_model(model)
    ids = query_input_ids.to(input_device(model))
    cache, pos, slots = graph_cache.prepare_query(ids, source_ids=source_ids, target_ids=target_ids)
    past_length = GraphKVCache._seq_len(cache)
    out = model(ids, past_key_values=cache, position_ids=pos, cache_position=slots,
                attention_mask=torch.ones((1, past_length + ids.shape[-1]), dtype=torch.long, device=ids.device),
                use_cache=True, return_dict=True)
    return out.logits[:, -1].detach().float()


@torch.no_grad()
def greedy_generate_graphkv(*, model, tokenizer, graph_cache, query="",
                            query_input_ids=None, max_new_tokens=256,
                            source_ids=None, target_ids=None, skip_special_tokens=True,
                            return_details=False, eos_token_id=None):
    _check_model(model)
    if not isinstance(max_new_tokens, int) or max_new_tokens < 0:
        raise ValueError("max_new_tokens must be a nonnegative integer")
    device = input_device(model)
    ids = (_tokenize_one(tokenizer, query, device) if query_input_ids is None
           else query_input_ids.to(device))
    past, pos, slots = graph_cache.prepare_query(ids, source_ids=source_ids, target_ids=target_ids)
    eos = eos_token_id
    if eos is None:
        eos = getattr(model.generation_config, "eos_token_id", None)
    if eos is None:
        eos = tokenizer.eos_token_id
    eos = set([] if eos is None else ([eos] if isinstance(eos, int) else eos))
    tokens, stopped = [], False
    for _ in range(max_new_tokens):
        total = GraphKVCache._seq_len(past) + ids.shape[-1]
        out = model(ids, past_key_values=past, position_ids=pos, cache_position=slots,
                    attention_mask=torch.ones((1, total), dtype=torch.long, device=device),
                    use_cache=True, return_dict=True)
        past = out.past_key_values
        ids = out.logits[:, -1].argmax(-1, keepdim=True)
        token = int(ids.item())
        tokens.append(token)
        if token in eos:
            stopped = True
            break
        pos, slots = pos[:, -1:] + 1, slots[-1:] + 1
    text = tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)
    result = GenerationResult(text, tokens, stopped)
    return result if return_details else text


def kl_divergence(logits_a, logits_b):
    """KL(A || B), reduced over vocabulary, averaged over batch."""
    log_a, log_b = logits_a.float().log_softmax(-1), logits_b.float().log_softmax(-1)
    return (log_a.exp() * (log_a - log_b)).sum(-1).mean().item()
