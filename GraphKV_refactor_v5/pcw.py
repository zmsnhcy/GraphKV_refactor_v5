"""RAG entry points preserving upstream call signatures; greedy inference only."""
import torch
from kv_compat import (rotate_half, apply_rotary_pos_emb,
    apply_pkv_rerotary_position_embeddings, apply_pkv_rotary_position_embeddings,
    cut_pkv, flatten_pkv, stack_pkv, concact_pkv, topk_pkv)
from graph_kv_adapter import (input_device, _tokenize_one, build_graphkv_from_ids,
                              greedy_generate_graphkv)


def _normalize_contexts(contexts):
    return [contexts] if isinstance(contexts, str) else list(contexts)


def _build_rag_graph_cache(tokenizer, model, emb, prefix, query, contexts,
                         *, top_k=None, max_length=8192):
    contexts = _normalize_contexts(contexts)
    if not contexts:
        raise ValueError("contexts must not be empty")
    if top_k is not None and (not isinstance(top_k, int) or top_k <= 0):
        raise ValueError("top_k must be a positive integer")
    if max_length <= 0:
        raise ValueError("Prompt leaves no context capacity")
    device = input_device(model)
    sources = {f"source::{i}": tokenizer(text, truncation=True, max_length=max_length,
        return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        for i, text in enumerate(contexts)}
    targets = {f"target::{i}": ids for i, ids in enumerate(sources.values())}
    chosen = list(sources) if top_k is None else list(sources)[-top_k:]
    gkv = build_graphkv_from_ids(model=model, source_inputs=sources, target_inputs=targets,
        edges=[(s, t) for t in targets for s in chosen],
        prefix_input_ids=_tokenize_one(tokenizer, prefix, device) if prefix else None)
    return gkv, _tokenize_one(tokenizer, query, device), list(sources), list(targets)


def _rag_generate(tokenizer, model, gkv, query_ids, source_ids, target_ids):
    return greedy_generate_graphkv(model=model, tokenizer=tokenizer, graph_cache=gkv,
        query_input_ids=query_ids, source_ids=source_ids, target_ids=target_ids)


def _rag(tokenizer, model, emb, prefix, middle, query, contexts, top_k):
    # Unlike upstream, a nonempty middle is not silently discarded.
    query = middle + query
    device = input_device(model)
    p = _tokenize_one(tokenizer, prefix, device).shape[-1]
    q = _tokenize_one(tokenizer, query, device).shape[-1]
    limit = min(8192, model.config.max_position_embeddings)
    # P + 2L + Q + generation budget must fit the logical window.
    max_length = (limit - p - q - 256) // 2
    data = _build_rag_graph_cache(tokenizer, model, emb, prefix, query, contexts,
                                 top_k=top_k, max_length=max_length)
    return _rag_generate(tokenizer, model, *data)


def gapemp(tokenizer, model, emb, prefix, middle, query, contexts,
           model_name, temperature, scale, mode):
    return _rag(tokenizer, model, emb, prefix, middle, query, contexts, None)


def gapemp_appr(tokenizer, model, emb, prefix, middle, query, contexts,
                model_name, temperature, scale, top_k):
    """Select LAST k input contexts. Caller must order by ascending relevance."""
    return _rag(tokenizer, model, emb, prefix, middle, query, contexts, top_k)


@torch.no_grad()
def vanilla(tokenizer, model, prompt, temperature=1, scale=1, mode=None):
    ids = _tokenize_one(tokenizer, prompt, input_device(model))
    out = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False,
                         use_cache=True, max_new_tokens=512)
    return tokenizer.decode(out[0, ids.shape[-1]:].tolist(), skip_special_tokens=False)
