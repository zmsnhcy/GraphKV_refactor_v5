"""Legacy helpers for old notebooks. Core v5 does not re-rotate K.

Supports Transformers 4.50 DynamicCache. Helpers retain upstream in-place
semantics, so never pass a registered node directly to a mutating helper.
"""
import torch
from graph_kv_cache import GraphKVCache

rotate_half = GraphKVCache._rotate_half


def _sync(cache):
    if hasattr(cache, "_seen_tokens"):
        cache._seen_tokens = GraphKVCache._seq_len(cache)
    return cache


def apply_rotary_pos_emb(k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos, sin = cos.unsqueeze(unsqueeze_dim), sin.unsqueeze(unsqueeze_dim)
    x = k.float()
    return (x * cos + rotate_half(x) * sin).to(k.dtype)


def _rotate(cache, emb, positions, inverse):
    GraphKVCache._validate_cache(cache, require_batch1=False)
    if getattr(emb, "rope_type", "default") not in ("default", "llama3", "linear"):
        raise ValueError("Legacy re-RoPE only supports fixed-frequency embeddings")
    local_modules = {}
    for i, key in enumerate(cache.key_cache):
        if key.device not in local_modules:
            local_modules[key.device] = GraphKVCache._device_local_rotary_embedding(emb, key.device)
        pos = (torch.arange(key.shape[-2], device=key.device) if positions is None
               else positions.to(device=key.device))
        if pos.ndim == 1:
            pos = pos.unsqueeze(0).expand(key.shape[0], -1)
        if pos.shape != (key.shape[0], key.shape[-2]):
            raise ValueError("positions must match cache batch and length")
        cos, sin = local_modules[key.device](key.float(), pos)
        if inverse:
            denominator = cos.square() + sin.square()
            cos, sin = cos / denominator, -sin / denominator
        cache.key_cache[i] = apply_rotary_pos_emb(key, cos, sin)
    return cache


def apply_pkv_rerotary_position_embeddings(pkv, emb, position_ids=None):
    return _rotate(pkv, emb, position_ids, True)


def apply_pkv_rotary_position_embeddings(pkv, emb, position_ids=None):
    return _rotate(pkv, emb, position_ids, False)


def cut_pkv(pkv, positions):
    for attr in ("key_cache", "value_cache"):
        setattr(pkv, attr, [x[:, :, positions.to(x.device) if torch.is_tensor(positions) else positions, :]
                            for x in getattr(pkv, attr)])
    return _sync(pkv)


def divide_pkv(pkv, split_id):
    after = GraphKVCache.slice_cache(pkv, split_id, GraphKVCache._seq_len(pkv))
    cut_pkv(pkv, slice(0, split_id))
    return pkv, after


def flatten_pkv(pkv, mask):
    for attr in ("key_cache", "value_cache"):
        setattr(pkv, attr, [x.transpose(1,2).flatten(0,1)[mask.to(x.device)].unsqueeze(0).transpose(1,2)
                            for x in getattr(pkv, attr)])
    return _sync(pkv)


def stack_pkv(pkv, bsz):
    if bsz <= 0: raise ValueError("bsz must be positive")
    for attr in ("key_cache", "value_cache"):
        setattr(pkv, attr, [x.repeat(bsz,1,1,1) for x in getattr(pkv, attr)])
    return _sync(pkv)


def _concat_into(destination, left, right):
    if len(left.key_cache) != len(right.key_cache):
        raise ValueError("cache layer counts differ")
    for attr in ("key_cache", "value_cache"):
        setattr(destination, attr, [torch.cat((a,b),2) for a,b in zip(getattr(left, attr), getattr(right, attr))])
    return _sync(destination)


def concact_pkv(pkv1, pkv2):
    return _concat_into(pkv2, pkv1, pkv2)


def concact_pkv_before(pkv1, pkv2):
    return _concat_into(pkv1, pkv1, pkv2)


def topk_pkv(pkv, top_k):
    if not isinstance(top_k, int) or top_k <= 0: raise ValueError("top_k must be positive")
    for attr in ("key_cache", "value_cache"):
        setattr(pkv, attr, [x[-top_k:] for x in getattr(pkv, attr)])
    return _sync(pkv)


def init_empty_pkv(pkv_example, total_length):
    if total_length < 0: raise ValueError("total_length must be nonnegative")
    out = type(pkv_example)()
    for attr in ("key_cache", "value_cache"):
        setattr(out, attr, [x.new_zeros(x.shape[0],x.shape[1],total_length,x.shape[3])
                            for x in getattr(pkv_example, attr)])
    return _sync(out)
