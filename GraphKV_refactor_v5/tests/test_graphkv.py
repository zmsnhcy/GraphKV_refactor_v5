"""Real tiny Llama tests: no downloaded weights, no GPU required."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from graph_kv_cache import GraphKVCache
from graph_kv_adapter import (build_graphkv_from_ids, graphkv_next_logits,
    greedy_generate_graphkv, _encode_independent_chunk, kl_divergence)

torch.set_num_threads(1)


def tiny(backend="eager", rope=None):
    torch.manual_seed(101)
    config = LlamaConfig(vocab_size=97, hidden_size=32, intermediate_size=64,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=256, attention_dropout=0.0, rope_scaling=rope)
    config._attn_implementation = backend
    return LlamaForCausalLM(config).eval()


def ids(*values):
    return torch.tensor([values], dtype=torch.long)


def fixture_data():
    return ids(1, 2), {"a": ids(3, 4, 5), "b": ids(6, 7)}, {
        "x": ids(8, 9, 10, 11, 12, 13), "y": ids(14, 15)}, [("a", "x"), ("b", "y")], ids(16, 17)


@torch.no_grad()
def dense_oracle(model, prefix, sources, targets, edges, query):
    """Independent reference: ONE full forward with a hand-built graph mask.

    Does not use GraphKVCache, rebase, slicing, or assembled KV caches.
    """
    p = prefix.shape[-1]
    span = max(x.shape[-1] for x in [*sources.values(), *targets.values()])
    chunks = [("prefix", prefix, 0)]
    chunks += [(k, v, p) for k, v in sources.items()]
    chunks += [(k, v, p + span) for k, v in targets.items()]
    chunks += [("query", query, p + (2 if targets else 1) * span)]
    ranges, positions, offset = {}, [], 0
    for name, chunk, start in chunks:
        ranges[name] = (offset, offset + chunk.shape[-1])
        offset += chunk.shape[-1]
        positions.extend(range(start, start + chunk.shape[-1]))
    allow = torch.zeros((offset, offset), dtype=torch.bool)
    for name, _, _ in chunks:
        start, end = ranges[name]
        allow[start:end, start:end] = torch.ones(end-start, end-start, dtype=torch.bool).tril()
        if name == "query":
            allow[start:end, :start] = True
        elif name in targets:
            for src, dst in edges:
                if dst == name:
                    a, b = ranges[src]
                    allow[start:end, a:b] = True
    mask = torch.zeros((1, 1, offset, offset)).masked_fill(~allow, torch.finfo(torch.float32).min)
    out = model(torch.cat([v for _, v, _ in chunks], -1),
        attention_mask=mask, position_ids=torch.tensor([positions]), use_cache=True)
    return out, ranges


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
@pytest.mark.parametrize("edges", [[("a", "x"), ("b", "y")],
    [(s,t) for s in ("a","b") for t in ("x","y")], []])
def test_cache_matches_dense_graph_mask(backend, edges):
    model = tiny(backend)
    prefix, sources, targets, _, query = fixture_data()
    g = build_graphkv_from_ids(model=model, prefix_input_ids=prefix,
        source_inputs=sources, target_inputs=targets, edges=edges)
    expected, ranges = dense_oracle(model, prefix, sources, targets, edges, query)
    actual = graphkv_next_logits(model=model, graph_cache=g, query_input_ids=query)
    torch.testing.assert_close(actual, expected.logits[:, -1], atol=2e-6, rtol=2e-5)
    for name in sources.keys() | targets.keys():
        a,b = ranges[name]
        for layer in range(3):
            for attr in ("key_cache", "value_cache"):
                torch.testing.assert_close(getattr(g.nodes[name], attr)[layer],
                    getattr(expected.past_key_values, attr)[layer][:,:,a:b], atol=2e-6, rtol=2e-5)
    assert g.meta["x"].logical_end <= g.query_position_ids(1)[0,0]


def test_permutation_and_probe_do_not_mutate_nodes():
    model = tiny()
    prefix,sources,targets,edges,query = fixture_data()
    g = build_graphkv_from_ids(model=model, source_inputs=sources, target_inputs=targets,
                               edges=edges, prefix_input_ids=prefix)
    before = {k:[t.clone() for t in v.key_cache] for k,v in g.nodes.items()}
    a = graphkv_next_logits(model=model, graph_cache=g, query_input_ids=query)
    b = graphkv_next_logits(model=model, graph_cache=g, query_input_ids=query,
                            source_ids=["b","a"], target_ids=["y","x"])
    torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)
    for name in before:
        for a,b in zip(before[name], g.nodes[name].key_cache):
            assert torch.equal(a,b)


def test_generation_matches_dense_recompute_each_step():
    model = tiny()
    prefix,sources,targets,edges,query = fixture_data()
    g = build_graphkv_from_ids(model=model, source_inputs=sources, target_inputs=targets,
                               edges=edges, prefix_input_ids=prefix)
    class Tokenizer:
        eos_token_id = None
        def decode(self, tokens, **kwargs): return str(tokens)
    generated=[]
    for _ in range(4):
        out,_ = dense_oracle(model,prefix,sources,targets,edges,query)
        token=int(out.logits[:,-1].argmax(-1).item())
        generated.append(token)
        query=torch.cat((query,ids(token)),-1)
    result=greedy_generate_graphkv(model=model,tokenizer=Tokenizer(),graph_cache=g,
        query_input_ids=ids(16,17),max_new_tokens=4,eos_token_id=[],return_details=True)
    assert result.token_ids == generated
    assert not result.stopped_on_eos
    stopped=greedy_generate_graphkv(model=model,tokenizer=Tokenizer(),graph_cache=g,
        query_input_ids=ids(16,17),max_new_tokens=4,eos_token_id=[generated[0]],return_details=True)
    assert stopped.token_ids == generated[:1] and stopped.stopped_on_eos


def test_layout_rejects_late_sources_and_unplanned_long_targets():
    model=tiny()
    g=GraphKVCache()
    g.add_source("a",_encode_independent_chunk(model,ids(1,2)))
    with pytest.raises(ValueError,match="chunk_span"):
        g.propagate_target(model,"t",ids(3,4,5),source_ids=["a"])
    with pytest.raises(RuntimeError,match="all sources"):
        g.add_source("b",_encode_independent_chunk(model,ids(6,7,8)))


def test_invalid_prefix_is_atomic_and_cache_metadata_is_valid():
    model=tiny()
    cache=_encode_independent_chunk(model,ids(1,2,3))
    g=GraphKVCache(2)
    with pytest.raises(ValueError): g.set_prefix_cache(cache)
    assert g.prefix_cache is None and g.prefix_token_count == 0
    sliced=GraphKVCache.slice_cache(cache,1,3)
    merged=GraphKVCache.concat_caches(sliced,sliced)
    assert merged.get_seq_length()==4 and merged._seen_tokens==4
    merged.key_cache[0].zero_()
    assert cache.key_cache[0].abs().sum()>0


def test_no_grad_round_and_duplicate_validation():
    model=tiny()
    g=build_graphkv_from_ids(model=model,source_inputs={"a":ids(1,2)},
        target_inputs={"t":ids(3,4)},edges=[("a","t")])
    assert all(not t.requires_grad for c in g.nodes.values() for t in c.key_cache)
    with pytest.raises(ValueError): g.target_position_ids(2,round_idx=0)
    with pytest.raises(ValueError): g.build_dependency_cache(["a","a"])
    with pytest.raises(ValueError): g.propagate_target(model,"u",ids(5),source_ids=["t"],round_idx=1)


@pytest.mark.parametrize("rope", [None, {"rope_type":"llama3","factor":8.,
    "low_freq_factor":1.,"high_freq_factor":4.,"original_max_position_embeddings":128}])
def test_fixed_rope_rebase_equals_direct_encoding(rope):
    model=tiny(rope=rope)
    a=_encode_independent_chunk(model,ids(1,2,3,4))
    b=_encode_independent_chunk(model,ids(1,2,3,4),position_start=9)
    GraphKVCache.rebase_key_positions(a,model.model.rotary_emb,9)
    for ka,kb,va,vb in zip(a.key_cache,b.key_cache,a.value_cache,b.value_cache):
        torch.testing.assert_close(ka,kb,atol=2e-6,rtol=2e-5)
        torch.testing.assert_close(va,vb,atol=2e-6,rtol=2e-5)


def test_kl_direction():
    a=torch.tensor([[0.8,0.2]]).log(); b=torch.tensor([[0.5,0.5]]).log()
    expected=(a.exp()*(a-b)).sum().item()
    assert kl_divergence(a,b)==pytest.approx(expected)
    assert kl_divergence(a,b)!=pytest.approx(kl_divergence(b,a))


class TinyTokenizer:
    eos_token_id=None
    def __call__(self,text,**kwargs):
        from types import SimpleNamespace
        values=[3+ord(c)%90 for c in text]
        if kwargs.get("truncation"): values=values[:kwargs["max_length"]]
        return SimpleNamespace(input_ids=torch.tensor([values],dtype=torch.long))
    def decode(self,tokens,**kwargs): return str(tokens)


def test_multigraph_plans_all_sources_and_preserves_edges():
    from pcw_parallel import _build_batch
    g,q,s,t=_build_batch(TinyTokenizer(),tiny(),"P",["C","long center"],
                        [["a"],["a longer neighbor"]],"Q")
    assert g.graph=={"graph0::center":["graph0::neighbor0"],"graph1::center":["graph1::neighbor0"]}
    assert len(t)==2 and len(s)==2
    assert all(m.logical_end<=g.query_position_ids(1)[0,0] for m in g.meta.values())
    with pytest.raises(ValueError): _build_batch(TinyTokenizer(),tiny(),"P",["C"],[],"Q")


def test_topk_zero_rejected_and_selection_explicit():
    from pcw import _build_rag_graph_cache
    args=(TinyTokenizer(),tiny(),None,"P","Q",["a","b","c"])
    with pytest.raises(ValueError): _build_rag_graph_cache(*args,top_k=0)
    g,*_=_build_rag_graph_cache(*args,top_k=1)
    assert all(v==["source::2"] for v in g.graph.values())


def test_legacy_helpers_preserve_precision_and_cache_counters():
    from kv_compat import (apply_pkv_rerotary_position_embeddings,
        apply_pkv_rotary_position_embeddings, divide_pkv, concact_pkv)
    model=tiny()
    cache=_encode_independent_chunk(model,ids(1,2,3,4))
    original=GraphKVCache._clone_cache(cache)
    apply_pkv_rerotary_position_embeddings(cache,model.model.rotary_emb)
    apply_pkv_rotary_position_embeddings(cache,model.model.rotary_emb)
    for a,b in zip(cache.key_cache, original.key_cache):
        assert a.dtype==torch.float32
        torch.testing.assert_close(a,b,atol=1e-7,rtol=1e-6)
    left,right=divide_pkv(cache,2)
    assert left._seen_tokens==right._seen_tokens==2
    merged=concact_pkv(left,right)
    assert merged.get_seq_length()==merged._seen_tokens==4


def test_empty_prefix_and_no_targets_match_dense():
    model=tiny()
    prefix,sources,_,_,query=fixture_data()
    g=build_graphkv_from_ids(model=model,source_inputs=sources,target_inputs={},edges=[])
    expected,_=dense_oracle(model,prefix[:,:0],sources,{},[],query)
    actual=graphkv_next_logits(model=model,graph_cache=g,query_input_ids=query)
    torch.testing.assert_close(actual,expected.logits[:,-1],atol=2e-6,rtol=2e-5)
