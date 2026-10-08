"""Run with the user's pinned environment. Never treat skips as model validation."""
import importlib.util
import unittest
from dataclasses import replace

AVAILABLE = all(importlib.util.find_spec(n) is not None for n in ('torch', 'transformers'))
if AVAILABLE:
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    from graph_kv_adapter import build_graphkv_from_ids, graphkv_next_logits
    from graphkv_safe.runtime import QueryEngine, cache_digest
    from graphkv_safe.policy import GraphSpec, NodeSpec
    from graphkv_safe.guard import FrozenGraphError, StaleCacheError


class TinyTokenizer:
    eos_token_id = None
    bos_token_id = 1
    pad_token_id = 0
    special_tokens_map = {}
    def __len__(self):
        return 101
    def encode(self, text, add_special_tokens=False):
        return [3 + ord(c) % 97 for c in text]
    def decode(self, ids, skip_special_tokens=True):
        return ' '.join(map(str, ids))


@unittest.skipUnless(AVAILABLE, 'torch/transformers unavailable: integration NOT executed')
class TorchIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(712)
        config = LlamaConfig(vocab_size=101, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                             max_position_embeddings=512, attention_dropout=0.0,
                             eos_token_id=None, pad_token_id=0)
        config._attn_implementation = 'eager'
        self.model = LlamaForCausalLM(config).float().eval()
        self.spec = GraphSpec((NodeSpec('a', 'source', 'alpha', ('A',)),
                               NodeSpec('b', 'source', 'beta', ('B',)),
                               NodeSpec('t', 'target', 't uses a', ('T',)),
                               NodeSpec('u', 'target', 'u uses b', ('U',))),
                              (('a', 't'), ('b', 'u')), 'prefix')
        self.engine = QueryEngine(self.model, TinyTokenizer(), self.spec,
                                  model_revision='tiny-random-test', context_budget=512)

    def first_logits(self, fn):
        captured = []
        def hook(module, args, output):
            if not captured:
                captured.append(output.logits[:, -1].detach().float().clone())
        handle = self.model.register_forward_hook(hook)
        try:
            result = fn()
        finally:
            handle.remove()
        return result, captured[0]

    def test_snapshot_matches_original_v5(self):
        e = self.engine
        original = build_graphkv_from_ids(model=self.model,
            source_inputs={n: e._tensor(e.node_ids[n]) for n in self.spec.ids('source')},
            target_inputs={n: e._tensor(e.node_ids[n]) for n in self.spec.ids('target')},
            edges=self.spec.edges, prefix_input_ids=e._tensor(e.prefix_ids))
        self.assertEqual(cache_digest(e.cache), cache_digest(original))
        q = e._tensor(e._query_ids('T?'))
        a = graphkv_next_logits(model=self.model, graph_cache=original, query_input_ids=q)
        b = graphkv_next_logits(model=self.model, graph_cache=e.cache, query_input_ids=q)
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)

    def test_full_mask_matches_adapter(self):
        e = self.engine
        a, la = self.first_logits(lambda: e.answer('T?', mode='graph_all', max_new_tokens=4))
        b, lb = self.first_logits(lambda: e.answer_masked('T?', tuple(e.node_ids), max_new_tokens=4))
        torch.testing.assert_close(la, lb, atol=2e-5, rtol=2e-5)
        self.assertEqual(a['token_ids'], b['token_ids'])

    def test_masked_subset_matches_physical_selection(self):
        e = self.engine
        a, la = self.first_logits(lambda: e.answer('T?', mode='graph_selected', max_new_tokens=4))
        b, lb = self.first_logits(lambda: e.answer_masked('T?', ('a', 't'), max_new_tokens=4))
        torch.testing.assert_close(la, lb, atol=2e-5, rtol=2e-5)
        self.assertEqual(a['token_ids'], b['token_ids'])
        self.assertGreater(b['physical_prompt_tokens'], a['physical_prompt_tokens'])
        self.assertEqual(b['query_logical_start'], a['query_logical_start'])

    def test_sequential_cache_matches_full_recompute(self):
        e = self.engine
        result = e.answer('T?', mode='sequential_selected', max_new_tokens=4)
        p = e.plan('T?')
        prompt = e.prefix_ids + tuple(x for n in p.source_ids + p.target_ids for x in e.node_ids[n]) + e._query_ids('T?')
        tokens = []
        with torch.no_grad():
            for _ in range(4):
                out = self.model(e._tensor(prompt + tuple(tokens)), use_cache=False, return_dict=True)
                tokens.append(int(out.logits[:, -1].argmax().item()))
        self.assertEqual(result['token_ids'], tokens)

    def test_all_paths_leave_original_cache_unchanged(self):
        e = self.engine
        before = cache_digest(e.cache)
        for mode in ('graph_all', 'graph_selected', 'sequential_selected', 'sequential_all', 'hybrid'):
            e.answer('U?', mode=mode, max_new_tokens=3)
        e.answer_masked('U?', ('b', 'u'), max_new_tokens=3)
        self.assertEqual(before, cache_digest(e.cache))

    def test_graph_change_and_stale_plan_are_rejected(self):
        e = self.engine
        with self.assertRaises(FrozenGraphError):
            e.cache.set_edges([('a', 'u'), ('b', 't')])
        with self.assertRaises(StaleCacheError):
            e.answer_plan(replace(e.plan('T?'), cache_revision='outdated'))
        with self.assertRaises(ValueError):
            e.answer_plan(replace(e.plan('T?'), source_ids=('b',)))
        e.assert_consistent()

    def test_model_update_and_cache_inplace_write_are_rejected(self):
        e = self.engine
        with torch.no_grad():
            next(self.model.parameters()).add_(0.001)
        with self.assertRaises(StaleCacheError):
            e.answer('T?')
        # Independent second engine uses the updated model; mutate its cache instead.
        e2 = QueryEngine(self.model, TinyTokenizer(), self.spec, model_revision='updated', context_budget=512)
        e2.cache.nodes['a'].key_cache[0].add_(0.001)
        with self.assertRaises(StaleCacheError):
            e2.answer('T?')

    def test_over_budget_never_calls_model(self):
        e = self.engine
        def forbidden(*args, **kwargs):
            raise AssertionError('Over-budget query called model')
        handle = self.model.register_forward_pre_hook(forbidden)
        try:
            result = e.answer('T?', max_new_tokens=1000)
            self.assertEqual(result['status'], 'budget_exceeded')
            self.assertIsNone(result['answer'])
        finally:
            handle.remove()

    def test_fallback_is_sequential_and_uses_all_nodes(self):
        e = self.engine
        result = e.answer('Unknown project?', max_new_tokens=2)
        self.assertEqual(result['path'], 'sequential')
        self.assertEqual(result['plan']['source_ids'], ('a', 'b'))
        self.assertEqual(result['plan']['target_ids'], ('t', 'u'))


if __name__ == '__main__':
    unittest.main()
