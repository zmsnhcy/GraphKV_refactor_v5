import unittest
from dataclasses import dataclass, replace
from types import SimpleNamespace
from graphkv_safe.policy import GraphSpec, NodeSpec, QueryPlanner
from graphkv_safe.fixtures import original_case, synthetic_case, material_screen
from graphkv_safe.guard import StaticGraphGuard, FrozenGraphError, StaleCacheError


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.spec, self.questions = original_case()
        self.planner = QueryPlanner(self.spec)

    def test_lyra_closure(self):
        p = self.planner.plan(self.questions[1]['question'], 'v1')
        self.assertEqual((p.mode, p.source_ids, p.target_ids), ('graph', ('5',), ('2',)))

    def test_multisource_closure(self):
        p = self.planner.plan(self.questions[2]['question'], 'v1')
        self.assertEqual((p.source_ids, p.target_ids), (('4', '5'), ('3',)))

    def test_direct_source(self):
        p = self.planner.plan('What does beta supply?', 'v1')
        self.assertEqual((p.source_ids, p.target_ids), (('5',), ()))

    def test_unknown_and_word_boundaries(self):
        for question in ('What does NOTLYRA use?', 'What is the material?', 'What does OMEGA use?'):
            p = self.planner.plan(question, 'v1')
            self.assertEqual((p.mode, p.reason), ('sequential', 'no_explicit_entity_match'))
            self.assertEqual(len(p.source_ids + p.target_ids), 5)

    def test_broad_questions_fall_back(self):
        for question in ('Compare LYRA and ORION.', 'What do all projects use?', 'LYRA不是用什么材料？'):
            self.assertEqual(self.planner.plan(question, 'v1').mode, 'sequential')

    def test_ambiguous_alias(self):
        ns = tuple(replace(n, aliases=('SAME',)) if n.kind == 'target' else n for n in self.spec.nodes)
        p = QueryPlanner(replace(self.spec, nodes=ns)).plan('What does SAME use?', 'v1')
        self.assertEqual(p.reason, 'ambiguous_entity_alias')

    def test_empty_question(self):
        with self.assertRaises(ValueError):
            self.planner.plan(' ', 'v1')

    def test_bad_graphs(self):
        for kwargs in (dict(nodes=self.spec.nodes + (self.spec.nodes[0],)),
                       dict(edges=(('missing', '2'),)), dict(edges=(('2', '4'),)),
                       dict(edges=(('4', '2'), ('4', '2')))):
            with self.assertRaises(ValueError):
                replace(self.spec, **kwargs)

    def test_malformed_edge_and_alias_containers(self):
        with self.assertRaises(ValueError):
            replace(self.spec, edges=('42',))
        with self.assertRaises(ValueError):
            NodeSpec('x', 'source', 'text', 'ALPHA')
        with self.assertRaises(ValueError):
            replace(self.planner.plan('LYRA?', 'v1'), source_ids='5')

    def test_revision_tracks_content_and_edges(self):
        n = replace(self.spec.nodes[0], text='Changed fact')
        self.assertNotEqual(self.spec.revision, replace(self.spec, nodes=(n,) + self.spec.nodes[1:]).revision)
        self.assertNotEqual(self.spec.revision, replace(self.spec, edges=(('4', '2'),)).revision)

    def test_synthetic_evidence_coverage(self):
        revisions = set()
        for split in ('dev', 'test'):
            for i in range(30):
                spec, qs = synthetic_case(i, split=split)
                self.assertNotIn(spec.revision, revisions)
                revisions.add(spec.revision)
                for q in qs:
                    p = QueryPlanner(spec).plan(q['question'], 'v1')
                    self.assertEqual(p.mode, 'graph')
                    self.assertEqual(set(p.source_ids + p.target_ids), set(q['evidence']))

    def test_scoring_does_not_claim_relations(self):
        result = material_screen('ALPHA supplies silver; BETA supplies copper.', ['silver', 'copper'])
        self.assertTrue(result['material_set_match'])
        self.assertIsNone(result['relation_and_citation_correct'])
        self.assertTrue(result['human_review_required'])


@dataclass(frozen=True)
class Meta:
    kind: str


class FakeTensor:
    shape = (1, 2, 3, 4)
    dtype = 'float32'
    device = 'cpu'
    _version = 0


class Base:
    """Tests lifecycle decisions only; actual tensor integration is in another file."""
    def __init__(self):
        self.graph = {'t': ['s']}
        self.meta = {'s': Meta('source'), 't': Meta('target')}
        self.nodes = {'s': SimpleNamespace(key_cache=[FakeTensor()], value_cache=[FakeTensor()])}
        self.prefix_cache = None
        self.source_position_start = 0
        self.chunk_span = 3
        self.prefix_token_count = 0
        self._frozen_source_span = 3
        self._max_round = 1
    def _freeze_layout(self):
        return 3
    def validate(self):
        return None
    def prepare_query(self, *a, **k):
        return 'prepared'
    def build_query_cache(self, *a, **k):
        return 'built'


class Protected(StaticGraphGuard, Base):
    pass


class GuardTests(unittest.TestCase):
    def test_encoded_dependency_cannot_change(self):
        g = Protected()
        with self.assertRaises(FrozenGraphError):
            g.set_edges([('other', 't')])
        self.assertEqual(g.graph, {'t': ['s']})

    def test_atomic_malformed_update(self):
        g = Protected()
        with self.assertRaises(ValueError):
            g.set_edges([('s', 't'), ('s', 'new'), ('broken',)])
        self.assertEqual(g.graph, {'t': ['s']})

    def test_new_target_edges_allowed_before_seal(self):
        g = Protected()
        g.add_edge('s', 'new')
        self.assertEqual(g.graph, {'t': ['s'], 'new': ['s']})

    def test_unsealed_query_rejected(self):
        with self.assertRaises(StaleCacheError):
            Protected().prepare_query()

    def test_seal_blocks_mutating_methods(self):
        g = Protected().seal()
        for fn in (lambda: g.set_edges([('s', 't')]), lambda: g.add_edge('s', 't'),
                   lambda: g.add_source('x'), lambda: g.propagate_target('x'), lambda: g.set_prefix_cache(None)):
            with self.assertRaises(FrozenGraphError):
                fn()
        with self.assertRaises(TypeError):
            g.graph['t'] = ['other']
        self.assertEqual(g.prepare_query(), 'prepared')
        self.assertEqual(g.build_query_cache(), 'built')

    def test_reassignment_and_layout_changes_detected(self):
        for attr, value in (('graph', {'t': ['other']}), ('chunk_span', 8), ('prefix_token_count', 10)):
            g = Protected().seal()
            setattr(g, attr, value)
            with self.assertRaises(StaleCacheError):
                g.validate()

    def test_tensor_inplace_change_detected(self):
        g = Protected().seal()
        g.nodes['s'].key_cache[0]._version += 1
        with self.assertRaises(StaleCacheError):
            g.prepare_query()

    def test_tensor_replacement_detected(self):
        g = Protected().seal()
        g.nodes['s'].value_cache[0] = FakeTensor()
        with self.assertRaises(StaleCacheError):
            g.build_query_cache()


if __name__ == '__main__':
    unittest.main()
