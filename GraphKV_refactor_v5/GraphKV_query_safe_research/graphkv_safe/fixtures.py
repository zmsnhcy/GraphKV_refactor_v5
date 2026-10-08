"""Synthetic evaluation data. Gold fields are NEVER passed to QueryPlanner."""
from dataclasses import asdict
import random
import re
from .policy import GraphSpec, NodeSpec

PREFIX = '<|user|>\nAnswer using the following documents. Give a short factual answer.\n'
MATERIALS = ('copper', 'silver', 'iron', 'gold', 'nickel', 'zinc', 'tin', 'aluminum')


def original_case():
    spec = GraphSpec((
        NodeSpec('4', 'source', '[Document 4] In this fictional dataset, ALPHA supplies copper. Its code is AX17.\n', ('ALPHA',)),
        NodeSpec('5', 'source', '[Document 5] In this fictional dataset, BETA supplies silver. Its code is BY29.\n', ('BETA',)),
        NodeSpec('1', 'target', '[Document 1] Project ORION uses the material supplied by ALPHA.\n', ('ORION',)),
        NodeSpec('2', 'target', '[Document 2] Project LYRA uses the material supplied by BETA.\n', ('LYRA',)),
        NodeSpec('3', 'target', '[Document 3] Project NOVA combines the materials supplied by ALPHA and BETA.\n', ('NOVA',)),
    ), (('4', '1'), ('4', '3'), ('5', '2'), ('5', '3')), PREFIX)
    questions = [
        dict(id='ORION', question='What material does ORION use? Answer with the material name only.', materials=['copper'], evidence=['4', '1']),
        dict(id='LYRA', question='What material does LYRA use? Answer with the material name only.', materials=['silver'], evidence=['5', '2']),
        dict(id='NOVA', question='Which two materials does NOVA combine? Answer with the material names only.', materials=['copper', 'silver'], evidence=['4', '5', '3']),
    ]
    return spec, questions


def synthetic_case(index, *, split='pilot'):
    """Each index is an independent graph, not a renaming of the saved LYRA graph.

    Same task template still limits generalization. Split is part of the seed;
    assess disjoint test graphs only after fixing policy on development graphs.
    """
    rng = random.Random(f'graphkv-safe-v1:{split}:{index}')
    def name(prefix):
        return prefix + ''.join(rng.sample('ABCDEFGHJKLMNPQRSTUVWXYZ', 6))
    suppliers, projects = [name('SUP') for _ in range(3)], [name('PRJ') for _ in range(2)]
    materials = rng.sample(MATERIALS, 3)
    docs = rng.sample(range(11, 990), 5)
    source_ids, target_ids = [f's{d}' for d in docs[:3]], [f't{d}' for d in docs[3:]]
    nodes = [NodeSpec(s, 'source', f'[Document {docs[i]}] In this fictional dataset, {suppliers[i]} supplies {materials[i]}.\n', (suppliers[i],))
             for i, s in enumerate(source_ids)]
    # Vary which supplier supports the single-project question independently of ordering.
    single = rng.randrange(3)
    combined = rng.sample(range(3), 2)
    nodes += [
        NodeSpec(target_ids[0], 'target', f'[Document {docs[3]}] Project {projects[0]} uses the material supplied by {suppliers[single]}.\n', (projects[0],)),
        NodeSpec(target_ids[1], 'target', f'[Document {docs[4]}] Project {projects[1]} combines the materials supplied by {suppliers[combined[0]]} and {suppliers[combined[1]]}.\n', (projects[1],))]
    rng.shuffle(nodes)
    edges = [(source_ids[single], target_ids[0])] + [(source_ids[i], target_ids[1]) for i in combined]
    spec = GraphSpec(tuple(nodes), tuple(edges), PREFIX)
    questions = [
        dict(id='single', question=f'What material does {projects[0]} use? Answer with the material name only.',
             materials=[materials[single]], evidence=[source_ids[single], target_ids[0]]),
        dict(id='combined', question=f'Which two materials does {projects[1]} combine? Answer with the material names only.',
             materials=[materials[i] for i in combined], evidence=[source_ids[i] for i in combined] + [target_ids[1]]),
        dict(id='direct', question=f'What material does {suppliers[2]} supply? Answer with the material name only.',
             materials=[materials[2]], evidence=[source_ids[2]])]
    return spec, questions


def material_screen(answer, expected):
    """A deliberately narrow score, NOT relation/citation correctness.

    All complete outputs must remain available for review. Even a single material
    can appear in a negated or otherwise wrong assertion.
    """
    answer = answer or ''
    found = sorted(m for m in MATERIALS if re.search(r'\b' + m + r'\b', answer, re.I))
    return dict(material_set_match=set(found) == set(expected), materials_found=found,
                relation_and_citation_correct=None, human_review_required=True)


def serialize_spec(spec):
    return asdict(spec)
