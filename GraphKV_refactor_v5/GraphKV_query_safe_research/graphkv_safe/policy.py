"""Conservative explicit-entity routing; never receives answers or gold nodes."""
from dataclasses import dataclass
import hashlib
import json
import re


@dataclass(frozen=True)
class NodeSpec:
    node_id: str
    kind: str
    text: str
    aliases: tuple = ()

    def __post_init__(self):
        if isinstance(self.aliases, str):
            raise ValueError('aliases must be a sequence of names, not one string')
        object.__setattr__(self, 'aliases', tuple(self.aliases))
        if not isinstance(self.node_id, str) or not self.node_id:
            raise ValueError('node_id must be a nonempty string')
        if self.kind not in ('source', 'target') or not isinstance(self.text, str) or not self.text:
            raise ValueError('Nodes need a source/target kind and nonempty text')
        if any(not isinstance(a, str) or not a.strip() for a in self.aliases):
            raise ValueError('Aliases must be nonempty strings')


@dataclass(frozen=True)
class GraphSpec:
    nodes: tuple
    edges: tuple
    prefix: str

    def __post_init__(self):
        object.__setattr__(self, 'nodes', tuple(self.nodes))
        edges = tuple(self.edges)
        if any(not isinstance(e, (tuple, list)) or len(e) != 2 or
               any(not isinstance(n, str) or not n for n in e) for e in edges):
            raise ValueError('Edges must be (source ID, target ID) pairs')
        object.__setattr__(self, 'edges', tuple(tuple(e) for e in edges))
        if any(not isinstance(n, NodeSpec) for n in self.nodes):
            raise ValueError('nodes must contain NodeSpec objects')
        by_id = {n.node_id: n for n in self.nodes}
        if len(by_id) != len(self.nodes):
            raise ValueError('Duplicate node IDs')
        if not any(n.kind == 'source' for n in self.nodes):
            raise ValueError('At least one source is required')
        if not isinstance(self.prefix, str) or not self.prefix:
            raise ValueError('This prototype requires a nonempty prefix')
        if len(set(self.edges)) != len(self.edges):
            raise ValueError('Duplicate edges')
        for e in self.edges:
            if len(e) != 2 or any(n not in by_id for n in e):
                raise ValueError(f'Invalid edge: {e}')
            if by_id[e[0]].kind != 'source' or by_id[e[1]].kind != 'target':
                raise ValueError('This prototype supports one-hop source -> target graphs only')

    @property
    def revision(self):
        payload = {'nodes': [(n.node_id, n.kind, n.text, n.aliases) for n in self.nodes],
                   'edges': self.edges, 'prefix': self.prefix}
        return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

    def ids(self, kind):
        return tuple(n.node_id for n in self.nodes if n.kind == kind)

    def sources_for(self, targets):
        wanted = {s for s, t in self.edges if t in targets}
        return tuple(n for n in self.ids('source') if n in wanted)


@dataclass(frozen=True)
class QueryPlan:
    question: str
    graph_revision: str
    cache_revision: str
    mode: str
    source_ids: tuple
    target_ids: tuple
    reason: str
    matched_aliases: tuple = ()

    def __post_init__(self):
        for field in ('source_ids', 'target_ids', 'matched_aliases'):
            value = getattr(self, field)
            if isinstance(value, str):
                raise ValueError(f'{field} must be a sequence, not one string')
            object.__setattr__(self, field, tuple(value))


class QueryPlanner:
    """An interpretable baseline, not a general semantic retriever.

    Broad/comparison/negation questions conservatively use sequential full text.
    Exact alias matches do not prove semantic completeness; evaluate coverage.
    """
    _broad = re.compile(r'\b(all|every|except|compare|versus|instead|not|and|or)\b', re.I)

    def __init__(self, spec):
        self.spec = spec

    @staticmethod
    def _matches(alias, question):
        a, q = alias.casefold(), question.casefold()
        if a.isascii():
            return re.search(r'(?<!\w)' + re.escape(a) + r'(?!\w)', q) is not None
        return a in q

    def plan(self, question, cache_revision):
        if not isinstance(question, str) or not question.strip():
            raise ValueError('Question must be a nonempty string')
        def full(reason):
            return QueryPlan(question, self.spec.revision, cache_revision, 'sequential',
                             self.spec.ids('source'), self.spec.ids('target'), reason)
        if self._broad.search(question) or any(w in question for w in ('所有', '全部', '比较', '除了', '不是', '或者')):
            return full('broad_or_complex_question')
        hits, aliases = set(), {}
        for n in self.spec.nodes:
            for alias in n.aliases:
                if self._matches(alias, question):
                    hits.add(n.node_id)
                    aliases.setdefault(alias.casefold(), set()).add(n.node_id)
        if not hits:
            return full('no_explicit_entity_match')
        if any(len(ids) > 1 for ids in aliases.values()):
            return full('ambiguous_entity_alias')
        targets = tuple(n for n in self.spec.ids('target') if n in hits)
        sources = set(self.spec.sources_for(targets)) | (hits & set(self.spec.ids('source')))
        return QueryPlan(question, self.spec.revision, cache_revision, 'graph',
                         tuple(n for n in self.spec.ids('source') if n in sources), targets,
                         'explicit_entities_and_graph_dependencies', tuple(sorted(aliases)))
