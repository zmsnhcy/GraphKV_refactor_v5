"""Static lifecycle protection, compatible with the existing v5 cache class.

Not a security boundary: malicious .data writes can bypass tensor version counters.
Use an explicit checksum audit for research runs, and never modify cached tensors.
"""
from types import MappingProxyType


class FrozenGraphError(RuntimeError):
    pass


class StaleCacheError(RuntimeError):
    pass


def tensor_stamp(x):
    try:
        version = x._version
    except (RuntimeError, AttributeError):
        version = None
    return (id(x), tuple(x.shape), str(x.dtype), str(x.device), version)


def cache_stamp(cache):
    if cache is None:
        return None
    return (id(cache), tuple(tensor_stamp(t) for t in cache.key_cache),
            tuple(tensor_stamp(t) for t in cache.value_cache))


class StaticGraphGuard:
    """Mixin before GraphKVCache in the MRO. Seal after construction."""
    def __init__(self, *args, **kwargs):
        self._sealed = False
        self._sealed_state = None
        super().__init__(*args, **kwargs)

    def _editable(self):
        if self._sealed:
            raise FrozenGraphError('The encoded graph is sealed. Build a new snapshot to change it.')

    def set_edges(self, edges):
        self._editable()
        # Validate the entire proposal before replacing any existing state.
        candidate = {}
        for e in edges:
            if not isinstance(e, (tuple, list)) or len(e) != 2 or any(not isinstance(x, str) or not x for x in e):
                raise ValueError('Edges must be nonempty string (source, target) pairs')
            s, t = e
            candidate.setdefault(t, [])
            if s not in candidate[t]:
                candidate[t].append(s)
        for n, meta in self.meta.items():
            if meta.kind == 'target' and tuple(candidate.get(n, ())) != tuple(self.graph.get(n, ())):
                raise FrozenGraphError(f'Cannot change the dependencies of encoded target {n!r}')
        self.graph = candidate

    def add_edge(self, source, target):
        self.set_edges([(s, t) for t, ss in self.graph.items() for s in ss] + [(source, target)])

    def add_source(self, *args, **kwargs):
        self._editable()
        return super().add_source(*args, **kwargs)

    def propagate_target(self, *args, **kwargs):
        self._editable()
        return super().propagate_target(*args, **kwargs)

    def set_prefix_cache(self, *args, **kwargs):
        self._editable()
        return super().set_prefix_cache(*args, **kwargs)

    def _state(self):
        return (self.source_position_start, self.chunk_span, self.prefix_token_count,
                self._frozen_source_span, self._max_round,
                tuple((t, tuple(ss)) for t, ss in self.graph.items()),
                tuple(self.meta.items()),
                tuple((n, cache_stamp(c)) for n, c in self.nodes.items()),
                cache_stamp(self.prefix_cache))

    def seal(self):
        if self._sealed:
            self.assert_consistent()
            return self
        self._freeze_layout()
        super().validate()
        self.graph = MappingProxyType({t: tuple(ss) for t, ss in self.graph.items()})
        self.nodes = MappingProxyType(dict(self.nodes))
        self.meta = MappingProxyType(dict(self.meta))
        self._sealed_state = self._state()
        self._sealed = True
        return self

    def assert_consistent(self):
        if not self._sealed:
            raise StaleCacheError('Seal the complete snapshot before querying')
        if self._state() != self._sealed_state:
            raise StaleCacheError('Cache tensors, graph, metadata, or layout changed after sealing')

    def prepare_query(self, *args, **kwargs):
        self.assert_consistent()
        return super().prepare_query(*args, **kwargs)

    def build_query_cache(self, *args, **kwargs):
        self.assert_consistent()
        return super().build_query_cache(*args, **kwargs)

    def validate(self):
        if self._sealed:
            self.assert_consistent()
        return super().validate()
