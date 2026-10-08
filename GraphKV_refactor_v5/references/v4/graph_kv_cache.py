"""Refactored Graph-KV cache core.

The implementation follows the paper's one-hop design while separating:

* logical RoPE positions (``position_ids``), and
* physical KV-cache positions (``cache_position``).

This module does not import Transformers at import time.  It can therefore be
unit-tested with a tiny cache object on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch


@dataclass(frozen=True)
class NodeMeta:
    node_id: str
    kind: str  # source / target
    token_count: int
    logical_start: int
    logical_end: int
    round_idx: int = 0


class GraphKVCache:
    """State manager for Graph-KV message passing.

    ``source_position_start`` lets callers reserve a non-graph prefix block
    before the graph.  With a prefix of length P, graph source nodes share
    ``[P, P+L)`` and first-hop targets use ``[P+L, P+2L)``.
    """

    def __init__(self, source_position_start: int = 0) -> None:
        if source_position_start < 0:
            raise ValueError("source_position_start must be >= 0")
        self.source_position_start = int(source_position_start)
        self.nodes: Dict[str, Any] = {}
        self.meta: Dict[str, NodeMeta] = {}
        # target -> [source, source, ...]
        self.graph: Dict[str, List[str]] = {}
        self.prefix_cache: Optional[Any] = None
        self.prefix_token_count: int = 0
        self._frozen_source_span: Optional[int] = None
        self._max_round: int = 0

    # ------------------------------------------------------------------
    # Generic cache helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _clone_cache(cache: Any) -> Any:
        new_cache = type(cache)()
        new_cache.key_cache = [x.clone().detach() for x in cache.key_cache]
        new_cache.value_cache = [x.clone().detach() for x in cache.value_cache]
        return new_cache

    @staticmethod
    def _empty_like(cache: Any) -> Any:
        new_cache = type(cache)()
        new_cache.key_cache = []
        new_cache.value_cache = []
        return new_cache

    @staticmethod
    def _seq_len(cache: Any) -> int:
        if not hasattr(cache, "key_cache") or not cache.key_cache:
            return 0
        return int(cache.key_cache[0].shape[-2])

    @staticmethod
    def _batch_size(cache: Any) -> int:
        if not hasattr(cache, "key_cache") or not cache.key_cache:
            return 0
        return int(cache.key_cache[0].shape[0])

    @classmethod
    def _validate_cache(cls, cache: Any, *, require_batch1: bool = True) -> None:
        if not hasattr(cache, "key_cache") or not hasattr(cache, "value_cache"):
            raise TypeError("cache must expose key_cache and value_cache lists")
        if len(cache.key_cache) != len(cache.value_cache):
            raise ValueError("key_cache and value_cache must have equal layer count")
        if not cache.key_cache:
            raise ValueError("cache must contain at least one layer")
        if require_batch1 and cls._batch_size(cache) != 1:
            raise ValueError("GraphKVCache expects batch size 1 per registered node")
        seq_len = None
        for k, v in zip(cache.key_cache, cache.value_cache):
            if k.ndim != 4 or v.ndim != 4:
                raise ValueError("KV tensors must have shape [batch, heads, seq, head_dim]")
            if k.shape[:-1] != v.shape[:-1]:
                raise ValueError("key/value cache shapes are inconsistent")
            if seq_len is None:
                seq_len = int(k.shape[-2])
            elif int(k.shape[-2]) != seq_len:
                raise ValueError("all KV layers must have the same sequence length")

    @classmethod
    def slice_cache(cls, cache: Any, start: int, end: int) -> Any:
        cls._validate_cache(cache, require_batch1=False)
        seq_len = cls._seq_len(cache)
        if not (0 <= start <= end <= seq_len):
            raise ValueError(f"invalid cache slice [{start}, {end}) for seq_len={seq_len}")
        out = cls._empty_like(cache)
        for k, v in zip(cache.key_cache, cache.value_cache):
            out.key_cache.append(k[:, :, start:end, :].clone().detach())
            out.value_cache.append(v[:, :, start:end, :].clone().detach())
        return out

    @classmethod
    def concat_caches(cls, *caches: Any) -> Any:
        if not caches:
            raise ValueError("concat_caches requires at least one cache")
        for cache in caches:
            cls._validate_cache(cache, require_batch1=True)
        layer_count = len(caches[0].key_cache)
        if any(len(c.key_cache) != layer_count for c in caches):
            raise ValueError("all caches must have the same number of layers")
        out = cls._empty_like(caches[0])
        for layer_idx in range(layer_count):
            out.key_cache.append(torch.cat([c.key_cache[layer_idx] for c in caches], dim=2))
            out.value_cache.append(torch.cat([c.value_cache[layer_idx] for c in caches], dim=2))
        return out

    # ------------------------------------------------------------------
    # Llama RoPE utility used by the original repository's ``emb`` object
    # ------------------------------------------------------------------
    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def _apply_rotary_tensor(
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Apply RoPE to one layer's key tensor.

        ``cos``/``sin`` must already live on ``k.device``. Keeping this
        operation layer-local is essential when ``device_map='auto'`` places
        different decoder layers on different GPUs.
        """
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        k_fp32 = k.to(dtype=torch.float32)
        rotated = (k_fp32 * cos) + (GraphKVCache._rotate_half(k_fp32) * sin)
        return rotated.to(dtype=k.dtype)

    @staticmethod
    def _device_local_rotary_embedding(rotary_embedding: Any, device: torch.device) -> Any:
        """Build a standalone RoPE module on ``device``.

        Never call ``.to(device)`` on a model-owned rotary module when the
        model has been dispatched with Accelerate/device_map. Such a module may
        carry a device-placement hook and can be moved back to its assigned
        device during forward. Instead, recreate the same RoPE class from its
        config on the requested device.
        """
        config = getattr(rotary_embedding, "config", None)
        if config is None:
            raise RuntimeError(
                "A device-local RoPE module requires rotary_embedding.config. "
                "For dispatched HuggingFace Llama models, pass the model's "
                "LlamaRotaryEmbedding instance."
            )
        rope_cls = type(rotary_embedding)
        try:
            local = rope_cls(config=config, device=device)
        except TypeError:
            local = rope_cls(config, device=device)
        return local.to(device)

    @classmethod
    def rebase_key_positions(
        cls,
        cache: Any,
        rotary_embedding: Any,
        target_start: int,
        *,
        original_start: int = 0,
    ) -> Any:
        """Move K from one contiguous RoPE range to another.

        This legacy helper remains available, but it is now safe for a
        multi-GPU DynamicCache: each transformer layer computes its RoPE on the
        same device that owns that layer's K tensor.
        """
        cls._validate_cache(cache)
        if target_start == original_start:
            return cache
        seq_len = cls._seq_len(cache)
        if seq_len <= 0:
            return cache

        rope_by_device: Dict[torch.device, Any] = {}
        for layer_idx, k in enumerate(cache.key_cache):
            device = k.device
            if device not in rope_by_device:
                rope_by_device[device] = cls._device_local_rotary_embedding(
                    rotary_embedding, device
                )
            rope = rope_by_device[device]

            positions_from = torch.arange(
                original_start,
                original_start + seq_len,
                dtype=torch.long,
                device=device,
            ).unsqueeze(0)
            positions_to = torch.arange(
                target_start,
                target_start + seq_len,
                dtype=torch.long,
                device=device,
            ).unsqueeze(0)

            k_float = k.to(dtype=torch.float32)
            cos_from, sin_from = rope(x=k_float, position_ids=positions_from)
            k_unrotated = cls._apply_rotary_tensor(k, cos_from, -sin_from)
            cos_to, sin_to = rope(
                x=k_unrotated.to(dtype=torch.float32),
                position_ids=positions_to,
            )
            cache.key_cache[layer_idx] = cls._apply_rotary_tensor(
                k_unrotated, cos_to, sin_to
            )
        return cache

    # ------------------------------------------------------------------
    # Optional non-graph prefix
    # ------------------------------------------------------------------
    def set_prefix_cache(self, cache: Any) -> None:
        """Set an external prefix cache that is visible only at query time."""
        self._validate_cache(cache)
        self.prefix_cache = self._clone_cache(cache)
        self.prefix_token_count = self._seq_len(cache)
        if self.prefix_token_count != self.source_position_start:
            raise ValueError(
                "prefix cache length must equal source_position_start so that "
                "graph source positions begin immediately after the prefix"
            )

    # ------------------------------------------------------------------
    # Graph definition and registration
    # ------------------------------------------------------------------
    def add_edge(self, source: str, target: str) -> None:
        self.graph.setdefault(target, [])
        if source not in self.graph[target]:
            self.graph[target].append(source)

    def set_edges(self, edges: Iterable[Tuple[str, str]]) -> None:
        self.graph = {}
        for source, target in edges:
            self.add_edge(source, target)

    def source_nodes_for(self, target: str) -> List[str]:
        return list(self.graph.get(target, []))

    def _register_node(
        self,
        node_id: str,
        cache: Any,
        *,
        kind: str,
        logical_start: int,
        round_idx: int,
    ) -> None:
        self._validate_cache(cache)
        if node_id in self.nodes:
            raise ValueError(f"Node {node_id!r} is already registered")
        length = self._seq_len(cache)
        if length <= 0:
            raise ValueError(f"Node {node_id!r} has an empty KV cache")
        self.nodes[node_id] = self._clone_cache(cache)
        self.meta[node_id] = NodeMeta(
            node_id=node_id,
            kind=kind,
            token_count=length,
            logical_start=int(logical_start),
            logical_end=int(logical_start + length),
            round_idx=int(round_idx),
        )

    def add_source(
        self,
        node_id: str,
        cache: Any,
        *,
        rotary_embedding: Optional[Any] = None,
        assume_positioned: bool = False,
    ) -> None:
        """Register a source node.

        By default, the cache is assumed to have local RoPE positions starting
        at zero. If ``source_position_start != 0``, callers can either:
        (1) pass ``rotary_embedding`` and let this class rebase the cache, or
        (2) set ``assume_positioned=True`` when the model already encoded the
        source using the desired absolute logical positions. The second path
        is preferred for multi-GPU HuggingFace models.
        """
        cache = self._clone_cache(cache)
        if self.source_position_start != 0 and not assume_positioned:
            if rotary_embedding is None:
                raise ValueError(
                    "rotary_embedding is required when source_position_start != 0 "
                    "unless assume_positioned=True"
                )
            self.rebase_key_positions(
                cache, rotary_embedding, self.source_position_start, original_start=0
            )
        self._register_node(
            node_id,
            cache,
            kind="source",
            logical_start=self.source_position_start,
            round_idx=0,
        )

    @property
    def source_span(self) -> int:
        lengths = [m.token_count for m in self.meta.values() if m.kind == "source"]
        if not lengths:
            raise RuntimeError("no source nodes have been registered")
        return max(lengths)

    def _freeze_layout(self) -> int:
        span = self.source_span
        if self._frozen_source_span is None:
            self._frozen_source_span = span
        elif self._frozen_source_span != span:
            raise RuntimeError("source span changed after Graph-KV layout was frozen")
        return self._frozen_source_span

    # ------------------------------------------------------------------
    # Physical KV assembly
    # ------------------------------------------------------------------
    def build_dependency_cache(self, node_ids: Sequence[str]) -> Any:
        ids = list(node_ids)
        if not ids:
            raise ValueError("node_ids must contain at least one node")
        caches = []
        for node_id in ids:
            if node_id not in self.nodes:
                raise KeyError(f"unknown node {node_id!r}")
            caches.append(self.nodes[node_id])
        return self.concat_caches(*caches)

    def build_source_cache(self, source_ids: Optional[Sequence[str]] = None) -> Any:
        if source_ids is None:
            source_ids = [m.node_id for m in self.meta.values() if m.kind == "source"]
        source_ids = list(source_ids)
        for node_id in source_ids:
            if node_id not in self.nodes:
                raise KeyError(f"unknown node {node_id!r}")
        return self.build_dependency_cache(source_ids)

    # ------------------------------------------------------------------
    # Position allocation
    # ------------------------------------------------------------------
    def target_position_ids(
        self,
        token_count: int,
        *,
        round_idx: int = 1,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if token_count <= 0:
            raise ValueError("token_count must be positive")
        span = self._freeze_layout()
        start = self.source_position_start + round_idx * span
        return (start + torch.arange(token_count, device=device)).unsqueeze(0)

    def query_position_ids(
        self,
        token_count: int,
        *,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if token_count <= 0:
            raise ValueError("token_count must be positive")
        span = self._freeze_layout()
        start = self.source_position_start + (self._max_round + 1) * span
        return (start + torch.arange(token_count, device=device)).unsqueeze(0)

    # ------------------------------------------------------------------
    # One message-passing step
    # ------------------------------------------------------------------
    def propagate_target(
        self,
        model: Any,
        target_id: str,
        target_input_ids: torch.Tensor,
        *,
        source_ids: Optional[Sequence[str]] = None,
        round_idx: int = 1,
        attention_mask: Optional[torch.Tensor] = None,
        **model_kwargs: Any,
    ) -> Any:
        if target_id in self.nodes:
            raise ValueError(f"target node {target_id!r} already exists")
        if target_input_ids.ndim != 2 or target_input_ids.shape[0] != 1:
            raise ValueError("target_input_ids must have shape [1, target_len]")
        if source_ids is None:
            source_ids = self.source_nodes_for(target_id)
        source_ids = list(source_ids)
        if not source_ids:
            raise ValueError(f"no source nodes specified for target {target_id!r}")

        source_cache = self.build_dependency_cache(source_ids)
        physical_source_len = self._seq_len(source_cache)
        target_len = int(target_input_ids.shape[-1])
        position_ids = self.target_position_ids(
            target_len, round_idx=round_idx, device=target_input_ids.device
        )
        cache_position = physical_source_len + torch.arange(
            target_len, device=target_input_ids.device
        )

        kwargs = dict(model_kwargs)
        kwargs.update(
            {
                "past_key_values": source_cache,
                "position_ids": position_ids,
                "cache_position": cache_position,
                "use_cache": True,
            }
        )
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask

        outputs = model(target_input_ids, **kwargs)
        full_cache = outputs.past_key_values
        full_len = self._seq_len(full_cache)
        expected_len = physical_source_len + target_len
        if full_len < expected_len:
            raise RuntimeError(
                f"model returned KV seq_len={full_len}, expected at least {expected_len}"
            )
        target_cache = self.slice_cache(full_cache, physical_source_len, expected_len)
        logical_start = self.source_position_start + round_idx * self._freeze_layout()
        self._register_node(
            target_id,
            target_cache,
            kind="target",
            logical_start=logical_start,
            round_idx=round_idx,
        )
        self._max_round = max(self._max_round, round_idx)
        self.graph.setdefault(target_id, list(source_ids))
        return outputs

    # ------------------------------------------------------------------
    # Query preparation
    # ------------------------------------------------------------------
    def build_query_cache(
        self,
        *,
        source_ids: Optional[Sequence[str]] = None,
        target_ids: Optional[Sequence[str]] = None,
    ) -> Any:
        source_ids = list(
            source_ids
            if source_ids is not None
            else [m.node_id for m in self.meta.values() if m.kind == "source"]
        )
        target_ids = list(
            target_ids
            if target_ids is not None
            else [m.node_id for m in self.meta.values() if m.kind == "target"]
        )
        caches = []
        if self.prefix_cache is not None:
            caches.append(self.prefix_cache)
        for node_id in source_ids + target_ids:
            if node_id not in self.nodes:
                raise KeyError(f"unknown node {node_id!r}")
            caches.append(self.nodes[node_id])
        if not caches:
            raise ValueError("no nodes selected for query cache")
        return self.concat_caches(*caches)

    def prepare_query(
        self,
        query_input_ids: torch.Tensor,
        *,
        source_ids: Optional[Sequence[str]] = None,
        target_ids: Optional[Sequence[str]] = None,
    ) -> Tuple[Any, torch.Tensor, torch.Tensor]:
        if query_input_ids.ndim != 2 or query_input_ids.shape[0] != 1:
            raise ValueError("query_input_ids must have shape [1, query_len]")
        cache = self.build_query_cache(source_ids=source_ids, target_ids=target_ids)
        physical_start = self._seq_len(cache)
        query_len = int(query_input_ids.shape[-1])
        position_ids = self.query_position_ids(
            query_len, device=query_input_ids.device
        )
        cache_position = physical_start + torch.arange(
            query_len, device=query_input_ids.device
        )
        return cache, position_ids, cache_position

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        return {
            "source_position_start": self.source_position_start,
            "prefix_token_count": self.prefix_token_count,
            "source_span": self._frozen_source_span,
            "max_round": self._max_round,
            "query_position_start": (
                None
                if self._frozen_source_span is None
                else self.source_position_start
                + (self._max_round + 1) * self._frozen_source_span
            ),
            "nodes": {
                node_id: {
                    "kind": meta.kind,
                    "token_count": meta.token_count,
                    "logical_range": (meta.logical_start, meta.logical_end),
                    "round": meta.round_idx,
                    "physical_seq_len": self._seq_len(self.nodes[node_id]),
                }
                for node_id, meta in self.meta.items()
            },
            "edges": {k: list(v) for k, v in self.graph.items()},
        }

    def validate(self) -> None:
        for node_id, cache in self.nodes.items():
            self._validate_cache(cache)
            meta = self.meta[node_id]
            if self._seq_len(cache) != meta.token_count:
                raise AssertionError(f"node {node_id} token-count metadata mismatch")
        if self.prefix_cache is not None:
            self._validate_cache(self.prefix_cache)
        for target, sources in self.graph.items():
            for source in sources:
                if source not in self.nodes:
                    raise AssertionError(f"graph references unknown source {source!r}")


__all__ = ["GraphKVCache", "NodeMeta"]
