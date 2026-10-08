"""Research inference wrapper around the unmodified v5, Transformers 4.50.0.

Single device, batch one, static one-hop graphs. This module needs torch and v5.
Do not share an engine between concurrent threads or mutate its model/tokenizer.
"""
from dataclasses import asdict, replace
import hashlib
import time
from types import MappingProxyType
import uuid

import torch
from graph_kv_cache import GraphKVCache
from graph_kv_adapter import (_check_model, _encode_independent_chunk, input_device,
                              greedy_generate_graphkv, GenerationResult)
from .guard import StaticGraphGuard, StaleCacheError, tensor_stamp
from .policy import QueryPlanner


class ConsistentGraphKVCache(StaticGraphGuard, GraphKVCache):
    pass


def build_snapshot(*, model, source_inputs, target_inputs, edges, prefix_input_ids):
    """Same positions and encoding calls as v5's build_graphkv_from_ids."""
    _check_model(model)
    if not source_inputs or set(source_inputs) & set(target_inputs):
        raise ValueError('Need sources and disjoint node IDs')
    chunks = {**source_inputs, **target_inputs}
    for ids in list(chunks.values()) + [prefix_input_ids]:
        if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] == 0:
            raise ValueError('Expected nonempty unpadded [1, length] token IDs')
    edges = tuple(edges)
    if any(s not in source_inputs or t not in target_inputs for s, t in edges):
        raise ValueError('Invalid one-hop edge')
    p, span = prefix_input_ids.shape[1], max(x.shape[1] for x in chunks.values())
    g = ConsistentGraphKVCache(p, chunk_span=span)
    # Ensure caches have mutation counters even if caller uses inference_mode.
    with torch.inference_mode(False), torch.no_grad():
        g.set_prefix_cache(_encode_independent_chunk(model, prefix_input_ids))
        g.set_edges(edges)
        for name, ids in source_inputs.items():
            g.add_source(name, _encode_independent_chunk(model, ids, position_start=p),
                         assume_positioned=True)
        for name, ids in target_inputs.items():
            g.propagate_target(model, name, ids)
    return g.seal()


def cache_digest(g):
    """Expensive explicit audit, includes all stored KV bytes; not per-token work."""
    h = hashlib.sha256()
    for name, cache in [('__prefix__', g.prefix_cache)] + list(g.nodes.items()):
        h.update(name.encode())
        if cache is None:
            continue
        for tensors in (cache.key_cache, cache.value_cache):
            for x in tensors:
                h.update(str((tuple(x.shape), x.dtype)).encode())
                h.update(x.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _sync(model):
    device = input_device(model)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


@torch.no_grad()
def _greedy(model, tokenizer, ids, limit, *, past=None, pos=None, slots=None,
            hidden_ranges=()):
    """Cached greedy decoding. A fixed visibility mask is applied on EVERY step."""
    device = ids.device
    if slots is None:
        slots = torch.arange(ids.shape[1], device=device)
    if pos is None:
        pos = slots.unsqueeze(0)
    eos = getattr(model.generation_config, 'eos_token_id', None)
    if eos is None:
        eos = tokenizer.eos_token_id
    eos = set([] if eos is None else ([eos] if isinstance(eos, int) else eos))
    tokens, stopped = [], False
    for _ in range(limit):
        past_len = 0 if past is None else GraphKVCache._seq_len(past)
        total = past_len + ids.shape[1]
        keys = torch.arange(total, device=device)
        allowed = keys[None, :] <= slots[:, None]
        for start, end in hidden_ranges:
            allowed[:, start:end] = False
        mask = torch.zeros((1, 1, ids.shape[1], total), dtype=model.dtype, device=device)
        mask.masked_fill_(~allowed[None, None, :, :], torch.finfo(model.dtype).min)
        out = model(ids, past_key_values=past, position_ids=pos, cache_position=slots,
                    attention_mask=mask, use_cache=True, return_dict=True)
        past = out.past_key_values
        ids = out.logits[:, -1].argmax(-1, keepdim=True)
        token = int(ids.item())
        tokens.append(token)
        if token in eos:
            stopped = True
            break
        pos, slots = pos[:, -1:] + 1, slots[-1:] + 1
    return GenerationResult(tokenizer.decode(tokens, skip_special_tokens=True), tokens, stopped)


class QueryEngine:
    """Frozen cache + explicit-entity plan + budgeted graph/sequential inference.

    No automatic correctness detector. Fallback decisions happen BEFORE generation.
    Wrong/incomplete graph edges remain an upstream data-quality problem.
    """
    def __init__(self, model, tokenizer, spec, *, model_revision,
                 context_budget=4096, native_assistant_newline=False):
        _check_model(model)
        if getattr(model.config, '_attn_implementation', '') != 'eager':
            raise ValueError('Research prototype requires attn_implementation="eager"')
        devices = {p.device for p in model.parameters()}
        if len(devices) != 1 or next(iter(devices)).type not in ('cpu', 'cuda'):
            raise ValueError('Use one CPU or one GPU; sharded/offloaded models are not validated')
        if not isinstance(context_budget, int) or context_budget <= 0:
            raise ValueError('context_budget must be positive')
        if not isinstance(model_revision, str) or not model_revision.strip():
            raise ValueError('Record a nonempty model revision/path')
        self.model, self.tokenizer, self.spec = model, tokenizer, spec
        self.context_budget = min(context_budget, int(model.config.max_position_embeddings))
        self.model_revision = model_revision
        self.native_assistant_newline = bool(native_assistant_newline)
        self.planner = QueryPlanner(spec)
        self.prefix_ids = self._tokenize(spec.prefix)
        self.node_ids = MappingProxyType({n.node_id: self._tokenize(n.text) for n in spec.nodes})
        self.cache_revision = uuid.uuid4().hex  # Plans never silently cross snapshots.
        # Fail before encoding if even an individual construction pass exceeds budget.
        for n in spec.nodes:
            dependencies = spec.sources_for((n.node_id,)) if n.kind == 'target' else ()
            length = len(self.node_ids[n.node_id]) + sum(len(self.node_ids[s]) for s in dependencies)
            if length > self.context_budget:
                raise ValueError(f'Node {n.node_id} and its dependencies exceed construction budget')
        span = max(map(len, self.node_ids.values()))
        if len(self.prefix_ids) + 2 * span > self.context_budget:
            raise ValueError('Graph logical position range exceeds construction budget')
        self.cache = build_snapshot(model=model,
            source_inputs={s: self._tensor(self.node_ids[s]) for s in spec.ids('source')},
            target_inputs={t: self._tensor(self.node_ids[t]) for t in spec.ids('target')},
            edges=spec.edges, prefix_input_ids=self._tensor(self.prefix_ids))
        self._model_state = self._model_stamp()
        self._tokenizer_state = self._tokenizer_stamp()
        self._layout_state = self._layout_stamp()

    def _tokenize(self, text):
        ids = tuple(self.tokenizer.encode(text, add_special_tokens=False))
        if not ids:
            raise ValueError('Empty tokenized block')
        return ids

    def _tensor(self, ids):
        return torch.tensor([ids], dtype=torch.long, device=input_device(self.model))

    def _model_stamp(self):
        return (id(self.model), self.model_revision, self.model.training,
                self.model.config.to_json_string(),
                getattr(self.model.config, '_attn_implementation', None),
                self.model.generation_config.to_json_string(),
                tuple((n, tensor_stamp(p)) for n, p in self.model.named_parameters()),
                tuple((n, tensor_stamp(b)) for n, b in self.model.named_buffers()))

    def _tokenizer_stamp(self):
        t = self.tokenizer
        return (id(t), len(t), t.bos_token_id, t.eos_token_id, t.pad_token_id,
                str(getattr(t, 'special_tokens_map', {})))

    def _layout_stamp(self):
        return (id(self.cache), self.spec.revision, self.cache_revision, self.prefix_ids,
                tuple(self.node_ids.items()), self.context_budget, self.native_assistant_newline)

    def assert_consistent(self):
        if self._model_stamp() != self._model_state:
            raise StaleCacheError('Model weights/config/mode changed; build a new engine')
        if self._tokenizer_stamp() != self._tokenizer_state or self._layout_stamp() != self._layout_state:
            raise StaleCacheError('Tokenizer or snapshot configuration changed; rebuild')
        self.cache.assert_consistent()

    def plan(self, question):
        self.assert_consistent()
        return self.planner.plan(question, self.cache_revision)

    def _query_ids(self, question):
        boundary = '\n\n' if self.native_assistant_newline else '\n'
        return self._tokenize('\nQuestion: ' + question + boundary + '<|assistant|>\n')

    def _validate_plan(self, plan):
        self.assert_consistent()
        if plan.graph_revision != self.spec.revision or plan.cache_revision != self.cache_revision:
            raise StaleCacheError('Plan belongs to a different graph/cache snapshot')
        if not isinstance(plan.question, str) or not plan.question.strip() or plan.mode not in ('graph', 'sequential'):
            raise ValueError('Invalid question or execution mode')
        chosen = tuple(plan.source_ids) + tuple(plan.target_ids)
        if not chosen or len(chosen) != len(set(chosen)):
            raise ValueError('Selection must be nonempty and have no duplicates')
        if not set(plan.source_ids) <= set(self.spec.ids('source')) or not set(plan.target_ids) <= set(self.spec.ids('target')):
            raise ValueError('Selection contains unknown or incorrectly typed nodes')
        if not set(self.spec.sources_for(plan.target_ids)) <= set(plan.source_ids):
            raise ValueError('Selection omits graph dependencies of a selected target')

    def _budget(self, plan, limit, *, full_physical=False):
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise ValueError('max_new_tokens must be a positive integer')
        q = self._query_ids(plan.question)
        chosen = (self.spec.ids('source') + self.spec.ids('target') if full_physical
                  else tuple(plan.source_ids) + tuple(plan.target_ids))
        past_len = len(self.prefix_ids) + sum(len(self.node_ids[n]) for n in chosen)
        logical_start = (self.cache.summary()['query_position_start'] if plan.mode == 'graph' else past_len)
        need = max(past_len + len(q) + limit, logical_start + len(q) + limit)
        return q, past_len, logical_start, need

    def answer(self, question, *, mode='hybrid', max_new_tokens=64):
        plan = self.plan(question)
        if mode in ('graph_all', 'sequential_all'):
            plan = replace(plan, mode='graph' if mode == 'graph_all' else 'sequential',
                           source_ids=self.spec.ids('source'), target_ids=self.spec.ids('target'),
                           reason='forced_full_comparison')
        elif mode == 'graph_selected':
            if plan.mode != 'graph':
                raise ValueError('Planner requested fallback; use hybrid or sequential_selected')
        elif mode == 'sequential_selected':
            plan = replace(plan, mode='sequential', reason='same_selection_sequential_comparison')
        elif mode != 'hybrid':
            raise ValueError('Unknown comparison mode')
        return self.answer_plan(plan, max_new_tokens=max_new_tokens)

    def answer_plan(self, plan, *, max_new_tokens=64):
        self._validate_plan(plan)
        q, past_len, logical_start, need = self._budget(plan, max_new_tokens)
        row = dict(plan=asdict(plan), path=plan.mode, physical_prompt_tokens=past_len + len(q),
                   query_logical_start=logical_start, required_budget=need,
                   context_budget=self.context_budget, max_new_tokens=max_new_tokens)
        if need > self.context_budget:
            return dict(row, status='budget_exceeded', answer=None, token_ids=[], elapsed_seconds=None)
        _sync(self.model)
        start = time.perf_counter()
        if plan.mode == 'graph':
            result = greedy_generate_graphkv(model=self.model, tokenizer=self.tokenizer,
                graph_cache=self.cache, query_input_ids=self._tensor(q),
                source_ids=plan.source_ids, target_ids=plan.target_ids,
                max_new_tokens=max_new_tokens, return_details=True)
        else:
            prompt = self.prefix_ids + tuple(x for n in plan.source_ids + plan.target_ids for x in self.node_ids[n]) + q
            result = _greedy(self.model, self.tokenizer, self._tensor(prompt), max_new_tokens)
        _sync(self.model)
        elapsed = time.perf_counter() - start
        self.assert_consistent()
        return dict(row, status='ok', answer=result.text, token_ids=result.token_ids,
                    stopped_on_eos=result.stopped_on_eos,
                    hit_limit=not result.stopped_on_eos and len(result.token_ids) == max_new_tokens,
                    elapsed_seconds=elapsed)

    def answer_masked(self, question, visible_ids, *, max_new_tokens=64):
        """Diagnostic only: retain the SAME full KV tensor shape, mask some nodes.

        Does not remove information already mixed into visible targets. No dependency
        closure requirement: deliberately supports causal-intervention conditions.
        """
        self.assert_consistent()
        visible_ids = tuple(visible_ids)
        if len(visible_ids) != len(set(visible_ids)) or not set(visible_ids) <= set(self.node_ids):
            raise ValueError('Unknown or duplicate visible node IDs')
        plan = replace(self.plan(question), mode='graph', source_ids=self.spec.ids('source'),
                       target_ids=self.spec.ids('target'), reason='fixed_shape_visibility_diagnostic')
        q, past_len, logical_start, need = self._budget(plan, max_new_tokens, full_physical=True)
        row = dict(plan=asdict(plan), path='graph_fixed_shape_mask', visible_ids=list(visible_ids),
                   physical_prompt_tokens=past_len + len(q), query_logical_start=logical_start,
                   required_budget=need, context_budget=self.context_budget, max_new_tokens=max_new_tokens)
        if need > self.context_budget:
            return dict(row, status='budget_exceeded', answer=None, token_ids=[], elapsed_seconds=None)
        hidden, offset = [], len(self.prefix_ids)
        for name in self.spec.ids('source') + self.spec.ids('target'):
            end = offset + len(self.node_ids[name])
            if name not in visible_ids:
                hidden.append((offset, end))
            offset = end
        _sync(self.model)
        start = time.perf_counter()
        ids = self._tensor(q)
        past, pos, slots = self.cache.prepare_query(ids)
        result = _greedy(self.model, self.tokenizer, ids, max_new_tokens,
                         past=past, pos=pos, slots=slots, hidden_ranges=hidden)
        _sync(self.model)
        elapsed = time.perf_counter() - start
        self.assert_consistent()
        return dict(row, status='ok', answer=result.text, token_ids=result.token_ids,
                    stopped_on_eos=result.stopped_on_eos,
                    hit_limit=not result.stopped_on_eos and len(result.token_ids) == max_new_tokens,
                    elapsed_seconds=elapsed)

    def manifest(self):
        return dict(graph_revision=self.spec.revision, cache_revision=self.cache_revision,
                    model_revision=self.model_revision, dtype=str(self.model.dtype),
                    device=str(input_device(self.model)), context_budget=self.context_budget,
                    token_ids={'prefix': self.prefix_ids, **dict(self.node_ids)},
                    layout=self.cache.summary(), native_assistant_newline=self.native_assistant_newline)
