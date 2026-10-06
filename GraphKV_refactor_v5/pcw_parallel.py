"""Citation graph entry points. 'batch' means disjoint graphs for ONE query."""
from kv_compat import (rotate_half, apply_rotary_pos_emb,
    apply_pkv_rerotary_position_embeddings, apply_pkv_rotary_position_embeddings,
    cut_pkv, divide_pkv, flatten_pkv, stack_pkv, concact_pkv,
    concact_pkv_before, topk_pkv, init_empty_pkv)
from graph_kv_adapter import (input_device, _tokenize_one, build_graphkv_from_ids,
                              greedy_generate_graphkv)

MIDDLE = '\nNow you will read the center paper and answer a related question: \n'


def _build_batch(tokenizer, model, prefix, centers, neighbors, query, *, graph=True):
    centers, neighbors = list(centers), list(neighbors)
    if not centers or len(centers) != len(neighbors):
        raise ValueError("Nonempty center and neighbor-group lists must have equal lengths")
    device = input_device(model)
    tok = lambda text: _tokenize_one(tokenizer, text, device)
    sources, targets, edges = {}, {}, []
    for i, (center, group) in enumerate(zip(centers, neighbors)):
        if isinstance(group, str):
            raise TypeError("Each neighbor group must be a list of texts")
        local = []
        for j, text in enumerate(group):
            sid = f"graph{i}::neighbor{j}"
            sources[sid] = tok(text)
            local.append(sid)
        tid = f"graph{i}::center"
        if graph:
            targets[tid] = tok(MIDDLE + center)
            edges.extend((s, tid) for s in local)
        else:
            sources[tid] = tok(MIDDLE + center)
    # Planning occurs only after ALL groups and centers have been tokenized.
    # An all-isolated graph is represented by independent center blocks.
    if not sources:
        sources, targets = targets, {}
    gkv = build_graphkv_from_ids(model=model, source_inputs=sources, target_inputs=targets,
        edges=edges, prefix_input_ids=tok(prefix) if prefix else None)
    return gkv, tok(query), list(sources), list(targets)


def _build_arxiv_graph_cache(tokenizer, model, emb, prefix, center_node, neighbor_nodes, query):
    return _build_batch(tokenizer, model, prefix, [center_node], [neighbor_nodes], query)


def _generate(tokenizer, model, data):
    gkv, query_ids, sources, targets = data
    return greedy_generate_graphkv(model=model, tokenizer=tokenizer, graph_cache=gkv,
        query_input_ids=query_ids, source_ids=sources, target_ids=targets)


def gapemp_graph(tokenizer, model, emb, prefix, center_node, neighbor_nodes, query,
                 model_name, temperature, scale, mode):
    return _generate(tokenizer, model, _build_arxiv_graph_cache(
        tokenizer, model, emb, prefix, center_node, neighbor_nodes, query))


def gapemp_graph_batch(tokenizer, model, emb, prefix, center_node_list, neighbor_nodes_list,
                       query, model_name, temperature, scale, mode):
    return _generate(tokenizer, model, _build_batch(
        tokenizer, model, prefix, center_node_list, neighbor_nodes_list, query))


def block(tokenizer, model, emb, prefix, center_node, neighbor_nodes, query,
          model_name, temperature, scale, mode):
    """Shared-position independent-block baseline, not upstream's serial re-RoPE baseline."""
    return _generate(tokenizer, model, _build_batch(
        tokenizer, model, prefix, [center_node], [neighbor_nodes], query, graph=False))


def block_batch(tokenizer, model, emb, prefix, center_node_list, neighbor_nodes_list,
                query, model_name, temperature, scale, mode):
    return _generate(tokenizer, model, _build_batch(
        tokenizer, model, prefix, center_node_list, neighbor_nodes_list, query, graph=False))
