"""Run from the bundle root: python tests/reproduce_v4.py."""
import importlib.util
import json
import sys
from pathlib import Path
from test_graphkv import tiny, ids
from graph_kv_adapter import _encode_independent_chunk

root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("original_v4", root / "references/v4/graph_kv_cache.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
model = tiny()
g = module.GraphKVCache()
g.add_source("s", _encode_independent_chunk(model, ids(1, 2, 3)))
out = g.propagate_target(model, "t", ids(4, 5, 6, 7, 8, 9), source_ids=["s"])
report = {
    "v4_target_range": [g.meta["t"].logical_start, g.meta["t"].logical_end],
    "v4_query_start": g.query_position_ids(1).item(),
    "v4_forward_logits_requires_grad": out.logits.requires_grad,
    "v4_clone_seen_tokens": g.nodes["s"]._seen_tokens,
    "actual_source_cache_length": g.nodes["s"].get_seq_length(),
}
try:
    g.add_source("later", _encode_independent_chunk(model, ids(1, 2, 3, 4)))
    g.propagate_target(model, "t2", ids(5), source_ids=["later"])
except Exception as exc:
    report["v4_late_longer_source_error"] = repr(exc)
assert report["v4_target_range"] == [3, 9] and report["v4_query_start"] == 6
(root / "v4_reproduction.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
