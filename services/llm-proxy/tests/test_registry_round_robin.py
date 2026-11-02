"""Round-robin peer selection — mirrors RegistryRouter._apply_round_robin."""


from collections import defaultdict
from typing import Any, Dict, List


def apply_round_robin(candidates: List[Dict[str, Any]], counters: Dict[str, int]) -> List[Dict[str, Any]]:
  if len(candidates) <= 1:
    return candidates
  by_lid: Dict[str, List[int]] = defaultdict(list)
  for i, c in enumerate(candidates):
    if c.get("backend_kind") in ("kserve", "external"):
      lid = c.get("logical_id") or c.get("model_name") or ""
      by_lid[lid].append(i)
  result = list(candidates)
  rotated = False
  for lid, idxs in by_lid.items():
    if len(idxs) <= 1:
      continue
    key = lid or "default"
    rot = counters.get(key, 0) % len(idxs)
    counters[key] = rot + 1
    first, chosen = idxs[0], idxs[rot]
    result[first], result[chosen] = result[chosen], result[first]
    rotated = True
  return result if rotated else candidates


def test_external_peers_included_in_round_robin_pool():
  lid = "Qwen"
  candidates = [
    {"backend_kind": "external", "logical_id": lid, "model_name": "a:Qwen", "node_id": "a"},
    {"backend_kind": "external", "logical_id": lid, "model_name": "b:Qwen", "node_id": "b"},
    {"backend_kind": "external", "logical_id": "Other", "model_name": "c:Other", "node_id": "c"},
  ]
  counters: Dict[str, int] = {}
  first = apply_round_robin(candidates, counters)
  second = apply_round_robin(candidates, counters)
  assert first[0]["node_id"] in ("a", "b")
  assert second[0]["node_id"] in ("a", "b")
  assert first[0]["node_id"] != second[0]["node_id"]


def test_two_node_failover_alternates():
  counters: Dict[str, int] = {}
  peers = [
    {"backend_kind": "external", "logical_id": "model", "model_name": "node_a:Model", "node_id": "node_a"},
    {"backend_kind": "external", "logical_id": "model", "model_name": "node_b:Model", "node_id": "node_b"},
  ]
  seen = {apply_round_robin(peers, counters)[0]["node_id"] for _ in range(4)}
  assert seen == {"node_a", "node_b"}


def test_independent_pools_rotate_separately():
  counters: Dict[str, int] = {}
  candidates = [
    {"backend_kind": "external", "logical_id": "a", "model_name": "a1", "node_id": "n1"},
    {"backend_kind": "external", "logical_id": "a", "model_name": "a2", "node_id": "n2"},
    {"backend_kind": "external", "logical_id": "b", "model_name": "b1", "node_id": "n3"},
    {"backend_kind": "external", "logical_id": "b", "model_name": "b2", "node_id": "n4"},
  ]
  out = apply_round_robin(candidates, counters)
  assert out[0]["logical_id"] == "a"
  assert out[2]["logical_id"] == "b"
