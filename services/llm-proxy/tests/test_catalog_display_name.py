"""Catalog display_name resolution for logical model pools."""

from __future__ import annotations

import unittest


def _resolve_catalog_display_name(logical_id: str, candidates: list) -> str:
    """Mirror of RegistryRouter._resolve_catalog_display_name (pure)."""
    lid = str(logical_id or "").strip()
    cleaned = [str(c).strip() for c in candidates if str(c or "").strip()]
    if not cleaned:
        return lid or "model"
    unique = list(dict.fromkeys(cleaned))
    if lid and any(c == lid for c in unique):
        return lid
    if len(unique) == 1:
        return unique[0]
    return lid or unique[0]


class CatalogDisplayNameTests(unittest.TestCase):
    def test_divergent_pool_served_names_use_logical_id(self):
        # Same pool: admin served qwen3-5-0-8b, node_2 served qwen-08b-pool
        got = _resolve_catalog_display_name(
            "qwen3-5-0-8b",
            ["qwen3-5-0-8b", "qwen-08b-pool"],
        )
        self.assertEqual(got, "qwen3-5-0-8b")

    def test_single_pretty_alias_kept(self):
        got = _resolve_catalog_display_name(
            "ornith-1-0-9b",
            ["Ornith-1.0-9B"],
        )
        self.assertEqual(got, "Ornith-1.0-9B")

    def test_empty_candidates_fall_back_to_logical_id(self):
        self.assertEqual(
            _resolve_catalog_display_name("qwen3-5-0-8b", []),
            "qwen3-5-0-8b",
        )


if __name__ == "__main__":
    unittest.main()
