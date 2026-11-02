"""Contract tests for llm_request_events tool_id + usage aggregate SQL shape.

No live Postgres — verifies insert column list and aggregate method signatures /
SQL fragments stay aligned with the identity+tool contract.
"""

from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[3]
PG = ROOT / "src" / "backends" / "postgres.py"
MAIN = ROOT / "src" / "main.py"
SCHEMA = ROOT / "src" / "schemas" / "__init__.py"
INIT_SQL = REPO / "init-scripts" / "23-llm-request-events-tool-id.sql"


class TestLlmUsageToolContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pg_src = PG.read_text(encoding="utf-8")
        cls.main_src = MAIN.read_text(encoding="utf-8")
        cls.schema_src = SCHEMA.read_text(encoding="utf-8")
        cls.init_src = INIT_SQL.read_text(encoding="utf-8")

    def test_migration_adds_tool_id(self):
        self.assertIn("ADD COLUMN IF NOT EXISTS tool_id", self.init_src)
        self.assertIn("idx_llm_request_events_tool_created", self.init_src)
        self.assertIn("WHERE COALESCE(tool_id, '') = ''", self.init_src)

    def test_schema_declares_tool_id(self):
        self.assertIn("tool_id TEXT NOT NULL DEFAULT ''", self.schema_src)

    def test_schema_declares_username(self):
        self.assertIn("username TEXT NOT NULL DEFAULT ''", self.schema_src)

    def test_backfill_does_not_invent_tool_from_caller(self):
        self.assertNotIn("legacy:' || caller", self.schema_src)
        self.assertNotIn("'legacy:' || caller", self.init_src)

    def test_insert_includes_tool_id(self):
        self.assertIn("tool_id, plugin_id", self.pg_src)
        # Live path must not default user_id to unknown in insert args.
        self.assertNotIn(
            'str(fields.get("user_id") or "unknown")',
            self.pg_src,
        )

    def test_aggregate_methods_exist(self):
        for name in (
            "llm_request_usage_by_tool",
            "llm_request_usage_by_user_tool",
            "llm_request_usage_by_model_tool",
            "llm_request_usage_by_tool_user",
            "llm_request_usage_by_tool_model",
        ):
            self.assertIn(f"async def {name}", self.pg_src, name)

    def test_storage_routes_expose_tool_aggregates(self):
        for path in (
            '"/llm/usage/by-tool"',
            '"/llm/usage/by-user-tool"',
            '"/llm/usage/by-model-tool"',
            '"/llm/usage/by-tool-user"',
            '"/llm/usage/by-tool-model"',
        ):
            self.assertIn(path, self.main_src, path)

    def test_list_supports_tool_id_filter(self):
        self.assertIn("tool_id: Optional[str] = None", self.main_src)
        self.assertIn("tool_id=tool_id", self.main_src)

    def test_summary_counts_tools(self):
        self.assertIn("AS tool_count", self.pg_src)


class TestLlmUsageSqlNoFallbackGuess(unittest.TestCase):
    def test_insert_does_not_guess_tool_from_route(self):
        # Insert body should take tool_id from fields, not CASE on route_key.
        insert_block = PG.read_text(encoding="utf-8")
        start = insert_block.find("async def llm_request_event_insert")
        end = insert_block.find("async def llm_request_events_list")
        body = insert_block[start:end]
        self.assertNotIn("split_part(route_key", body)
        self.assertIn('str(fields.get("tool_id") or "").strip()', body)


if __name__ == "__main__":
    unittest.main()
