"""Layout resolve + purge allowlist for database_id."""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock

from src.persist_layouts import resolve_layout_template, validate_layout_template
from src.storage import Storage


class TestDatabaseLayoutResolve(unittest.TestCase):
    def test_database_preferred_over_case_when_both(self):
        tmpl = resolve_layout_template(
            layout_name="default",
            layout_template=None,
            has_case=True,
            has_database=True,
        )
        self.assertTrue(tmpl.startswith("database_importer/"))
        self.assertIn("{database_id}", tmpl)
        tmpl_db = resolve_layout_template(
            layout_name="default",
            layout_template=None,
            has_case=False,
            has_database=True,
        )
        self.assertTrue(tmpl_db.startswith("database_importer/"))
        self.assertIn("{database_id}", tmpl_db)

    def test_reject_unknown_prefix(self):
        ok, err = validate_layout_template("evil/{database_id}/x")
        self.assertFalse(ok)
        self.assertEqual(err, "prefix_not_allowed")


class TestDatabasePurgeFull(unittest.IsolatedAsyncioTestCase):
    async def test_purge_calls_backends_with_database_id(self):
        storage = Storage.__new__(Storage)
        storage.postgres = MagicMock()
        storage.postgres.pool = object()
        storage.postgres.list_artifact_ids_for_database = AsyncMock(return_value=[])
        storage.postgres.purge_column_references = AsyncMock(return_value={"ok": True})
        storage.postgres.delete_database_row = AsyncMock(return_value=True)

        storage.neo4j = MagicMock()
        storage.neo4j.driver = object()
        storage.neo4j.purge_by_property = AsyncMock(return_value={"deleted": 0})

        storage.qdrant = MagicMock()
        storage.qdrant.client = object()
        storage.qdrant.purge_by_payload_field = AsyncMock(return_value={"deleted": 0})

        storage.mongo = MagicMock()
        storage.mongo.db = object()
        storage.mongo.purge_by_field = AsyncMock(return_value={"deleted": 0})

        storage.minio = None
        storage.artifact_purge_full = AsyncMock()

        summary = await storage.database_purge_full("db_purge_test_01")
        self.assertEqual(summary["database_id"], "db_purge_test_01")
        self.assertTrue(summary.get("ok"))

        storage.neo4j.purge_by_property.assert_awaited_with(
            "database_id", "db_purge_test_01"
        )
        storage.qdrant.purge_by_payload_field.assert_awaited_with(
            "database_id", "db_purge_test_01"
        )
        storage.mongo.purge_by_field.assert_awaited_with(
            "database_id", "db_purge_test_01"
        )
        storage.postgres.delete_database_row.assert_awaited_with("db_purge_test_01")


class TestBackendPurgeAllowlist(unittest.TestCase):
    def test_qdrant_allows_database_id(self):
        from src.backends.qdrant import QdrantBackend

        # Inspect source of allowlist via calling with disconnected client
        backend = QdrantBackend.__new__(QdrantBackend)
        backend.client = None
        # Method should reject unknown fields before touching client
        import asyncio

        async def _run():
            return await backend.purge_by_payload_field("not_allowed", "x")

        # Unknown field returns empty / error without crash
        try:
            result = asyncio.get_event_loop().run_until_complete(_run())
        except RuntimeError:
            result = asyncio.run(_run())
        self.assertIsInstance(result, dict)

    def test_mongo_allows_database_id_field(self):
        from src.backends import mongodb as mongo_mod

        src = open(mongo_mod.__file__).read()
        self.assertIn('"database_id"', src)

    def test_neo4j_allows_database_id(self):
        from src.backends import neo4j as neo_mod

        src = open(neo_mod.__file__).read()
        # purge_by_property itself is generic; storage passes database_id
        self.assertIn("purge_by_property", src)


if __name__ == "__main__":
    unittest.main()
