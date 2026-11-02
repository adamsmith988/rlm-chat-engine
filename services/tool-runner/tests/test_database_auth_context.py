"""X-Database-Id flows through AuthzContext and storage headers."""

from __future__ import annotations

import unittest
import unittest.mock
from unittest.mock import MagicMock

from src.authz import AuthzContext, _internal_context
from src import storage_context as sc


class TestDatabaseAuthzContext(unittest.TestCase):
    def test_as_tool_context_includes_database_id(self):
        ctx = AuthzContext(
            allowed=True,
            mode="internal",
            principal="service:case-importer",
            user_id="u1",
            database_id="db_aabbccddeeff",
            service_name="case-importer",
        )
        tool_ctx = ctx.as_tool_context()
        self.assertEqual(tool_ctx["database_id"], "db_aabbccddeeff")

    def test_internal_context_reads_x_database_id(self):
        request = MagicMock()
        headers = {
            "x-internal-key": "test-key",
            "x-service-name": "case-importer",
            "x-user-id": "u1",
            "x-database-id": "db_aabbccddeeff",
        }
        request.headers.get.side_effect = lambda n: headers.get(n.lower())

        with unittest.mock.patch("src.authz.settings") as settings:
            settings.INTERNAL_API_KEY = "test-key"
            settings.TOOL_RUNNER_TRUSTED_SERVICES = "case-importer,api-gateway"
            ctx = _internal_context(request, "tools:execute")
        self.assertIsNotNone(ctx)
        self.assertEqual(ctx.database_id, "db_aabbccddeeff")
        self.assertIsNone(ctx.case_id)

    def test_set_storage_context_sets_x_database_id(self):
        token = sc.set_storage_context(
            {
                "user_id": "u1",
                "database_id": "db_aabbccddeeff",
                "artifact_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            }
        )
        try:
            headers = sc._storage_headers.get()
            self.assertEqual(headers.get("X-Database-Id"), "db_aabbccddeeff")
            self.assertNotIn("X-Case-Id", headers)
        finally:
            sc.reset_storage_context(token)


if __name__ == "__main__":
    unittest.main()
