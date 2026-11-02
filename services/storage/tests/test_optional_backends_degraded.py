"""Optional backends (Neo4j/Mongo) must degrade, never abort Storage.connect().

Regression: a Neo4j outage used to raise out of connect(), leaving every
Postgres repository (users/roles/chats/...) uninitialized and the whole
storage API answering 503 even though Postgres was healthy.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

STORAGE_PY = Path(__file__).resolve().parents[1] / "src" / "storage.py"


class TestOptionalBackendsDegraded(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = STORAGE_PY.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def _connect_fn(self) -> ast.AsyncFunctionDef:
        for node in ast.walk(self.tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "connect":
                return node
        self.fail("Storage.connect not found")

    def test_neo4j_and_mongo_connects_are_guarded(self):
        fn = self._connect_fn()
        src = ast.get_source_segment(self.source, fn) or ""
        # Both optional backends connect inside try/except and reset to None.
        self.assertIn("self.neo4j = None", src)
        self.assertIn("self.mongo = None", src)
        self.assertIn("self.qdrant = None", src)  # pre-existing Qdrant guard

    def test_repositories_initialized_in_connect(self):
        fn = self._connect_fn()
        src = ast.get_source_segment(self.source, fn) or ""
        self.assertIn("self.roles = RoleRepository(self.postgres)", src)
        self.assertIn("self.users = UserRepository(self.postgres)", src)


if __name__ == "__main__":
    unittest.main()
