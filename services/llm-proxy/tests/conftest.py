"""Make ``src.*`` importable regardless of the working directory.

``src.config`` builds its Settings singleton at import time from ``./.env``, so
it is imported once from a directory without one (tests only need defaults).
The cwd is restored immediately — other suites in the same pytest session must
not see it change.
"""

import os
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
SERVICE_ROOT = TESTS_DIR.parent

if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

_cwd = os.getcwd()
os.chdir(TESTS_DIR)
try:
    import src.config  # noqa: F401
finally:
    os.chdir(_cwd)
