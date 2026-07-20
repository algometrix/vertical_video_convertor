"""Make `src/` importable for the test suite without installing the package.

Putting it first also guarantees the tests exercise THIS checkout even when
some other version of vertical_video_converter is installed in the active
environment.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
