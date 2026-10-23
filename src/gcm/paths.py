"""Dataset locations shared by experiment entry points."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def data_path(relative=""):
    """Resolve a dataset under LOOM_DATA_ROOT, or the repository's data folder."""
    root = Path(os.environ.get("LOOM_DATA_ROOT", PROJECT_ROOT / "data"))
    return root.expanduser().resolve() / relative
