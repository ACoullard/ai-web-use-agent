import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

DEFAULT_TRACES_DIR = Path("traces")

load_dotenv(find_dotenv(usecwd=True), override=False)


def traces_root() -> Path:
    return Path(os.environ.get("TRACES_DIR") or DEFAULT_TRACES_DIR)


def live_traces_dir() -> Path:
    """Where interactive `webagent run` traces go: <traces_root>/live."""
    return traces_root() / "live"


def eval_traces_dir() -> Path:
    """Where eval-harness traces go: <traces_root>/evals."""
    return traces_root() / "evals"
