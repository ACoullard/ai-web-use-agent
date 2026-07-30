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


def capture_full_input() -> bool:
    """Whether to capture the exact model input of every request (`TRACE_FULL_INPUT`)."""
    return os.environ.get("TRACE_FULL_INPUT", "false").lower() == "true"


def request_capture_path(trace_id: str) -> Path:
    """Where the exact model input for `trace_id` goes: <traces_root>/requests/<id>.requests.txt."""
    return traces_root() / "requests" / f"{trace_id}.requests.txt"
