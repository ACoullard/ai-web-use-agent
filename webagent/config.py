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


def request_capture_path(trace_id: str) -> Path | None:
    """Where the exact model input for `trace_id` goes, or None when capture is disabled.

    Enabled by `TRACE_FULL_INPUT`; see `webagent/traces/requests.py`. Off by default
    because the capture holds the full accumulated history for every request, so it runs
    several times the size of the trace itself. Named by trace id rather than sharing the
    trace's timestamped filename: that's what `webagent trace show` addresses traces by
    anyway, and it keeps the writer from needing to know which trace directory is in use.
    """
    if (os.environ.get("TRACE_FULL_INPUT") or "").strip().lower() not in ("1", "true", "yes", "on"):
        return None
    return traces_root() / "requests" / f"{trace_id}.requests.txt"
