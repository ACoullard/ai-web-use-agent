"""A tiny, dependency-free local website for browsing traces on disk.

A stdlib HTTP server bound to loopback serves the static page in `web/` plus a small
read-only JSON API over the traces in a directory. The page has two levels: a
filterable/sortable list of runs, and a drill-down detail view of one run's step timeline
(generation vs tool observations, expandable to the exact input/reasoning/output).

Path resolution is kept pure - path in, (status, content-type, body) out - so it can be
unit-tested without opening a socket; the HTTP handler is a thin adapter over it.
"""

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from pydantic import ValidationError

from webagent.traces.trace import Trace, find_trace

logger = logging.getLogger(__name__)

_API_PREFIX = "/api/traces"
_STATIC_PREFIX = "/static/"
_WEB_DIR = Path(__file__).parent / "web"
_STATIC_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
}


def _serve_static(name: str) -> tuple[int, str, bytes] | None:
    """Read one file straight out of `web/`, or None if it isn't a servable asset.

    Only direct children of that directory with a known extension are served, so a
    crafted name (`../`, an absolute path) resolves outside `_WEB_DIR` and is rejected.
    Files are read per request rather than cached at import, so editing the page during
    development just needs a browser refresh.
    """
    path = (_WEB_DIR / name).resolve()
    if path.parent != _WEB_DIR.resolve() or path.suffix not in _STATIC_TYPES or not path.is_file():
        return None
    return 200, _STATIC_TYPES[path.suffix], path.read_bytes()


_TRACE_CACHE: dict[Path, tuple[tuple[int, int], Trace]] = {}


def clear_cache() -> None:
    """Drop the parsed-trace cache. For tests; the server never needs it."""
    _TRACE_CACHE.clear()


def _load_traces_cached(directory: Path) -> list[Trace]:
    """`load_traces`, but re-parsing only the files whose mtime/size changed.

    Without this, every poll would re-parse every trace ever recorded
    just to serve one. Keyed on (mtime_ns, size); malformed files are skipped, matching
    `load_traces`.
    """
    if not directory.exists():
        return []
    traces: list[Trace] = []
    seen: set[Path] = set()
    for path in directory.rglob("*.json"):
        try:
            stat = path.stat()
        except OSError:
            continue
        seen.add(path)
        key = (stat.st_mtime_ns, stat.st_size)
        cached = _TRACE_CACHE.get(path)
        if cached is not None and cached[0] == key:
            traces.append(cached[1])
            continue
        try:
            trace = Trace.model_validate_json(path.read_text(encoding="utf-8"))
        except (ValidationError, ValueError, OSError):
            # Windows briefly refuses to open a file while its atomic replace lands, so a
            # live run is unreadable for a moment every step. Serving the last good parse
            # keeps it from flickering out of the list; the cache entry keeps its old key,
            # so the next poll re-parses. Only a file we've never read is dropped, which
            # is what `load_traces` does with a malformed trace.
            if cached is not None:
                traces.append(cached[1])
            continue
        _TRACE_CACHE[path] = (key, trace)
        traces.append(trace)
    for gone in _TRACE_CACHE.keys() - seen:
        del _TRACE_CACHE[gone]
    traces.sort(key=lambda t: t.created_at, reverse=True)
    return traces


def handle_request(trace_dir: Path, path: str) -> tuple[int, str, bytes]:
    """Resolve one GET path to (status, content_type, body). Traces are re-read per request
    (cache-checked against mtime) so a run in progress shows up as it advances. Lookup is by
    trace id within the loaded set - never by a client-supplied filesystem path - so there's
    no path traversal."""
    path = path.split("?", 1)[0]
    if path in ("/", "/index.html"):
        return _serve_static("index.html") or (500, "text/plain; charset=utf-8", b"page assets missing")

    if path.startswith(_STATIC_PREFIX):
        asset = _serve_static(path[len(_STATIC_PREFIX) :])
        if asset is not None:
            return asset
        return 404, "text/plain; charset=utf-8", b"not found"

    if path == _API_PREFIX:
        summaries = [t.summary() for t in _load_traces_cached(trace_dir)]
        return 200, "application/json", json.dumps(summaries).encode("utf-8")

    if path.startswith(_API_PREFIX + "/"):
        trace_id = path[len(_API_PREFIX) + 1 :]
        trace = find_trace(_load_traces_cached(trace_dir), trace_id)
        if trace is None:
            return 404, "application/json", b'{"error": "trace not found"}'
        return 200, "application/json", trace.model_dump_json().encode("utf-8")

    return 404, "text/plain; charset=utf-8", b"not found"


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
        status, content_type, body = handle_request(self.server.trace_dir, self.path)  # type: ignore[attr-defined]
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:  # noqa: A002 (stdlib signature)
        logger.debug("trace-web %s - %s", self.address_string(), format % args)


def make_server(trace_dir: Path, *, host: str = "127.0.0.1", port: int = 8756) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), _Handler)
    server.trace_dir = trace_dir  # type: ignore[attr-defined]
    return server


def serve(trace_dir: Path, *, host: str = "127.0.0.1", port: int = 8756, open_browser: bool = True) -> None:
    """Run the trace browser until interrupted (Ctrl-C)."""
    server = make_server(trace_dir, host=host, port=port)
    url = f"http://{host}:{server.server_address[1]}"
    print(f"Serving traces from {trace_dir} at {url}  (Ctrl-C to stop)")
    if open_browser:
        import webbrowser

        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
