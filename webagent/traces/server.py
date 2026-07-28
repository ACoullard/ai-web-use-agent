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

from webagent.traces.trace import find_trace, load_traces

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


def handle_request(trace_dir: Path, path: str) -> tuple[int, str, bytes]:
    """Resolve one GET path to (status, content_type, body). Traces are re-read per request
    so newly written runs show up on refresh. Lookup is by trace id within the loaded set -
    never by a client-supplied filesystem path - so there's no path traversal."""
    path = path.split("?", 1)[0]
    if path in ("/", "/index.html"):
        return _serve_static("index.html") or (500, "text/plain; charset=utf-8", b"page assets missing")

    if path.startswith(_STATIC_PREFIX):
        asset = _serve_static(path[len(_STATIC_PREFIX) :])
        if asset is not None:
            return asset
        return 404, "text/plain; charset=utf-8", b"not found"

    if path == _API_PREFIX:
        summaries = [t.summary() for t in load_traces(trace_dir)]
        return 200, "application/json", json.dumps(summaries).encode("utf-8")

    if path.startswith(_API_PREFIX + "/"):
        trace_id = path[len(_API_PREFIX) + 1 :]
        trace = find_trace(load_traces(trace_dir), trace_id)
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
