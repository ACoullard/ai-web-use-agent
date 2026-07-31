import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from webagent.traces import Generation, Trace, save_trace
from webagent.traces.server import clear_cache, handle_request, make_server


@pytest.fixture(autouse=True)
def _clean_trace_cache():
    """The parsed-trace cache is module state; keep it from leaking across tests."""
    clear_cache()
    yield
    clear_cache()


def _write_trace(directory, **overrides):
    fields = dict(
        trace_id="feedface0000",
        created_at=datetime(2026, 7, 24, 9, 0, tzinfo=timezone.utc),
        task="find pricing",
        url="https://example.com",
        model="test:model",
        thinking="medium",
        output_mode="freeform",
        status="success",
        steps_taken=1,
        observations=[
            Generation(name="decide_action", step=0, duration_seconds=1.0, model="test:model",
                       input_prompt="p", output={"type": "finish", "answer": "42"})
        ],
    )
    fields.update(overrides)
    trace = Trace(**fields)
    save_trace(trace, directory)
    return trace


def test_root_serves_html(tmp_path):
    status, content_type, body = handle_request(tmp_path, "/")
    assert status == 200
    assert "text/html" in content_type
    assert body.startswith(b"<!DOCTYPE")


def test_static_assets_are_served(tmp_path):
    for name, expected_type in (("app.css", "text/css"), ("app.js", "text/javascript")):
        status, content_type, body = handle_request(tmp_path, f"/static/{name}")
        assert status == 200
        assert expected_type in content_type
        assert body


def test_static_route_rejects_traversal(tmp_path):
    for path in ("/static/../trace_web.py", "/static/js/extract_elements.js", "/static/nope.js"):
        status, _, _ = handle_request(tmp_path, path)
        assert status == 404, path


def test_api_traces_returns_metadata_without_heavy_content(tmp_path):
    _write_trace(tmp_path)
    status, content_type, body = handle_request(tmp_path, "/api/traces")
    assert status == 200
    assert content_type == "application/json"
    data = json.loads(body)
    assert len(data) == 1
    assert data[0]["task"] == "find pricing"
    assert "observations" not in data[0]  # list payload stays light
    assert "system_prompt" not in data[0]


def test_api_traces_strips_query_string(tmp_path):
    _write_trace(tmp_path)
    status, _, body = handle_request(tmp_path, "/api/traces?sort=created")
    assert status == 200
    assert len(json.loads(body)) == 1


def test_api_single_trace_returns_full_detail(tmp_path):
    trace = _write_trace(tmp_path)
    status, _, body = handle_request(tmp_path, f"/api/traces/{trace.trace_id}")
    assert status == 200
    data = json.loads(body)
    assert data["trace_id"] == trace.trace_id
    assert data["observations"][0]["output"] == {"type": "finish", "answer": "42"}


def test_api_single_trace_accepts_prefix(tmp_path):
    trace = _write_trace(tmp_path)
    status, _, _ = handle_request(tmp_path, f"/api/traces/{trace.trace_id[:8]}")
    assert status == 200


def test_api_unknown_trace_404(tmp_path):
    status, _, _ = handle_request(tmp_path, "/api/traces/does-not-exist")
    assert status == 404


def test_rewritten_trace_is_re_read_not_served_stale(tmp_path):
    """The whole point of polling: a live run rewrites its file every step, and both
    endpoints have to show the new content rather than a cached parse."""
    _write_trace(tmp_path, status="running", steps_taken=1)
    assert json.loads(handle_request(tmp_path, "/api/traces")[2])[0]["status"] == "running"

    _write_trace(tmp_path, status="running", steps_taken=2, observations=[
        Generation(name="decide_action", step=0, duration_seconds=1.0, model="test:model",
                   input_prompt="p", output={"type": "click", "index": 3}),
        Generation(name="decide_action", step=1, duration_seconds=1.0, model="test:model",
                   input_prompt="p", output={"type": "finish", "answer": "42"}),
    ])
    summary = json.loads(handle_request(tmp_path, "/api/traces")[2])[0]
    assert summary["steps_taken"] == 2
    detail = json.loads(handle_request(tmp_path, "/api/traces/feedface0000")[2])
    assert len(detail["observations"]) == 2


def test_a_momentarily_unreadable_trace_falls_back_to_the_last_good_parse(tmp_path, monkeypatch):
    """Windows refuses to open a file while its atomic replace lands, so a live run is
    briefly unreadable every step. Dropping it would make the run flicker out of the
    list mid-poll; the cached parse covers the gap."""
    _write_trace(tmp_path, status="running", steps_taken=3)
    assert json.loads(handle_request(tmp_path, "/api/traces")[2])[0]["steps_taken"] == 3

    real_read_text = Path.read_text

    def deny(self, *args, **kwargs):
        if self.suffix == ".json":
            raise PermissionError("being replaced")
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", deny)
    # the file's mtime is untouched, so force a re-parse attempt by invalidating the key
    Path(next(tmp_path.glob("*.json"))).touch()
    assert json.loads(handle_request(tmp_path, "/api/traces")[2])[0]["steps_taken"] == 3


def test_an_unreadable_trace_never_parsed_is_skipped(tmp_path):
    """The fallback only covers files we've read before - a genuinely malformed trace is
    still skipped, as load_traces does."""
    _write_trace(tmp_path)
    (tmp_path / "junk.json").write_text("not json", encoding="utf-8")
    assert len(json.loads(handle_request(tmp_path, "/api/traces")[2])) == 1


def test_deleted_trace_is_dropped_from_the_cache(tmp_path):
    trace = _write_trace(tmp_path)
    assert len(json.loads(handle_request(tmp_path, "/api/traces")[2])) == 1

    next(tmp_path.glob("*.json")).unlink()
    assert json.loads(handle_request(tmp_path, "/api/traces")[2]) == []
    assert handle_request(tmp_path, f"/api/traces/{trace.trace_id}")[0] == 404


def test_unknown_path_404(tmp_path):
    status, _, _ = handle_request(tmp_path, "/nope")
    assert status == 404


def test_make_server_binds_loopback_only(tmp_path):
    server = make_server(tmp_path, port=0)
    try:
        assert server.server_address[0] == "127.0.0.1"
        assert server.trace_dir == tmp_path
    finally:
        server.server_close()
