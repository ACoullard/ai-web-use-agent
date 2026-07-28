import json
from datetime import datetime, timezone

from webagent.traces import Generation, Trace, save_trace
from webagent.traces.server import handle_request, make_server


def _write_trace(directory):
    trace = Trace(
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
