from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    ThinkingPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.usage import RequestUsage

from webagent.actions import ClickAction
from webagent.output_spec import SelfCheckVerdict
from webagent.traces import (
    FileTracer,
    Generation,
    NullTracer,
    ToolCall,
    Trace,
    TraceRecorder,
    filter_traces,
    find_trace,
    load_traces,
    save_trace,
)


class _FakeResult:
    """Stand-in for a Pydantic AI AgentRunResult in recorder tests."""

    def __init__(self, messages, output):
        self._messages = messages
        self.output = output

    def new_messages(self):
        return self._messages


def _recorder(**overrides) -> TraceRecorder:
    kwargs = dict(
        task="find the price",
        url="https://example.com",
        model="test:model",
        thinking="medium",
        output_mode="freeform",
        system_prompt="you are an agent",
    )
    kwargs.update(overrides)
    return TraceRecorder(**kwargs)


def test_record_generation_extracts_reasoning_action_and_tokens():
    request = ModelRequest(parts=[UserPromptPart(content="observe the page")])
    response = ModelResponse(
        parts=[
            ThinkingPart(content="the Pricing link is element 4"),
            ToolCallPart(tool_name="final_result", args={"type": "click", "index": 4}),
        ],
        usage=RequestUsage(input_tokens=100, output_tokens=20, details={"reasoning_tokens": 15}),
        model_name="test:model-x",
        finish_reason="stop",
        provider_response_id="resp_1",
    )
    rec = _recorder()
    rec.record_generation(0, "observe the page", _FakeResult([request, response], ClickAction(index=4, memory="clicked Pricing, 1 of 2 done")), 1.5)

    assert len(rec.observations) == 1
    gen = rec.observations[0]
    assert isinstance(gen, Generation)
    assert gen.name == "decide_action"
    assert gen.reasoning == "the Pricing link is element 4"
    assert gen.reasoning_encrypted is False
    assert gen.output == {"type": "click", "index": 4}  # memory promoted out of the action
    assert gen.memory == "clicked Pricing, 1 of 2 done"
    assert (gen.input_tokens, gen.output_tokens, gen.reasoning_tokens) == (100, 20, 15)
    assert gen.finish_reason == "stop"
    assert gen.provider_response_id == "resp_1"
    assert gen.model == "test:model-x"
    assert gen.input_prompt == "observe the page"


def test_record_generation_flags_encrypted_reasoning():
    request = ModelRequest(parts=[UserPromptPart(content="p")])
    response = ModelResponse(
        parts=[
            ThinkingPart(content="", signature="enc-blob", provider_name="openai"),
            ToolCallPart(tool_name="final_result", args={"type": "click", "index": 1}),
        ],
        usage=RequestUsage(input_tokens=10, output_tokens=5, details={"reasoning_tokens": 42}),
    )
    rec = _recorder()
    rec.record_generation(0, "p", _FakeResult([request, response], ClickAction(index=1, memory="")), 0.5)

    gen = rec.observations[0]
    assert gen.reasoning is None
    assert gen.reasoning_encrypted is True
    assert gen.reasoning_tokens == 42


def test_record_generation_records_one_per_response_on_retry():
    req1 = ModelRequest(parts=[UserPromptPart(content="first prompt")])
    resp_failed = ModelResponse(
        parts=[ToolCallPart(tool_name="final_result", args={"type": "click", "index": 9})],
        usage=RequestUsage(input_tokens=10, output_tokens=2),
    )
    req_retry = ModelRequest(parts=[RetryPromptPart(content="index 9 is invalid, try again")])
    resp_ok = ModelResponse(
        parts=[ToolCallPart(tool_name="final_result", args={"type": "click", "index": 4})],
        usage=RequestUsage(input_tokens=12, output_tokens=3),
    )
    rec = _recorder()
    rec.record_generation(
        0, "first prompt", _FakeResult([req1, resp_failed, req_retry, resp_ok], ClickAction(index=4, memory="retried")), 2.0
    )

    assert len(rec.observations) == 2
    first, final = rec.observations
    # the intermediate (failed) response shows what it attempted; only the last carries the accepted output
    assert first.output == {"tool_name": "final_result", "args": {"type": "click", "index": 9}}
    assert first.duration_seconds == 0.0
    assert final.output == {"type": "click", "index": 4}
    assert final.duration_seconds == 2.0
    assert first.input_prompt == "first prompt"
    assert final.input_prompt != "first prompt"  # reconstructed from the retry request


def test_record_tool_ok_and_error():
    rec = _recorder()
    rec.record_tool(ClickAction(index=4, memory="note"), 1, 0.2, status="ok", result="a search hit")
    rec.record_tool(ClickAction(index=7, memory="note"), 2, 0.1, status="error", error="no element 7")

    ok, err = rec.observations
    assert isinstance(ok, ToolCall)
    assert ok.name == "click" and ok.args == {"index": 4} and ok.status == "ok" and ok.result == "a search hit"
    assert err.status == "error" and err.error == "no element 7" and err.result is None


def test_record_self_check_captures_verdict_reasoning_and_tokens():
    request = ModelRequest(parts=[UserPromptPart(content="does it satisfy?")])
    response = ModelResponse(
        parts=[
            ThinkingPart(content="the answer names a concrete price"),
            ToolCallPart(tool_name="final_result", args={"passes": True, "reason": "looks good"}),
        ],
        usage=RequestUsage(input_tokens=30, output_tokens=8, details={"reasoning_tokens": 12}),
    )
    rec = _recorder()
    rec.record_self_check(
        3,
        "does it satisfy?",
        _FakeResult([request, response], SelfCheckVerdict(passes=True, reason="looks good")),
        0.4,
    )

    gen = rec.observations[0]
    assert gen.name == "self_check"
    assert gen.output == {"passes": True, "reason": "looks good"}
    assert gen.reasoning == "the answer names a concrete price"
    assert (gen.input_tokens, gen.output_tokens, gen.reasoning_tokens) == (30, 8, 12)


def test_finish_sums_tokens_and_carries_metadata():
    rec = _recorder(fixture_id="find-pricing", run_id="run123")
    rec.observations.append(Generation(name="decide_action", step=0, duration_seconds=1.0, model="m",
                                        input_prompt="a", input_tokens=100, output_tokens=20))
    rec.observations.append(Generation(name="decide_action", step=1, duration_seconds=1.0, model="m",
                                        input_prompt="b", input_tokens=50, output_tokens=10))
    trace = rec.finish(status="success", steps_taken=2, duration=3.3)

    assert trace.total_input_tokens == 150
    assert trace.total_output_tokens == 30
    assert trace.status == "success" and trace.steps_taken == 2
    assert trace.fixture_id == "find-pricing" and trace.run_id == "run123"
    assert trace.system_prompt == "you are an agent"


_START_METADATA = dict(
    task="find the price",
    url="https://example.com",
    model="test:model",
    thinking="medium",
    output_mode="freeform",
    system_prompt="you are an agent",
)


def test_file_tracer_writes_on_finish(tmp_path):
    recording = FileTracer(tmp_path).start(**_START_METADATA)
    recording.record_tool(ClickAction(index=1, memory=""), 0, 0.1, status="ok")
    assert load_traces(tmp_path) == []  # nothing on disk until the run is closed
    recording.finish(status="success", steps_taken=1, duration=1.0)

    loaded = load_traces(tmp_path)
    assert [t.trace_id for t in loaded] == [recording.trace_id]
    assert loaded[0].task == "find the price"


def test_live_tracer_writes_a_snapshot_after_every_observation(tmp_path):
    recording = FileTracer(tmp_path, live=True).start(**_START_METADATA)

    recording.record_tool(ClickAction(index=1, memory=""), 0, 0.1, status="ok")
    mid = load_traces(tmp_path)
    assert [t.status for t in mid] == ["running"]
    assert len(mid[0].observations) == 1

    recording.record_tool(ClickAction(index=2, memory=""), 1, 0.1, status="ok")
    mid = load_traces(tmp_path)
    assert len(mid) == 1  # snapshots overwrite one file, they don't accumulate
    assert len(mid[0].observations) == 2
    assert mid[0].steps_taken == 2  # tracks the highest step seen so far

    recording.finish(status="success", steps_taken=2, duration=1.0)
    final = load_traces(tmp_path)
    assert [t.status for t in final] == ["success"]
    assert len(final[0].observations) == 2


def test_finish_is_idempotent(tmp_path):
    """The cleanup path in run_task calls finish() unconditionally, so a run that already
    returned normally must not have its status overwritten."""
    recording = FileTracer(tmp_path, live=True).start(**_START_METADATA)
    recording.record_tool(ClickAction(index=1, memory=""), 0, 0.1, status="ok")
    first = recording.finish(status="success", steps_taken=1, duration=1.0)

    again = recording.finish(status="interrupted", steps_taken=99, duration=99.0)
    assert again is first
    assert load_traces(tmp_path)[0].status == "success"


def test_snapshots_stop_once_the_run_is_finished(tmp_path):
    recording = FileTracer(tmp_path, live=True).start(**_START_METADATA)
    recording.finish(status="success", steps_taken=0, duration=1.0)

    recording.record_tool(ClickAction(index=1, memory=""), 0, 0.1, status="ok")
    assert load_traces(tmp_path)[0].status == "success"


def test_file_tracer_with_labels_stamps_metadata_and_leaves_original_alone(tmp_path):
    base = FileTracer(tmp_path)
    labelled = base.with_labels(run_id="run123").with_labels(fixture_id="find-pricing")
    labelled.start(**_START_METADATA).finish(status="success", steps_taken=1, duration=1.0)

    trace = load_traces(tmp_path)[0]
    assert (trace.run_id, trace.fixture_id) == ("run123", "find-pricing")
    assert base.labels == {}  # deriving a labelled tracer must not mutate its parent


def test_with_labels_preserves_live(tmp_path):
    labelled = FileTracer(tmp_path, live=True).with_labels(run_id="run123")
    assert labelled.live is True
    labelled.start(**_START_METADATA).record_tool(ClickAction(index=1, memory=""), 0, 0.1, status="ok")
    assert load_traces(tmp_path)[0].status == "running"


def test_null_tracer_writes_nothing_and_has_no_trace_id(tmp_path):
    recording = NullTracer().start(**_START_METADATA)
    recording.record_tool(ClickAction(index=1, memory=""), 0, 0.1, status="ok")

    assert recording.finish(status="success", steps_taken=1, duration=1.0) is None
    assert recording.trace_id is None
    assert load_traces(tmp_path) == []


def _sample_trace(**overrides) -> Trace:
    rec = _recorder(**overrides)
    rec.observations.append(Generation(name="decide_action", step=0, duration_seconds=1.0, model="test:model",
                                        input_prompt="prompt", output={"type": "finish", "answer": "42"},
                                        input_tokens=10, output_tokens=2))
    return rec.finish(status="success", steps_taken=1, duration=1.2)


def test_save_and_load_round_trip(tmp_path):
    trace = _sample_trace()
    path = save_trace(trace, tmp_path)
    assert path.exists()

    loaded = load_traces(tmp_path)
    assert len(loaded) == 1
    assert loaded[0].trace_id == trace.trace_id
    assert loaded[0].observations[0].output == {"type": "finish", "answer": "42"}


def test_save_trace_overwrites_in_place_and_leaves_no_temp_files(tmp_path):
    """A live run rewrites one path every step; the temp file the atomic write goes
    through must never be left behind for load_traces to trip over."""
    trace = _sample_trace()
    first = save_trace(trace, tmp_path)
    trace.status = "success"
    trace.steps_taken = 7
    second = save_trace(trace, tmp_path)

    assert first == second
    assert [p.name for p in tmp_path.iterdir()] == [first.name]
    assert load_traces(tmp_path)[0].steps_taken == 7


def test_load_traces_skips_malformed_files(tmp_path):
    save_trace(_sample_trace(), tmp_path)
    (tmp_path / "junk.json").write_text("not json", encoding="utf-8")
    assert len(load_traces(tmp_path)) == 1


def test_load_traces_missing_dir_returns_empty(tmp_path):
    assert load_traces(tmp_path / "nope") == []


def test_filter_traces():
    traces = [
        _sample_trace(task="find the pricing page", model="openai:gpt", thinking="high"),
        _sample_trace(task="check stock status", model="anthropic:claude", thinking="off"),
    ]
    assert len(filter_traces(traces, task="PRICING")) == 1
    assert len(filter_traces(traces, model="anthropic:claude")) == 1
    assert len(filter_traces(traces, thinking="off")) == 1
    assert filter_traces(traces, status="success") == traces
    assert filter_traces(traces, task="nonexistent") == []


def test_find_trace_by_prefix():
    a = _sample_trace()
    b = _sample_trace()
    traces = [a, b]
    assert find_trace(traces, a.trace_id) is a
    assert find_trace(traces, a.trace_id[:8]) is a
    assert find_trace(traces, "zzzzzzzz") is None
