"""Durable, human- and AI-legible traces of an agent run."""

from webagent.traces.requests import CapturingModel, render_request
from webagent.traces.server import serve
from webagent.traces.trace import (
    FileTracer,
    Generation,
    NullRecording,
    NullTracer,
    Observation,
    Recording,
    ToolCall,
    Trace,
    TraceRecorder,
    Tracer,
    filter_traces,
    find_trace,
    load_traces,
    save_trace,
)
from webagent.traces.view import render_list, render_trace

__all__ = [
    # model
    "Generation",
    "ToolCall",
    "Observation",
    "Trace",
    # recording / injection
    "Tracer",
    "Recording",
    "FileTracer",
    "NullTracer",
    "NullRecording",
    "TraceRecorder",
    "CapturingModel",
    "render_request",
    # storage + queries
    "save_trace",
    "load_traces",
    "filter_traces",
    "find_trace",
    # presentation
    "render_list",
    "render_trace",
    "serve",
]
