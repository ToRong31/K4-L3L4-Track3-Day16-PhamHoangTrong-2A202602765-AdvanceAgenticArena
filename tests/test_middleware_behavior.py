"""Contracts and run lifetime for bound middleware chains."""
import json
from types import SimpleNamespace

import pytest

from arena.model import ModelResponse
from arena.tools import ToolResult
from harness.middleware import LoggingMiddleware, Middleware, MiddlewareStack
from tests.fixtures_briefs import BRIEF_SLA
from tests.test_middleware import _agent


@pytest.mark.parametrize("hook,bad,invoke,match", [
    ("before_model", None, lambda s: s.before_model(None, []), "before_model.*list"),
    ("after_model", SimpleNamespace(text="ok"), lambda s: s.after_model(None, ModelResponse("ok", 1, 1)), "after_model.*ModelResponse"),
    ("wrap_model_call", None, lambda s: s.wrap_model_call(None, lambda m: None)([]), "wrap_model_call.*ModelResponse"),
    ("wrap_tool_call", SimpleNamespace(ok=1, content="", error=None), lambda s: s.wrap_tool_call(None, lambda *a: None)("search", {}), "wrap_tool_call.*ToolResult"),
    ("after_agent", [], lambda s: s.after_agent(None, {}), "after_agent.*dict"),
])
def test_validation_identifies_responsible_hook(hook, bad, invoke, match):
    layer = Middleware()
    layer.name = "broken"
    setattr(layer, hook, lambda *args: bad)
    with pytest.raises(TypeError, match="broken\\." + match):
        invoke(MiddlewareStack([layer]))


def test_structural_adapters_and_valid_tool_failure_are_accepted():
    stack = MiddlewareStack([Middleware()])
    model, tool = stack.bind_calls(None, lambda m: SimpleNamespace(text="ok", prompt_tokens=1, completion_tokens=2),
                                   lambda *a: SimpleNamespace(ok=False, content="", error="timeout"))
    assert model([]).text == "ok"
    assert tool("search", {}).error == "timeout"


def test_bound_context_is_live_and_chains_are_isolated_across_runs():
    class ReadState(Middleware):
        def wrap_tool_call(self, ctx, call, name, args):
            return ToolResult(True, str(ctx.step))

    stack = MiddlewareStack([ReadState()])
    first, second = SimpleNamespace(step=1), SimpleNamespace(step=2)
    _, a = stack.bind_calls(first, None, None)
    _, b = stack.bind_calls(second, None, None)
    first.step = 9
    assert a("search", {}).content == "9"
    assert b("search", {}).content == "2"


def test_agent_binds_once_per_run_and_logging_resets_without_changing_baseline():
    plain, _, plain_trace = _agent()
    expected = plain.run(BRIEF_SLA)
    logger = LoggingMiddleware()
    agent, _, trace = _agent(middleware=[logger])
    calls = []
    original = agent.middleware.bind_calls

    def bind(*args):
        calls.append(args[0])
        return original(*args)

    agent.middleware.bind_calls = bind
    assert agent.run(BRIEF_SLA) == expected
    def semantic_events(source):
        return [{key: value for key, value in event.items() if key != "seq"}
                for event in map(json.loads, source.to_jsonl().splitlines())
                if event["event"] != "layer"]

    assert semantic_events(trace) == semantic_events(plain_trace)
    first_context = agent.last_context
    first_events = list(logger.events)
    assert len(calls) == 1
    assert agent.run(BRIEF_SLA)
    assert len(calls) == 2 and calls[-1] is not first_context
    assert logger.events.count("before_agent") == 1
    assert first_events[0] == logger.events[0] == "before_agent"


def test_bind_does_not_call_inner_and_short_circuit_remains_valid():
    class Stop(Middleware):
        def wrap_tool_call(self, ctx, call, name, args):
            return ToolResult(False, "", "blocked")

    def forbidden(*args):
        pytest.fail("short circuit called inner")

    _, tool = MiddlewareStack([Stop(), Middleware()]).bind_calls(None, forbidden, forbidden)
    assert tool("search", {}).error == "blocked"


def test_bind_once_preserves_legacy_dispatch_trace():
    bound, _, bound_trace = _agent(middleware=[Middleware() for _ in range(5)])
    legacy, _, legacy_trace = _agent(middleware=[Middleware() for _ in range(5)])

    def bind_each_call(ctx, model, tool):
        return (lambda messages: legacy.middleware.wrap_model_call(ctx, model)(messages),
                lambda name, args: legacy.middleware.wrap_tool_call(ctx, tool)(name, args))

    legacy.middleware.bind_calls = bind_each_call
    assert bound.run(BRIEF_SLA) == legacy.run(BRIEF_SLA)
    assert bound_trace.to_jsonl() == legacy_trace.to_jsonl()
