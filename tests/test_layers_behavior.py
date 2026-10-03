"""Behavioral tests with independent evidence, without public brief labels."""
import json
from types import SimpleNamespace

import pytest

from arena.corpus import INJECTION_CANARY
from arena.tools import ToolResult
from harness.agent import AgentContext
from harness.layers.budget_policy import BudgetPolicy, NUDGE
from harness.layers.citation_checker import CitationChecker
from harness.layers.critic import Critic
from harness.layers.injection_guard import BLOCK_END, BLOCK_START, PLACEHOLDER, InjectionGuard
from harness.layers.retry import Retry
from harness.middleware import MiddlewareStack


def context(bodies=(), observed=None, limit=8, calls=0):
    docs = [SimpleNamespace(doc_id=f"source-{i}", body=body) for i, body in enumerate(bodies)]
    return AgentContext(
        brief={"budget": {} if limit is None else {"max_tool_calls": limit}},
        tools=SimpleNamespace(calls=calls), trace=None,
        corpus=SimpleNamespace(docs=docs),
        observations=list(bodies) if observed is None else observed,
    )


def report(text, doc_id="source-0", **extra):
    return {"claims": [{"text": text, "doc_id": doc_id}], "answer": text,
            "abstain": False, "verdict": "Chấp thuận", **extra}


def test_reattribute_without_mutating_or_rewriting_model_quote():
    text = "Thời gian xử lý yêu cầu là 72 giờ."
    ctx = context(["Một nguồn khác.", text])
    original = report(text)
    result = CitationChecker().after_agent(ctx, original)
    assert result["claims"] == [{"text": text, "doc_id": "source-1"}]
    assert original["claims"][0]["doc_id"] == "source-0"
    assert Critic().after_agent(ctx, result)["verdict"] == "Chấp thuận"


def test_duplicate_sources_prefer_existing_citation_else_sorted_id():
    text = "Yêu cầu được xác nhận trong 24 giờ."
    ctx = context([text, text])
    checker = CitationChecker()
    assert checker.after_agent(ctx, report(text, "source-1"))["claims"][0]["doc_id"] == "source-1"
    assert checker.after_agent(ctx, report(text, "missing-source"))["claims"][0]["doc_id"] == "source-0"


@pytest.mark.parametrize("observed", [[], ["Thời gian xử lý"], ["Thời gian xử lý yêu cầu là 72 giờ."]])
def test_partial_or_unread_source_is_not_evidence(observed):
    body = "Thời gian xử lý yêu cầu là 72 giờ.\nLiên hệ nhóm vận hành."
    ctx = context([body], observed=observed)
    stack = MiddlewareStack([Critic(), CitationChecker()])
    result = stack.after_agent(ctx, report(body.splitlines()[0]))
    assert result["claims"] == [] and result["abstain"]
    assert "verdict" not in result


def test_cross_line_claim_dropped_even_when_observation_contains_it():
    text = "Dòng thứ nhất.\nDòng thứ hai."
    result = Critic().after_agent(context([text]), report(text))
    assert result["claims"] == []


def test_splice_preserves_literal_substrings_and_abstains():
    left, right = "Nhân viên làm việc từ xa hai ngày", "Nhân viên phải có mặt mỗi ngày"
    text = left + " và " + right
    result = Critic().after_agent(context([left, right]), report(text))
    assert result["claims"] == [{"text": left, "doc_id": "source-0"},
                                {"text": right, "doc_id": "source-1"}]
    assert all(c["text"] in text for c in result["claims"])
    assert result["abstain"] and "verdict" not in result


@pytest.mark.parametrize("body", ["Nhóm kiểm tra và phê duyệt yêu cầu.", "Hai nhóm phối hợp xử lý."])
def test_conjunction_alone_does_not_trigger_abstention(body):
    original = report(body)
    assert Critic().after_agent(context([body]), original)["claims"] == original["claims"]
    assert not Critic().after_agent(context([body]), original)["abstain"]


@pytest.mark.parametrize("claims", [None, {}, "bad", [None, {"text": 3}], [], [{"text": "", "doc_id": "source-0"}]])
def test_malformed_claims_abstain(claims):
    result = Critic().after_agent(context(), {"claims": claims, "verdict": "yes"})
    assert result["abstain"] and result["claims"] == [] and result["citations"] == []
    assert "verdict" not in result


def test_drop_fabrication_rebuilds_answer_but_keeps_verified_claim():
    body = "Hồ sơ được lưu trong ba năm."
    original = report(body)
    original["claims"].append({"text": "Tất cả hồ sơ được lưu mãi mãi.", "doc_id": "source-0"})
    original["answer"] += " Tất cả hồ sơ được lưu mãi mãi."
    result = Critic().after_agent(context([body]), original)
    assert len(result["claims"]) == 1
    assert "mãi mãi" not in result["answer"]
    assert len(original["claims"]) == 2


@pytest.mark.parametrize("content,expected", [
    ("ordinary", "ordinary"),
    (f"a{BLOCK_START} evil {BLOCK_END}b", f"a{PLACEHOLDER}b"),
    (f"a{BLOCK_START} cut", f"a{PLACEHOLDER}"),
    (f"{BLOCK_START}x{BLOCK_END}{BLOCK_START}y", PLACEHOLDER * 2),
])
def test_guard_filters_blocks_once_and_preserves_status(content, expected):
    calls = []
    original = ToolResult(False, content, "upstream error")
    result = InjectionGuard().wrap_tool_call(None, lambda *args: calls.append(args) or original, "fetch_doc", {})
    assert len(calls) == 1
    assert result == ToolResult(False, expected, "upstream error")
    assert original.content == content


def test_guard_removes_contaminated_claims_and_nested_extra_fields():
    original = report("Verified quotation.", metadata={"nested": [INJECTION_CANARY]})
    original["answer"] += INJECTION_CANARY
    original["claims"].append({"text": INJECTION_CANARY, "doc_id": "source-0"})
    original[INJECTION_CANARY] = "bad key"
    result = InjectionGuard().after_agent(None, original)
    assert INJECTION_CANARY not in json.dumps(result)
    assert result["claims"] == [original["claims"][0]]
    assert result["citations"] == ["source-0"]
    assert "metadata" not in result
    assert INJECTION_CANARY in original["answer"]


def test_guard_abstains_when_all_claims_are_contaminated():
    result = InjectionGuard().after_agent(None, report(INJECTION_CANARY))
    assert result["abstain"] and result["claims"] == [] and result["citations"] == []
    assert "verdict" not in result and INJECTION_CANARY not in json.dumps(result)


def test_guard_filtered_body_is_not_full_source_evidence():
    body = f"Verified quotation.\n{BLOCK_START} evil {BLOCK_END}"
    guard = InjectionGuard()
    filtered = guard.wrap_tool_call(None, lambda *a: ToolResult(True, body), "fetch_doc", {})
    ctx = context([body], observed=[filtered.content])
    result = MiddlewareStack([guard, Critic(), CitationChecker()]).after_agent(ctx, report("Verified quotation."))
    assert result["abstain"] and result["claims"] == []


@pytest.mark.parametrize("limit,calls,spent", [(None, 100, False), (0, 0, True), (1, 0, True), (8, 6, False), (8, 7, True)])
def test_budget_nudge_and_hard_gate(limit, calls, spent):
    ctx = context(limit=limit, calls=calls)
    layer = BudgetPolicy()
    messages = [{"role": "user", "content": "original question"}]
    outbound = layer.before_model(ctx, messages)
    assert messages == [{"role": "user", "content": "original question"}]
    assert (outbound == messages + [{"role": "user", "content": NUDGE}]) if spent else outbound is messages
    invoked = []
    result = layer.wrap_tool_call(ctx, lambda *a: invoked.append(a) or ToolResult(True, "ok"), "search", {})
    assert bool(invoked) is not spent
    assert result.ok is not spent


@pytest.mark.parametrize("sequence,limit,calls,expected", [
    ([ToolResult(False, "", "timeout"), ToolResult(True, "clean")], 8, 0, 2),
    ([ToolResult(True, "[TRUNCATED: content]"), ToolResult(True, "clean")], 8, 0, 2),
    ([ToolResult(False, "", "timeout")] * 4, 8, 0, 3),
    ([ToolResult(False, "", "timeout")] * 4, 8, 6, 1),
    ([ToolResult(True, "clean")], 1, 0, 0),
    ([ToolResult(True, "clean")], 0, 0, 0),
    ([ToolResult(False, "", "timeout")] * 4, None, 99, 3),
])
def test_retry_counts_real_calls_and_preserves_submit_reserve(sequence, limit, calls, expected):
    ctx = context(limit=limit, calls=calls)
    invoked = []
    args = {"doc_id": "independent-source"}

    def tool(name, received):
        assert name == "fetch_doc" and received == args
        ctx.tools.calls += 1
        result = sequence[len(invoked)]
        invoked.append(result)
        return result

    result = Retry().wrap_tool_call(ctx, tool, "fetch_doc", args)
    assert len(invoked) == expected
    assert ctx.state["retry_attempts"] == max(0, expected - 1)
    assert result == (sequence[expected - 1] if expected else ToolResult(False, "", "Hết ngân sách công cụ"))
    if limit is not None and limit >= 1:
        assert ctx.tools.calls + 1 <= limit


@pytest.mark.parametrize("reverse", [False, True])
def test_budget_and_retry_composition_never_consumes_submit_reserve(reverse):
    ctx = context(limit=8, calls=6)
    layers = [BudgetPolicy(), Retry()]
    if reverse:
        layers.reverse()
    invoked = []

    def failed_tool(name, args):
        invoked.append(name)
        ctx.tools.calls += 1
        return ToolResult(False, "", "timeout")

    call = MiddlewareStack(layers).wrap_tool_call(ctx, failed_tool)
    assert not call("fetch_doc", {}).ok
    assert len(invoked) == 1 and ctx.tools.calls == 7
    assert not call("fetch_doc", {}).ok
    assert len(invoked) == 1 and ctx.tools.calls + 1 == 8
