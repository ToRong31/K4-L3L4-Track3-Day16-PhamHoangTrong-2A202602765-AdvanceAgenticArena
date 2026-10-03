"""The six-hook middleware model — the whole harness in one file.

STUDENT-OWNED. Everything under `harness/` is yours: read it, change it,
replace it. Everything under `arena/` is frozen scaffold you may read but
never edit.

The six hook names below are CONTRACTUAL. They are the names on the Day
16 deck's §7 middleware diagram and they are the names the reference
solution, the instructor pack and every worked example use:

    before_agent      once, before the loop starts
    before_model      every turn, on the way OUT to the model
    wrap_model_call   every turn, AROUND the model call itself
    after_model       every turn, on the way BACK from the model
    wrap_tool_call    every tool call, AROUND the tool itself
    after_agent       once, after the loop ends and BEFORE `tools.submit`

WHERE EACH HOOK SITS
====================

One turn of `ReActAgent.run`, drawn as the onion it is::

    before_agent (once)
    ┌─ loop ────────────────────────────────────────────────┐
    │  messages = before_model(messages)                    │
    │  ┌ wrap_model_call ────────────────────────────────┐  │
    │  │      response = model.complete(messages)        │  │
    │  └─────────────────────────────────────────────────┘  │
    │  <the runner stamps `model_call` on the trace here>   │
    │  response = after_model(response)                     │
    │  if FINAL -> leave the loop                           │
    │  ┌ wrap_tool_call ─────────────────────────────────┐  │
    │  │      result = tools.<name>(**args)              │  │
    │  └─────────────────────────────────────────────────┘  │
    └───────────────────────────────────────────────────────┘
    report = after_agent(report)
    tools.submit(report)          # <- what after_agent returned

ORDER, EXACTLY
==============

With `middleware=[A, B, C]`:

* `before_agent` and `before_model` run **in list order** — A, B, C.
  Each `before_model` receives what the previous one returned.
* `wrap_model_call` and `wrap_tool_call` **nest, A outermost**: A's hook
  is called first and the `call` it is handed is "B wrapping C wrapping
  the real thing". A layer that never calls `call(...)` short-circuits
  every layer inside it — that is a feature (see `budget_policy`), and it
  is also the single easiest way to break a run by accident.
* `after_model` and `after_agent` run **in reverse list order** — C, B,
  A. That is what makes the picture an onion rather than a pipeline: you
  come back out through the layers in the order you went in.

Practical consequence for the reference stack: the layer whose
`after_agent` must run LAST goes FIRST in the list. `injection_guard`
does the final sweep of `report["answer"]`, so it leads; `citation_checker`
re-attributes claims before `critic` judges them, so it trails.

    [injection_guard, critic, citation_checker, budget_policy, retry]

That is the order to reason in, not a scoring trick: with the reference
implementations every permutation of the five measures the SAME 92.5184
(0 canary leaks), because none of them reads another's output. It starts
to matter the moment one of yours does — a `critic` that trusts `doc_id`,
for instance, needs `citation_checker` to have run first.

WHAT A HOOK MAY DO TO THE REPORT (this is worth points)
=======================================================

`after_agent` is where four of the five layers earn their score, and the
frozen scorer prices edits by KIND, not by intent:

* RE-ATTRIBUTE a claim (change its `doc_id`)      -> `citation_checker`
* DELETE a claim, or set `abstain`                -> `critic`
* REWRITE `report["answer"]`                      -> `injection_guard`
* TRIM a claim's text (a substring is still a quotation)

**Any non-substring edit of claim TEXT loses the claim.** A claim is
credited only if it is text the MODEL produced and a verbatim quotation
of one LINE of the document it cites. Adding a full stop, straightening a
quote mark, normalising spacing, or repairing a truncated sentence from
the corpus all break both tests at once. Measured on the full reference
stack (92.52 on the trap-spanning set): trimming to 120 characters costs
8.11 points (that loss is recall, and it is legal); adding a trailing
period costs 47.16 and lands the run on the abstain-everything floor.
Editing `answer` is free; editing a quotation is not.

RAISING
=======

Hooks are not sandboxed. If your layer raises, the run dies and the
entry scores zero — there is no `try/except` around your code, on purpose:
a layer that silently swallowed its own bugs would be worse than one that
fails loudly during practice.
"""

from __future__ import annotations


class Middleware:
    """Base class for a harness layer. Every hook is a no-op by default.

    Subclass it and override only the hooks you need — an unimplemented
    hook costs nothing and changes nothing. `name` is used in trace
    events and error messages; it defaults to the class name.

    Every hook receives `ctx`, the shared `AgentContext` (see
    `harness/agent.py`) carrying the brief, the tools, the trace, the
    corpus and the observations seen so far.
    """

    #: Human-readable label for this layer. Override in a subclass, or
    #: leave it None to use the class name.
    name: str | None = None

    @property
    def label(self) -> str:
        return self.name or type(self).__name__

    # -- once per run --------------------------------------------------

    def before_agent(self, ctx) -> None:
        """Called once before the loop starts. Return value ignored.

        Use it to set up state on `ctx.state` (a plain dict, yours to
        use), read `ctx.brief["budget"]`, or record a starting point.
        """

    def after_agent(self, ctx, report: dict) -> dict:
        """Called once after the loop ends, BEFORE `tools.submit`.

        Whatever you return here is what gets submitted and scored, so
        this is the last place a layer can fix — or ruin — the report.
        Must return a dict.
        """
        return report

    # -- once per model turn -------------------------------------------

    def before_model(self, ctx, messages: list[dict]) -> list[dict]:
        """Transform the message list on its way to the model.

        Return the list to send. The agent keeps its own canonical
        history separately, so returning `messages + [something]` adds a
        one-turn nudge rather than a permanent message.

        TWO MEASURED CAUTIONS, both silent:

        * `MockModel` will only quote a document whose sentence is
          VERBATIM in the message list it is handed right now. A
          "helpful" context-compression layer that summarises or drops
          old observations therefore removes the model's ability to cite
          what it read — measured 47.16 points off the full stack (92.52
          -> 45.36), with no error anywhere.
        * A message you append here can be MISTAKEN FOR THE BRIEF.
          `arena.model._first_user_content` reads the last user message
          before the first assistant turn as the question and skips only
          candidates carrying `FINALIZE_SENTINEL`. A bare "answer now"
          nudge becomes the search query for the whole run.
        """
        return messages

    def wrap_model_call(self, ctx, call, messages: list[dict]):
        """Wrap the model call. `call(messages)` returns a ModelResponse.

        Call `call(...)` exactly once in the normal case; call it more
        than once to implement a model-level retry; skip it entirely to
        short-circuit the turn (you must then return a ModelResponse of
        your own).
        """
        return call(messages)

    def after_model(self, ctx, response):
        """Inspect or replace the ModelResponse on its way back.

        The trace has already recorded the RAW response by the time this
        runs — the model's own output is stamped before any hook can see
        it, so rewriting `response.text` here changes what your agent
        does, never what the scorer believes the model said.
        """
        return response

    # -- once per tool call --------------------------------------------

    def wrap_tool_call(self, ctx, call, name: str, args: dict):
        """Wrap one tool call. `call(name, args)` returns a ToolResult.

        This is the boundary where untrusted document text enters your
        agent, and the boundary where a failed call can be retried before
        the model ever sees it. Both `retry` (§7) and `injection_guard`
        (§10) live here in the reference solution.
        """
        return call(name, args)


class LoggingMiddleware(Middleware):
    """A fully worked example: implements all six hooks, changes nothing.

    It writes one `layer` trace event per hook fired, which is the
    cheapest possible observability layer and a useful debugging tool
    during practice — run with it installed and `to_jsonl()` shows you
    exactly which of your layers ran, in what order, on which turn.

    Two rules it demonstrates, both of which bite in real layers:

    1. **Never pass a mutable to `Trace.emit`.** `emit` stores a
       REFERENCE to what you hand it. Log a list and then append to that
       same list, and the trace retroactively changes — measured turning
       a 1-element list into a 112,061-character line that failed the
       conformance gate. Everything below is an `int` or a `str`.
    2. **Never pass `seq`, `run_id`, `seed` or `event` as a field.** They
       are reserved; `emit` raises rather than let a layer clobber the
       monotonic sequence the gate checks.

    `layer` events are free: the gate accepts them and the scorer ignores
    them. They cost nothing but trace bytes.
    """

    name = "logging"

    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose
        self.events: list[str] = []

    def _log(self, ctx, hook: str, **fields) -> None:
        self.events.append(hook)
        if not self.verbose:
            return
        # Only immutable scalars past this point — see rule 1 above.
        ctx.trace.emit("layer", layer=self.label, hook=hook, step=ctx.step, **fields)

    def before_agent(self, ctx) -> None:
        self.events.clear()
        self._log(ctx, "before_agent", brief_id=str(ctx.brief.get("brief_id", "")))

    def before_model(self, ctx, messages):
        self._log(ctx, "before_model", n_messages=len(messages))
        return messages

    def wrap_model_call(self, ctx, call, messages):
        self._log(ctx, "wrap_model_call", n_messages=len(messages))
        response = call(messages)
        self._log(ctx, "wrap_model_call_done", completion_tokens=response.completion_tokens)
        return response

    def after_model(self, ctx, response):
        self._log(ctx, "after_model", n_chars=len(response.text))
        return response

    def wrap_tool_call(self, ctx, call, name, args):
        self._log(ctx, "wrap_tool_call", tool=str(name))
        result = call(name, args)
        self._log(ctx, "wrap_tool_call_done", tool=str(name), ok=bool(result.ok))
        return result

    def after_agent(self, ctx, report):
        claims = report.get("claims")
        self._log(
            ctx,
            "after_agent",
            n_claims=len(claims) if isinstance(claims, list) else 0,
            abstain=bool(report.get("abstain")),
        )
        return report


# ---------------------------------------------------------------------------
# Composition. `ReActAgent` drives the hooks through this one object, so
# the ordering rules documented above live in exactly one place.
# ---------------------------------------------------------------------------


class MiddlewareStack:
    """An ordered list of `Middleware`, composed per the ordering rules.

    Not something you normally construct yourself — `ReActAgent` builds
    one from the `middleware=[...]` you pass it. It is public because
    testing your own layer against it directly is often easier than
    running a whole session.
    """

    def __init__(self, middleware=None) -> None:
        self.middleware: list[Middleware] = list(middleware or [])

    def __len__(self) -> int:
        return len(self.middleware)

    def __iter__(self):
        return iter(self.middleware)

    # -- straight passes -----------------------------------------------

    def before_agent(self, ctx) -> None:
        for layer in self.middleware:
            layer.before_agent(ctx)

    def before_model(self, ctx, messages: list[dict]) -> list[dict]:
        for layer in self.middleware:
            messages = layer.before_model(ctx, messages)
            if not isinstance(messages, list):
                raise TypeError(f"{layer.label}.before_model must return a list")
        return messages

    def after_model(self, ctx, response):
        for layer in reversed(self.middleware):
            response = layer.after_model(ctx, response)
            _validate_model_response(layer, "after_model", response)
        return response

    def after_agent(self, ctx, report: dict) -> dict:
        for layer in reversed(self.middleware):
            report = layer.after_agent(ctx, report)
            if not isinstance(report, dict):
                raise TypeError(
                    f"{layer.label}.after_agent must return a dict, got "
                    f"{type(report).__name__}"
                )
        return report

    # -- nested wraps --------------------------------------------------

    def bind_calls(self, ctx, model_call, tool_call):
        """Bind both chains to this run; topology stays fixed until it ends."""
        return self.wrap_model_call(ctx, model_call), self.wrap_tool_call(ctx, tool_call)

    def wrap_model_call(self, ctx, call):
        """Return `call` wrapped by every layer, list-order outermost."""
        for layer in reversed(self.middleware):
            call = _bind_model_wrap(layer, ctx, call)
        return call

    def wrap_tool_call(self, ctx, call):
        """Return `call` wrapped by every layer, list-order outermost."""
        for layer in reversed(self.middleware):
            call = _bind_tool_wrap(layer, ctx, call)
        return call


def _bind_model_wrap(layer: Middleware, ctx, inner):
    # A factory, not a lambda in the loop: closing over the loop variable
    # directly is the classic late-binding bug and every layer would end
    # up calling the last one.
    def wrapped(messages):
        response = layer.wrap_model_call(ctx, inner, messages)
        _validate_model_response(layer, "wrap_model_call", response)
        return response

    return wrapped


def _bind_tool_wrap(layer: Middleware, ctx, inner):
    def wrapped(name, args):
        result = layer.wrap_tool_call(ctx, inner, name, args)
        _validate_tool_result(layer, result)
        return result

    return wrapped


def _validate_model_response(layer, hook, response):
    if not (
        isinstance(getattr(response, "text", None), str)
        and hasattr(response, "prompt_tokens")
        and hasattr(response, "completion_tokens")
    ):
        raise TypeError(
            f"{layer.label}.{hook} must return a ModelResponse with str text "
            "and prompt_tokens/completion_tokens"
        )


def _validate_tool_result(layer, result):
    if not (
        isinstance(getattr(result, "ok", None), bool)
        and isinstance(getattr(result, "content", None), str)
        and hasattr(result, "error")
        and (result.error is None or isinstance(result.error, str))
    ):
        raise TypeError(
            f"{layer.label}.wrap_tool_call must return a ToolResult "
            "with bool ok, str content and str/None error"
        )
