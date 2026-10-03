"""LỚP `budget_policy` — bài giảng Day 16, §3 (Budgets & Control Flow).

NHIỆM VỤ: kế hoạch của mô hình dài đúng 11 lượt gọi công cụ, bất kể brief
cho ngân sách bao nhiêu — và BỐN lượt cuối là rác có chủ ý: một lần search
lặp lại, một phép tính vô nghĩa, hai lần fetch lại tài liệu đã có trong
tay. Phần việc hữu ích nằm ở ĐẦU kế hoạch, nên cắt phần đuôi không mất một
điểm grounding nào mà lấy trọn phần điểm efficiency về tool call và token.

TÍN HIỆU:

    ctx.tools.calls >= ctx.max_tool_calls - reserve

CÁCH DỪNG: thêm `FINALIZE_SENTINEL` vào bên trong MỘT CÂU tiếng Việt bình
thường và đẩy vào cuối danh sách message trong `before_model`. `MockModel`
khoá theo token; một mô hình thật thì nghe câu tiếng Việt bao quanh nó.
Viết như vậy để cùng một lớp chạy được trên cả hai đường.

SENTINEL KHÔNG PHẢI TUỲ CHỌN — và không chỉ vì chuyện dừng.
`arena.model._first_user_content` lấy message user CUỐI CÙNG trước lượt
assistant đầu tiên làm câu hỏi của brief, và nó bỏ qua đúng những message
có mang `FINALIZE_SENTINEL`. Nếu bạn chèn một câu nhắc trơn không có
sentinel, mô hình sẽ đi search CHÍNH CÂU NHẮC ĐÓ: mọi brief truy xuất
cùng một mớ tài liệu vô can, mọi bậc thang điểm dịch chuyển đúng 0.00, và
không có một dòng lỗi nào báo cho bạn biết.

TRẢ VỀ `messages + [...]`, ĐỪNG `messages.append(...)`. Agent áp dụng
`before_model` lên một BẢN SAO của lịch sử, nên trả về danh sách mới nghĩa
là "nhắc trong đúng lượt này"; append vào chính danh sách được truyền vào
thì lời nhắc dính vĩnh viễn.

`reserve` KHÔNG PHẢI TRANG TRÍ: `Tools.calls` ĐẾM CẢ `submit`, và scorer
cũng đếm như vậy. Brief cho `max_tool_calls: 8` nghĩa là bảy lượt hữu ích
cộng một lượt submit. Dừng ở `calls >= 8` là tiêu lố đúng một lượt, lần
nào cũng lố.

MỘT HOOK LÀ CHƯA ĐỦ — ĐÃ ĐO. `before_model` chỉ chặn được khi mỗi lượt
model tiêu đúng MỘT lượt công cụ. Không phải vậy: lớp `retry` (§7) có thể
tiêu ba lượt trong CÙNG một vòng, nên một vòng bắt đầu khi còn thiếu đúng
một lượt vẫn kết thúc ở trên ngưỡng. Đo trên full stack: 34/120 lượt chạy
kết thúc ở 9+ lượt gọi trong khi brief cho 8, efficiency 12.06 thay vì
14.24 — trong khi `budget_policy` chạy MỘT MÌNH thì sạch cả 120 lượt.
Vì thế lớp này có thêm `wrap_tool_call`: khi ngân sách chỉ còn phần dự
trữ, TỪ CHỐI gọi công cụ (trả về `ToolResult(ok=False, ...)`, đừng raise —
agent phải sống để còn chốt FINAL). Nửa còn lại nằm ở `retry`: hook
`wrap_tool_call` của `budget_policy` bọc NGOÀI vòng lặp thử lại nên không
nhìn thấy các lượt gọi lại; chỉ chính `retry` mới chặn được `retry`.

CẢNH BÁO ĐÃ ĐO ĐƯỢC — ĐỪNG NÉN NGỮ CẢNH Ở ĐÂY. `before_model` trông rất
hợp lý để "tóm tắt cho gọn", nhưng `MockModel` chỉ trích được câu nào
xuất hiện NGUYÊN VĂN trong danh sách message NÓ ĐANG NHẬN. Một lớp nén
ngữ cảnh tử tế làm mô hình mất khả năng trích dẫn chính những tài liệu nó
vừa đọc: -47.16 điểm trên full stack (92.52 -> 45.36), không có một
thông báo lỗi nào.

CÔNG CỤ CÓ SẴN:
    from arena.model import FINALIZE_SENTINEL
    from arena.tools import ToolResult
    ctx.tools.calls      -> số lượt gọi công cụ đã dùng (kể cả submit)
    ctx.max_tool_calls   -> ngân sách của brief, hoặc None nếu brief không đặt

Cài đặt:  ReActAgent(..., middleware=[..., BudgetPolicy(), ...])
Xem `harness/middleware.py` để biết thứ tự các hook.
"""

from __future__ import annotations

from arena.model import FINALIZE_SENTINEL
from arena.tools import ToolResult

from harness.middleware import Middleware

#: Dành lại cho lượt `submit` mà agent vẫn còn phải gọi.
DEFAULT_RESERVE = 1

NUDGE = (
    "Ngân sách công cụ đã hết. Hãy trả lời ngay bằng bằng chứng đang có, "
    f"không gọi thêm công cụ nào nữa. {FINALIZE_SENTINEL}"
)


def token_estimate(ctx, messages=None):
    """Soft forecast from actual prior usage and context growth, not a hard cap."""
    messages = ctx.messages if messages is None else messages
    previous = ctx.state.get("last_prompt_tokens", 0)
    previous_chars = ctx.state.get("last_prompt_chars", 0)
    chars = sum(len(str(message.get("content", ""))) for message in messages)
    ratio = previous / previous_chars if previous_chars else 0.5
    prompt = previous + int(max(0, chars - previous_chars) * ratio + 0.999)
    if not previous:
        prompt = int(chars * ratio + 0.999)
    completion = max(512, min(1500, ctx.state.get("last_completion_tokens", 0)))
    return prompt + completion


def can_repair_final(ctx):
    limit = ctx.budget.get("max_tokens")
    if isinstance(limit, bool) or not isinstance(limit, (int, float)):
        return True
    # Reserve a little room for the repair feedback itself.
    return ctx.state.get("model_tokens", 0) + token_estimate(ctx) + 256 <= limit


class BudgetPolicy(Middleware):
    """Ép mô hình chốt FINAL ngay khi ngân sách công cụ đã tiêu hết."""

    name = "budget_policy"

    def __init__(self, reserve: int = DEFAULT_RESERVE, token_headroom: float = 0.1) -> None:
        self.reserve = max(0, int(reserve))
        # Usage forecasting is soft. A small tolerance permits a useful last
        # fetch when conservative completion reserves would otherwise stop it.
        self.token_headroom = max(0.0, min(0.2, float(token_headroom)))

    def _spent(self, ctx) -> bool:
        limit = ctx.max_tool_calls
        return limit is not None and ctx.tools.calls >= limit - self.reserve

    def before_model(self, ctx, messages):
        if self._spent(ctx):
            return messages + [{"role": "user", "content": NUDGE}]
        if ctx.state.get("real_model_features"):
            if "priority_fetch_step" in ctx.state and ctx.step > ctx.state["priority_fetch_step"]:
                return messages + [{"role": "user", "content": NUDGE}]
            limit = ctx.budget.get("max_tokens")
            estimate = token_estimate(ctx, messages)
            ctx.state["next_token_estimate"] = estimate
            if (isinstance(limit, (int, float)) and not isinstance(limit, bool)
                    and ctx.state.get("model_tokens", 0) + 2 * estimate >= limit * (1 + self.token_headroom)):
                if (ctx.state.get("pending_policy_sources") and "priority_fetch_step" not in ctx.state
                        and ctx.state.get("model_tokens", 0) + estimate <= limit * (1 + self.token_headroom)):
                    ctx.state["priority_fetch_step"] = ctx.step
                    if ctx.trace is not None:
                        ctx.trace.emit("layer", layer=self.name, hook="priority_fetch",
                                       candidates=len(ctx.state["pending_policy_sources"]),
                                       spent=ctx.state.get("model_tokens", 0), estimated_next=estimate)
                    return messages + [{"role": "system", "content":
                        "Chỉ còn một lượt đọc nguồn trước FINAL. Kết quả search đã có ứng viên "
                        "Văn bản chính thức; chọn đúng chủ đề và fetch_doc NGAY, không search thêm. "
                        "Sau fetch phải chốt FINAL bằng quote nguyên dòng vừa đọc. "
                        "Nếu không có ứng viên phù hợp thì abstain, không đoán."}]
                if ctx.trace is not None:
                    ctx.trace.emit("layer", layer=self.name, hook="token_nudge",
                                   spent=ctx.state.get("model_tokens", 0), estimated_next=estimate)
                return messages + [{"role": "user", "content":
                    "Ngân sách token gần hết; chốt FINAL ngay bằng bằng chứng hiện có, "
                    f"abstain nếu thiếu căn cứ, không gọi thêm công cụ. {FINALIZE_SENTINEL}"}]
            if ctx.state.get("searches_since_fetch", 0) >= 2:
                return messages + [{"role": "system", "content":
                    "Bạn đã search liên tiếp mà chưa đọc nguồn. Hãy đối chiếu TIÊU ĐỀ với "
                    "chủ đề nghiệp vụ và loại văn bản được hỏi, đọc ứng viên phù hợp; "
                    "nếu chưa có, đổi sang tên quy trình/chính sách nội bộ và mở rộng k=8–10. "
                    "Không tiếp tục lặp mô tả sự việc bằng từ đồng nghĩa."}]
        return messages

    def wrap_tool_call(self, ctx, call, name, args):
        if self._spent(ctx):
            return ToolResult(ok=False, content="", error="Hết ngân sách công cụ")
        return call(name, args)
