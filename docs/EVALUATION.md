> Audit sau v1: xem `ROOT_CAUSE_AND_IMPROVEMENT_PLAN.md` cho lỗi critic overlap, root cause Windows và phương pháp đánh giá corpus khác đã được hiệu chỉnh.

# Kết quả triển khai và kiểm chứng v1

Đã triển khai theo `TECHNICAL_PLAN.md`: single ReAct agent, 5 middleware xác định, bind wrapper một lần mỗi phiên, validation tại ranh giới hook và reset log theo phiên. Không thay prompt, parser, MAX_STEPS, dependencies runtime, `arena/`, `data/` hoặc test sẵn có. Chưa commit/push.

## Hành vi đã triển khai

- Citation checker sửa doc_id bằng nguồn đã quan sát đầy đủ, ưu tiên nguồn hiện tại rồi doc_id đã sort; giữ nguyên text.
- Critic loại claim không xác minh được, thử tách literal ` và ` thành hai substring có nguồn khác nhau; abstain và bỏ verdict khi thiếu bằng chứng hoặc có splice.
- Injection guard thay block độc, kể cả block thiếu dấu đóng; quét toàn report cuối, bỏ claim/trường phụ nhiễm canary, không sửa chữ quotation.
- Budget policy nhắc FINAL bằng sentinel và chặn tool khi chỉ còn reserve submit.
- Retry tối đa 3 attempts, nhận biết lỗi lẫn degraded content, kiểm budget trước mọi attempt, trả lỗi thật và đếm retries trong ctx.state.
- MiddlewareStack giữ 6 hook, thứ tự onion và short-circuit. Bind theo run đọc ctx hiện tại; không cache xuyên phiên, không thêm trace model/tool giả.

## Kiểm tra

**Ubuntu/WSL, Python 3.12, bản sao LF: 804 passed; verify 21/21 đạt.** Bản sao kiểm chứng đặt ngoài workspace, lấy scaffold/test từ Git và overlay harness/test mới hiện tại; chỉ chuẩn hoá xuống dòng trong bản sao. Pytest và thư viện test được đưa vào môi trường Python riêng; không thêm dependencies cho harness.

**47 test mới** kiểm chứng bằng corpus nhỏ tự tạo: nguồn chưa đọc/snippet-only, quotation vắt dòng, citation trùng nguồn, bịa xen claim thật, splice và liên từ hợp lệ, malformed report, nhiều block độc/block cụt, canary trong nested field/key, budget None/0/1/8, degraded response, retry tiêu reserve, thứ tự budget/retry, adapter hợp lệ, lỗi hook, bind lifetime và short-circuit.

Test hồi quy so report và trace giữa bind-once với ghép wrapper mỗi lần: giống hoàn toàn. Logging thêm event nhưng không đổi các model/tool/submit event còn lại sau khi bỏ event layer và đánh lại seq. Kiểm tra diff xác nhận scaffold/data/test cũ không thay đổi.

Suite Windows tại workspace vẫn có nhóm lỗi môi trường giống baseline: hash raw CRLF, dấu phân cách đường dẫn, Unicode subprocess và giới hạn độ dài environment variable. Lần chạy với 42 test mới: 785 passed, 9 failed, 8 errors, 1 skipped. Sau 5 test bổ sung, cả 47 test mới qua riêng trên Windows; suite đầy đủ cuối cùng được xác nhận trên Ubuntu/LF. Không sửa hash expectation hoặc test để che lỗi.

## Điểm luyện tập và leave-one-out

MockModel, public 9 briefs, corpus seed 42, cùng 5 base seed **11/23/37/51/71**, mỗi cấu hình 45 lượt. Điểm chính thức với model thật/private briefs chưa được đo.

| Cấu hình | Mean /100 | Giảm so full | Gate | Max tool calls gồm submit |
|---|---:|---:|---:|---:|
| Baseline | 25.1997 | 56.5119 | 45/45 | 13 |
| Full stack | 81.7116 | — | 45/45 | 8 |
| Bỏ injection guard | 72.6430 | 9.0686 | 45/45 | 8 |
| Bỏ critic | 69.7712 | 11.9404 | 45/45 | 8 |
| Bỏ citation checker | 41.2051 | 40.5065 | 45/45 | 8 |
| Bỏ budget policy | 78.9247 | 2.7869 | 45/45 | 8 |
| Bỏ retry | 76.8146 | 4.8970 | 45/45 | 8 |

Full stack: **0 canary leak, 0 lượt có claim HALLUCINATED** trong 45 lượt. Lượt mặc định seed 11: baseline **24.2747**, full **81.7116**. Mọi layer đều có đóng góp đo được theo leave-one-out; đây không phải đóng góp cộng tuyến tính.

Độ lệch chuẩn giữa các brief chứa cả khác biệt độ khó. Để kiểm ảnh hưởng retry theo seed, tính population std trên 5 seed cho từng brief rồi lấy trung bình: full **0.0000**, bỏ retry **7.7318**. Kết quả chỉ áp dụng cho 5 seed đã đo, không chứng minh phương sai bằng 0 trên mọi seed.

Dữ liệu tổng hợp máy đọc: `evaluation_summary.json`. Chi tiết điểm cũng đã được kiểm qua `scripts/selfeval.py`.

## Đo tối ưu middleware

Workload giả gồm 5 middleware pass-through, mỗi vòng gọi model và tool một lần, 20.000 vòng/trial, warm-up trước và 5 trials; cả hai cấu hình đều dùng cùng validation:

| Cấu hình | Median ns/cặp gọi | Min–max |
|---|---:|---:|
| Ghép chain mỗi lần | 5326.2 | 4778.1–7487.1 |
| Bind một lần | 2947.7 | 2500.2–3444.6 |

Đo end-to-end cùng mock brief và 5 pass-through layers, gồm tạo agent/run, 10 runs/trial × 5 trials: rebuild median **10.308 ms** (8.891–17.347), bind median **9.832 ms** (8.231–11.669). Khoảng đo chồng lấp; chưa đủ để tuyên bố cải thiện end-to-end đáng kể. Lợi ích điểm số đến từ 5 layer, không từ vài microsecond wrapper.

## Giới hạn còn lại

- Chính sách full-body bảo thủ đã chọn: snippet, bản cụt hoặc body bị guard lọc không đủ chứng minh nguồn đầy đủ; có thể mất claim đúng. Test xác nhận hành vi này, chưa mở rộng evidence tracking.
- Splice là heuristic hai substring/hai nguồn; không phải bộ phân tích mâu thuẫn ngữ nghĩa hay Reflexion feedback loop. Audit sau v1 tìm được lỗi separator overlap ` và và `: critic bỏ sót ranh giới hợp lệ, khiến pub-04 mất cả hai claim dù tổng điểm không giảm. Chưa sửa code; hướng sửa trong `ROOT_CAUSE_AND_IMPROVEMENT_PLAN.md`.
- Pub-08 và pub-09 ở seed mặc định đều **40.1471**: thiếu dữ kiện cần thiết; pub-09 còn thiếu verdict do model viết với bằng chứng hỗ trợ. Đối chứng sau v1 xác nhận nguồn cần đọc có rank 56/83 và mock chỉ fetch 5 nguồn đầu. Riêng verdict: formatter MockModel không tạo trường này, không phải chỉ thiếu hướng dẫn prompt. Middleware không điền required_facts hoặc tự chọn đáp án.
- Smoke test corpus seed 73/base seed 37: **61.7606**, 9/9 gate, tối đa 8 calls, không canary/hallucination. Public briefs vẫn cố định, nên phép đổi corpus này chỉ kiểm khả năng chạy và an toàn, không đại diện đánh giá private briefs. Không dùng mức điểm giảm để kết luận retrieval kém hơn: public expected facts thuộc seed 42, một số fact không còn khớp corpus 73. Đối chứng sau v1 xác nhận pub-02/03/07 đã đọc đúng ID nhưng expected quote tại ID đó không tồn tại trong corpus mới.

## Chạy lại

```powershell
$env:PYTHONUTF8='1'
python -m pytest tests/test_layers_behavior.py tests/test_middleware_behavior.py -q
python scripts/run_practice.py --layers all --seed 11 --entry me --out runs/me.json
python scripts/selfeval.py --run runs/me.json --summary
```

Trên Linux/LF: `python -m pytest -q` và `python scripts/verify.py`. Với Windows, đối chiếu lỗi nền ở trên; không thay luật chấm hoặc module đóng băng.
