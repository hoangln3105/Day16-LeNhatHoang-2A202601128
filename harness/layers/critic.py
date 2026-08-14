"""LỚP `critic` — bài giảng Day 16, §2 (Reflection & Self-Critique).

NHIỆM VỤ: mô hình KHÔNG BAO GIỜ nói "tôi không biết". `abstain` bị gán
cứng `False`, và nó bịa theo ba kiểu khác nhau:

  (a) brief `absent`  -> bịa ra một con số không có trong tài liệu nào.
  (b) không có bằng chứng -> bịa ra một câu chung chung vô thưởng vô phạt.
  (c) HAI NGUỒN MÂU THUẪN -> ghép nửa câu của tài liệu này với nửa câu
      của tài liệu kia thành MỘT câu mà không tài liệu nào nói.

TÍN HIỆU (chỉ một dòng): câu trong `claim["text"]` có xuất hiện NGUYÊN VĂN
trong bằng chứng agent đã thực sự đọc hay không —

    text in ctx.observed_text

Trên một brief có bằng chứng tốt thì mọi claim đều thoả điều kiện này,
nên critic xây trên tín hiệu đó không báo động giả.

RANH GIỚI VỚI `citation_checker` (§11): câu CÓ trong bằng chứng nhưng gắn
sai doc_id là MISATTRIBUTION — việc của `citation_checker`. Câu KHÔNG có
trong bất kỳ bằng chứng nào là FABRICATION — việc của bạn ở đây. Hai điều
kiện loại trừ nhau, đừng làm phần việc của lớp kia.

ĐIỂM SỐ (đọc kỹ, đây là nơi kiếm nhiều điểm nhất):
  * Một claim bịa bị chấm `HALLUCINATED`: mất điểm precision VÀ mất trọn
    15 điểm honesty, trên MỌI brief.
  * Trên brief `is_absent`, `abstain: true` được 0.75 recall + trọn 15
    điểm honesty. "Không có số liệu" CHÍNH LÀ câu trả lời đúng.
  * Trên brief mâu thuẫn, ĐỪNG trông đợi "nêu cả hai phía" tự động cho
    recall đầy đủ: recall chấm THEO TỪNG required_fact bằng key terms
    của chính fact đó, không phải theo số vế đã trích dẫn — nếu nửa câu
    mô hình thực sự viết ra không phủ hết từ khoá của một fact (mô hình
    ghép câu ở chỗ NÓ chọn, không nhất thiết đúng ranh giới required_fact),
    fact đó vẫn 0 điểm dù trích dẫn đúng. Trên `pub-04-lam-viec-tu-xa` cụ
    thể, trần recall là 0.5 với MỌI harness đúng luật, vì đúng lý do đó —
    đo được, không phải suy đoán. Vẫn nên làm: `abstain: true` sau khi nêu
    cả hai phía được 0.5 recall + trọn 15 điểm honesty, và điểm recall lấy
    theo `max(...)` nên làm cả hai không bao giờ THIỆT — chỉ đừng trông
    đợi nó vượt sàn 0.5 trên brief này.
  * Xoá claim là hợp lệ. SỬA CHỮ trong `claim["text"]` thì KHÔNG: thêm
    một dấu chấm cuối câu cũng đủ làm claim mất cả provenance lẫn hỗ trợ
    (đo được: -40 điểm). Chỉ được xoá, giữ nguyên, hoặc cắt bớt.

GỢI Ý cho trường hợp (c): câu bị ghép là hai đoạn DO CHÍNH MÔ HÌNH viết,
dán với nhau bằng một liên từ (" và "). Cắt đúng chỗ dán thì hai nửa vẫn
là chữ của mô hình — vẫn qua được kiểm tra provenance. Muốn biết cắt đúng
chưa: cả hai nửa phải xuất hiện nguyên văn trong `ctx.observed_text` và
phải thuộc HAI tài liệu khác nhau. Cắt sai thì một nửa sẽ vắt qua hai tài
liệu và không quan sát nào chứa nó.

CÔNG CỤ CÓ SẴN:
    ctx.observed_text  -> toàn bộ quan sát agent đã thấy, nối lại
    ctx.saw(text)      -> text có trong quan sát không
    ctx.corpus.docs    -> danh sách Doc (doc_id, title, body); qua
                          `ctx.corpus`, `Doc.tags` LUÔN RỖNG — CẢ Ở VÒNG
                          LUYỆN TẬP LẪN VÒNG CHẤM ĐIỂM, vì corpus mà code
                          của bạn cầm bị gỡ nhãn bẫy ('outdated',
                          'contradiction', 'injection'…) ngay khi runner
                          dựng lên nó, không phải chỉ lúc chấm điểm. Đọc
                          nhãn là tra bảng chứ không phải kỹ năng lab này
                          chấm. Ở vòng LUYỆN TẬP seed 42 thì file TRÊN ĐĨA
                          `data/corpus/*.json` (khác với `ctx.corpus`)
                          vẫn có nhãn: hard-code được từ đó, và điều đó
                          được nói thẳng ra ở đây thay vì giấu đi.
    ctx.state          -> dict tuỳ bạn dùng để ghi số liệu gỡ lỗi

Cài đặt:  ReActAgent(..., middleware=[InjectionGuard(), Critic(), ...])
Xem `harness/middleware.py` để biết thứ tự các hook.
"""

from __future__ import annotations

from harness.middleware import Middleware

from harness.layers._quoting import norm, source_doc

#: Chỗ mô hình dán hai nửa câu của hai tài liệu khác nhau — trường hợp (c).
CONJUNCTION = " và "

#: Câu thay thế khi không còn claim nào đứng vững. Cố tình KHÔNG dùng cụm
#: "không đủ căn cứ": trên brief `is_synthesis` cụm đó thường là một phương
#: án verdict được khai báo sẵn, và tự tay khẳng định thêm một phương án là
#: cách biến một lượt chạy im lặng thành một lượt chạy đoán bừa.
NO_EVIDENCE_ANSWER = (
    "Chưa thu thập được bằng chứng đỡ cho một kết luận nào, nên tôi không "
    "đưa ra khẳng định về câu hỏi này."
)


class Critic(Middleware):
    """Xoá những gì bằng chứng không đỡ; abstain khi không còn gì."""

    name = "critic"

    def after_agent(self, ctx, report):
        claims = report.get("claims")
        if not isinstance(claims, list) or not claims:
            return report

        kept: list = []
        dropped = 0
        for claim in claims:
            if not isinstance(claim, dict):
                dropped += 1  # MALFORMED: phạt trọn 1.0, xoá rẻ hơn giữ
                continue
            text = claim.get("text")
            if not isinstance(text, str) or not text.strip():
                dropped += 1
                continue

            if _seen(ctx, text):
                kept.append(claim)  # chữ thật — KHÔNG đụng vào text
                continue

            halves = _split_fused(ctx, claim, text)
            if halves is not None:
                # Hai nguồn mâu thuẫn. Nêu cả hai phía VÀ không chọn bên
                # nào: trên brief `is_contradiction` cách này giữ trọn 15
                # điểm honesty và recall lấy theo max(...) nên không bao
                # giờ thiệt so với việc im lặng.
                kept.extend(halves)
                report["abstain"] = True
                continue

            dropped += 1  # không quan sát nào chứa câu này -> bịa

        ctx.state["critic_dropped"] = dropped
        report["claims"] = kept
        if not kept:
            report["abstain"] = True
            report["citations"] = []
            report["answer"] = NO_EVIDENCE_ANSWER
            return report

        report["citations"] = sorted(
            {
                claim["doc_id"]
                for claim in kept
                if isinstance(claim, dict)
                and isinstance(claim.get("doc_id"), str)
                and claim["doc_id"]
            }
        )
        return report


def _seen(ctx, text: str) -> bool:
    """Câu này có nằm trong bằng chứng agent đã thực sự đọc không?

    `ctx.saw` so khớp nguyên xi, đúng như docstring mô tả và đủ cho mock.
    Bản chuẩn hoá đứng sau là để dành cho mô hình thật: nó có thể đổi hoa
    thường hay gộp khoảng trắng khi chép lại, và scorer THA cho đúng hai
    thứ đó. Xoá nhầm một claim mà scorer vốn định cho điểm cũng đắt như
    giữ lại một claim bịa.
    """
    if ctx.saw(text):
        return True
    normalised = norm(text)
    return bool(normalised) and normalised in norm(ctx.observed_text)


def _split_fused(ctx, claim: dict, text: str):
    """Tách câu ghép của trường hợp (c), hoặc `None` nếu không tách được.

    Cắt tại chỗ dán (" và ") và chỉ chấp nhận khi CẢ HAI nửa đều là trích
    dẫn nguyên văn một dòng của HAI tài liệu KHÁC NHAU đã đọc sạch. Hai
    nửa vẫn là substring của chữ mô hình viết, nên vẫn qua được kiểm tra
    provenance — cắt thì được, sửa thì không.
    """
    corpus = getattr(ctx, "corpus", None)
    if corpus is None:
        return None

    observed = ctx.observed_text
    start = 0
    while True:
        pos = text.find(CONJUNCTION, start)
        if pos < 0:
            return None
        start = pos + 1

        left = text[:pos].strip()
        right = text[pos + len(CONJUNCTION):].strip()
        left_doc = source_doc(corpus, left, observed)
        right_doc = source_doc(corpus, right, observed)
        if left_doc is None or right_doc is None:
            continue
        if left_doc.doc_id == right_doc.doc_id:
            # Cùng một tài liệu nghĩa là cắt sai chỗ: câu này vốn không bị
            # ghép từ hai nguồn. Thử chỗ dán tiếp theo.
            continue

        return [
            {**claim, "text": left, "doc_id": left_doc.doc_id},
            {**claim, "text": right, "doc_id": right_doc.doc_id},
        ]
