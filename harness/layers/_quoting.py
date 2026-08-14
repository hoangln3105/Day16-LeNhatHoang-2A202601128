"""Dùng chung cho `critic` (§2) và `citation_checker` (§11).

Cả hai lớp đều phải trả lời ĐÚNG MỘT câu hỏi, và phải trả lời nó GIỐNG HỆT
cách scorer hỏi: "câu này có phải trích dẫn nguyên văn MỘT DÒNG của tài
liệu kia không?"

Vì sao phải khớp từng chi tiết với `arena.scorer._supports`: nếu bộ so
khớp ở đây CHẶT HƠN scorer, `critic` sẽ xoá những claim mà scorer vốn đã
định cho điểm; nếu LỎNG HƠN, `citation_checker` giữ nguyên một trích dẫn
mà scorer chấm `HALLUCINATED`. Cả hai chiều đều mất điểm im lặng, nên ba
hàm dưới đây sao lại đúng ba quy tắc của scorer:

  * `_norm`      -> NFC, casefold, gộp khoảng trắng   (scorer.py:961)
  * `_norm_lines`-> tách theo DÒNG, bỏ dòng rỗng      (scorer.py:981)
  * `_supports`  -> chứa nguyên văn trong MỘT dòng,
                    và claim phải dài >= 12 ký tự     (scorer.py:1333)

Chép lại thay vì `from arena.scorer import _norm` là có chủ ý: `arena/`
đóng băng và chỉ để đọc, nên phụ thuộc vào tên riêng tư của nó là thứ
`scripts/verify.py` không bảo vệ được. Ba hàm này ngắn và đứng yên cùng
scorer.
"""

from __future__ import annotations

import re
import unicodedata

#: Dưới ngưỡng này scorer coi là quá ngắn để gọi là trích dẫn
#: (`arena.scorer.MIN_SUPPORT_CHARS`).
MIN_SUPPORT_CHARS = 12

_WS_RE = re.compile(r"\s+")


def norm(text) -> str:
    """Bản chuẩn hoá mà mọi phép so chuỗi của scorer chạy trên đó."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    return _WS_RE.sub(" ", unicodedata.normalize("NFC", text).casefold()).strip()


def norm_lines(text) -> tuple:
    """Mỗi DÒNG của tài liệu thành một chuỗi đã chuẩn hoá, bỏ dòng rỗng."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    return tuple(line for line in (norm(raw) for raw in text.splitlines()) if line)


def quotes_a_line(text, body) -> bool:
    """`text` có nằm gọn trong MỘT dòng của `body` không?

    Chữ DÒNG là phần chịu lực: `text in body` coi một câu vắt qua tiêu đề,
    một dòng trắng và nửa đoạn văn là hợp lệ, còn scorer thì không.
    """
    normalised = norm(text)
    if len(normalised) < MIN_SUPPORT_CHARS:
        return False
    return any(normalised in line for line in norm_lines(body))


def source_doc(corpus, text, observed_text: str):
    """Tài liệu ĐÃ ĐỌC SẠCH đầu tiên có một dòng chứa nguyên văn `text`.

    Hai điều kiện, và điều kiện thứ nhất mới là điều kiện đắt:

    1. `doc.body in observed_text` — tài liệu này đã về NGUYÊN VẸN từ một
       lần `fetch_doc` sạch. Một snippet của `search` hay một bản
       `[TRUNCATED: ...]` không tính. Đây là cách rẻ nhất để không bao giờ
       gắn claim vào một tài liệu lượt chạy chưa từng đọc — scorer chấm
       cái đó là `UNRETRIEVED`, phạt 0.75.
    2. một dòng của nó chứa nguyên văn `text`.

    Trả về `Doc` hoặc `None`. Không bao giờ đoán: không tìm được thì trả
    `None` để lớp gọi tự quyết định bỏ claim đi.
    """
    if corpus is None or not text:
        return None
    for doc in getattr(corpus, "docs", ()):
        body = getattr(doc, "body", "")
        if body and body in observed_text and quotes_a_line(text, body):
            return doc
    return None
