from typing import Any, Dict, List
from berry.utils.text_utils import compact_text, format_choice_text, is_yes_no_question, normalize_choices, normalize_vi_text

def build_prompt(item: Dict[str, Any], retrieved_examples: List[Dict[str, Any]], retrieved_laws: List[Dict[str, Any]], image_description: str) -> str:
    def format_relevant_articles(x: Any) -> str:
        if not x:
            return ""
        if isinstance(x, list):
            parts = []
            for a in x:
                if isinstance(a, dict):
                    law_id = str(a.get("law_id", "")).strip()
                    article_id = str(a.get("article_id", "")).strip()
                    if law_id and article_id:
                        parts.append(f"{law_id} - Điều {article_id}")
                    elif article_id:
                        parts.append(f"Điều {article_id}")
                    elif law_id:
                        parts.append(law_id)
                else:
                    parts.append(str(a).strip())
            return "; ".join([p for p in parts if p])
        return str(x).strip()

    def summarize_visual_cues(desc: str) -> List[str]:
        d = normalize_vi_text(desc)
        cues = []
        priority_rules = [
            ("biển ngày chẵn", "Có dấu hiệu ngày chẵn"),
            ("biển ngày lẻ", "Có dấu hiệu ngày lẻ"),
            ("biển cấm dừng xe và đỗ xe", "Có dấu hiệu cấm dừng xe và đỗ xe"),
            ("biển cấm đỗ xe", "Có dấu hiệu cấm đỗ xe"),
            ("vạch chéo đỏ", "Có vạch chéo đỏ"),
            ("biển tròn xanh", "Có biển tròn xanh"),
            ("mũi tên", "Có mũi tên chỉ hướng"),
            ("làn dành cho ô tô", "Có dấu hiệu làn dành cho ô tô"),
            ("làn dành cho xe buýt", "Có dấu hiệu làn dành cho xe buýt"),
            ("làn dành cho xe mô tô", "Có dấu hiệu làn dành cho xe mô tô"),
        ]
        for key, msg in priority_rules:
            if key in d:
                cues.append(msg)
        return cues[:4]

    def extract_variant_hints(retrieved_laws: List[Dict[str, Any]]) -> List[str]:
        hints = []
        for hit in retrieved_laws[:3]:
            p = hit.get("payload", {})
            text_norm = normalize_vi_text(str(p.get("text", "") or p.get("full_text", "")))
            title = str(p.get("title", "")).strip()
            if "p.131" in normalize_vi_text(title) or "cấm đỗ xe" in normalize_vi_text(title):
                if "ngày lẻ" in text_norm and "ngày chẵn" in text_norm:
                    hints.append("Trong luật nhóm biển P.131, biến thể ngày lẻ/ngày chẵn là các biến thể khác nhau, không được gộp chung.")
                if "p.131b" in text_norm and "ngày lẻ" in text_norm:
                    hints.append("P.131b tương ứng cấm đỗ xe vào ngày lẻ.")
                if "p.131c" in text_norm and "ngày chẵn" in text_norm:
                    hints.append("P.131c tương ứng cấm đỗ xe vào ngày chẵn.")
        return list(dict.fromkeys(hints))[:4]

    question = str(item.get("question", "")).strip()
    qtype = str(item.get("question_type", "")).strip()
    choices = normalize_choices(item.get("choices", []))
    yes_no = is_yes_no_question(qtype)
    visual_cues = summarize_visual_cues(image_description)
    variant_hints = extract_variant_hints(retrieved_laws)

    lines = [
        "Bạn là trợ lý giải bài MLQA-TSR về luật giao thông Việt Nam.",
        "Ưu tiên tuyệt đối điều luật và đặc trưng trực quan then chốt của ảnh.",
        "Nếu luật mô tả một nhóm biển có nhiều biến thể (a/b/c...), phải chọn đúng biến thể cụ thể; không được chọn tên gọi chung nếu đáp án có phương án chi tiết hơn.",
    ]
    if yes_no:
        lines += ["Nhiệm vụ: xác định phát biểu là ĐÚNG hay SAI.", "Chỉ trả lời bằng đúng một từ: ĐÚNG hoặc SAI."]
    else:
        lines += ["Nhiệm vụ: chọn đúng một đáp án trong các lựa chọn.", "Chỉ trả lời bằng đúng một chữ cái: A, B, C hoặc D.", "Quy tắc bắt buộc: nếu ảnh có tín hiệu đặc thù như 'ngày chẵn' hoặc 'ngày lẻ', phải ưu tiên đáp án chi tiết tương ứng, không chọn đáp án tổng quát."]
    lines.append("\n# DẤU HIỆU TRỰC QUAN CHÍNH")
    if visual_cues:
        lines.extend(f"- {cue}" for cue in visual_cues)
    else:
        lines.append(f"- {image_description}")

    if variant_hints:
        lines.append("\n# GỢI Ý BIẾN THỂ QUAN TRỌNG")
        lines.extend(f"- {hint}" for hint in variant_hints)

    if retrieved_laws:
        lines.append("\n# ĐIỀU LUẬT THAM KHẢO")
        for i, hit in enumerate(retrieved_laws[:5], 1):
            p = hit.get("payload", {})
            meta = " | ".join([x for x in [str(p.get('law_id', '')).strip(), str(p.get('article_id', '')).strip(), str(p.get('title', '')).strip()] if x])
            lines.append(f"[LAW {i}] {meta}")
            text = compact_text(p.get("text", "") or p.get("full_text", ""), 900)
            if text:
                lines.append(text)
            matched_phrases = (hit.get("debug", {}) or {}).get("matched_phrases", []) or []
            if matched_phrases:
                lines.append(f"Gợi ý khớp lựa chọn: {', '.join(matched_phrases[:5])}")
            lines.append("")

    if retrieved_examples:
        lines.append("\n# VÍ DỤ THAM KHẢO")
        for i, hit in enumerate(retrieved_examples[:2], 1):
            p = hit.get("payload", {})
            lines.append(f"[EX {i}]")
            if p.get("question_type"): lines.append(f"Loại: {p.get('question_type')}")
            if p.get("question"): lines.append(f"Câu hỏi: {p.get('question')}")
            ex_choices = normalize_choices(p.get("choices", []))
            ex_choice_text = format_choice_text(ex_choices)
            if ex_choice_text:
                lines += ["Lựa chọn:", ex_choice_text]
            ex_articles = format_relevant_articles(p.get("relevant_articles", []))
            if ex_articles: lines.append(f"Điều luật liên quan: {ex_articles}")
            if p.get("answer"): lines.append(f"Đáp án: {p.get('answer')}")
            lines.append("")

    lines += ["\n# CÂU HỎI CẦN TRẢ LỜI"]
    if qtype: lines.append(f"Loại: {qtype}")
    lines.append(f"Câu hỏi: {question}")
    if not yes_no:
        choice_text = format_choice_text(choices)
        if choice_text:
            lines += ["Lựa chọn:", choice_text]
    lines += [
        "\n# QUY TẮC SUY LUẬN BẮT BUỘC",
        "- Đối chiếu từng lựa chọn với điều luật.",
        "- Nếu có đáp án tổng quát và đáp án cụ thể hơn, ưu tiên đáp án cụ thể đúng với đặc trưng ảnh.",
        "- Nếu có dấu hiệu 'ngày chẵn', không chọn đáp án 'Cấm' chung chung.",
        "- Nếu có dấu hiệu 'ngày lẻ', không chọn đáp án 'Cấm' chung chung.",
        "- Chỉ chọn đáp án tổng quát khi không có dấu hiệu nào đủ để xác định biến thể cụ thể.",
        "\n# OUTPUT",
        "Chỉ ghi đúng một từ: ĐÚNG hoặc SAI" if yes_no else "Chỉ ghi đúng một chữ cái: A, B, C hoặc D",
    ]
    return "\n".join(lines)
