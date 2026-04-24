from typing import Any, Dict
from berry.semantics import parse_question_intent
from berry.utils.text_utils import build_choice_map

def build_query_text(item: Dict[str, Any]) -> str:
    lines = []
    if item.get("question"):
        lines.append(f"Câu hỏi: {item['question']}")
    if item.get("question_type"):
        lines.append(f"Loại câu hỏi: {item['question_type']}")
    intent = parse_question_intent(item)
    # lines.append(f"Topic suy luận: {intent.get('topic')}")
    # if intent.get("side"):
    #     lines.append(f"Vị trí/làn quan tâm: {intent.get('side')}")
    # if intent.get("vehicles"):
    #     lines.append("Loại phương tiện liên quan: " + ", ".join(intent.get("vehicles", [])))
    # if intent.get("features"):
    #     lines.append("Đặc trưng cần ưu tiên: " + ", ".join(intent.get("features", [])))
    choice_map = build_choice_map(item)
    if choice_map:
        lines.append("Lựa chọn:")
        for label, text in choice_map.items():
            lines.append(f"- {label}. {text}")
    # relevant_articles = item.get("relevant_articles") or []
    # if relevant_articles:
    #     article_strs = []
    #     for a in relevant_articles:
    #         if isinstance(a, dict):
    #             law_id = str(a.get("law_id", "")).strip()
    #             article_id = str(a.get("article_id", "")).strip()
    #             if law_id and article_id:
    #                 article_strs.append(f"{law_id} - Điều {article_id}")
    #             elif article_id:
    #                 article_strs.append(f"Điều {article_id}")
    #         else:
    #             article_strs.append(str(a).strip())
    #     if article_strs:
    #         lines.append("Điều luật gợi ý:")
    #         lines.extend(f"- {x}" for x in article_strs)
    return "\n".join(lines).strip()
