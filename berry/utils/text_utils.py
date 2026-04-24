import re
from typing import Any, Dict, Iterable, List

def compact_text(text: Any, max_len: int = 300) -> str:
    s = str(text or "").strip()
    return s if len(s) <= max_len else s[:max_len].rstrip() + "..."

def normalize_vi_text(s: str) -> str:
    s = str(s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def unique_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out

def extract_sign_codes(text: str) -> List[str]:
    s = str(text or "")
    return sorted(set(re.findall(r"\b([A-Z]\.\d+[a-zA-Z]?)\b", s)))

def extract_sign_codes_from_text(text: str) -> List[str]:
    return unique_keep_order(re.findall(r"\b([A-Z]\.\d{1,3}[a-z]?)\b", text or ""))

def find_choice_label(choice_idx: int) -> str:
    return chr(65 + choice_idx)

def normalize_choices(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, dict):
        return [str(value[k]).strip() for k in sorted(value.keys())]
    if isinstance(value, list):
        return [str(x).strip() for x in value]
    return []

def build_choice_map(item: Dict[str, Any]) -> Dict[str, str]:
    raw_choices = item.get("choices", [])
    if isinstance(raw_choices, dict):
        return {str(k).strip().upper(): str(v).strip() for k, v in sorted(raw_choices.items())}
    if isinstance(raw_choices, list):
        return {find_choice_label(i): str(v).strip() for i, v in enumerate(raw_choices)}
    return {}

def format_choice_text(choices: List[str]) -> str:
    return "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(choices))

def is_yes_no_question(question_type: str) -> bool:
    q = str(question_type or "").strip().lower()
    return q in {"yes/no", "yes no", "true/false", "boolean"}

def truncate_for_embedding(text: str, max_chars: int = 5500) -> str:
    text = str(text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars]