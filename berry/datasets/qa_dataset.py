import json
from typing import Any, Dict, Iterable, List
from berry.utils.text_utils import normalize_choices

class QaDataset:
    def __init__(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.items = self._normalize_items(raw)

    @staticmethod
    def _normalize_items(raw: Any) -> List[Dict[str, Any]]:
        if isinstance(raw, list):
            data = raw
        elif isinstance(raw, dict):
            for key in ("data", "items", "examples", "samples"):
                if key in raw and isinstance(raw[key], list):
                    data = raw[key]
                    break
            else:
                raise ValueError("Không tìm thấy list dữ liệu trong JSON.")
        else:
            raise ValueError("JSON dataset không đúng định dạng.")

        normalized = []
        for item in data:
            normalized.append(
                {
                    "id": item.get("id"),
                    "image_id": item.get("image_id") or item.get("image"),
                    "question": item.get("question", ""),
                    "choices": normalize_choices(item.get("choices", [])),
                    "question_type": item.get("question_type", ""),
                    "answer": item.get("answer"),
                    "relevant_articles": item.get("relevant_articles", []),
                    "raw": item,
                }
            )
        return normalized

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        yield from self.items

    def __len__(self) -> int:
        return len(self.items)
