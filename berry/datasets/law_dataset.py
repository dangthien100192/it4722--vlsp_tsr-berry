import json
from typing import Any, Dict, Iterable, List
from berry.runtime import MAX_JINA_TEXT_LENGTH
from berry.utils.text_utils import extract_sign_codes

class LawDataset:
    def __init__(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.items = self._normalize_items(raw)

    @staticmethod
    def _normalize_items(raw: Any) -> List[Dict[str, Any]]:
        if not isinstance(raw, list):
            raise ValueError("LAW_JSON phải là list các văn bản luật.")

        normalized: List[Dict[str, Any]] = []
        for law_idx, law_doc in enumerate(raw):
            law_id = str(law_doc.get("id", "")).strip() or f"LAW_{law_idx}"
            law_title = str(law_doc.get("title", "")).strip()
            articles = law_doc.get("articles", [])
            if not isinstance(articles, list):
                continue

            for art_idx, article in enumerate(articles):
                article_id = str(article.get("id", "")).strip() or f"{art_idx + 1}"
                article_title = str(article.get("title", "")).strip()
                article_text = str(article.get("text", "")).strip()
                full_id = f"{law_id}::{article_id}"

                parts = []
                if law_title:
                    parts.append(f"Văn bản: {law_title}")
                if law_id:
                    parts.append(f"Mã văn bản: {law_id}")
                if article_title:
                    parts.append(f"Điều/Phụ lục: {article_title}")
                if article_id:
                    parts.append(f"Mã điều: {article_id}")
                if article_text:
                    parts.append(article_text)

                full_text = "\n".join(parts).strip()
                embed_text_value = full_text[:MAX_JINA_TEXT_LENGTH] if len(full_text) > MAX_JINA_TEXT_LENGTH else full_text
                sign_codes = extract_sign_codes(f"{article_title}\n{article_text}")

                normalized.append(
                    {
                        "id": full_id,
                        "law_id": law_id,
                        "article_id": article_id,
                        "full_id": full_id,
                        "law_title": law_title,
                        "title": article_title,
                        "text": embed_text_value,
                        "full_text": full_text,
                        "sign_codes": sign_codes,
                        "image_id": None,
                        "raw": article,
                    }
                )
        return normalized

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        yield from self.items

    def __len__(self) -> int:
        return len(self.items)
