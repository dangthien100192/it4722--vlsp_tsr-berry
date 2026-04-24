import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from qdrant_client import QdrantClient

from berry.config import config
from berry.retrieval.retrieve import retrieve_examples_and_laws


DEFAULT_INPUT = getattr(config, "submission_task1", "./dataset/submission_task1_no_labels.json")
DEFAULT_OUTPUT = getattr(config, "output_task1", "./dataset/submission_task1.json")


def load_json_list(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Không tìm thấy input file: {p}")

    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Input phải là JSON list: {p}")

    return data


def save_json_list(path: str, rows: List[Dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(rows, ensure_ascii=False, indent=4),
        encoding="utf-8",
    )


def get_qdrant_client() -> QdrantClient:
    url = getattr(config, "qdrant_url", None) or getattr(config, "QDRANT_URL", None)
    host = getattr(config, "qdrant_host", None)
    port = getattr(config, "qdrant_port", None)

    if url:
        return QdrantClient(url=url)

    if host and port:
        return QdrantClient(host=host, port=port)

    return QdrantClient(url="http://localhost:6333")


def _normalize_article_id(value: Any) -> str:
    return str(value or "").strip()


def _article_from_payload(payload: Dict[str, Any]) -> Optional[Dict[str, str]]:
    law_id = _normalize_article_id(payload.get("law_id"))
    article_id = _normalize_article_id(
        payload.get("article_id")
        or payload.get("parent_article_id")
        or payload.get("id")
    )

    # Nếu article_id đang là chunk id kiểu QCVN...::text::..., ưu tiên parent/article thật.
    if "::" in article_id:
        article_id = _normalize_article_id(payload.get("article_id"))

    if not law_id or not article_id:
        return None

    return {
        "law_id": law_id,
        "article_id": article_id,
    }


def build_relevant_articles(retrieved_laws: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, str]]:
    seen: Set[Tuple[str, str]] = set()
    articles: List[Dict[str, str]] = []

    for hit in retrieved_laws:
        payload = hit.get("payload", {}) or {}
        article = _article_from_payload(payload)
        if not article:
            continue

        key = (article["law_id"], article["article_id"])
        if key in seen:
            continue

        seen.add(key)
        articles.append(article)

        if len(articles) >= top_k:
            break

    return articles


def make_task1_submission_item(
    item: Dict[str, Any],
    relevant_articles: List[Dict[str, str]],
) -> Dict[str, Any]:
    # Task 1: multimodal legal retrieval.
    # Output giữ các trường nhận diện/câu hỏi và điền relevant_articles.
    return {
        "id": item.get("id"),
        "image_id": item.get("image_id"),
        "question": item.get("question"),
        "relevant_articles": relevant_articles,
    }


def run_retrieval_submission(
    input_file: str = DEFAULT_INPUT,
    output_file: str = DEFAULT_OUTPUT,
    top_k_laws: int = 5,
) -> List[Dict[str, Any]]:
    client = get_qdrant_client()
    dataset = load_json_list(input_file)

    results: List[Dict[str, Any]] = []
    total = len(dataset)

    for idx, item in enumerate(dataset, 1):
        print(f"[TASK1][RETRIEVAL] {idx}/{total} | id={item.get('id')} image_id={item.get('image_id')}")

        # Hàm hiện tại trả cả examples/laws/image_description.
        # Task 1 chỉ dùng retrieved_laws.
        _, retrieved_laws, _ = retrieve_examples_and_laws(client, item)

        relevant_articles = build_relevant_articles(
            retrieved_laws,
            top_k=top_k_laws,
        )

        result = make_task1_submission_item(
            item,
            relevant_articles,
        )

        results.append(result)

    save_json_list(output_file, results)
    print(f"[DONE][TASK1] saved: {output_file} | total={len(results)}")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 1 - Multimodal legal retrieval submission")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input no-label JSON file")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output submission JSON file")
    parser.add_argument("--top-k-laws", type=int, default=5, help="Number of unique relevant_articles to save")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_retrieval_submission(
        input_file=args.input,
        output_file=args.output,
        top_k_laws=args.top_k_laws,
    )


if __name__ == "__main__":
    main()
