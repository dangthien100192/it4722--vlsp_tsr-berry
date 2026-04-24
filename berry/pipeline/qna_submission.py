import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from qdrant_client import QdrantClient

from berry.config import config
from berry.llm.fusion import compute_llm_confidence, compute_rule_confidence, fuse_predictions
from berry.llm.llm_client import call_llm_openai
from berry.llm.prompt_builder import build_prompt
from berry.retrieval.retrieve import retrieve_examples_and_laws
from berry.retrieval.rerank import choose_by_law_priority


DEFAULT_INPUT = getattr(config, "submission_task2", "./dataset/submission_task2_no_labels.json")
DEFAULT_OUTPUT = getattr(config, "output_task2", "./dataset/submission_task2.json")


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


def build_relevant_examples(retrieved_examples: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    for hit in retrieved_examples:
        payload = hit.get("payload", {}) or {}
        example_id = str(payload.get("id") or "").strip()

        if not example_id or example_id in seen:
            continue

        seen.add(example_id)

        examples.append({
            "id": example_id,
            "image_id": payload.get("image_id"),
            "question_type": payload.get("question_type"),
            "answer": payload.get("answer"),
            "score": hit.get("score"),
        })

        if len(examples) >= top_k:
            break

    return examples


def normalize_answer(prediction: Any, question_type: str = "") -> str:
    text = str(prediction or "").strip()

    if not text:
        return ""

    qtype = str(question_type or "").lower()

    # Multiple choice: submission nên là A/B/C/D.
    if "multiple" in qtype or "choice" in qtype:
        m = re.search(r"\b([ABCD])\b", text.upper())
        if m:
            return m.group(1)

        m = re.match(r"^\s*([ABCD])[\.\):\s-]", text.upper())
        if m:
            return m.group(1)

    # Yes/No: giữ tiếng Việt thống nhất.
    yes_words = ["đúng", "dung", "yes", "true", "có", "co"]
    no_words = ["sai", "no", "false", "không", "khong"]

    lowered = text.lower()
    if any(w in lowered for w in yes_words) and not any(w in lowered for w in no_words):
        return "Đúng"
    if any(w in lowered for w in no_words):
        return "Sai"

    return text


def make_task2_submission_item(
    item: Dict[str, Any],
    relevant_articles: List[Dict[str, str]],
    relevant_examples: List[Dict[str, Any]],
    answer: str,
    include_debug: bool = False,
    debug: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    # Task 2: multimodal legal Q/A.
    # Giữ format gần input no-label, điền relevant_articles, relevant_examples, answer.
    result: Dict[str, Any] = {
        "id": item.get("id"),
        "image_id": item.get("image_id"),
        "question": item.get("question"),
        "relevant_articles": relevant_articles,
        "question_type": item.get("question_type"),
    }

    if "choices" in item:
        result["choices"] = item.get("choices")

    result["relevant_examples"] = relevant_examples
    result["answer"] = answer

    if include_debug and debug:
        result["debug"] = debug

    return result


def run_qna_submission(
    input_file: str = DEFAULT_INPUT,
    output_file: str = DEFAULT_OUTPUT,
    top_k_laws: int = 5,
    top_k_examples: int = 5,
    use_llm: bool = True,
    include_debug: bool = False,
) -> List[Dict[str, Any]]:
    client = get_qdrant_client()
    dataset = load_json_list(input_file)

    results: List[Dict[str, Any]] = []
    total = len(dataset)

    for idx, item in enumerate(dataset, 1):
        sample_id = item.get("id")
        print(f"[TASK2][QNA] {idx}/{total} | id={sample_id} image_id={item.get('image_id')}")

        retrieved_examples, retrieved_laws, image_description = retrieve_examples_and_laws(client, item)

        relevant_articles = build_relevant_articles(
            retrieved_laws,
            top_k=top_k_laws,
        )
        relevant_examples = build_relevant_examples(
            retrieved_examples,
            top_k=top_k_examples,
        )

        item_for_prompt = dict(item)
        item_for_prompt["image_description"] = image_description

        rule_prediction, rule_debug = choose_by_law_priority(
            item_for_prompt,
            retrieved_laws,
        )

        llm_prediction = ""
        raw_output = ""
        decision_source = "rule_only"
        fused_scores = {}
        fused_margin = 0.0

        if use_llm:
            prompt = build_prompt(
                item_for_prompt,
                retrieved_examples,
                retrieved_laws,
                image_description,
            )

            llm_prediction, raw_output = call_llm_openai(
                prompt,
                question_type=item.get("question_type", ""),
                sample_id=str(sample_id or ""),
            )

            rule_confidence = compute_rule_confidence(
                item_for_prompt,
                rule_prediction,
                rule_debug,
                retrieved_laws,
            )
            llm_confidence = compute_llm_confidence(
                item_for_prompt,
                llm_prediction,
                raw_output,
                retrieved_laws,
            )

            prediction, decision_source, fused_scores, fused_margin = fuse_predictions(
                item_for_prompt,
                rule_prediction,
                llm_prediction,
                rule_confidence,
                llm_confidence,
            )
        else:
            prediction = rule_prediction

        answer = normalize_answer(
            prediction,
            question_type=item.get("question_type", ""),
        )

        debug = {
            "rule_prediction": rule_prediction,
            "llm_prediction": llm_prediction,
            "decision_source": decision_source,
            "fused_scores": fused_scores,
            "fused_margin": fused_margin,
            "rule_debug": rule_debug,
            "llm_raw_output": raw_output,
            "image_description": image_description,
        }

        result = make_task2_submission_item(
            item=item,
            relevant_articles=relevant_articles,
            relevant_examples=relevant_examples,
            answer=answer,
            include_debug=include_debug,
            debug=debug,
        )

        results.append(result)

    save_json_list(output_file, results)
    print(f"[DONE][TASK2] saved: {output_file} | total={len(results)}")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 2 - Multimodal legal Q/A submission")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input no-label JSON file")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output submission JSON file")
    parser.add_argument("--top-k-laws", type=int, default=5, help="Number of unique relevant_articles to save")
    parser.add_argument("--top-k-examples", type=int, default=5, help="Number of relevant_examples to save")
    parser.add_argument("--no-llm", action="store_true", help="Only use rule prediction, do not call LLM")
    parser.add_argument("--debug", action="store_true", help="Include debug field in output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_qna_submission(
        input_file=args.input,
        output_file=args.output,
        top_k_laws=args.top_k_laws,
        top_k_examples=args.top_k_examples,
        use_llm=not args.no_llm,
        include_debug=args.debug,
    )


if __name__ == "__main__":
    main()
