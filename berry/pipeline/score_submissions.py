import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


def load_json_list(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {p}")

    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"File phải là JSON list: {p}")

    return data


def normalize_text(value: Any) -> str:
    return str(value or "").strip()


def article_key(article: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    if not isinstance(article, dict):
        return None

    law_id = normalize_text(article.get("law_id"))
    article_id = normalize_text(article.get("article_id"))

    if not law_id or not article_id:
        return None

    return law_id, article_id


def article_list(item: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    Giữ nguyên thứ tự relevant_articles để tính hit@k và MRR.
    Không dùng set vì set làm mất ranking.
    """
    rows = item.get("relevant_articles") or []
    result: List[Tuple[str, str]] = []

    if not isinstance(rows, list):
        return result

    seen: Set[Tuple[str, str]] = set()

    for article in rows:
        key = article_key(article)
        if key and key not in seen:
            result.append(key)
            seen.add(key)

    return result


def index_by_id(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}

    for item in rows:
        sid = normalize_text(item.get("id"))
        if sid:
            by_id[sid] = item

    return by_id


def f_beta(precision: float, recall: float, beta: float = 2.0) -> float:
    if precision <= 0 and recall <= 0:
        return 0.0

    beta2 = beta * beta
    denom = beta2 * precision + recall

    if denom == 0:
        return 0.0

    return (1 + beta2) * precision * recall / denom


def score_task1_law_retrieval(
    pred_file: str,
    gold_file: str,
    k: int = 5,
    beta: float = 2.0,
    debug_file: Optional[str] = None,
) -> Dict[str, Any]:
    pred_rows = load_json_list(pred_file)
    gold_rows = load_json_list(gold_file)

    pred_by_id = index_by_id(pred_rows)
    gold_by_id = index_by_id(gold_rows)

    total = len(gold_by_id)

    hit1_sum = 0.0
    hit3_sum = 0.0
    hit5_sum = 0.0
    precision5_sum = 0.0
    recall5_sum = 0.0
    f2_5_sum = 0.0
    mrr_sum = 0.0

    details: List[Dict[str, Any]] = []
    missing_prediction_ids: List[str] = []

    for sid, gold_item in gold_by_id.items():
        pred_item = pred_by_id.get(sid)

        if pred_item is None:
            missing_prediction_ids.append(sid)
            pred_item = {
                "id": sid,
                "relevant_articles": [],
            }

        pred_articles = article_list(pred_item)
        gold_articles = article_list(gold_item)
        gold_set = set(gold_articles)

        top1 = pred_articles[:1]
        top3 = pred_articles[:3]
        top5 = pred_articles[:k]

        hit1 = 1.0 if any(a in gold_set for a in top1) else 0.0
        hit3 = 1.0 if any(a in gold_set for a in top3) else 0.0
        hit5 = 1.0 if any(a in gold_set for a in top5) else 0.0

        correct_top5 = [a for a in top5 if a in gold_set]

        precision5 = len(correct_top5) / k if k > 0 else 0.0
        recall5 = len(correct_top5) / len(gold_set) if gold_set else 0.0
        f2_5 = f_beta(precision5, recall5, beta=beta)

        reciprocal_rank = 0.0
        for rank, article in enumerate(pred_articles, start=1):
            if article in gold_set:
                reciprocal_rank = 1.0 / rank
                break

        hit1_sum += hit1
        hit3_sum += hit3
        hit5_sum += hit5
        precision5_sum += precision5
        recall5_sum += recall5
        f2_5_sum += f2_5
        mrr_sum += reciprocal_rank

        details.append({
            "id": sid,
            "image_id": gold_item.get("image_id"),
            "question": gold_item.get("question"),
            "gold_articles": [
                {"law_id": law_id, "article_id": article_id}
                for law_id, article_id in gold_articles
            ],
            "pred_articles": [
                {"law_id": law_id, "article_id": article_id}
                for law_id, article_id in pred_articles
            ],
            "correct_top5": [
                {"law_id": law_id, "article_id": article_id}
                for law_id, article_id in correct_top5
            ],
            "hit@1": hit1,
            "hit@3": hit3,
            "hit@5": hit5,
            "precision@5": precision5,
            "recall@5": recall5,
            "f2@5": f2_5,
            "rr": reciprocal_rank,
        })

    if total == 0:
        strict = {
            "hit@1": 0.0,
            "hit@3": 0.0,
            "hit@5": 0.0,
            "precision@5": 0.0,
            "recall@5": 0.0,
            "f2@5": 0.0,
            "mrr": 0.0,
        }
    else:
        strict = {
            "hit@1": hit1_sum / total,
            "hit@3": hit3_sum / total,
            "hit@5": hit5_sum / total,
            "precision@5": precision5_sum / total,
            "recall@5": recall5_sum / total,
            "f2@5": f2_5_sum / total,
            "mrr": mrr_sum / total,
        }

    result = {
        "law_retrieval": {
            "total": total,
            "strict": strict,
        }
    }

    if missing_prediction_ids:
        result["law_retrieval"]["missing_prediction_count"] = len(missing_prediction_ids)
        result["law_retrieval"]["missing_prediction_ids"] = missing_prediction_ids

    if debug_file:
        debug_path = Path(debug_file)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_path.write_text(
            json.dumps(details, ensure_ascii=False, indent=4),
            encoding="utf-8",
        )

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score Task 1 - Multimodal Legal Retrieval"
    )

    parser.add_argument(
        "--pred",
        default=os.getenv("OUTPUT_TASK1", "./dataset/submission_task1.json"),
        help="File dự đoán task 1, ví dụ ./dataset/submission_task1.json",
    )

    parser.add_argument(
        "--gold",
        default=os.getenv("SUBMISSION_TASK2", "./dataset/submission_task2_no_labels.json"),
        help="Gold task 1, chính là ./dataset/submission_task2_no_labels.json",
    )

    parser.add_argument(
        "--output",
        default="./outputs/task1_law_retrieval_score.json",
        help="File lưu score summary",
    )

    parser.add_argument(
        "--debug",
        default="./outputs/score_debug/task1_law_retrieval_details.json",
        help="File lưu chi tiết từng câu",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="K dùng cho precision@k/recall@k/f2@k. Mặc định 5.",
    )

    parser.add_argument(
        "--beta",
        type=float,
        default=2.0,
        help="Beta cho F-beta. Mặc định 2.0.",
    )

    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Không xuất file debug chi tiết.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    debug_file = None if args.no_debug else args.debug

    result = score_task1_law_retrieval(
        pred_file=args.pred,
        gold_file=args.gold,
        k=args.top_k,
        beta=args.beta,
        debug_file=debug_file,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[DONE] saved score: {output_path}")

    if debug_file:
        print(f"[DONE] saved debug: {debug_file}")


if __name__ == "__main__":
    main()
