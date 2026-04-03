import json
import os
from typing import Any, Dict, List, Tuple, Set
from dotenv import load_dotenv


# =========================
# Utils
# =========================

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_dataset(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, list):
        data = raw
    elif isinstance(raw, dict):
        for key in ("data", "items", "examples", "samples"):
            if key in raw and isinstance(raw[key], list):
                data = raw[key]
                break
        else:
            raise ValueError(f"Không tìm thấy list dữ liệu trong file JSON: keys={list(raw.keys())}")
    else:
        raise ValueError("JSON không đúng định dạng")

    normalized = []
    for item in data:
        choices = item.get("choices", [])
        if isinstance(choices, dict):
            choices = [v for _, v in sorted(choices.items())]

        normalized.append({
            "id": item.get("id"),
            "image_id": item.get("image_id") or item.get("image"),
            "question": item.get("question", ""),
            "choices": choices,
            "question_type": item.get("question_type", ""),
            "answer": item.get("answer"),
            "relevant_articles": item.get("relevant_articles", []),
            "raw": item,
        })
    return normalized


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def normalize_answer(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip().upper()


# =========================
# GOLD LAW KEYS
# =========================

def build_gold_law_keys(relevant_articles: List[Dict[str, Any]]) -> Tuple[Set[str], Set[str]]:
    """
    strict_keys: law_id::article_id
    loose_keys: article_id
    """
    strict_keys = set()
    loose_keys = set()

    for x in relevant_articles or []:
        if not isinstance(x, dict):
            continue

        law_id = str(x.get("law_id", "")).strip()
        article_id = str(x.get("article_id", "")).strip()

        if article_id:
            loose_keys.add(article_id)

        if law_id and article_id:
            strict_keys.add(f"{law_id}::{article_id}")

    return strict_keys, loose_keys


# =========================
# PRED LAW KEYS
# =========================

def normalize_pred_law_entry(x: Any) -> Tuple[str, str]:
    """
    Trả về:
      strict_key = law_id::article_id  (nếu có)
      loose_key  = article_id          (nếu có)

    Hỗ trợ cả:
    - kiểu cũ: retrieved_law_ids = ["22", "B.31"]
    - kiểu mới: retrieved_laws = [{"law_id": ..., "article_id": ..., "full_id": ...}, ...]
    """
    strict_key = ""
    loose_key = ""

    if isinstance(x, dict):
        law_id = str(x.get("law_id", "")).strip()
        article_id = str(x.get("article_id", "")).strip()
        full_id = str(x.get("full_id", "")).strip()
        raw_id = str(x.get("id", "")).strip()

        if law_id and article_id:
            strict_key = f"{law_id}::{article_id}"
            loose_key = article_id
            return strict_key, loose_key

        if full_id:
            if "::" in full_id:
                strict_key = full_id
                loose_key = full_id.split("::", 1)[1].strip()
                return strict_key, loose_key
            loose_key = full_id
            return "", loose_key

        if article_id:
            loose_key = article_id
            return "", loose_key

        if raw_id:
            if "::" in raw_id:
                strict_key = raw_id
                loose_key = raw_id.split("::", 1)[1].strip()
                return strict_key, loose_key
            loose_key = raw_id
            return "", loose_key

        return "", ""

    s = str(x).strip()
    if not s:
        return "", ""

    if "::" in s:
        return s, s.split("::", 1)[1].strip()

    return "", s


def extract_predicted_law_keys(pred: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """
    Ưu tiên đọc retrieved_laws (kiểu mới).
    Nếu không có thì fallback sang retrieved_law_ids (kiểu cũ).
    """
    strict_keys: List[str] = []
    loose_keys: List[str] = []

    retrieved_laws = pred.get("retrieved_laws")
    if isinstance(retrieved_laws, list) and retrieved_laws:
        for x in retrieved_laws:
            strict_key, loose_key = normalize_pred_law_entry(x)
            if strict_key:
                strict_keys.append(strict_key)
            if loose_key:
                loose_keys.append(loose_key)
        return strict_keys, loose_keys

    retrieved_law_ids = pred.get("retrieved_law_ids") or []
    for x in retrieved_law_ids:
        strict_key, loose_key = normalize_pred_law_entry(x)
        if strict_key:
            strict_keys.append(strict_key)
        if loose_key:
            loose_keys.append(loose_key)

    return strict_keys, loose_keys


# =========================
# METRICS
# =========================

def hit_at_k(pred_ids: List[str], gold_set: Set[str], k: int) -> int:
    if not gold_set:
        return 0
    return int(any(x in gold_set for x in pred_ids[:k]))


def recall_at_k(pred_ids: List[str], gold_set: Set[str], k: int) -> float:
    if not gold_set:
        return 0.0
    topk = set(pred_ids[:k])
    return len(topk & gold_set) / len(gold_set)


def reciprocal_rank(pred_ids: List[str], gold_set: Set[str]) -> float:
    if not gold_set:
        return 0.0
    for i, x in enumerate(pred_ids, start=1):
        if x in gold_set:
            return 1.0 / i
    return 0.0


# =========================
# LAW EVALUATION
# =========================

def evaluate_laws(predictions: List[Dict[str, Any]], eval_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    eval_by_id = {x["id"]: x for x in eval_items}

    total = 0
    strict_h1 = strict_h3 = strict_h5 = 0
    loose_h1 = loose_h3 = loose_h5 = 0
    strict_r5 = loose_r5 = 0.0
    strict_mrr = loose_mrr = 0.0

    debug_rows = []

    for pred in predictions:
        qid = pred.get("id")
        if qid not in eval_by_id:
            continue

        item = eval_by_id[qid]
        gold_strict, gold_loose = build_gold_law_keys(item.get("relevant_articles", []))
        if not gold_loose:
            continue

        pred_strict, pred_loose = extract_predicted_law_keys(pred)

        total += 1

        strict_h1 += hit_at_k(pred_strict, gold_strict, 1)
        strict_h3 += hit_at_k(pred_strict, gold_strict, 3)
        strict_h5 += hit_at_k(pred_strict, gold_strict, 5)
        strict_r5 += recall_at_k(pred_strict, gold_strict, 5)
        strict_mrr += reciprocal_rank(pred_strict, gold_strict)

        loose_h1 += hit_at_k(pred_loose, gold_loose, 1)
        loose_h3 += hit_at_k(pred_loose, gold_loose, 3)
        loose_h5 += hit_at_k(pred_loose, gold_loose, 5)
        loose_r5 += recall_at_k(pred_loose, gold_loose, 5)
        loose_mrr += reciprocal_rank(pred_loose, gold_loose)

        if len(debug_rows) < 20:
            debug_rows.append({
                "id": qid,
                "gold_strict": sorted(gold_strict),
                "gold_loose": sorted(gold_loose),
                "pred_strict_top5": pred_strict[:5],
                "pred_loose_top5": pred_loose[:5],
                "strict_hit@5": bool(set(pred_strict[:5]) & gold_strict),
                "loose_hit@5": bool(set(pred_loose[:5]) & gold_loose),
            })

    return {
        "total": total,
        "strict": {
            "hit@1": safe_div(strict_h1, total),
            "hit@3": safe_div(strict_h3, total),
            "hit@5": safe_div(strict_h5, total),
            "recall@5": safe_div(strict_r5, total),
            "mrr": safe_div(strict_mrr, total),
        },
        "loose": {
            "hit@1": safe_div(loose_h1, total),
            "hit@3": safe_div(loose_h3, total),
            "hit@5": safe_div(loose_h5, total),
            "recall@5": safe_div(loose_r5, total),
            "mrr": safe_div(loose_mrr, total),
        },
        "sample_debug": debug_rows,
    }


# =========================
# EXAMPLE EVALUATION
# =========================

def article_id_set(item: Dict[str, Any]) -> Set[str]:
    _, article_keys = build_gold_law_keys(item.get("relevant_articles", []))
    return article_keys


def evaluate_examples(
    predictions: List[Dict[str, Any]],
    eval_items: List[Dict[str, Any]],
    train_items: List[Dict[str, Any]]
) -> Dict[str, Any]:
    eval_by_id = {x["id"]: x for x in eval_items}
    train_by_id = {x["id"]: x for x in train_items}

    total = 0
    qtype_score = 0.0
    answer_score = 0.0
    overlap_hit = 0

    debug_rows = []

    for pred in predictions:
        qid = pred.get("id")
        if qid not in eval_by_id:
            continue

        query = eval_by_id[qid]
        retrieved_ids = pred.get("retrieved_example_ids") or []

        if not retrieved_ids:
            continue

        total += 1

        qtype = query.get("question_type")
        answer = query.get("answer")
        articles = article_id_set(query)

        qtype_match = 0
        answer_match = 0
        overlap = False

        for ex_id in retrieved_ids[:5]:
            ex = train_by_id.get(ex_id)
            if not ex:
                continue

            if ex.get("question_type") == qtype:
                qtype_match += 1

            if normalize_answer(ex.get("answer")) == normalize_answer(answer):
                answer_match += 1

            if articles & article_id_set(ex):
                overlap = True

        denom = max(1, min(5, len(retrieved_ids)))
        qtype_score += qtype_match / denom
        answer_score += answer_match / denom
        overlap_hit += int(overlap)

        if len(debug_rows) < 20:
            debug_rows.append({
                "id": qid,
                "retrieved_example_ids_top5": retrieved_ids[:5],
                "qtype_match_count_top5": qtype_match,
                "answer_match_count_top5": answer_match,
                "article_overlap_hit@5": overlap,
            })

    return {
        "total": total,
        "avg_qtype_match": safe_div(qtype_score, total),
        "avg_answer_match": safe_div(answer_score, total),
        "article_overlap_hit@5": safe_div(overlap_hit, total),
        "sample_debug": debug_rows,
    }


# =========================
# FINAL ANSWER EVALUATION
# =========================

def evaluate_answers(predictions: List[Dict[str, Any]], eval_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    eval_by_id = {x["id"]: x for x in eval_items}

    total = 0
    correct = 0
    missing_prediction = 0
    invalid_prediction = 0

    per_type_total: Dict[str, int] = {}
    per_type_correct: Dict[str, int] = {}

    debug_rows = []

    for pred in predictions:
        qid = pred.get("id")
        if qid not in eval_by_id:
            continue

        gold_item = eval_by_id[qid]
        gold_answer = normalize_answer(gold_item.get("answer"))
        pred_answer = normalize_answer(pred.get("prediction"))

        if not gold_answer:
            continue

        total += 1

        qtype = str(gold_item.get("question_type", "")).strip() or "UNKNOWN"
        per_type_total[qtype] = per_type_total.get(qtype, 0) + 1

        if not pred_answer:
            missing_prediction += 1
            is_correct = False
        else:
            if pred_answer not in {"A", "B", "C", "D"}:
                invalid_prediction += 1
            is_correct = (pred_answer == gold_answer)

        if is_correct:
            correct += 1
            per_type_correct[qtype] = per_type_correct.get(qtype, 0) + 1

        if len(debug_rows) < 30:
            debug_rows.append({
                "id": qid,
                "question_type": qtype,
                "gold_answer": gold_answer,
                "pred_answer": pred_answer,
                "correct": is_correct,
            })

    per_type_accuracy = {
        qtype: safe_div(per_type_correct.get(qtype, 0), cnt)
        for qtype, cnt in per_type_total.items()
    }

    return {
        "total": total,
        "correct": correct,
        "accuracy": safe_div(correct, total),
        "missing_prediction": missing_prediction,
        "invalid_prediction": invalid_prediction,
        "per_question_type_accuracy": per_type_accuracy,
        "sample_debug": debug_rows,
    }


# =========================
# MAIN
# =========================

def main():
    load_dotenv()

    predictions_path = os.getenv("OUTPUT_FILE")
    eval_path = os.getenv("EVAL_JSON")
    train_path = os.getenv("TRAIN_JSON")
    output_path = os.getenv("REPORT_PATH", "retrieval_report.json")

    if not predictions_path or not eval_path or not train_path:
        raise ValueError("Thiếu biến môi trường trong .env")

    predictions = load_json(predictions_path)
    eval_items = normalize_dataset(load_json(eval_path))
    train_items = normalize_dataset(load_json(train_path))

    law_report = evaluate_laws(predictions, eval_items)
    example_report = evaluate_examples(predictions, eval_items, train_items)
    answer_report = evaluate_answers(predictions, eval_items)

    report = {
        "law_retrieval": law_report,
        "example_retrieval": example_report,
        "answer_evaluation": answer_report,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n===== LAW RETRIEVAL =====")
    print(json.dumps(law_report, indent=2, ensure_ascii=False))

    print("\n===== EXAMPLE RETRIEVAL =====")
    print(json.dumps(example_report, indent=2, ensure_ascii=False))

    print("\n===== ANSWER EVALUATION =====")
    print(json.dumps(answer_report, indent=2, ensure_ascii=False))

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()