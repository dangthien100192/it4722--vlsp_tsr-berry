from typing import Dict, Optional, Tuple
from berry.utils.text_utils import build_choice_map, is_yes_no_question

def compute_rule_confidence(item: Dict, rule_prediction: Optional[str], rule_debug: Dict, retrieved_laws):
    if not rule_prediction:
        return 0.0
    debug = rule_debug or {}
    reason = str(debug.get("reason", "") or "")
    if reason in {"not_applicable", "no_force", "no_choice_scores", "weak_gap"}:
        return 0.0
    conf = 0.0
    yes_no = is_yes_no_question(item.get("question_type", ""))
    if yes_no:
        support_true = float(debug.get("support_true", 0.0))
        support_false = float(debug.get("support_false", 0.0))
        support_gap = abs(support_true - support_false)
        conf += min(support_gap / 0.50, 1.0) * 0.60
        conf += min(max(support_true, support_false) / 0.60, 1.0) * 0.25
        matched_terms = 0
        overlap_sum = 0.0
        sign_code_hits = 0
        for hit in retrieved_laws[:3]:
            dbg = hit.get("debug", {}) or {}
            matched_terms += len(dbg.get("matched_terms", []) or [])
            overlap_sum += float(dbg.get("overlap_score", 0.0) or 0.0)
            sign_code_hits += len(dbg.get("sign_codes", []) or [])
        conf += min(matched_terms / 3.0, 1.0) * 0.10
        conf += min(overlap_sum / 0.50, 1.0) * 0.05
        conf += min(sign_code_hits / 2.0, 1.0) * 0.05
    else:
        ranked = debug.get("ranked_choice_scores", []) or []
        choice_scores = debug.get("choice_scores", {}) or {}
        best_score = float(ranked[0][1] if ranked else 0.0)
        second_score = float(ranked[1][1] if len(ranked) > 1 else 0.0)
        choice_gap = float(debug.get("choice_gap", best_score - second_score) or 0.0)
        top_gap = float(debug.get("top_gap", 0.0) or 0.0)
        exact_title_match = bool(debug.get("exact_title_match", False))
        conf += min(best_score / 1.50, 1.0) * 0.35
        conf += min(max(choice_gap, 0.0) / 0.50, 1.0) * 0.30
        conf += min(max(top_gap, 0.0) / 0.20, 1.0) * 0.15
        conf += min(len(choice_scores) / 4.0, 1.0) * 0.05
        if exact_title_match:
            conf += 0.15
        if reason == "exact_title_match":
            conf += 0.10
        elif reason == "strong_choice_score_gap":
            conf += 0.08
        elif reason == "single_matched_choice":
            conf += 0.05
    return max(0.0, min(conf, 1.0))

def compute_llm_confidence(item: Dict, llm_prediction: Optional[str], raw_output: str, retrieved_laws):
    if not llm_prediction:
        return 0.0
    conf = 0.35
    raw = str(raw_output or "").strip()
    yes_no = is_yes_no_question(item.get("question_type", ""))
    valid_labels = {"ĐÚNG", "SAI"} if yes_no else set(build_choice_map(item).keys()) or {"A", "B", "C", "D"}
    if llm_prediction in valid_labels:
        conf += 0.15
    if raw:
        upper = raw.upper().strip()
        if upper in {"A", "B", "C", "D", "ĐÚNG", "SAI", "DUNG"}:
            conf += 0.20
        elif len(raw) <= 12:
            conf += 0.12
        elif len(raw) <= 64:
            conf += 0.06
    else:
        conf -= 0.20
    if retrieved_laws:
        top_score = float(retrieved_laws[0].get("score", 0.0) or 0.0)
        second_score = float(retrieved_laws[1].get("score", 0.0) or 0.0) if len(retrieved_laws) > 1 else 0.0
        conf += min(top_score / 1.0, 1.0) * 0.15
        conf += min(max(top_score - second_score, 0.0) / 0.15, 1.0) * 0.05
    if yes_no and any(tok in raw.upper() for tok in ["ĐÚNG", "SAI", "TRUE", "FALSE"]):
        conf += 0.05
    return max(0.0, min(conf, 1.0))

def fuse_predictions(item: Dict, rule_prediction: Optional[str], llm_prediction: Optional[str], rule_confidence: float, llm_confidence: float) -> Tuple[str, str, Dict[str, float], float]:
    yes_no = is_yes_no_question(item.get("question_type", ""))
    labels = ["ĐÚNG", "SAI"] if yes_no else (list(build_choice_map(item).keys()) or ["A", "B", "C", "D"])
    fallback = "SAI" if yes_no else labels[0]
    scores = {label: 0.0 for label in labels}
    if rule_prediction in scores and rule_confidence > 0.0: scores[rule_prediction] += rule_confidence
    if llm_prediction in scores and llm_confidence > 0.0: scores[llm_prediction] += llm_confidence
    if rule_prediction and llm_prediction and rule_prediction == llm_prediction and rule_prediction in scores:
        scores[rule_prediction] += 0.05
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_label, best_score = ranked[0] if ranked else (fallback, 0.0)
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = best_score - second_score
    if rule_prediction and rule_confidence >= 0.80 and llm_confidence < 0.90:
        return rule_prediction, "rule_strong", scores, margin
    if rule_prediction and llm_prediction and rule_prediction == llm_prediction and rule_confidence >= 0.50:
        return rule_prediction, "rule_llm_agree", scores, margin
    if best_score > 0.0:
        source = "fusion_rule" if rule_prediction == best_label and rule_confidence > llm_confidence else "fusion_llm" if llm_prediction == best_label else "fusion"
        return best_label, source, scores, margin
    if llm_prediction: return llm_prediction, "llm_fallback", scores, margin
    if rule_prediction: return rule_prediction, "rule_fallback", scores, margin
    return fallback, "default_fallback", scores, margin
