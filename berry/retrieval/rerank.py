from typing import Any, Dict, List, Optional, Tuple
from berry.semantics import parse_choice_semantics
from berry.utils.text_utils import build_choice_map, extract_sign_codes, is_yes_no_question, normalize_vi_text

def score_law_against_choices(item: Dict[str, Any], payload: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    choice_map = build_choice_map(item)
    combined = " ".join([str(payload.get("title", "")), str(payload.get("text", "")), str(payload.get("law_title", "")), str(payload.get("full_text", ""))])
    combined_norm = normalize_vi_text(combined)
    score_boost = 0.0
    matched_choices, matched_phrases = [], []
    discriminative_phrases = ["cấm dừng xe và đỗ xe", "cấm đỗ xe", "cấm đỗ xe vào ngày lẻ", "cấm đỗ xe vào ngày chẵn", "ngày lẻ", "ngày chẵn", "nơi đỗ xe", "chú ý xe đỗ"]
    for label, choice_text in choice_map.items():
        c_norm = normalize_vi_text(choice_text)
        if len(c_norm) >= 4 and c_norm in combined_norm:
            score_boost += 0.18
            matched_choices.append(label)
            matched_phrases.append(choice_text)
    for phrase in discriminative_phrases:
        if phrase in combined_norm:
            for label, choice_text in choice_map.items():
                if phrase in normalize_vi_text(choice_text):
                    score_boost += 0.10
                    if label not in matched_choices:
                        matched_choices.append(label)
                    if choice_text not in matched_phrases:
                        matched_phrases.append(choice_text)
    question_norm = normalize_vi_text(item.get("question", ""))
    title_norm = normalize_vi_text(payload.get("title", ""))
    codes = extract_sign_codes(str(payload.get("title", "")))
    if "bien bao gi" in question_norm and any(code.lower() in title_norm.lower() for code in codes):
        score_boost += 0.03
    return score_boost, {"matched_choices": matched_choices, "matched_phrases": matched_phrases, "sign_codes": codes}

def score_yes_no_law_hit(item: Dict[str, Any], payload: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    question = str(item.get("question", "") or "")
    title = str(payload.get("title", "") or "")
    text = str(payload.get("text", "") or "")
    full_text = str(payload.get("full_text", "") or "")
    q = normalize_vi_text(question)
    doc = normalize_vi_text(" ".join([title, text, full_text]))
    debug: Dict[str, Any] = {"question_norm": q, "matched_terms": [], "support_true": 0.0, "support_false": 0.0, "predicted_label": None, "overlap_score": 0.0, "sign_codes": extract_sign_codes(str(title)), "reason": "yesno_scoring"}
    if not q or not doc:
        debug["reason"] = "empty_question_or_doc"
        return 0.0, debug
    candidate_terms: List[str] = []
    import re
    for pat in [r'"([^"]+)"', r"“([^”]+)”", r"'([^']+)'"]:
        for m in re.findall(pat, question):
            term = normalize_vi_text(m)
            if term and len(term) >= 3:
                candidate_terms.append(term)
    hand_terms = ["giữ khoảng cách an toàn", "chữ màu vàng", "nền đen", "nền vàng", "chữ đen", "chữ trắng", "biển chỉ dẫn", "biển cảnh báo", "biển báo cấm", "biển hiệu lệnh", "biển viết bằng chữ"]
    for term in hand_terms:
        if term in q:
            candidate_terms.append(term)
    overlap_score, matched_terms = 0.0, []
    for term in dict.fromkeys(candidate_terms):
        if term in doc:
            overlap_score += 0.12
            matched_terms.append(term)
    support_true = support_false = 0.0
    has_yellow_text_claim = "chu mau vang" in q or "chu vang" in q
    has_black_bg_claim = "nen den" in q
    has_yellow_bg_claim = "nen vang" in q
    has_black_text_claim = "chu den" in q
    has_white_text_claim = "chu trang" in q
    asks_safe_distance = "giu khoang cach an toan" in q
    law_mentions_safe_distance = "giu khoang cach an toan" in doc
    law_mentions_guide_sign = "bien chi dan" in doc or "chi dan" in doc
    doc_has_yellow_bg_black_text = "nen vang" in doc and "chu den" in doc
    doc_has_black_bg_yellow_text = "nen den" in doc and ("chu vang" in doc or "chu mau vang" in doc)
    doc_has_blue_bg_white_text = ("nen xanh" in doc or "nen mau xanh" in doc) and "chu trang" in doc
    doc_has_red_bg_white_text = ("nen do" in doc or "nen mau do" in doc) and "chu trang" in doc
    if asks_safe_distance and law_mentions_safe_distance: support_true += 0.18
    if asks_safe_distance and law_mentions_guide_sign: support_true += 0.10
    if has_yellow_bg_claim and has_black_text_claim and doc_has_yellow_bg_black_text: support_true += 0.50
    if has_black_bg_claim and has_yellow_text_claim and doc_has_black_bg_yellow_text: support_true += 0.50
    if has_black_bg_claim and has_yellow_text_claim and doc_has_yellow_bg_black_text: support_false += 0.70
    if has_yellow_bg_claim and has_black_text_claim and doc_has_black_bg_yellow_text: support_false += 0.70
    if (has_yellow_text_claim or has_black_bg_claim or has_yellow_bg_claim or has_black_text_claim) and doc_has_blue_bg_white_text: support_false += 0.35
    if (has_yellow_text_claim or has_black_bg_claim or has_yellow_bg_claim or has_black_text_claim) and doc_has_red_bg_white_text: support_false += 0.20
    if has_white_text_claim and "chu trang" in doc: support_true += 0.18
    final_support = overlap_score + max(support_true, support_false)
    debug.update({"matched_terms": matched_terms, "support_true": round(support_true, 6), "support_false": round(support_false, 6), "predicted_label": "ĐÚNG" if support_true > support_false else "SAI" if support_false > support_true else None, "overlap_score": round(overlap_score, 6)})
    return final_support, debug

def rerank_law_hits(item: Dict[str, Any], law_hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    reranked = []
    yes_no_mode = is_yes_no_question(item.get("question_type", ""))
    for hit in law_hits:
        payload = hit.get("payload", {})
        base_score = float(hit.get("score", 0.0))
        boost, debug = score_yes_no_law_hit(item, payload) if yes_no_mode else score_law_against_choices(item, payload)
        new_hit = dict(hit)
        new_hit["base_score"] = base_score
        new_hit["choice_boost"] = boost
        new_hit["score"] = base_score + boost
        new_hit["debug"] = debug
        reranked.append(new_hit)
    return sorted(reranked, key=lambda x: x["score"], reverse=True)

def score_choices_from_laws(item: Dict[str, Any], retrieved_laws: List[Dict[str, Any]], image_description: str = "") -> Dict[str, float]:
    choice_map = build_choice_map(item)
    scores = {label: 0.0 for label in choice_map}
    if not choice_map or not retrieved_laws:
        return scores
    norm_choice_map = {label: normalize_vi_text(text) for label, text in choice_map.items()}
    choice_sem_map = {label: parse_choice_semantics(text) for label, text in choice_map.items()}
    image_desc_norm = normalize_vi_text(image_description)
    if "ngày chẵn" in image_desc_norm:
        for label, text in norm_choice_map.items():
            if "ngày chẵn" in text:
                scores[label] += 3.0
    if "ngày lẻ" in image_desc_norm:
        for label, text in norm_choice_map.items():
            if "ngày lẻ" in text:
                scores[label] += 3.0
    for rank, hit in enumerate(retrieved_laws[:5], 1):
        payload = hit.get("payload", {})
        debug = hit.get("debug", {}) or {}
        rank_weight = 1.0 / rank
        fused_score = float(hit.get("score", 0.0))
        combined_norm = f"{normalize_vi_text(payload.get('title', ''))} {normalize_vi_text(payload.get('text', '') or payload.get('full_text', ''))}"
        matched_choices = set(debug.get("matched_choices", []) or [])
        for label, choice_norm in norm_choice_map.items():
            if choice_norm and choice_norm in combined_norm:
                scores[label] += 0.3 * rank_weight
        for label in matched_choices:
            if label in scores:
                scores[label] += 0.25 * rank_weight
        variants = payload.get("semantics", {}).get("variants", [])
        for v in variants:
            constraints = v.get("constraints", {})
            if constraints.get("day_parity") == "even":
                for label, text in norm_choice_map.items():
                    if "ngày chẵn" in text:
                        scores[label] += 1.5 * rank_weight
            if constraints.get("day_parity") == "odd":
                for label, text in norm_choice_map.items():
                    if "ngày lẻ" in text:
                        scores[label] += 1.5 * rank_weight
        base_intents = payload.get("semantics", {}).get("base_intents", [])
        for label, sem in choice_sem_map.items():
            if sem.get("base_intent") in base_intents:
                scores[label] += 0.5 * rank_weight
        for label in scores:
            scores[label] += min(fused_score, 2.0) * 0.1 * rank_weight
    has_day_variant = any("ngày chẵn" in normalize_vi_text(c) or "ngày lẻ" in normalize_vi_text(c) for c in choice_map.values())
    if has_day_variant:
        for label, text in norm_choice_map.items():
            if text == "cấm đỗ xe":
                scores[label] -= 1.0
    if "ngày chẵn" in image_desc_norm:
        for label, text in norm_choice_map.items():
            if "ngày chẵn" not in text:
                scores[label] -= 0.5
    if "ngày lẻ" in image_desc_norm:
        for label, text in norm_choice_map.items():
            if "ngày lẻ" not in text:
                scores[label] -= 0.5
    return scores

def choose_by_law_priority(item: Dict[str, Any], retrieved_laws: List[Dict[str, Any]]) -> Tuple[Optional[str], Dict[str, Any]]:
    if not retrieved_laws:
        return None, {"reason": "not_applicable", "choice_scores": {}}
    if is_yes_no_question(item.get("question_type", "")):
        support_true = support_false = 0.0
        for hit in retrieved_laws[:3]:
            dbg = hit.get("debug", {}) or {}
            fused_score = float(hit.get("score", 0.0))
            st = float(dbg.get("support_true", 0.0))
            sf = float(dbg.get("support_false", 0.0))
            overlap = float(dbg.get("overlap_score", 0.0))
            support_true += fused_score * max(st + 0.20 * overlap, 0.0)
            support_false += fused_score * max(sf + 0.20 * overlap, 0.0)
        debug = {"reason": "yesno_aggregate", "support_true": round(support_true, 6), "support_false": round(support_false, 6)}
        if support_true == 0.0 and support_false == 0.0:
            debug["reason"] = "not_applicable"
            return None, debug
        gap = abs(support_true - support_false)
        debug["support_gap"] = round(gap, 6)
        if gap < 0.05:
            debug["reason"] = "weak_gap"
            return None, debug
        return ("ĐÚNG", debug) if support_true > support_false else ("SAI", debug)
    choice_map = build_choice_map(item)
    if not choice_map:
        return None, {"reason": "not_applicable", "choice_scores": {}}
    choice_scores = score_choices_from_laws(item, retrieved_laws, image_description=item.get("image_description", ""))
    ranked = sorted(choice_scores.items(), key=lambda kv: kv[1], reverse=True)
    if not ranked:
        return None, {"reason": "no_choice_scores", "choice_scores": choice_scores}
    best_label, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else -999.0
    top1 = retrieved_laws[0]
    top2 = retrieved_laws[1] if len(retrieved_laws) > 1 else None
    top1_title_norm = normalize_vi_text(top1.get("payload", {}).get("title", ""))
    best_choice_norm = normalize_vi_text(choice_map.get(best_label, ""))
    exact_title_match = bool(best_choice_norm and best_choice_norm == top1_title_norm)
    top_gap = float(top1.get("score", 0.0)) - float(top2.get("score", 0.0)) if top2 else float(top1.get("score", 0.0))
    choice_gap = best_score - second_score
    debug = {"choice_scores": choice_scores, "ranked_choice_scores": ranked, "top_gap": top_gap, "choice_gap": choice_gap, "exact_title_match": exact_title_match, "top1_title": top1.get("payload", {}).get("title", "")}
    if exact_title_match and top_gap >= 0.10:
        debug["reason"] = "exact_title_match"
        return best_label, debug
    if best_score >= 1.15 and choice_gap >= 0.35:
        debug["reason"] = "strong_choice_score_gap"
        return best_label, debug
    matched_choices = top1.get("debug", {}).get("matched_choices", []) or []
    if len(matched_choices) == 1 and matched_choices[0] in choice_map and top_gap >= 0.08:
        debug["reason"] = "single_matched_choice"
        return matched_choices[0], debug
    debug["reason"] = "no_force"
    return None, debug
