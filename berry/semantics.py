import re
from typing import Any, Dict, List, Tuple
from berry.utils.text_utils import (
    build_choice_map,
    extract_sign_codes_from_text,
    is_yes_no_question,
    normalize_vi_text,
    unique_keep_order,
)

def parse_choice_semantics(choice_text: str) -> Dict[str, Any]:
    c = normalize_vi_text(choice_text)
    semantics = {"base_intent": None, "constraints": {}}
    if "cấm dừng xe và đỗ xe" in c:
        semantics["base_intent"] = "no_stopping_no_parking"
    elif "cấm đỗ xe" in c:
        semantics["base_intent"] = "no_parking"
    elif "nơi đỗ xe" in c:
        semantics["base_intent"] = "parking_place"
    elif "chú ý xe đỗ" in c:
        semantics["base_intent"] = "watch_parked_vehicle"

    if "ngày lẻ" in c:
        semantics["constraints"]["day_parity"] = "odd"
    elif "ngày chẵn" in c:
        semantics["constraints"]["day_parity"] = "even"
    return semantics

def law_supports_choice_semantics(choice_sem: Dict[str, Any], payload: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    law_sem = payload.get("semantics", {}) or {}
    base_intents = set(law_sem.get("base_intents", []) or [])
    variants = law_sem.get("variants", []) or []
    global_constraints = law_sem.get("global_constraints", {}) or {}
    score = 0.0
    debug = {"semantic_base_match": False, "semantic_variant_match": False, "semantic_day_matches": [], "semantic_penalty": 0.0}
    base_intent = choice_sem.get("base_intent")
    constraints = choice_sem.get("constraints", {}) or {}
    day_parity = constraints.get("day_parity")
    if not base_intent:
        return score, debug
    if base_intent in base_intents:
        debug["semantic_base_match"] = True
        score += 0.55
    if base_intent == "no_parking":
        law_days = []
        for v in variants:
            c = v.get("constraints", {}) or {}
            if c.get("day_parity") in {"odd", "even"}:
                law_days.append(c.get("day_parity"))
        law_days.extend(global_constraints.get("day_parity", []) or [])
        law_days = list(dict.fromkeys([x for x in law_days if x]))
        if day_parity:
            if day_parity in law_days:
                score += 1.15
                debug["semantic_variant_match"] = True
                debug["semantic_day_matches"].append(day_parity)
            elif law_days:
                score -= 0.20
                debug["semantic_penalty"] -= 0.20
        elif law_days:
            score -= 0.35
            debug["semantic_penalty"] -= 0.35
    return score, debug

def parse_question_intent(item: Dict[str, Any]) -> Dict[str, Any]:
    question = str(item.get("question", "") or "")
    q_norm = normalize_vi_text(question)
    choice_map = build_choice_map(item)
    choice_norms = {k: normalize_vi_text(v) for k, v in choice_map.items()}
    topic = "generic_sign"
    if any(x in q_norm for x in ["làn đường", "bên tay trái", "bên trái", "bên phải", "phân làn", "dành riêng cho"]):
        topic = "lane_assignment"
    elif any(x in q_norm for x in ["đỗ xe", "dừng xe", "ngày chẵn", "ngày lẻ", "cấm đỗ", "cấm dừng"]):
        topic = "parking_restriction"
    elif any(x in q_norm for x in ["tốc độ", "km/h", "vận tốc"]):
        topic = "speed_limit"
    elif any(x in q_norm for x in ["nhường đường", "ưu tiên", "đường ưu tiên"]):
        topic = "priority"
    elif any(x in q_norm for x in ["hướng đi", "rẽ trái", "rẽ phải", "đi thẳng", "quay đầu"]):
        topic = "direction_mandate"

    vehicles: List[str] = []
    vehicle_map = [
        ("ô tô", "car"),
        ("xe tải", "truck"),
        ("xe buýt", "bus"),
        ("xe khách", "bus"),
        ("xe máy", "motorcycle"),
        ("xe mô tô", "motorcycle"),
        ("người đi bộ", "pedestrian"),
    ]
    for phrase, label in vehicle_map:
        if phrase in q_norm or any(phrase in c for c in choice_norms.values()):
            vehicles.append(label)

    side = None
    if any(x in q_norm for x in ["bên tay trái", "bên trái", "phía trái", "làn trái"]):
        side = "left"
    elif any(x in q_norm for x in ["bên tay phải", "bên phải", "phía phải", "làn phải"]):
        side = "right"

    features: List[str] = []
    if "ngày chẵn" in q_norm:
        features.append("even_day")
    if "ngày lẻ" in q_norm:
        features.append("odd_day")
    if "chỉ dành cho" in q_norm or "dành riêng cho" in q_norm:
        features.append("exclusive")
    if "ô tô" in q_norm:
        features.append("car")

    return {
        "question_norm": q_norm,
        "topic": topic,
        "vehicles": unique_keep_order(vehicles),
        "side": side,
        "features": unique_keep_order(features),
        "is_yes_no": is_yes_no_question(item.get("question_type", "")),
    }

def infer_base_intents(text_norm: str) -> List[str]:
    intents: List[str] = []
    rules = [
        ("cấm dừng xe và đỗ xe", "no_stopping_no_parking"),
        ("cấm đỗ xe", "no_parking"),
        ("nơi đỗ xe", "parking_place"),
        ("chú ý xe đỗ", "watch_parked_vehicle"),
        ("đường cấm", "no_entry"),
        ("cấm đi ngược chiều", "no_wrong_way"),
        ("rẽ trái", "turn_left"),
        ("rẽ phải", "turn_right"),
        ("đi thẳng", "go_straight"),
        ("quay đầu xe", "u_turn"),
        ("cấm quay đầu xe", "no_u_turn"),
        ("tốc độ tối đa", "max_speed"),
        ("tốc độ tối thiểu", "min_speed"),
        ("nhường đường", "yield"),
        ("dừng lại", "stop"),
        ("đường ưu tiên", "priority_road"),
        ("hết đường ưu tiên", "end_priority_road"),
        ("cấm ô tô", "no_car"),
        ("cấm xe mô tô", "no_motorcycle"),
        ("cấm xe tải", "no_truck"),
        ("cấm xe khách", "no_bus"),
        ("cấm người đi bộ", "no_pedestrian"),
    ]
    for phrase, label in rules:
        if phrase in text_norm:
            intents.append(label)
    return unique_keep_order(intents)

def infer_entities(text_norm: str) -> Dict[str, List[str]]:
    applies_to: List[str] = []
    mapping = [
        ("xe cơ giới", "motor_vehicle"),
        ("ô tô", "car"),
        ("xe tải", "truck"),
        ("xe khách", "bus"),
        ("xe mô tô", "motorcycle"),
        ("người đi bộ", "pedestrian"),
        ("xe ưu tiên", "priority_vehicle"),
        ("xe thô sơ", "non_motor_vehicle"),
    ]
    for phrase, label in mapping:
        if phrase in text_norm:
            applies_to.append(label)
    return {"applies_to": unique_keep_order(applies_to)}

def infer_global_constraints(text_norm: str) -> Dict[str, Any]:
    constraints: Dict[str, Any] = {}
    has_odd = "ngày lẻ" in text_norm
    has_even = "ngày chẵn" in text_norm
    if has_odd and not has_even:
        constraints["day_parity"] = ["odd"]
    elif has_even and not has_odd:
        constraints["day_parity"] = ["even"]
    elif has_odd and has_even:
        constraints["day_parity"] = ["odd", "even"]

    side_values: List[str] = []
    if "bên trái" in text_norm:
        side_values.append("left")
    if "bên phải" in text_norm:
        side_values.append("right")
    if "phía đường có đặt biển" in text_norm:
        side_values.append("same_side_as_sign")
    if side_values:
        constraints["applies_side"] = unique_keep_order(side_values)

    scope_values: List[str] = []
    if "trong khu vực" in text_norm:
        scope_values.append("zone")
    if "trên đoạn đường" in text_norm:
        scope_values.append("road_segment")
    if "giao nhau" in text_norm or "ngã ba" in text_norm or "ngã tư" in text_norm:
        scope_values.append("intersection")
    if scope_values:
        constraints["scope_type"] = unique_keep_order(scope_values)
    return constraints

def split_variant_sentences(text: str) -> Dict[str, str]:
    if not text:
        return {}
    raw = re.sub(r"\s+", " ", text)
    matches = list(re.finditer(r"\b([A-Z]\.\d{1,3}[a-z])\b", raw))
    if not matches:
        return {}
    out: Dict[str, str] = {}
    for i, m in enumerate(matches):
        code = m.group(1)
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        out[code] = raw[start:end].strip(" ;,.")
    return out

def infer_variant_constraints_from_text(code: str, variant_text_norm: str) -> Dict[str, Any]:
    constraints: Dict[str, Any] = {}
    if "ngày lẻ" in variant_text_norm:
        constraints["day_parity"] = "odd"
    elif "ngày chẵn" in variant_text_norm:
        constraints["day_parity"] = "even"
    if "bên trái" in variant_text_norm:
        constraints["applies_side"] = "left"
    elif "bên phải" in variant_text_norm:
        constraints["applies_side"] = "right"
    elif "phía đường có đặt biển" in variant_text_norm:
        constraints["applies_side"] = "same_side_as_sign"
    if "trái" in variant_text_norm and "phải" not in variant_text_norm:
        constraints.setdefault("direction", "left")
    elif "phải" in variant_text_norm and "trái" not in variant_text_norm:
        constraints.setdefault("direction", "right")
    elif "đi thẳng" in variant_text_norm:
        constraints.setdefault("direction", "straight")
    return constraints

def build_variants(sign_codes: List[str], title: str, text: str) -> List[Dict[str, Any]]:
    full = f"{title}\n{text}"
    full_norm = normalize_vi_text(full)
    variant_chunks = split_variant_sentences(full)
    variants: List[Dict[str, Any]] = []
    for code in sign_codes:
        constraints: Dict[str, Any] = {}
        chunk = variant_chunks.get(code, "")
        chunk_norm = normalize_vi_text(chunk)
        if chunk_norm:
            constraints.update(infer_variant_constraints_from_text(code, chunk_norm))
        else:
            code_norm = code.lower()
            if code_norm.endswith("b") and "ngày lẻ" in full_norm and "ngày chẵn" not in full_norm:
                constraints["day_parity"] = "odd"
            elif code_norm.endswith("c") and "ngày chẵn" in full_norm and "ngày lẻ" not in full_norm:
                constraints["day_parity"] = "even"
        variants.append({"variant_id": code, "constraints": constraints, "text": chunk if chunk else None})
    seen = set()
    deduped = []
    for v in variants:
        if v["variant_id"] not in seen:
            seen.add(v["variant_id"])
            deduped.append(v)
    return deduped

def build_law_semantics(item: Dict[str, Any]) -> Dict[str, Any]:
    law_title = str(item.get("law_title", "") or "")
    title = str(item.get("title", "") or "")
    text = str(item.get("text", "") or "")
    full_text = str(item.get("full_text", "") or "")
    full = "\n".join([x for x in [law_title, title, text, full_text] if x])
    text_norm = normalize_vi_text(full)
    raw_codes = list(item.get("sign_codes", []) or [])
    extracted_codes = extract_sign_codes_from_text(full)
    sign_codes = unique_keep_order(raw_codes + extracted_codes)
    return {
        "base_intents": infer_base_intents(text_norm),
        "global_constraints": infer_global_constraints(text_norm),
        "variants": build_variants(sign_codes, title=title, text=full_text or text),
        "sign_codes": sign_codes,
        "applies_to": infer_entities(text_norm).get("applies_to", []),
    }
