from typing import Dict, List, Optional

from berry.config import config
from berry.runtime import OBJECT_DIM

from berry.embeddings.text_embedding import embed_text_passage
from berry.models.owl import detect_objects

from berry.semantics import parse_question_intent
from berry.utils.math_utils import zero_vec
from berry.utils.math_utils import l2_normalize
from berry.utils.text_utils import unique_keep_order

# =========================
# Label mapping
# =========================

OWL_LABEL_VI_MAP: Dict[str, str] = {
    "traffic sign": "biển báo giao thông",
    "no parking sign": "biển cấm đỗ xe",
    "no stopping and parking sign": "biển cấm dừng xe và đỗ xe",
    "parking sign": "biển nơi đỗ xe",
    "warning sign": "biển cảnh báo",
    "priority sign": "biển ưu tiên",
    "prohibitory sign": "biển báo cấm",
    "regulatory sign": "biển hiệu lệnh",
    "blue circle sign": "biển tròn xanh",
    "red slash sign": "vạch chéo đỏ",
    "even day sign": "biển ngày chẵn",
    "odd day sign": "biển ngày lẻ",
    "lane assignment sign": "biển phân làn",
    "lane control sign": "biển điều khiển làn đường",
    "car lane sign": "biển làn dành cho ô tô",
    "bus lane sign": "biển làn dành cho xe buýt",
    "truck lane sign": "biển làn dành cho xe tải",
    "motorcycle lane sign": "biển làn dành cho xe mô tô",
    "direction arrow sign": "biển mũi tên chỉ hướng",
    "mandatory turn sign": "biển bắt buộc rẽ",
    "speed limit sign": "biển hạn chế tốc độ tối đa",
    "minimum speed sign": "biển tốc độ tối thiểu",
}

# =========================
# Query builder
# =========================

def get_question_guided_owl_queries(item: Dict) -> List[str]:
    intent = parse_question_intent(item)
    topic = intent.get("topic")

    queries: List[str] = ["traffic sign"]

    if topic == "parking_restriction":
        queries += [
            "no parking sign", "no stopping and parking sign",
            "parking sign", "red slash sign",
            "even day sign", "odd day sign",
            "prohibitory sign", "regulatory sign"
        ]

    elif topic == "lane_assignment":
        queries += [
            "lane assignment sign", "lane control sign",
            "car lane sign", "bus lane sign",
            "truck lane sign", "motorcycle lane sign",
            "direction arrow sign", "blue circle sign",
            "regulatory sign"
        ]

    elif topic == "direction_mandate":
        queries += [
            "direction arrow sign", "mandatory turn sign",
            "blue circle sign", "regulatory sign"
        ]

    elif topic == "priority":
        queries += ["priority sign", "warning sign", "regulatory sign"]

    elif topic == "speed_limit":
        queries += ["speed limit sign", "minimum speed sign", "regulatory sign"]

    else:
        queries += list(config.owl_queries)

    return unique_keep_order(queries)

# =========================
# Filter + translate
# =========================

def filter_detected_labels_by_intent(labels: List[str], item: Dict) -> List[str]:
    if not labels:
        return []

    topic = parse_question_intent(item).get("topic")

    keyword_groups = {
        "parking_restriction": ["parking", "stopping", "odd day", "even day", "red slash"],
        "lane_assignment": ["lane", "direction arrow", "blue circle"],
        "direction_mandate": ["direction", "arrow"],
        "priority": ["priority", "warning"],
        "speed_limit": ["speed"],
    }

    keywords = keyword_groups.get(topic, [])

    filtered = [
        label for label in labels
        if (not keywords) or any(k in label.lower() for k in keywords)
    ]

    if filtered:
        return unique_keep_order(filtered)

    return unique_keep_order(labels[:3])


def translate_labels_to_vi(labels: List[str]) -> List[str]:
    out, seen = [], set()

    for label in labels or []:
        vi = OWL_LABEL_VI_MAP.get(label.strip(), label.strip())
        if vi and vi not in seen:
            seen.add(vi)
            out.append(vi)

    return out

# =========================
# MAIN EMBEDDING
# =========================

def embed_objects(
    image_path: Optional[str],
    item: Optional[Dict] = None,
    labels: Optional[List[str]] = None,
) -> List[float]:

    # if not config.use_object_models:
    #     return zero_vec(OBJECT_DIM)

    labels = list(labels or [])

    # detect if needed
    if not labels:
        queries = get_question_guided_owl_queries(item or {}) if item else None
        labels_en = detect_objects(image_path, text_queries=queries)

        if item:
            labels_en = filter_detected_labels_by_intent(labels_en, item)

        labels = translate_labels_to_vi(labels_en)

    else:
        labels = translate_labels_to_vi(labels)

    if not labels:
        return zero_vec(OBJECT_DIM)

    try:
        object_text = "Đặc trưng nhận diện: " + ", ".join(labels)

        vec = embed_text_passage(object_text)

        if len(vec) < OBJECT_DIM:
            vec += [0.0] * (OBJECT_DIM - len(vec))
        elif len(vec) > OBJECT_DIM:
            vec = vec[:OBJECT_DIM]

        return l2_normalize(vec)

    except Exception as e:
        print(f"[WARN][OBJECT] embed_objects failed: {e}")
        return zero_vec(OBJECT_DIM)