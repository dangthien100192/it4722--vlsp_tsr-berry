import json
import os
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import requests
import torch
import torchvision.transforms as T
from PIL import Image
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from transformers import AutoModel, Owlv2ForObjectDetection, Owlv2Processor


# =========================================================
# Load environment early
# =========================================================
load_dotenv()


# =========================================================
# Config
# =========================================================
@dataclass
class Config:
    # Input files
    train_json: str
    eval_json: str
    train_image_dir: str
    test_image_dir: str
    law_json: str

    # Optional law image folder (currently not used directly)
    law_image_dir: Optional[str] = None

    # Storage / output
    qdrant_url: str = "http://localhost:6333"
    collection_examples: str = "berry_examples"
    collection_law: str = "berry_law"
    output_file: str = "predictions.json"

    # Retrieval settings
    top_k_examples: int = 5
    top_k_laws: int = 5
    recreate_on_dim_mismatch: bool = False
    debug_retrieval: bool = False

    # Jina embedding settings
    jina_api_key: str = ""
    jina_model: str = "jina-embeddings-v3"
    embed_url: str = "https://api.jina.ai/v1/embeddings"
    embed_timeout: int = 60

    # LLM / Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"
    llm_timeout: int = 300
    llm_temperature: float = 0.0
    llm_num_predict: int = 16

    # Image model flags
    use_image_models: bool = False
    cradio_repo: str = "nvidia/C-RADIOv2-B"
    owlv2_repo: str = "google/owlv2-base-patch16-ensemble"

    # Object detection labels for traffic signs
    owl_queries: Tuple[str, ...] = (
        "traffic sign",
        "no parking sign",
        "no stopping and parking sign",
        "parking sign",
        "warning sign",
        "priority sign",
        "prohibitory sign",
        "regulatory sign",
        "blue circle sign",
        "red slash sign",
        "even day sign",
        "odd day sign",
    )

    @classmethod
    def from_env(cls) -> "Config":
        cfg = cls(
            train_json=os.getenv("TRAIN_JSON", "./dataset/train.json"),
            eval_json=os.getenv("EVAL_JSON", "./dataset/public_test.json"),
            train_image_dir=os.getenv("TRAIN_IMAGE_DIR", "./dataset/train_images"),
            test_image_dir=os.getenv("TEST_IMAGE_DIR", "./dataset/public_test_images"),
            law_json=os.getenv("LAW_JSON", "./vlsp2025_law.json"),
            law_image_dir=os.getenv("LAW_IMAGE_DIR", ""),
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            collection_examples=os.getenv("EXAMPLE_COLLECTION", "berry_examples"),
            collection_law=os.getenv("LAW_COLLECTION", "berry_law"),
            output_file=os.getenv("OUTPUT_FILE", "predictions.json"),
            top_k_examples=int(os.getenv("TOP_K_EXAMPLES", "5")),
            top_k_laws=int(os.getenv("TOP_K_LAWS", "5")),
            recreate_on_dim_mismatch=os.getenv("RECREATE_ON_DIM_MISMATCH", "false").lower() == "true",
            debug_retrieval=os.getenv("DEBUG_RETRIEVAL", "false").lower() == "true",
            jina_api_key=os.getenv("JINA_API_KEY", ""),
            jina_model=os.getenv("JINA_MODEL", "jina-embeddings-v3"),
            embed_url=os.getenv("EMBED_URL", "https://api.jina.ai/v1/embeddings"),
            embed_timeout=int(os.getenv("EMBED_TIMEOUT", "60")),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            ollama_model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
            llm_timeout=int(os.getenv("LLM_TIMEOUT", "300")),
            llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
            llm_num_predict=int(os.getenv("LLM_NUM_PREDICT", "16")),
            use_image_models=os.getenv("USE_IMAGE_MODELS", "false").lower() == "true",
            cradio_repo=os.getenv("CRADIO_REPO", "nvidia/C-RADIOv2-B"),
            owlv2_repo=os.getenv("OWLV2_REPO", "google/owlv2-base-patch16-ensemble"),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        missing = []
        for name in ["TRAIN_JSON", "EVAL_JSON", "TRAIN_IMAGE_DIR", "TEST_IMAGE_DIR", "LAW_JSON"]:
            if not getattr(self, name.lower()):
                missing.append(name)
        if not self.jina_api_key:
            missing.append("JINA_API_KEY")
        if missing:
            raise ValueError(f"Thiếu biến môi trường: {', '.join(missing)}")


config = Config.from_env()


# =========================================================
# Globals / constants
# =========================================================
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
MAX_JINA_TEXT_LENGTH = int(os.getenv("MAX_JINA_TEXT_LENGTH", "8192"))

# Text dim is validated at runtime from Jina.
TEXT_DIM = 0

# Fixed defaults for image/object vectors.
# Change via env if you already have an indexed collection with specific dims.
IMAGE_DIM = int(os.getenv("IMAGE_DIM", "1024"))
OBJECT_DIM = int(os.getenv("OBJECT_DIM", "1024"))

# Lazy-loaded model objects
CRADIO_MODEL = None
OWL_PROCESSOR = None
OWL_MODEL = None

# Simple transform for C-RADIO.
CRADIO_TRANSFORM = T.Compose(
    [
        T.Resize((378, 378)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)


# =========================================================
# Utility helpers
# =========================================================
def normalize_vi_text(text: Any) -> str:
    """Lowercase + remove accents for softer matching / reranking."""
    s = str(text or "").strip().lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = re.sub(r"[^a-z0-9\s./:-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def compact_text(text: Any, max_len: int = 300) -> str:
    s = str(text or "").strip()
    if len(s) <= max_len:
        return s
    return s[:max_len].rstrip() + "..."


def l2_normalize(vec: List[float]) -> List[float]:
    arr = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm == 0:
        return arr.tolist()
    return (arr / norm).tolist()


def zero_vec(dim: int) -> List[float]:
    return [0.0] * dim


def extract_sign_codes(text: str) -> List[str]:
    """Extract codes like P.131, W.247, I.408 for debugging / reranking."""
    s = str(text or "")
    return sorted(set(re.findall(r"\b([A-Z]\.\d+[A-Z]?)\b", s)))


def find_choice_label(choice_idx: int) -> str:
    return chr(65 + choice_idx)


def normalize_choices(value: Any) -> List[str]:
    """Convert dict/list choices into ordered list [A_text, B_text, ...]."""
    if not value:
        return []
    if isinstance(value, dict):
        # Prefer alphabetic A/B/C/D ordering.
        return [str(value[k]).strip() for k in sorted(value.keys())]
    if isinstance(value, list):
        return [str(x).strip() for x in value]
    return []


def build_choice_map(item: Dict[str, Any]) -> Dict[str, str]:
    raw_choices = item.get("choices", [])
    if isinstance(raw_choices, dict):
        out: Dict[str, str] = {}
        for k in sorted(raw_choices.keys()):
            out[str(k).strip().upper()] = str(raw_choices[k]).strip()
        return out
    if isinstance(raw_choices, list):
        return {find_choice_label(i): str(v).strip() for i, v in enumerate(raw_choices)}
    return {}


def format_choice_text(choices: List[str]) -> str:
    return "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(choices))


def is_yes_no_question(question_type: str) -> bool:
    q = str(question_type or "").strip().lower()
    return q in {"yes/no", "yes no", "true/false", "boolean"}


def get_qa_image_path(base_dir: str, item: Dict[str, Any]) -> Optional[str]:
    """
    Resolve image path from image_id.
    We try a few common extensions to make the script easier to run.
    """
    image_id = str(item.get("image_id") or "").strip()
    if not image_id:
        return None

    candidates = [
        os.path.join(base_dir, image_id),
        os.path.join(base_dir, f"{image_id}.jpg"),
        os.path.join(base_dir, f"{image_id}.jpeg"),
        os.path.join(base_dir, f"{image_id}.png"),
        os.path.join(base_dir, f"{image_id}.webp"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


# =========================================================
# Dataset readers
# =========================================================
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

        normalized: List[Dict[str, Any]] = []
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

                # Keep full text so prompt / rerank can use richer context.
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
                embed_text = full_text[:MAX_JINA_TEXT_LENGTH] if len(full_text) > MAX_JINA_TEXT_LENGTH else full_text
                sign_codes = extract_sign_codes(f"{article_title}\n{article_text}")

                normalized.append(
                    {
                        "id": full_id,
                        "law_id": law_id,
                        "article_id": article_id,
                        "full_id": full_id,
                        "law_title": law_title,
                        "title": article_title,
                        "text": embed_text,
                        "full_text": full_text,
                        "sign_codes": sign_codes,
                        "image_id": None,
                        "raw": article,
                    }
                )

        return normalized

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        yield from self.items


# =========================================================
# Jina text embeddings
# =========================================================
def _post_jina_embeddings(texts: List[str], task: str) -> List[List[float]]:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.jina_api_key}",
    }
    payload = {
        "model": config.jina_model,
        "task": task,
        "input": texts,
    }
    response = requests.post(config.embed_url, headers=headers, json=payload, timeout=config.embed_timeout)
    response.raise_for_status()
    data = response.json()
    rows = data.get("data", [])
    if not rows:
        raise RuntimeError(f"Jina trả về rỗng cho task={task}")
    rows = sorted(rows, key=lambda x: x["index"])
    return [row["embedding"] for row in rows]


def embed_text_query(text: str) -> List[float]:
    return _post_jina_embeddings([text], task="retrieval.query")[0]


def embed_text_passage(text: str) -> List[float]:
    return _post_jina_embeddings([text], task="retrieval.passage")[0]


def validate_embedding_dims() -> int:
    query_dim = len(embed_text_query("test dimension"))
    passage_dim = len(embed_text_passage("test dimension"))
    if query_dim != passage_dim:
        raise RuntimeError(
            f"Text embedding dim không khớp: retrieval.query={query_dim}, retrieval.passage={passage_dim}"
        )
    return query_dim


# =========================================================
# Optional image models
# =========================================================
@lru_cache(maxsize=1)
def get_c_radio():
    model = AutoModel.from_pretrained(
        r"D:\hf_models\C-RADIOv2-B",
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval().to(DEVICE)
    return model

@lru_cache(maxsize=1)
def get_owl():
    processor = Owlv2Processor.from_pretrained(config.owlv2_repo)
    model = Owlv2ForObjectDetection.from_pretrained(config.owlv2_repo)
    model.eval().to(DEVICE)
    return processor, model


def preprocess_c_radio_image(image: Image.Image) -> torch.Tensor:
    x = CRADIO_TRANSFORM(image).unsqueeze(0)  # (1, 3, H, W)
    return x.to(DEVICE)


def load_image(image_path: Optional[str]) -> Optional[Image.Image]:
    if not image_path or not os.path.exists(image_path):
        return None
    try:
        return Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[WARN][IMAGE] Không mở được ảnh: {image_path} | error={e}")
        return None


def embed_image(image_path: Optional[str]) -> List[float]:
    """
    Embed image using C-RADIO.
    If image models are disabled or image missing, return zero vector.
    """
    if not config.use_image_models:
        return zero_vec(IMAGE_DIM)

    image = load_image(image_path)
    if image is None:
        return zero_vec(IMAGE_DIM)

    try:
        model = get_c_radio()
        x = preprocess_c_radio_image(image)
        with torch.no_grad():
            outputs = model(x)

        # Different remote-code models may expose embeddings differently.
        if isinstance(outputs, torch.Tensor):
            vec = outputs.squeeze(0).detach().cpu().float().flatten().tolist()
        elif hasattr(outputs, "pooler_output"):
            vec = outputs.pooler_output.squeeze(0).detach().cpu().float().flatten().tolist()
        elif hasattr(outputs, "last_hidden_state"):
            vec = outputs.last_hidden_state.mean(dim=1).squeeze(0).detach().cpu().float().flatten().tolist()
        else:
            raise RuntimeError(f"Không hiểu output của C-RADIO: {type(outputs)}")

        # Keep vector size stable for Qdrant.
        if len(vec) < IMAGE_DIM:
            vec = vec + [0.0] * (IMAGE_DIM - len(vec))
        elif len(vec) > IMAGE_DIM:
            vec = vec[:IMAGE_DIM]
        return l2_normalize(vec)

    except Exception as e:
        print(f"[WARN][IMAGE] embed_image failed for {image_path}: {e}")
        return zero_vec(IMAGE_DIM)


def detect_objects(image_path: Optional[str], threshold: float = 0.10) -> List[str]:
    """
    Detect traffic-sign-related labels with OWLv2.
    Returns matched text labels, later re-embedded as object text vector.
    """
    if not config.use_image_models:
        return []

    image = load_image(image_path)
    if image is None:
        return []

    try:
        processor, model = get_owl()
        text_queries = list(config.owl_queries)
        inputs = processor(text=text_queries, images=image, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]], device=DEVICE)
        results = processor.post_process_object_detection(outputs=outputs, threshold=threshold, target_sizes=target_sizes)

        labels: List[str] = []
        if results:
            res = results[0]
            for label_idx in res["labels"].detach().cpu().tolist():
                if 0 <= label_idx < len(text_queries):
                    labels.append(text_queries[label_idx])

        # Deduplicate but preserve order.
        deduped: List[str] = []
        seen = set()
        for x in labels:
            if x not in seen:
                deduped.append(x)
                seen.add(x)
        return deduped

    except Exception as e:
        print(f"[WARN][OBJECT] detect_objects failed for {image_path}: {e}")
        return []


def embed_objects(image_path: Optional[str]) -> List[float]:
    """
    Turn detected object labels into a vector.
    We reuse Jina passage embedding on object-label text for a stable vector space.
    """
    labels = detect_objects(image_path)
    if not labels:
        return zero_vec(OBJECT_DIM)

    try:
        object_text = "Detected objects: " + ", ".join(labels)
        vec = embed_text_passage(object_text)
        if len(vec) < OBJECT_DIM:
            vec = vec + [0.0] * (OBJECT_DIM - len(vec))
        elif len(vec) > OBJECT_DIM:
            vec = vec[:OBJECT_DIM]
        return l2_normalize(vec)
    except Exception as e:
        print(f"[WARN][OBJECT] embed_objects failed for {image_path}: {e}")
        return zero_vec(OBJECT_DIM)


def build_image_description(image_path: Optional[str]) -> str:
    labels = detect_objects(image_path)
    if not labels:
        return "Không nhận diện được đặc trưng biển báo rõ ràng từ module object detection."
    return "Các đối tượng/đặc trưng nhận diện được: " + ", ".join(labels)


# =========================================================
# Qdrant helpers
# =========================================================
def collection_exists(client: QdrantClient, collection_name: str) -> bool:
    collections = client.get_collections().collections
    return any(c.name == collection_name for c in collections)


def collection_has_data(client: QdrantClient, collection_name: str) -> bool:
    if not collection_exists(client, collection_name):
        return False
    info = client.get_collection(collection_name)
    return int(getattr(info, "points_count", 0) or 0) > 0


def ensure_collection(client: QdrantClient, collection_name: str) -> None:
    """
    Ensure collection exists with named vectors: text / image / objects.
    If dims mismatch and RECREATE_ON_DIM_MISMATCH=true, the collection is recreated.
    """
    expected_vectors = {
        "text": VectorParams(size=TEXT_DIM, distance=Distance.COSINE),
        "image": VectorParams(size=IMAGE_DIM, distance=Distance.COSINE),
        "objects": VectorParams(size=OBJECT_DIM, distance=Distance.COSINE),
    }

    if not collection_exists(client, collection_name):
        client.create_collection(collection_name=collection_name, vectors_config=expected_vectors)
        print(f"[INFO][QDRANT] Created collection: {collection_name}")
        return

    info = client.get_collection(collection_name)
    current = getattr(info.config.params, "vectors", None)
    mismatch = False

    try:
        if isinstance(current, dict):
            current_dict = current
        else:
            current_dict = getattr(current, "params_map", {}) or {}

        for name, vp in expected_vectors.items():
            cur = current_dict.get(name)
            cur_size = getattr(cur, "size", None)
            if cur_size != vp.size:
                mismatch = True
                print(
                    f"[WARN][QDRANT] Vector dim mismatch in {collection_name} | "
                    f"name={name} expected={vp.size} got={cur_size}"
                )
    except Exception:
        mismatch = True
        print(f"[WARN][QDRANT] Không đọc được schema vectors của collection {collection_name}")

    if mismatch:
        if config.recreate_on_dim_mismatch:
            print(f"[INFO][QDRANT] Recreating collection due to dim mismatch: {collection_name}")
            client.delete_collection(collection_name=collection_name)
            client.create_collection(collection_name=collection_name, vectors_config=expected_vectors)
        else:
            raise RuntimeError(
                f"Collection {collection_name} có dim không khớp. "
                f"Bật RECREATE_ON_DIM_MISMATCH=true hoặc đổi tên collection mới."
            )


def search_named_vector(
    client: QdrantClient,
    collection_name: str,
    vector_name: str,
    vector: List[float],
    limit: int,
) -> List[Dict[str, Any]]:
    """
    Search Qdrant collection by one named vector.
    We support both newer and older qdrant-client call styles.
    """
    if not vector or not any(abs(v) > 1e-12 for v in vector):
        return []

    try:
        response = client.query_points(
            collection_name=collection_name,
            query=vector,
            using=vector_name,
            with_payload=True,
            limit=limit,
        )
        points = response.points
    except Exception:
        response = client.search(
            collection_name=collection_name,
            query_vector=(vector_name, vector),
            with_payload=True,
            limit=limit,
        )
        points = response

    hits: List[Dict[str, Any]] = []
    for p in points:
        payload = getattr(p, "payload", None) or {}
        score = float(getattr(p, "score", 0.0) or 0.0)
        pid = payload.get("id") or getattr(p, "id", None)
        hits.append({"id": pid, "score": score, "payload": payload})
    return hits


def fuse_hits(
    text_hits: List[Dict[str, Any]],
    image_hits: List[Dict[str, Any]],
    object_hits: List[Dict[str, Any]],
    limit: int,
    weights: Tuple[float, float, float] = (0.60, 0.25, 0.15),
) -> List[Dict[str, Any]]:
    """
    Weighted late fusion on hit ids.
    Scores from separate retrieval channels are summed after multiplying weights.
    """
    fused: Dict[str, Dict[str, Any]] = {}

    def _add(hits: List[Dict[str, Any]], weight: float, channel: str) -> None:
        for h in hits:
            hid = str(h.get("id"))
            if hid not in fused:
                fused[hid] = {
                    "id": hid,
                    "score": 0.0,
                    "payload": h.get("payload", {}),
                    "channel_scores": {},
                }
            fused[hid]["score"] += float(h.get("score", 0.0)) * weight
            fused[hid]["channel_scores"][channel] = float(h.get("score", 0.0))

    _add(text_hits, weights[0], "text")
    _add(image_hits, weights[1], "image")
    _add(object_hits, weights[2], "objects")

    ranked = sorted(fused.values(), key=lambda x: x["score"], reverse=True)
    return ranked[:limit]


# =========================================================
# Build query text for retrieval
# =========================================================
def build_query_text(item: Dict[str, Any]) -> str:
    lines = []
    if item.get("question"):
        lines.append(f"Câu hỏi: {item['question']}")
    if item.get("question_type"):
        lines.append(f"Loại câu hỏi: {item['question_type']}")

    choice_map = build_choice_map(item)
    if choice_map:
        lines.append("Lựa chọn:")
        for label, text in choice_map.items():
            lines.append(f"- {label}. {text}")

    relevant_articles = item.get("relevant_articles") or []
    if relevant_articles:
        article_strs = []
        for a in relevant_articles:
            if isinstance(a, dict):
                law_id = str(a.get("law_id", "")).strip()
                article_id = str(a.get("article_id", "")).strip()
                if law_id and article_id:
                    article_strs.append(f"{law_id} - Điều {article_id}")
                elif article_id:
                    article_strs.append(f"Điều {article_id}")
            else:
                article_strs.append(str(a).strip())
        if article_strs:
            lines.append("Điều luật gợi ý:")
            lines.extend(f"- {x}" for x in article_strs)

    return "\n".join(lines).strip()


# =========================================================
# Law reranking: boost hits that align with answer choices
# =========================================================
def score_law_against_choices(item: Dict[str, Any], payload: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """
    Soft reranking after vector retrieval.
    Main idea:
    - keep semantic base score from vector DB
    - add small boosts when law title/text directly matches choice text
    - add extra boosts for highly discriminative phrases
    """
    choice_map = build_choice_map(item)
    combined = " ".join(
        [
            str(payload.get("title", "")),
            str(payload.get("text", "")),
            str(payload.get("law_title", "")),
            str(payload.get("full_text", "")),
        ]
    )
    combined_norm = normalize_vi_text(combined)

    score_boost = 0.0
    matched_choices: List[str] = []
    matched_phrases: List[str] = []

    discriminative_phrases = [
        "cam dung xe va do xe",
        "cam do xe",
        "cam do xe vao ngay le",
        "cam do xe vao ngay chan",
        "ngay le",
        "ngay chan",
        "noi do xe",
        "chu y xe do",
    ]

    # 1) direct option-text match
    for label, choice_text in choice_map.items():
        c_norm = normalize_vi_text(choice_text)
        if len(c_norm) >= 4 and c_norm in combined_norm:
            score_boost += 0.18
            matched_choices.append(label)
            matched_phrases.append(choice_text)

    # 2) discriminative phrase match for finer-grained variants
    for phrase in discriminative_phrases:
        if phrase in combined_norm:
            for label, choice_text in choice_map.items():
                if phrase in normalize_vi_text(choice_text):
                    score_boost += 0.10
                    if label not in matched_choices:
                        matched_choices.append(label)
                    if choice_text not in matched_phrases:
                        matched_phrases.append(choice_text)

    # 3) small hint if question asks directly about sign identity
    question_norm = normalize_vi_text(item.get("question", ""))
    title_norm = normalize_vi_text(payload.get("title", ""))
    codes = extract_sign_codes(str(payload.get("title", "")))
    if "bien bao gi" in question_norm and any(code.lower() in title_norm.lower() for code in codes):
        score_boost += 0.03

    debug = {
        "matched_choices": matched_choices,
        "matched_phrases": matched_phrases,
        "sign_codes": codes,
    }
    return score_boost, debug


def rerank_law_hits(item: Dict[str, Any], law_hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    reranked: List[Dict[str, Any]] = []
    for hit in law_hits:
        payload = hit.get("payload", {})
        base_score = float(hit.get("score", 0.0))
        boost, debug = score_law_against_choices(item, payload)

        new_hit = dict(hit)
        new_hit["base_score"] = base_score
        new_hit["choice_boost"] = boost
        new_hit["score"] = base_score + boost
        new_hit["debug"] = debug
        reranked.append(new_hit)

    reranked.sort(key=lambda x: x["score"], reverse=True)
    return reranked


# =========================================================
# Indexing
# =========================================================
def index_examples(client: QdrantClient, dataset: QaDataset) -> None:
    ensure_collection(client, config.collection_examples)
    points: List[PointStruct] = []

    for idx, item in enumerate(dataset):
        image_path = get_qa_image_path(config.train_image_dir, item)
        query_text = build_query_text(item)

        try:
            text_vec = embed_text_passage(query_text)
            image_vec = embed_image(image_path)
            object_vec = embed_objects(image_path)

            payload = {
                "kind": "qa_example",
                "id": item.get("id"),
                "image_id": item.get("image_id"),
                "question": item.get("question"),
                "choices": item.get("choices"),
                "question_type": item.get("question_type"),
                "answer": item.get("answer"),
                "relevant_articles": item.get("relevant_articles", []),
            }

            points.append(
                PointStruct(
                    id=idx,
                    vector={"text": text_vec, "image": image_vec, "objects": object_vec},
                    payload=payload,
                )
            )
        except Exception as e:
            print(f"[ERROR][INDEX_EXAMPLE] idx={idx} id={item.get('id')} error={e}")

    if points:
        print(f"[INFO][EXAMPLE] Upserting {len(points)} example points...")
        client.upsert(collection_name=config.collection_examples, points=points)
        print(f"[OK][EXAMPLE] Indexed {len(points)} examples")


def index_laws(client: QdrantClient, dataset: LawDataset) -> None:
    ensure_collection(client, config.collection_law)
    points: List[PointStruct] = []
    error_count = 0

    for idx, item in enumerate(dataset):
        law_id = item.get("law_id")
        article_id = item.get("article_id")
        full_id = item.get("full_id")
        print(f"[PROCESSING][LAW] idx={idx} | {law_id}::{article_id}")

        try:
            text = str(item.get("text", "")).strip() or "[EMPTY]"
            text_vec = embed_text_passage(text)

            full_text = str(item.get("full_text", "")).strip()
            payload_text = full_text[:4000] if len(full_text) > 4000 else full_text

            payload = {
                "kind": "law_article",
                "id": item.get("id"),
                "law_id": item.get("law_id"),
                "article_id": item.get("article_id"),
                "full_id": item.get("full_id"),
                "law_title": item.get("law_title"),
                "title": item.get("title"),
                "text": payload_text,
                "full_text": payload_text,
                "sign_codes": item.get("sign_codes", []),
                "image_id": item.get("image_id"),
            }

            points.append(
                PointStruct(
                    id=idx,
                    vector={
                        "text": text_vec,
                        "image": zero_vec(IMAGE_DIM),
                        "objects": zero_vec(OBJECT_DIM),
                    },
                    payload=payload,
                )
            )
        except Exception as e:
            print("\n" + "=" * 90)
            print(f"[ERROR][LAW] idx={idx}")
            print(f"law_id={law_id} article_id={article_id} full_id={full_id}")
            print(f"error={e}")
            print("=" * 90 + "\n")
            error_count += 1
            continue

    if points:
        print(f"[INFO][LAW] Upserting {len(points)} law points...")
        client.upsert(collection_name=config.collection_law, points=points)
        print(f"[OK][LAW] Indexed {len(points)} law articles")

    print(f"[SUMMARY][LAW] Total indexing errors: {error_count}")


# =========================================================
# Retrieval
# =========================================================
def retrieve_examples_and_laws(
    client: QdrantClient,
    item: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str]:
    image_path = get_qa_image_path(config.test_image_dir, item)
    text_vec = embed_text_query(build_query_text(item))

    if len(text_vec) != TEXT_DIM:
        raise RuntimeError(
            f"Embedding dim sai khi retrieve query id={item.get('id')}: expected={TEXT_DIM}, got={len(text_vec)}"
        )

    has_image = bool(image_path and os.path.exists(image_path))
    image_vec = embed_image(image_path) if has_image else zero_vec(IMAGE_DIM)
    object_vec = embed_objects(image_path) if has_image else zero_vec(OBJECT_DIM)
    image_desc = build_image_description(image_path) if has_image else "Không tìm thấy ảnh để phân tích."

    # Example retrieval uses all three modalities.
    example_hits = fuse_hits(
        search_named_vector(client, config.collection_examples, "text", text_vec, config.top_k_examples * 3),
        search_named_vector(client, config.collection_examples, "image", image_vec, config.top_k_examples * 3),
        search_named_vector(client, config.collection_examples, "objects", object_vec, config.top_k_examples * 3),
        config.top_k_examples,
        weights=(0.60, 0.25, 0.15),
    )

    # Law retrieval currently relies on text retrieval, then reranks using choices.
    raw_law_hits = fuse_hits(
        search_named_vector(client, config.collection_law, "text", text_vec, config.top_k_laws * 8),
        [],
        [],
        config.top_k_laws * 8,
        weights=(1.0, 0.0, 0.0),
    )
    reranked_law_hits = rerank_law_hits(item, raw_law_hits)
    law_hits = reranked_law_hits[: config.top_k_laws]

    if config.debug_retrieval:
        print("=" * 100)
        print(f"[DEBUG][RETRIEVE] QUERY ID: {item.get('id')}")
        print(f"[DEBUG][RETRIEVE] QUESTION: {item.get('question', '')}")
        print(f"[DEBUG][RETRIEVE] QUESTION TYPE: {item.get('question_type', '')}")
        print(f"[DEBUG][RETRIEVE] IMAGE PATH: {image_path}")
        print(f"[DEBUG][RETRIEVE] HAS IMAGE: {has_image}")
        print(f"[DEBUG][RETRIEVE] IMAGE DESCRIPTION: {image_desc}")
        print(f"[DEBUG][RETRIEVE] GOLD ARTICLES: {item.get('relevant_articles', [])}")

        choice_map = build_choice_map(item)
        if choice_map:
            print("[DEBUG][RETRIEVE] CHOICES:")
            for label, text in choice_map.items():
                print(f"  - {label}: {text}")

        print("[DEBUG][RETRIEVE] TOP LAW HITS AFTER RERANK:")
        for i, hit in enumerate(law_hits, 1):
            p = hit.get("payload", {})
            dbg = hit.get("debug", {})
            print(
                f"  {i}. final={hit.get('score', 0.0):.6f} "
                f"base={hit.get('base_score', 0.0):.6f} "
                f"boost={hit.get('choice_boost', 0.0):.6f} "
                f"id={p.get('full_id')} title={compact_text(p.get('title', ''), 120)}"
            )
            print(f"     matched_choices={dbg.get('matched_choices', [])}")
            print(f"     matched_phrases={dbg.get('matched_phrases', [])}")
            print(f"     sign_codes={dbg.get('sign_codes', [])}")

        print("[DEBUG][RETRIEVE] TOP EXAMPLE HITS:")
        for i, hit in enumerate(example_hits[:5], 1):
            p = hit.get("payload", {})
            print(
                f"  {i}. score={hit.get('score', 0.0):.6f} "
                f"id={p.get('id')} answer={p.get('answer')} "
                f"question={compact_text(p.get('question', ''), 100)}"
            )

    return example_hits, law_hits, image_desc


# =========================================================
# Prompt builder
# =========================================================
def build_prompt(
    item: Dict[str, Any],
    retrieved_examples: List[Dict[str, Any]],
    retrieved_laws: List[Dict[str, Any]],
    image_description: str,
) -> str:
    def format_relevant_articles(x: Any) -> str:
        if not x:
            return ""
        if isinstance(x, list):
            parts = []
            for a in x:
                if isinstance(a, dict):
                    law_id = str(a.get("law_id", "")).strip()
                    article_id = str(a.get("article_id", "")).strip()
                    if law_id and article_id:
                        parts.append(f"{law_id} - Điều {article_id}")
                    elif article_id:
                        parts.append(f"Điều {article_id}")
                    elif law_id:
                        parts.append(law_id)
                else:
                    parts.append(str(a).strip())
            return "; ".join([p for p in parts if p])
        return str(x).strip()

    question = str(item.get("question", "")).strip()
    qtype = str(item.get("question_type", "")).strip()
    choices = normalize_choices(item.get("choices", []))
    yes_no = is_yes_no_question(qtype)

    lines: List[str] = []
    lines.append("Bạn là trợ lý giải bài MLQA-TSR về luật giao thông Việt Nam.")
    lines.append("Ưu tiên bám sát điều luật được cung cấp. Ví dụ truy hồi chỉ để tham khảo phụ trợ.")
    lines.append(
        "Nếu điều luật mô tả một nhóm biển có nhiều biến thể (ví dụ a/b/c), phải chọn đúng biến thể khớp nhất với lựa chọn; "
        "không chọn tên gọi chung nếu lựa chọn yêu cầu mức chi tiết hơn."
    )

    if yes_no:
        lines.append("Nhiệm vụ: xác định phát biểu là ĐÚNG hay SAI.")
        lines.append("Chỉ trả lời bằng đúng một từ: ĐÚNG hoặc SAI.")
    else:
        lines.append("Nhiệm vụ: chọn đúng một đáp án trong các lựa chọn.")
        lines.append(
            "Phải phân biệt kỹ các lựa chọn gần nghĩa như: cấm dừng/đỗ, ngày chẵn/ngày lẻ, nơi đỗ xe/chú ý xe đỗ."
        )
        lines.append("Chỉ trả lời bằng đúng một chữ cái: A, B, C hoặc D.")

    lines.append("\n# BIỂN BÁO / ẢNH")
    lines.append(image_description)

    if retrieved_laws:
        lines.append("\n# ĐIỀU LUẬT THAM KHẢO")
        for i, hit in enumerate(retrieved_laws[:5], 1):
            p = hit.get("payload", {})
            title = str(p.get("title", "")).strip()
            law_id = str(p.get("law_id", "")).strip()
            article_id = str(p.get("article_id", "")).strip()
            text = compact_text(p.get("text", "") or p.get("full_text", ""), 1200)
            meta = " | ".join([x for x in [law_id, article_id, title] if x])
            lines.append(f"[LAW {i}] {meta}")
            if text:
                lines.append(text)
            dbg = hit.get("debug", {})
            if dbg.get("matched_phrases"):
                lines.append(f"Gợi ý khớp lựa chọn: {', '.join(dbg.get('matched_phrases', []))}")
            lines.append("")

    if retrieved_examples:
        lines.append("\n# VÍ DỤ THAM KHẢO")
        for i, hit in enumerate(retrieved_examples[:3], 1):
            p = hit.get("payload", {})
            ex_question = str(p.get("question", "")).strip()
            ex_qtype = str(p.get("question_type", "")).strip()
            ex_answer = str(p.get("answer", "")).strip()
            ex_choices = normalize_choices(p.get("choices", []))
            ex_choice_text = format_choice_text(ex_choices)
            ex_articles = format_relevant_articles(p.get("relevant_articles", []))

            lines.append(f"[EX {i}]")
            if ex_qtype:
                lines.append(f"Loại: {ex_qtype}")
            if ex_question:
                lines.append(f"Câu hỏi: {ex_question}")
            if ex_choice_text:
                lines.append("Lựa chọn:")
                lines.append(ex_choice_text)
            if ex_articles:
                lines.append(f"Điều luật liên quan: {ex_articles}")
            if ex_answer:
                lines.append(f"Đáp án: {ex_answer}")
            lines.append("")

    lines.append("\n# CÂU HỎI CẦN TRẢ LỜI")
    if qtype:
        lines.append(f"Loại: {qtype}")
    lines.append(f"Câu hỏi: {question}")

    if not yes_no:
        choice_text = format_choice_text(choices)
        if choice_text:
            lines.append("Lựa chọn:")
            lines.append(choice_text)

        lines.append("\n# YÊU CẦU SUY LUẬN NGẦM")
        lines.append("- Đối chiếu từng lựa chọn với điều luật.")
        lines.append("- Nếu có lựa chọn tổng quát và lựa chọn cụ thể hơn, ưu tiên lựa chọn cụ thể đúng với mô tả luật.")
        lines.append("- Đặc biệt chú ý các khác biệt: 'cấm đỗ xe' vs 'cấm đỗ xe ngày lẻ/ngày chẵn', 'cấm dừng xe và đỗ xe' vs 'cấm đỗ xe'.")

    lines.append("\n# OUTPUT")
    if yes_no:
        lines.append("Chỉ ghi đúng một từ: ĐÚNG hoặc SAI")
    else:
        lines.append("Chỉ ghi đúng một chữ cái: A, B, C hoặc D")

    return "\n".join(lines)


# =========================================================
# LLM call / output extraction
# =========================================================
def extract_choice(text: str, question_type: str = "") -> str:
    """
    Normalize model output:
    - Multiple choice -> A/B/C/D
    - Yes/No -> ĐÚNG/SAI
    """
    raw = str(text or "").strip()
    upper = raw.upper()
    yes_no = is_yes_no_question(question_type)

    if yes_no:
        if "ĐÚNG" in upper or "DUNG" in upper or re.search(r"\bTRUE\b", upper):
            return "ĐÚNG"
        if "SAI" in upper or re.search(r"\bFALSE\b", upper):
            return "SAI"
        return "SAI"

    match = re.search(r"\b([ABCD])\b", upper)
    if match:
        return match.group(1)
    if upper and upper[0] in "ABCD":
        return upper[0]
    return "A"


def call_llm(prompt: str, question_type: str = "") -> Tuple[str, str]:
    """
    Call local Ollama and return:
    - normalized prediction
    - raw model output for debugging
    """
    payload = {
        "model": config.ollama_model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": config.llm_temperature,
            "num_predict": config.llm_num_predict,
        },
    }

    try:
        response = requests.post(
            f"{config.ollama_base_url}/api/generate",
            json=payload,
            timeout=config.llm_timeout,
        )

        if response.status_code != 200:
            print(f"[OLLAMA ERROR] status={response.status_code} body={response.text[:1000]}")
            fallback = "SAI" if is_yes_no_question(question_type) else "A"
            return fallback, ""

        data = response.json()
        raw_text = str(data.get("response", "")).strip()
        pred = extract_choice(raw_text, question_type=question_type)

        if config.debug_retrieval:
            print(f"[DEBUG][LLM] raw_output={raw_text[:500]}")
            print(f"[DEBUG][LLM] normalized_prediction={pred}")

        return pred, raw_text

    except Exception as e:
        print(f"[OLLAMA ERROR] {e}")
        fallback = "SAI" if is_yes_no_question(question_type) else "A"
        return fallback, ""


# =========================================================
# Evaluation run
# =========================================================
def run_eval(client: QdrantClient, dataset: QaDataset) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    for idx, item in enumerate(dataset, 1):
        print("=" * 100)
        print(f"[EVAL] {idx}/{len(dataset.items)} | id={item.get('id')} | image_id={item.get('image_id')}")

        retrieved_examples, retrieved_laws, image_description = retrieve_examples_and_laws(client, item)
        prompt = build_prompt(item, retrieved_examples, retrieved_laws, image_description)
        prediction, raw_output = call_llm(prompt, question_type=item.get("question_type", ""))

        if config.debug_retrieval:
            print(f"[DEBUG][PROMPT]\n{compact_text(prompt, 2500)}")

        results.append(
            {
                "id": item.get("id"),
                "image_id": item.get("image_id"),
                "question_type": item.get("question_type"),
                "prediction": prediction,
                "llm_raw_output": raw_output,
                "retrieved_example_ids": [x["payload"].get("id") for x in retrieved_examples],
                "retrieved_law_ids": [
                    x["payload"].get("article_id") or x["payload"].get("id") for x in retrieved_laws
                ],
                "retrieved_laws": [
                    {
                        "id": x["payload"].get("id"),
                        "law_id": x["payload"].get("law_id"),
                        "article_id": x["payload"].get("article_id"),
                        "full_id": x["payload"].get("full_id"),
                        "score": x.get("score"),
                        "base_score": x.get("base_score"),
                        "choice_boost": x.get("choice_boost"),
                        "law_title": x["payload"].get("law_title"),
                        "title": x["payload"].get("title"),
                        "debug": x.get("debug", {}),
                    }
                    for x in retrieved_laws
                ],
                "image_description": image_description,
            }
        )

    return results


# =========================================================
# Main
# =========================================================
def main() -> None:
    global TEXT_DIM, CRADIO_MODEL, OWL_PROCESSOR, OWL_MODEL

    print("[INFO] Validating text embedding dimensions...")
    TEXT_DIM = validate_embedding_dims()
    print(f"[INFO] TEXT_DIM={TEXT_DIM} | IMAGE_DIM={IMAGE_DIM} | OBJECT_DIM={OBJECT_DIM}")

    if config.use_image_models:
        print("[INFO] Loading C-RADIOv2...")
        CRADIO_MODEL = get_c_radio()
        print("[INFO] Loading OWLv2...")
        OWL_PROCESSOR, OWL_MODEL = get_owl()
    else:
        print("[INFO] Image models disabled -> image/object vectors will be zero or text-derived placeholders.")

    client = QdrantClient(url=config.qdrant_url)

    train_ds = QaDataset(config.train_json)
    eval_ds = QaDataset(config.eval_json)
    law_ds = LawDataset(config.law_json)

    print(f"[INFO] Train items: {len(train_ds.items)}")
    print(f"[INFO] Eval items: {len(eval_ds.items)}")
    print(f"[INFO] Flattened law articles: {len(law_ds.items)}")
    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Use image models: {config.use_image_models}")
    print(f"[INFO] Qdrant URL: {config.qdrant_url}")
    print(f"[INFO] Ollama base URL: {config.ollama_base_url}")
    print(f"[INFO] Ollama model: {config.ollama_model}")

    if not collection_has_data(client, config.collection_examples):
        print("[INFO] Indexing examples...")
        index_examples(client, train_ds)
    else:
        print("[SKIP] examples already indexed")

    if not collection_has_data(client, config.collection_law):
        print("[INFO] Indexing laws...")
        index_laws(client, law_ds)
    else:
        print("[SKIP] laws already indexed")

    print("[INFO] Running evaluation / prediction...")
    results = run_eval(client, eval_ds)

    with open(config.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Saved predictions to {config.output_file}")


if __name__ == "__main__":
    main()
