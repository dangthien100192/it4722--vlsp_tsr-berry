import json
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Tuple
import re
from together import Together
import numpy as np
import requests
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from transformers import (
    AutoModel,
    Owlv2ForObjectDetection,
    Owlv2Processor,
)
load_dotenv()


# =========================================================
# Config
# =========================================================
@dataclass
class Config:
    train_json: str
    eval_json: str
    train_image_dir: str
    test_image_dir: str
    qdrant_url: str
    output_file: str
    jina_api_key: str
    law_json: Optional[str] = None
    law_image_dir: Optional[str] = None
    collection_examples: str = "berry_examples"
    collection_law: str = "berry_law"
    top_k_examples: int = 5
    top_k_laws: int = 5
    jina_model: str = "jina-embeddings-v3"
    embed_timeout: int = 60
    recreate_on_dim_mismatch: bool = False
    debug_retrieval: bool = False

    @classmethod
    def from_env(cls) -> "Config":
        cfg = cls(
            train_json=os.getenv("TRAIN_JSON", "./dataset/vlsp_2025_train.json"),
            eval_json=os.getenv("EVAL_JSON", "./dataset/vlsp_2025_public_test.json"),
            train_image_dir=os.getenv("TRAIN_IMAGE_DIR", "./dataset/train_images"),
            test_image_dir=os.getenv("TEST_IMAGE_DIR", "./dataset/public_test_images"),
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            output_file=os.getenv("OUTPUT_FILE", "predictions.json"),
            jina_api_key=os.getenv("JINA_API_KEY", ""),
            law_json=os.getenv("LAW_JSON", "./vlsp2025_law.json"),
            law_image_dir=os.getenv("LAW_IMAGE_DIR", ""),
            collection_examples=os.getenv("EXAMPLE_COLLECTION", "berry_examples"),
            collection_law=os.getenv("LAW_COLLECTION", "berry_law"),
            top_k_examples=int(os.getenv("TOP_K_EXAMPLES", "5")),
            top_k_laws=int(os.getenv("TOP_K_LAWS", "5")),
            jina_model=os.getenv("JINA_MODEL", "jina-embeddings-v3"),
            embed_timeout=int(os.getenv("EMBED_TIMEOUT", "60")),
            recreate_on_dim_mismatch=os.getenv("RECREATE_ON_DIM_MISMATCH", "false").lower() == "true",
            debug_retrieval=os.getenv("DEBUG_RETRIEVAL", "false").lower() == "true",
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        missing = []
        if not self.train_json:
            missing.append("TRAIN_JSON")
        if not self.eval_json:
            missing.append("EVAL_JSON")
        if not self.test_image_dir:
            missing.append("TEST_IMAGE_DIR")
        if not self.train_image_dir:
            missing.append("TRAIN_IMAGE_DIR")
        if not self.jina_api_key:
            missing.append("JINA_API_KEY")
        if missing:
            raise ValueError(f"Thiếu biến môi trường: {', '.join(missing)}")


config = Config.from_env()
together_client = Together(api_key=os.getenv("TOGETHER_API_KEY"))

# =========================================================
# Global constants
# =========================================================
EMBED_URL = os.getenv("EMBED_URL","https://api.jina.ai/v1/embeddings")
_embedding_dims: Dict[str, int] = {}

DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

CRADIO_REPO = os.getenv("CRADIO_REPO", "nvidia/C-RADIOv2-B")
OWLV2_REPO = os.getenv("OWLV2_REPO", "google/owlv2-base-patch16-ensemble")

USE_IMAGE_MODELS = os.getenv("USE_IMAGE_MODELS", "true").lower() == "true"
OWLV2_THRESHOLD = float(os.getenv("OWLV2_THRESHOLD", "0.12"))
MAX_OBJECT_BOXES = int(os.getenv("MAX_OBJECT_BOXES", "8"))

# Có thể giữ 1024 để thống nhất Qdrant vectors
IMAGE_DIM = int(os.getenv("IMAGE_DIM", "1024"))
OBJECT_DIM = int(os.getenv("OBJECT_DIM", "1024"))

# Query text cho OWLv2 detect traffic sign
OWL_TEXT_QUERIES = [
    "a traffic sign",
    "a road sign",
    "a prohibition traffic sign",
    "a warning traffic sign",
    "a mandatory traffic sign",
    "a guide traffic sign",
    "a no parking sign",
    "a no stopping sign",
    "a speed limit sign",
    "a one way sign",
    "a turn left sign",
    "a turn right sign",
    "a u-turn sign",
    "a pedestrian crossing sign",
    "a roundabout sign",
    "a stop sign",
]

MAX_JINA_TEXT_LENGTH = int(os.getenv("MAX_JINA_TEXT_LENGTH", "8000"))

CRADIO_IMAGE_SIZE = int(os.getenv("CRADIO_IMAGE_SIZE", "224"))

CRADIO_TRANSFORM = T.Compose([
    T.Resize((CRADIO_IMAGE_SIZE, CRADIO_IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

def extract_choice(text: str) -> str:
    text = (text or "").strip().upper()
    match = re.search(r"\b([ABCD])\b", text)
    if match:
        return match.group(1)
    return "A"
# =========================================================
# Jina text embedding
# =========================================================
def _post_embedding(text: str, task: str) -> List[float]:
    text = (text or "").strip() or "[EMPTY]"

    if len(text) > MAX_JINA_TEXT_LENGTH:
        text = text[:MAX_JINA_TEXT_LENGTH]

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.jina_api_key}",
    }
    payload = {
        "model": config.jina_model,
        "task": task,
        "input": [text],
    }

    try:
        res = requests.post(
            EMBED_URL,
            headers=headers,
            json=payload,
            timeout=config.embed_timeout,
        )
        if res.status_code != 200:
            print("=" * 80)
            print(f"[JINA ERROR] task={task}")
            print(f"[JINA ERROR] status={res.status_code}")
            print(f"[JINA ERROR] text_length={len(text)}")
            print(f"[JINA ERROR] text_sample={text[:500]}")
            print(f"[JINA ERROR] response={res.text[:1000]}")
            print("=" * 80)
        res.raise_for_status()
        data = res.json()
        vec = data["data"][0]["embedding"]
    except Exception as e:
        raise RuntimeError(f"Lỗi embedding Jina. task={task}, error={e}") from e

    if not isinstance(vec, list) or not vec:
        raise RuntimeError(f"Embedding không hợp lệ. task={task}")

    return [float(x) for x in vec]


def embed_text_query(text: str) -> List[float]:
    return _post_embedding(text, task="retrieval.query")


def embed_text_passage(text: str) -> List[float]:
    return _post_embedding(text, task="retrieval.passage")


def get_embedding_dim(task: str = "retrieval.query") -> int:
    if task not in _embedding_dims:
        _embedding_dims[task] = len(_post_embedding("test dimension", task))
        print(f"[DEBUG] embedding dim {task} = {_embedding_dims[task]}")
    return _embedding_dims[task]


def validate_embedding_dims() -> int:
    query_dim = get_embedding_dim("retrieval.query")
    passage_dim = get_embedding_dim("retrieval.passage")
    if query_dim != passage_dim:
        raise RuntimeError(
            f"Embedding dim không khớp: query={query_dim}, passage={passage_dim}"
        )
    print(f"[OK] Jina dim = {query_dim}")
    return query_dim

def preprocess_c_radio_image(image: Image.Image) -> torch.Tensor:
    x = CRADIO_TRANSFORM(image).unsqueeze(0)  # (1, 3, H, W)
    return x.to(DEVICE)

TEXT_DIM = validate_embedding_dims()


# =========================================================
# Vision model loaders
# =========================================================
@lru_cache(maxsize=1)
def get_c_radio():
    model = AutoModel.from_pretrained(
        CRADIO_REPO,
        trust_remote_code=True,
    )
    model.eval().to(DEVICE)
    return model

@lru_cache(maxsize=1)
def get_owlv2():
    processor = Owlv2Processor.from_pretrained(OWLV2_REPO)
    model = Owlv2ForObjectDetection.from_pretrained(OWLV2_REPO)
    model.eval().to(DEVICE)
    return processor, model


def l2_normalize_np(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x)
    if norm <= 1e-12:
        return x
    return x / norm


def ensure_dim(vec: np.ndarray, dim: int) -> List[float]:
    out = np.zeros(dim, dtype=np.float32)
    n = min(dim, vec.shape[0])
    out[:n] = vec[:n]
    return out.tolist()


def open_rgb_image(image_path: str) -> Optional[Image.Image]:
    if not image_path or not os.path.exists(image_path):
        return None
    try:
        return Image.open(image_path).convert("RGB")
    except Exception:
        return None


# =========================================================
# Image / object embedding
# =========================================================
def embed_image(image_path: str) -> List[float]:
    if not USE_IMAGE_MODELS:
        return [0.0] * IMAGE_DIM

    image = open_rgb_image(image_path)
    if image is None:
        return [0.0] * IMAGE_DIM

    try:
        processor = CRADIO_PROCESSOR
        model = CRADIO_MODEL

        with torch.no_grad():
            pixel_values = processor(
                images=image,
                return_tensors="pt",
                do_resize=True,
            ).pixel_values.to(DEVICE)

            summary, _ = model(pixel_values)
            summary = summary[0].detach().cpu().numpy()

        return ensure_dim(l2_normalize_np(summary), IMAGE_DIM)

    except Exception as e:
        print(f"[WARN] embed_image failed: {e}")
        return [0.0] * IMAGE_DIM

def embed_objects(image_path: str) -> List[float]:
    if not USE_IMAGE_MODELS:
        return [0.0] * OBJECT_DIM

    image = open_rgb_image(image_path)
    if image is None:
        return [0.0] * OBJECT_DIM

    try:
        processor, model = get_owlv2()

        text_queries = [OWL_TEXT_QUERIES]
        inputs = processor(text=text_queries, images=image, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]], device=DEVICE)
        results = processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=OWLV2_THRESHOLD,
        )

        boxes = results[0]["boxes"]
        scores = results[0]["scores"]

        if boxes.numel() == 0:
            return [0.0] * OBJECT_DIM

        order = torch.argsort(scores, descending=True)[:MAX_OBJECT_BOXES]
        boxes = boxes[order]
        scores = scores[order]

        w, h = image.size
        feat_rows = []
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box.detach().float().cpu().tolist()
            x1n = max(0.0, min(1.0, x1 / max(w, 1)))
            y1n = max(0.0, min(1.0, y1 / max(h, 1)))
            x2n = max(0.0, min(1.0, x2 / max(w, 1)))
            y2n = max(0.0, min(1.0, y2 / max(h, 1)))
            area = max(0.0, (x2n - x1n) * (y2n - y1n))
            feat_rows.extend([x1n, y1n, x2n, y2n, area, float(score.item())])

        feat = np.array(feat_rows, dtype=np.float32)
        feat = l2_normalize_np(feat)
        return ensure_dim(feat, OBJECT_DIM)

    except Exception as e:
        print(f"[WARN] embed_objects failed for {image_path}: {e}")
        return [0.0] * OBJECT_DIM


# =========================================================
# Dataset
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

        normalized = []
        for item in data:
            choices = item.get("choices", [])
            if isinstance(choices, dict):
                choices = [v for _, v in sorted(choices.items())]

            normalized.append(
                {
                    "id": item.get("id"),
                    "image_id": item.get("image_id") or item.get("image"),
                    "question": item.get("question", ""),
                    "choices": choices,
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

                text_parts = []
                if law_title:
                    text_parts.append(f"Văn bản: {law_title}")
                if law_id:
                    text_parts.append(f"Mã văn bản: {law_id}")
                if article_title:
                    text_parts.append(f"Điều/Phụ lục: {article_title}")
                if article_id:
                    text_parts.append(f"Mã điều: {article_id}")
                if article_text:
                    text_parts.append(article_text)

                text = "\n".join(text_parts).strip()
                if len(text) > MAX_JINA_TEXT_LENGTH:
                    text = text[:MAX_JINA_TEXT_LENGTH]

                normalized.append(
                    {
                        "id": full_id,
                        "law_id": law_id,
                        "article_id": article_id,
                        "full_id": full_id,
                        "law_title": law_title,
                        "title": article_title,
                        "text": text,
                        "image_id": None,
                        "raw": article,
                    }
                )

        return normalized

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        yield from self.items


# =========================================================
# Path utils
# =========================================================
def get_qa_image_path(base_dir: str, item: Dict[str, Any]) -> str:
    image_id = item.get("image_id", "")
    if not image_id:
        return ""

    image_id = str(image_id)
    filename = image_id if image_id.endswith(".jpg") else f"{image_id}.jpg"

    path = os.path.join(base_dir, filename)

    if not os.path.exists(path):
        print(f"[WARN] image not found: {path}")
        return ""

    return path


# =========================================================
# Text builders
# =========================================================
def build_example_text(item: Dict[str, Any]) -> str:
    parts = [f"Câu hỏi: {item.get('question', '')}".strip()]
    qtype = item.get("question_type", "")
    if qtype:
        parts.append(f"Loại câu hỏi: {qtype}")
    choices = item.get("choices", [])
    if choices:
        parts.append("Lựa chọn:\n" + "\n".join(f"- {c}" for c in choices))
    rel = item.get("relevant_articles", [])
    if rel:
        parts.append("Điều luật liên quan:\n" + "\n".join(str(x) for x in rel))
    answer = item.get("answer")
    if answer not in (None, ""):
        parts.append(f"Đáp án: {answer}")
    return "\n".join(parts)


def build_query_text(item: Dict[str, Any]) -> str:
    parts = []
    question = str(item.get("question", "")).strip()
    qtype = str(item.get("question_type", "")).strip()
    choices = item.get("choices", [])

    if question:
        parts.append(f"Câu hỏi giao thông: {question}")
    if qtype:
        parts.append(f"Loại câu hỏi: {qtype}")
    if choices:
        parts.append("Các phương án trả lời:")
        parts.extend(f"- {c}" for c in choices)

    parts.append("Tìm điều luật, biển báo, quy chuẩn hoặc quy định giao thông liên quan nhất.")
    return "\n".join(parts)


# =========================================================
# Qdrant helpers
# =========================================================
def get_existing_named_vector_dim(
    client: QdrantClient,
    collection_name: str,
    vector_name: str,
) -> Optional[int]:
    try:
        info = client.get_collection(collection_name)
    except Exception:
        return None

    vectors = getattr(info.config.params, "vectors", None)
    if vectors is None:
        return None

    if hasattr(vectors, "size"):
        return int(vectors.size) if vector_name == "text" else None

    if isinstance(vectors, dict):
        vec_cfg = vectors.get(vector_name)
        if vec_cfg and hasattr(vec_cfg, "size"):
            return int(vec_cfg.size)

    try:
        vec_cfg = vectors[vector_name]
        if hasattr(vec_cfg, "size"):
            return int(vec_cfg.size)
    except Exception:
        pass

    return None


def ensure_collection(client: QdrantClient, collection_name: str) -> None:
    existing_text_dim = get_existing_named_vector_dim(client, collection_name, "text")

    if existing_text_dim is None:
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config={
                "text": VectorParams(size=TEXT_DIM, distance=Distance.COSINE),
                "image": VectorParams(size=IMAGE_DIM, distance=Distance.COSINE),
                "objects": VectorParams(size=OBJECT_DIM, distance=Distance.COSINE),
            },
        )
        print(f"[INFO] Created collection: {collection_name}")
        return

    if existing_text_dim != TEXT_DIM:
        msg = (
            f"Collection '{collection_name}' có text dim={existing_text_dim}, "
            f"nhưng model trả về dim={TEXT_DIM}"
        )
        if config.recreate_on_dim_mismatch:
            client.recreate_collection(
                collection_name=collection_name,
                vectors_config={
                    "text": VectorParams(size=TEXT_DIM, distance=Distance.COSINE),
                    "image": VectorParams(size=IMAGE_DIM, distance=Distance.COSINE),
                    "objects": VectorParams(size=OBJECT_DIM, distance=Distance.COSINE),
                },
            )
            print(f"[WARN] Recreated collection: {collection_name}")
        else:
            raise RuntimeError(msg)
    else:
        print(f"[OK] Collection '{collection_name}' dim khớp: {TEXT_DIM}")

def collection_has_data(client: QdrantClient, collection_name: str) -> bool:
    try:
        info = client.get_collection(collection_name)
        return info.points_count > 0
    except Exception:
        return False
# =========================================================
# Indexing
# =========================================================
def index_examples(client: QdrantClient, dataset: QaDataset) -> None:
    ensure_collection(client, config.collection_examples)
    points: List[PointStruct] = []

    for idx, item in enumerate(dataset):
        image_path = get_qa_image_path(config.train_image_dir, item)
        text_vec = embed_text_passage(build_example_text(item))
        image_vec = embed_image(image_path)
        object_vec = embed_objects(image_path)

        payload = {
            "kind": "qa_example",
            "id": item.get("id"),
            "image_id": item.get("image_id"),
            "question": item.get("question"),
            "choices": item.get("choices", []),
            "question_type": item.get("question_type", ""),
            "answer": item.get("answer"),
            "relevant_articles": item.get("relevant_articles", []),
        }

        points.append(
            PointStruct(
                id=idx,
                vector={
                    "text": text_vec,
                    "image": image_vec,
                    "objects": object_vec,
                },
                payload=payload,
            )
        )

        if idx % 100 == 0:
            print(f"[INFO] index_examples processing idx={idx}")

    if points:
        client.upsert(collection_name=config.collection_examples, points=points)
        print(f"[OK] Indexed {len(points)} examples")


def index_laws(client: QdrantClient, dataset: LawDataset) -> None:
    ensure_collection(client, config.collection_law)
    points: List[PointStruct] = []
    error_count = 0

    for idx, item in enumerate(dataset):
        law_id = item.get("law_id")
        article_id = item.get("article_id")
        full_id = item.get("full_id")

        print(f"[PROCESSING] idx={idx} | {law_id}::{article_id}")

        try:
            text = item.get("text", "")
            if not isinstance(text, str):
                print(f"[WARN] text không phải string | idx={idx}")
                text = str(text)

            text = text.strip()
            if not text:
                print(f"[WARN] text rỗng | {full_id}")
                text = "[EMPTY]"

            if len(text) > MAX_JINA_TEXT_LENGTH:
                print(f"[WARN] text quá dài ({len(text)}) | {full_id}")
                text = text[:MAX_JINA_TEXT_LENGTH]

            text_vec = embed_text_passage(text)

            payload = {
                "kind": "law_article",
                "id": item.get("id"),
                "law_id": item.get("law_id"),
                "article_id": item.get("article_id"),
                "full_id": item.get("full_id"),
                "law_title": item.get("law_title"),
                "title": item.get("title"),
                "text": text[:1000],
                "image_id": item.get("image_id"),
            }

            points.append(
                PointStruct(
                    id=idx,
                    vector={
                        "text": text_vec,
                        "image": [0.0] * IMAGE_DIM,
                        "objects": [0.0] * OBJECT_DIM,
                    },
                    payload=payload,
                )
            )

        except Exception as e:
            print("\n" + "=" * 80)
            print(f"[ERROR] idx={idx}")
            print(f"law_id={law_id} article_id={article_id}")
            print(f"error={e}")
            print("=" * 80 + "\n")
            error_count += 1
            continue

    if points:
        print(f"[INFO] Upserting {len(points)} law points...")
        client.upsert(collection_name=config.collection_law, points=points)
        print(f"[OK] Indexed {len(points)} law articles")

    print(f"[SUMMARY] Total law indexing errors: {error_count}")


# =========================================================
# Retrieval
# =========================================================
def search_named_vector(
    client: QdrantClient,
    collection_name: str,
    vector_name: str,
    vector: List[float],
    limit: int,
) -> List[Dict[str, Any]]:
    response = client.query_points(
        collection_name=collection_name,
        query=vector,
        using=vector_name,
        limit=limit,
        with_payload=True,
    )
    points = response.points if hasattr(response, "points") else response
    return [{"score": p.score, "payload": p.payload} for p in points]


def fuse_hits(
    text_hits: List[Dict[str, Any]],
    image_hits: List[Dict[str, Any]],
    object_hits: List[Dict[str, Any]],
    top_k: int,
    weights: Tuple[float, float, float] = (0.60, 0.25, 0.15),
) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for weight, hits in zip(weights, (text_hits, image_hits, object_hits)):
        if weight == 0:
            continue
        for hit in hits:
            payload = hit["payload"]
            key = f'{payload.get("kind")}::{payload.get("id")}'
            if key not in merged:
                merged[key] = {"score": 0.0, "payload": payload}
            merged[key]["score"] += weight * float(hit["score"])

    ranked = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
    return ranked[:top_k]


def retrieve_examples_and_laws(
    client: QdrantClient,
    item: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    image_path = get_qa_image_path(config.test_image_dir, item)
    text_vec = embed_text_query(build_query_text(item))

    if len(text_vec) != TEXT_DIM:
        raise RuntimeError(
            f"Embedding dim sai khi retrieve query id={item.get('id')}: "
            f"expected={TEXT_DIM}, got={len(text_vec)}"
        )

    image_vec = embed_image(image_path) if image_path and os.path.exists(image_path) else [0.0] * IMAGE_DIM
    object_vec = embed_objects(image_path) if image_path and os.path.exists(image_path) else [0.0] * OBJECT_DIM

    example_hits = fuse_hits(
        search_named_vector(client, config.collection_examples, "text", text_vec, config.top_k_examples * 3),
        search_named_vector(client, config.collection_examples, "image", image_vec, config.top_k_examples * 3),
        search_named_vector(client, config.collection_examples, "objects", object_vec, config.top_k_examples * 3),
        config.top_k_examples,
        weights=(0.60, 0.25, 0.15),
    )

    # Law retrieval: text-only
    law_hits = fuse_hits(
        search_named_vector(client, config.collection_law, "text", text_vec, config.top_k_laws * 3),
        [],
        [],
        config.top_k_laws,
        weights=(1.0, 0.0, 0.0),
    )

    if config.debug_retrieval:
        print("=" * 80)
        print(f"[DEBUG] QUERY ID: {item.get('id')}")
        print(f"[DEBUG] QUESTION: {item.get('question', '')}")
        print(f"[DEBUG] IMAGE PATH: {image_path}")
        print(f"[DEBUG] GOLD ARTICLES: {item.get('relevant_articles', [])}")
        print("[DEBUG] TOP LAW HITS:")
        for i, hit in enumerate(law_hits[:5], 1):
            p = hit["payload"]
            print(
                f"  {i}. score={hit['score']:.6f} "
                f"law_id={p.get('law_id')} article_id={p.get('article_id')} full_id={p.get('full_id')}"
            )

    return example_hits, law_hits


# =========================================================
# Prompt / LLM
# =========================================================
def build_prompt(
    item: Dict[str, Any],
    retrieved_examples: List[Dict[str, Any]],
    retrieved_laws: List[Dict[str, Any]],
) -> str:
    lines: List[str] = []

    if retrieved_laws:
        lines.append("Điều luật tham khảo:")
        for i, hit in enumerate(retrieved_laws, 1):
            p = hit["payload"]
            law_id = p.get("law_id", "")
            article_id = p.get("article_id", "")
            title = p.get("title", "")
            snippet = str(p.get("text", "")).strip()[:600]

            lines.append(
                f"[LAW {i}] {law_id} | {article_id} | {title}\n{snippet}"
            )

    if retrieved_examples:
        lines.append("\nVí dụ gần nhất:")
        for i, hit in enumerate(retrieved_examples, 1):
            p = hit["payload"]
            question = p.get("question", "")
            choices = p.get("choices", [])
            answer = p.get("answer", "")
            qtype = p.get("question_type", "")
            relevant_articles = p.get("relevant_articles", [])

            choice_text = "\n".join(
                f"- {chr(65 + idx)}. {choice}"
                for idx, choice in enumerate(choices)
            )

            lines.append(
                f"[EX {i}]"
                f"\nLoại: {qtype}"
                f"\nCâu hỏi: {question}"
                f"\nLựa chọn:\n{choice_text}"
                f"\nĐiều luật liên quan: {relevant_articles}"
                f"\nĐáp án: {answer}"
            )

    question = item.get("question", "")
    choices = item.get("choices", [])
    qtype = item.get("question_type", "")

    choice_text = "\n".join(
        f"{chr(65 + idx)}. {choice}"
        for idx, choice in enumerate(choices)
    )

    lines.append("\nCâu hỏi cần trả lời:")
    if qtype:
        lines.append(f"Loại: {qtype}")
    lines.append(f"Câu hỏi: {question}")
    lines.append("Lựa chọn:")
    lines.append(choice_text)
    lines.append("\nHãy chọn đáp án đúng nhất. Chỉ trả lời một chữ cái A, B, C hoặc D.")

    return "\n".join(lines)


def call_llm(prompt: str) -> str:
    try:
        response = together_client.chat.completions.create(
            model=os.getenv(
                "LLM_MODEL",
                "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
            ),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Bạn là trợ lý giải bài VLSP 2025 MLQA-TSR về luật giao thông Việt Nam. "
                        "Nhiệm vụ của bạn là chọn đúng một đáp án trong các lựa chọn A, B, C, D. "
                        "Chỉ trả lời bằng đúng một chữ cái: A, B, C hoặc D. "
                        "Không giải thích, không thêm bất kỳ từ nào khác."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0,
            max_tokens=4,
        )

        text = response.choices[0].message.content or ""
        return extract_choice(text)

    except Exception as e:
        print(f"[LLM ERROR] {e}")
        return "A"
# =========================================================
# Eval
# =========================================================
def run_eval(client: QdrantClient, dataset: QaDataset) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    for item in dataset:
        retrieved_examples, retrieved_laws = retrieve_examples_and_laws(client, item)
        prompt = build_prompt(item, retrieved_examples, retrieved_laws)
        prediction = call_llm(prompt)

        results.append(
            {
                "id": item.get("id"),
                "image_id": item.get("image_id"),
                "question_type": item.get("question_type"),
                "prediction": prediction,
                "retrieved_example_ids": [x["payload"].get("id") for x in retrieved_examples],
                "retrieved_law_ids": [
                    x["payload"].get("article_id") or x["payload"].get("id")
                    for x in retrieved_laws
                ],
                "retrieved_laws": [
                    {
                        "id": x["payload"].get("id"),
                        "law_id": x["payload"].get("law_id"),
                        "article_id": x["payload"].get("article_id"),
                        "full_id": x["payload"].get("full_id"),
                        "score": x.get("score"),
                        "law_title": x["payload"].get("law_title"),
                        "title": x["payload"].get("title"),
                    }
                    for x in retrieved_laws
                ],
            }
        )

    return results


# =========================================================
# Main
# =========================================================
def main() -> None:
    global CRADIO_MODEL, CRADIO_PROCESSOR

    if USE_IMAGE_MODELS:
        print("[INFO] Loading C-RADIOv2...")
        CRADIO_PROCESSOR, CRADIO_MODEL = get_c_radio()
    client = QdrantClient(url=config.qdrant_url)

    train_ds = QaDataset(config.train_json)
    eval_ds = QaDataset(config.eval_json)
    law_ds = LawDataset(config.law_json)

    print(f"[INFO] Train items: {len(train_ds.items)}")
    print(f"[INFO] Eval items: {len(eval_ds.items)}")
    print(f"[INFO] Flattened law articles: {len(law_ds.items)}")
    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Use image models: {USE_IMAGE_MODELS}")

    # Index train examples
    if not collection_has_data(client, config.collection_examples):
        print("[INFO] Indexing examples...")
        index_examples(client, train_ds)
    else:
        print("[SKIP] examples already indexed")
    # Index laws
    if not collection_has_data(client, config.collection_law):
        print("[INFO] Indexing laws...")
        index_laws(client, law_ds)
    else:
        print("[SKIP] laws already indexed")

    # Predict from public test
    predictions = run_eval(client, eval_ds)

    with open(config.output_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(predictions)} predictions to {config.output_file}")


if __name__ == "__main__":
    main()