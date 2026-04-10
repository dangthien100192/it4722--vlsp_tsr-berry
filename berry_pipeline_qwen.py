import os

# =========================================================
# GLOBAL HF CACHE CONTROL (set BEFORE importing transformers)
# =========================================================
HF_CACHE_ROOT = os.getenv("HF_CACHE_ROOT", r"D:\hf_cache")

os.environ["HF_HOME"] = HF_CACHE_ROOT
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(HF_CACHE_ROOT, "hub")
os.environ["TRANSFORMERS_DYNAMIC_MODULE_NAME"] = "local"

import json
import re
import shutil
import unicodedata
import importlib.util
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import requests
import torch
import torchvision.transforms as T
from PIL import Image
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from transformers import AutoModel, Owlv2ForObjectDetection, Owlv2Processor

def auto_clear_hf_dynamic_cache(repo_name: str = "C-RADIOv2-B") -> None:
    """
    Xóa các dynamic module cache lỗi của transformers liên quan tới repo_name.
    """
    hf_modules = Path(os.environ["HF_HOME"]) / "modules" / "transformers_modules"
    if not hf_modules.exists():
        return

    repo_name_norm = repo_name.lower().replace("-", "").replace("_", "")
    removed = 0

    for d in hf_modules.rglob("*"):
        if not d.is_dir():
            continue
        dn = d.name.lower().replace("-", "").replace("_", "")
        if repo_name_norm in dn:
            try:
                print(f"[AUTO-FIX][HF] Removing broken dynamic cache: {d}")
                shutil.rmtree(d, ignore_errors=True)
                removed += 1
            except Exception as e:
                print(f"[WARN][HF] Cannot remove {d}: {e}")

    if removed > 0:
        print(f"[AUTO-FIX][HF] Removed {removed} broken cache folder(s)")


def auto_fix_local_model_dir(model_dir: str) -> None:
    """
    Nếu model local thiếu file quan trọng thì xóa để tải lại sạch.
    """
    p = Path(model_dir).expanduser().resolve()
    required = [
        "config.json",
        "hf_model.py",
        "radio_model.py",
        "dual_hybrid_vit.py",
    ]

    if not p.exists():
        return

    missing = [f for f in required if not (p / f).exists()]
    if missing:
        print(f"[AUTO-FIX][HF] Local model thiếu file {missing} -> removing {p}")
        shutil.rmtree(p, ignore_errors=True)




def ensure_cradio_dynamic_module(model_dir: str, repo_name: str = "C-RADIOv2-B") -> Path:
    """
    Copy các file python của repo C-RADIO sang dynamic module cache để
    transformers với trust_remote_code=True không bị thiếu file phụ thuộc.
    """
    src = Path(model_dir).expanduser().resolve()
    dst = Path(os.environ["HF_HOME"]) / "modules" / "transformers_modules" / repo_name
    dst.mkdir(parents=True, exist_ok=True)

    required_py = [
        "hf_model.py",
        "radio_model.py",
        "dual_hybrid_vit.py",
    ]
    optional_py = [
        "__init__.py",
        "configuration_hf.py",
        "configuration_radio.py",
        "modeling_hf.py",
        "model.py",
    ]

    copied = []
    for name in required_py + optional_py:
        s = src / name
        d = dst / name
        if s.exists():
            shutil.copy2(s, d)
            copied.append(name)

    init_file = dst / "__init__.py"
    if not init_file.exists():
        init_file.write_text("", encoding="utf-8")

    missing = [f for f in required_py if not (dst / f).exists()]
    if missing:
        raise RuntimeError(f"Dynamic cache của C-RADIO còn thiếu file: {missing} | dst={dst}")

    print(f"[OK][HF] Prepared dynamic module cache at: {dst}")
    if copied:
        print(f"[OK][HF] Copied files: {copied}")
    return dst
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
    cradio_local_dir: str = r"D:\hf_models\C-RADIOv2-B"
    owlv2_repo: str = "google/owlv2-base-patch16-ensemble"
    owlv2_local_dir: str = r"D:\hf_models\owlv2-base-patch16-ensemble"

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
            cradio_local_dir=os.getenv("CRADIO_LOCAL_DIR", r"D:\hf_models\C-RADIOv2-B"),
            owlv2_repo=os.getenv("OWLV2_REPO", "google/owlv2-base-patch16-ensemble"),
            owlv2_local_dir=os.getenv("OWLV2_LOCAL_DIR", r"D:\hf_models\owlv2-base-patch16-ensemble"),
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
DEVICE = torch.device(os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
MAX_JINA_TEXT_LENGTH = int(os.getenv("MAX_JINA_TEXT_LENGTH", "8192"))

# Text dim is validated at runtime from Jina.
TEXT_DIM = 0

# Fixed defaults for image/object vectors.
IMAGE_DIM = int(os.getenv("IMAGE_DIM", "1024"))
OBJECT_DIM = int(os.getenv("OBJECT_DIM", "1024"))

# Lazy-loaded model objects
CRADIO_MODEL = None
OWL_PROCESSOR = None
OWL_MODEL = None

CRADIO_TRANSFORM = T.Compose(
    [
        T.Resize((384, 384)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)


def has_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


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
    s = str(text or "")
    return sorted(set(re.findall(r"\b([A-Z]\.\d+[a-zA-Z]?)\b", s)))


def find_choice_label(choice_idx: int) -> str:
    return chr(65 + choice_idx)


def normalize_choices(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, dict):
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


def parse_choice_semantics(choice_text: str) -> Dict[str, Any]:
    c = normalize_vi_text(choice_text)
    semantics = {
        "base_intent": None,
        "constraints": {},
    }

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
    debug = {
        "semantic_base_match": False,
        "semantic_variant_match": False,
        "semantic_day_matches": [],
        "semantic_penalty": 0.0,
    }

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
        else:
            if law_days:
                score -= 0.35
                debug["semantic_penalty"] -= 0.35

    return score, debug


def format_choice_text(choices: List[str]) -> str:
    return "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(choices))


def is_yes_no_question(question_type: str) -> bool:
    q = str(question_type or "").strip().lower()
    return q in {"yes/no", "yes no", "true/false", "boolean"}


def get_qa_image_path(base_dir: str, item: Dict[str, Any]) -> Optional[str]:
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
                embed_text_value = full_text[:MAX_JINA_TEXT_LENGTH] if len(full_text) > MAX_JINA_TEXT_LENGTH else full_text
                sign_codes = extract_sign_codes(f"{article_title}\n{article_text}")

                normalized.append(
                    {
                        "id": full_id,
                        "law_id": law_id,
                        "article_id": article_id,
                        "full_id": full_id,
                        "law_title": law_title,
                        "title": article_title,
                        "text": embed_text_value,
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
# Hugging Face local model helpers
# =========================================================
def _is_valid_hf_model_dir(model_dir: Path) -> bool:
    if not model_dir.exists() or not model_dir.is_dir():
        return False

    config_ok = (model_dir / "config.json").exists()
    weight_ok = (
        (model_dir / "model.safetensors").exists()
        or (model_dir / "pytorch_model.bin").exists()
        or (model_dir / "pytorch_model.bin.index.json").exists()
        or any(model_dir.glob("*.safetensors"))
    )
    return config_ok and weight_ok


def _ensure_hf_repo_local(
    repo_id: str,
    local_dir: str,
    repo_type: str = "model",
) -> str:
    model_path = Path(local_dir).expanduser().resolve()

    if _is_valid_hf_model_dir(model_path):
        print(f"[INFO][HF] Using cached local model: {model_path}")
    else:
        print(f"[INFO][HF] Model chưa có hoặc chưa đầy đủ tại: {model_path}")
        print(f"[INFO][HF] Downloading repo '{repo_id}' to '{model_path}' ...")

        model_path.mkdir(parents=True, exist_ok=True)

        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            local_dir=str(model_path),
            local_dir_use_symlinks=False,
        )

        if not _is_valid_hf_model_dir(model_path):
            raise RuntimeError(
                f"Đã tải repo '{repo_id}' nhưng thư mục '{model_path}' vẫn không hợp lệ."
            )

        print(f"[OK][HF] Model ready at: {model_path}")

    if repo_id == config.cradio_repo:
        required_code = [
            "config.json",
            "hf_model.py",
            "radio_model.py",
            "dual_hybrid_vit.py",
        ]
        missing = [f for f in required_code if not (model_path / f).exists()]
        if missing:
            raise RuntimeError(
                f"Repo '{repo_id}' tại '{model_path}' còn thiếu file: {missing}"
            )

    return str(model_path)


# =========================================================
# Optional image models
# =========================================================
@lru_cache(maxsize=1)
def get_c_radio():
    last_error = None

    for attempt in range(1, 3):
        try:
            auto_fix_local_model_dir(config.cradio_local_dir)
            auto_clear_hf_dynamic_cache("C-RADIOv2-B")

            model_dir = _ensure_hf_repo_local(
                repo_id=config.cradio_repo,
                local_dir=config.cradio_local_dir,
                repo_type="model",
            )

            dyn_dir = ensure_cradio_dynamic_module(model_dir, "C-RADIOv2-B")

            print(f"[INFO][HF] Loading C-RADIO from local dir: {model_dir}")
            print(f"[INFO][HF] Dynamic module dir: {dyn_dir}")

            model = AutoModel.from_pretrained(
                model_dir,
                trust_remote_code=True,
                local_files_only=True,
            )

            model = model.to(DEVICE)
            model.eval()
            print("[OK][HF] C-RADIO loaded successfully")
            return model

        except Exception as e:
            last_error = e
            print(f"[WARN][HF] get_c_radio attempt {attempt} failed: {e}")

            try:
                model_dir = Path(config.cradio_local_dir).expanduser().resolve()
                if model_dir.exists():
                    print(f"[INFO][HF] Removing local broken model dir: {model_dir}")
                    shutil.rmtree(model_dir, ignore_errors=True)
            except Exception as cleanup_err:
                print(f"[WARN][HF] Cannot remove local model dir: {cleanup_err}")

            try:
                auto_clear_hf_dynamic_cache("C-RADIOv2-B")
            except Exception as cleanup_err:
                print(f"[WARN][HF] Cannot clear transformers cache: {cleanup_err}")

    raise RuntimeError(f"Không thể load C-RADIO sau 2 lần thử. Last error: {last_error}")
@lru_cache(maxsize=1)
def get_owl():
    if not has_module("scipy"):
        raise RuntimeError(
            "Thiếu scipy cho OWLv2. Cài bằng: pip install scipy"
        )

    model_dir = _ensure_hf_repo_local(
        repo_id=config.owlv2_repo,
        local_dir=config.owlv2_local_dir,
        repo_type="model",
    )

    processor = Owlv2Processor.from_pretrained(
        model_dir,
        local_files_only=True,
    )
    model = Owlv2ForObjectDetection.from_pretrained(
        model_dir,
        local_files_only=True,
    )
    model = model.to(DEVICE)
    model.eval()
    return processor, model


def preprocess_c_radio_image(image: Image.Image) -> torch.Tensor:
    x = CRADIO_TRANSFORM(image).unsqueeze(0)
    return x.to(DEVICE)


def load_image(image_path: Optional[str]) -> Optional[Image.Image]:
    if not image_path or not os.path.exists(image_path):
        return None
    try:
        return Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[WARN][IMAGE] Không mở được ảnh: {image_path} | error={e}")
        return None


def _extract_cradio_vector(outputs: Any) -> List[float]:
    """
    Robust extractor for C-RADIO outputs.
    Supports:
    - torch.Tensor
    - HF style outputs with pooler_output / last_hidden_state
    - C-RADIO RadioOutput with summary / features
    - dict / tuple / list fallbacks
    """
    tensor = None

    if isinstance(outputs, torch.Tensor):
        tensor = outputs
    elif hasattr(outputs, "summary") and getattr(outputs, "summary") is not None:
        tensor = getattr(outputs, "summary")
    elif hasattr(outputs, "features") and getattr(outputs, "features") is not None:
        feats = getattr(outputs, "features")
        if isinstance(feats, torch.Tensor):
            tensor = feats.mean(dim=1) if feats.ndim >= 3 else feats
    elif hasattr(outputs, "pooler_output") and getattr(outputs, "pooler_output") is not None:
        tensor = getattr(outputs, "pooler_output")
    elif hasattr(outputs, "last_hidden_state") and getattr(outputs, "last_hidden_state") is not None:
        lhs = getattr(outputs, "last_hidden_state")
        if isinstance(lhs, torch.Tensor):
            tensor = lhs.mean(dim=1) if lhs.ndim >= 3 else lhs
    elif isinstance(outputs, dict):
        for key in ["summary", "pooler_output", "last_hidden_state", "features"]:
            val = outputs.get(key)
            if val is None:
                continue
            if isinstance(val, torch.Tensor):
                tensor = val.mean(dim=1) if (key in ["last_hidden_state", "features"] and val.ndim >= 3) else val
                break
    elif isinstance(outputs, (tuple, list)) and len(outputs) > 0:
        first = outputs[0]
        if isinstance(first, torch.Tensor):
            tensor = first.mean(dim=1) if first.ndim >= 3 else first

    if tensor is None:
        attrs = [a for a in dir(outputs) if not a.startswith("__")][:30]
        raise RuntimeError(
            f"Không extract được vector từ C-RADIO output: type={type(outputs)}, attrs={attrs}"
        )

    if not isinstance(tensor, torch.Tensor):
        raise RuntimeError(f"C-RADIO output không phải tensor sau extract: {type(tensor)}")

    vec = tensor.squeeze(0).detach().cpu().float().flatten().tolist()
    if len(vec) < IMAGE_DIM:
        vec = vec + [0.0] * (IMAGE_DIM - len(vec))
    elif len(vec) > IMAGE_DIM:
        vec = vec[:IMAGE_DIM]
    return l2_normalize(vec)



def embed_image(image_path: Optional[str]) -> List[float]:
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

        return _extract_cradio_vector(outputs)

    except Exception as e:
        print(f"[WARN][IMAGE] C-RADIO failed permanently for {image_path}: {e}")
        return zero_vec(IMAGE_DIM)

def parse_question_intent(item: Dict[str, Any]) -> Dict[str, Any]:
    question = str(item.get("question", "") or "")
    q_norm = normalize_vi_text(question)  # ✅ dùng có dấu

    choice_map = build_choice_map(item)
    choice_norms = {k: normalize_vi_text(v) for k, v in choice_map.items()}

    topic = "generic_sign"

    if any(x in q_norm for x in [
        "làn đường", "bên tay trái", "bên trái", "bên phải",
        "phân làn", "dành riêng cho"
    ]):
        topic = "lane_assignment"

    elif any(x in q_norm for x in [
        "đỗ xe", "dừng xe", "ngày chẵn", "ngày lẻ",
        "cấm đỗ", "cấm dừng"
    ]):
        topic = "parking_restriction"

    elif any(x in q_norm for x in [
        "tốc độ", "km/h", "vận tốc"
    ]):
        topic = "speed_limit"

    elif any(x in q_norm for x in [
        "nhường đường", "ưu tiên", "đường ưu tiên"
    ]):
        topic = "priority"

    elif any(x in q_norm for x in [
        "hướng đi", "rẽ trái", "rẽ phải",
        "đi thẳng", "quay đầu"
    ]):
        topic = "direction_mandate"

    # =========================
    # VEHICLES
    # =========================
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

    # =========================
    # SIDE
    # =========================
    side = None
    if any(x in q_norm for x in [
        "bên tay trái", "bên trái", "phía trái", "làn trái"
    ]):
        side = "left"

    elif any(x in q_norm for x in [
        "bên tay phải", "bên phải", "phía phải", "làn phải"
    ]):
        side = "right"

    # =========================
    # FEATURES
    # =========================
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

def get_question_guided_owl_queries(item: Dict[str, Any]) -> List[str]:
    intent = parse_question_intent(item)
    topic = intent.get("topic")
    queries: List[str] = ["traffic sign"]

    if topic == "parking_restriction":
        queries.extend([
            "no parking sign",
            "no stopping and parking sign",
            "parking sign",
            "red slash sign",
            "even day sign",
            "odd day sign",
            "prohibitory sign",
            "regulatory sign",
        ])
    elif topic == "lane_assignment":
        queries.extend([
            "lane assignment sign",
            "lane control sign",
            "car lane sign",
            "bus lane sign",
            "truck lane sign",
            "motorcycle lane sign",
            "direction arrow sign",
            "blue circle sign",
            "regulatory sign",
        ])
    elif topic == "direction_mandate":
        queries.extend([
            "direction arrow sign",
            "mandatory turn sign",
            "blue circle sign",
            "regulatory sign",
        ])
    elif topic == "priority":
        queries.extend([
            "priority sign",
            "warning sign",
            "regulatory sign",
        ])
    elif topic == "speed_limit":
        queries.extend([
            "speed limit sign",
            "minimum speed sign",
            "regulatory sign",
        ])
    else:
        queries.extend(list(config.owl_queries))

    return unique_keep_order(queries)


def filter_detected_labels_by_intent(labels: List[str], item: Dict[str, Any]) -> List[str]:
    if not labels:
        return []

    topic = parse_question_intent(item).get("topic")
    keyword_groups = {
        "parking_restriction": ["parking", "stopping", "odd day", "even day", "red slash", "prohibitory", "regulatory"],
        "lane_assignment": ["lane", "direction arrow", "blue circle", "regulatory", "car lane", "bus lane", "truck lane", "motorcycle lane"],
        "direction_mandate": ["direction", "arrow", "blue circle", "regulatory"],
        "priority": ["priority", "warning", "regulatory"],
        "speed_limit": ["speed", "regulatory"],
    }

    keywords = keyword_groups.get(topic, [])
    filtered = [label for label in labels if (not keywords) or any(k in label.lower() for k in keywords)]
    if filtered:
        return unique_keep_order(filtered)

    fallbacks = [x for x in labels if x.lower() in {"traffic sign", "regulatory sign", "warning sign", "priority sign", "prohibitory sign"}]
    return unique_keep_order(fallbacks[:3] or labels[:3])


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


def translate_detected_labels_to_vi(labels: List[str]) -> List[str]:
    translated: List[str] = []
    seen = set()
    for label in labels or []:
        key = str(label or "").strip()
        vi = OWL_LABEL_VI_MAP.get(key, key)
        if vi and vi not in seen:
            seen.add(vi)
            translated.append(vi)
    return translated


def detect_objects(
    image_path: Optional[str],
    threshold: float = 0.10,
    text_queries: Optional[List[str]] = None,
) -> List[str]:
    if not config.use_image_models:
        return []

    if not has_module("scipy"):
        print("[WARN][OBJECT] scipy chưa được cài -> bỏ qua object detection. Cài bằng: pip install scipy")
        return []

    image = load_image(image_path)
    if image is None:
        return []

    try:
        processor, model = get_owl()
        text_queries = list(text_queries or config.owl_queries)
        inputs = processor(text=text_queries, images=image, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]], device=DEVICE)
        results = processor.post_process_grounded_object_detection(outputs=outputs, threshold=threshold, target_sizes=target_sizes)

        labels: List[str] = []
        if results:
            res = results[0]
            for label_idx in res["labels"].detach().cpu().tolist():
                if 0 <= label_idx < len(text_queries):
                    labels.append(text_queries[label_idx])

        return unique_keep_order(labels)

    except Exception as e:
        print(f"[WARN][OBJECT] detect_objects failed for {image_path}: {e}")
        return []


def embed_objects(
    image_path: Optional[str],
    item: Optional[Dict[str, Any]] = None,
    labels: Optional[List[str]] = None,
) -> List[float]:
    labels = list(labels or [])
    if not labels:
        guided_queries = get_question_guided_owl_queries(item or {}) if item else None
        labels_en = detect_objects(image_path, text_queries=guided_queries)
        if item:
            labels_en = filter_detected_labels_by_intent(labels_en, item)
        labels = translate_detected_labels_to_vi(labels_en)
    else:
        labels = translate_detected_labels_to_vi(labels)

    if not labels:
        return zero_vec(OBJECT_DIM)

    try:
        object_text = "Đặc trưng nhận diện được trong ảnh: " + ", ".join(labels)
        vec = embed_text_passage(object_text)
        if len(vec) < OBJECT_DIM:
            vec = vec + [0.0] * (OBJECT_DIM - len(vec))
        elif len(vec) > OBJECT_DIM:
            vec = vec[:OBJECT_DIM]
        return l2_normalize(vec)
    except Exception as e:
        print(f"[WARN][OBJECT] embed_objects failed for {image_path}: {e}")
        return zero_vec(OBJECT_DIM)


def build_image_description(
    image_path: Optional[str],
    item: Optional[Dict[str, Any]] = None,
    labels: Optional[List[str]] = None,
) -> str:
    labels = list(labels or [])
    if not labels:
        guided_queries = get_question_guided_owl_queries(item or {}) if item else None
        labels_en = detect_objects(image_path, text_queries=guided_queries)
        if item:
            labels_en = filter_detected_labels_by_intent(labels_en, item)
        labels = translate_detected_labels_to_vi(labels_en)
    else:
        labels = translate_detected_labels_to_vi(labels)

    if not labels:
        return "Không nhận diện được đặc trưng phù hợp với câu hỏi từ module object detection."

    if item:
        intent = parse_question_intent(item)
        return f"Các đối tượng/đặc trưng phù hợp với câu hỏi (topic={intent.get('topic')}): " + ", ".join(labels)
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

    intent = parse_question_intent(item)
    lines.append(f"Topic suy luận: {intent.get('topic')}")
    if intent.get("side"):
        lines.append(f"Vị trí/làn quan tâm: {intent.get('side')}")
    if intent.get("vehicles"):
        lines.append("Loại phương tiện liên quan: " + ", ".join(intent.get("vehicles", [])))
    if intent.get("features"):
        lines.append("Đặc trưng cần ưu tiên: " + ", ".join(intent.get("features", [])))

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
# Law reranking
# =========================================================
def score_law_against_choices(item: Dict[str, Any], payload: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
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
        "cấm dừng xe và đỗ xe",
        "cấm đỗ xe",
        "cấm đỗ xe vào ngày lẻ",
        "cấm đỗ xe vào ngày chẵn",
        "ngày lẻ",
        "ngày chẵn",
        "nơi đỗ xe",
        "chú ý xe đỗ",
    ]

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

    debug = {
        "matched_choices": matched_choices,
        "matched_phrases": matched_phrases,
        "sign_codes": codes,
    }
    return score_boost, debug


def score_yes_no_law_hit(item: Dict[str, Any], payload: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """
    Chấm law hit cho câu Đúng/Sai.
    Mục tiêu:
    - lấy term chính từ câu hỏi
    - tìm overlap trong luật
    - suy ra support_true / support_false từ một số pattern phổ biến
    """
    question = str(item.get("question", "") or "")
    title = str(payload.get("title", "") or "")
    text = str(payload.get("text", "") or "")
    full_text = str(payload.get("full_text", "") or "")

    q = normalize_vi_text(question)
    doc = normalize_vi_text(" ".join([title, text, full_text]))

    debug: Dict[str, Any] = {
        "question_norm": q,
        "matched_terms": [],
        "support_true": 0.0,
        "support_false": 0.0,
        "predicted_label": None,
        "overlap_score": 0.0,
        "sign_codes": extract_sign_codes(str(title)),
        "reason": "yesno_scoring",
    }

    if not q or not doc:
        debug["reason"] = "empty_question_or_doc"
        return 0.0, debug

    candidate_terms: List[str] = []

    # Trích cụm trong ngoặc kép nếu có
    for pat in [r'"([^"]+)"', r"“([^”]+)”", r"'([^']+)'"]:
        for m in re.findall(pat, question):
            term = normalize_vi_text(m)
            if term and len(term) >= 3:
                candidate_terms.append(term)

    hand_terms = [
        "giữ khoảng cách an toàn",
        "chữ màu vàng",
        "nền đen",
        "nền vàng",
        "chữ đen",
        "chữ trắng",
        "biển chỉ dẫn",
        "biển cảnh báo",
        "biển báo cấm",
        "biển hiệu lệnh",
        "biển viết bằng chữ",
    ]
    for term in hand_terms:
        if term in q:
            candidate_terms.append(term)

    candidate_terms = list(dict.fromkeys(candidate_terms))

    overlap_score = 0.0
    matched_terms: List[str] = []
    for term in candidate_terms:
        if term in doc:
            overlap_score += 0.12
            matched_terms.append(term)

    support_true = 0.0
    support_false = 0.0

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

    if asks_safe_distance and law_mentions_safe_distance:
        support_true += 0.18

    if asks_safe_distance and law_mentions_guide_sign:
        support_true += 0.10

    if has_yellow_bg_claim and has_black_text_claim and doc_has_yellow_bg_black_text:
        support_true += 0.50
    if has_black_bg_claim and has_yellow_text_claim and doc_has_black_bg_yellow_text:
        support_true += 0.50

    if has_black_bg_claim and has_yellow_text_claim and doc_has_yellow_bg_black_text:
        support_false += 0.70
    if has_yellow_bg_claim and has_black_text_claim and doc_has_black_bg_yellow_text:
        support_false += 0.70

    if (has_yellow_text_claim or has_black_bg_claim or has_yellow_bg_claim or has_black_text_claim) and doc_has_blue_bg_white_text:
        support_false += 0.35

    if (has_yellow_text_claim or has_black_bg_claim or has_yellow_bg_claim or has_black_text_claim) and doc_has_red_bg_white_text:
        support_false += 0.20

    # Một số hỗ trợ đúng trực tiếp cho claim nền/chữ
    if has_white_text_claim and "chu trang" in doc:
        support_true += 0.18

    final_support = overlap_score + max(support_true, support_false)

    debug["matched_terms"] = matched_terms
    debug["support_true"] = round(support_true, 6)
    debug["support_false"] = round(support_false, 6)
    debug["predicted_label"] = "ĐÚNG" if support_true > support_false else "SAI" if support_false > support_true else None
    debug["overlap_score"] = round(overlap_score, 6)

    return final_support, debug


def rerank_law_hits(item: Dict[str, Any], law_hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    reranked: List[Dict[str, Any]] = []
    yes_no_mode = is_yes_no_question(item.get("question_type", ""))

    for hit in law_hits:
        payload = hit.get("payload", {})
        base_score = float(hit.get("score", 0.0))

        if yes_no_mode:
            boost, debug = score_yes_no_law_hit(item, payload)
        else:
            boost, debug = score_law_against_choices(item, payload)

        new_hit = dict(hit)
        new_hit["base_score"] = base_score
        new_hit["choice_boost"] = boost
        new_hit["score"] = base_score + boost
        new_hit["debug"] = debug
        reranked.append(new_hit)

    reranked.sort(key=lambda x: x["score"], reverse=True)
    return reranked


def score_choices_from_laws(
    item: Dict[str, Any],
    retrieved_laws: List[Dict[str, Any]],
    image_description: str = "",
) -> Dict[str, float]:

    choice_map = build_choice_map(item)
    scores: Dict[str, float] = {label: 0.0 for label in choice_map}

    if not choice_map or not retrieved_laws:
        return scores

    norm_choice_map = {label: normalize_vi_text(text) for label, text in choice_map.items()}
    choice_sem_map = {label: parse_choice_semantics(text) for label, text in choice_map.items()}

    image_desc_norm = normalize_vi_text(image_description)

    # =====================================================
    # 1. IMAGE SIGNAL (rất mạnh)
    # =====================================================
    if "ngày chẵn" in image_desc_norm:
        for label, text in norm_choice_map.items():
            if "ngày chẵn" in text:
                scores[label] += 3.0

    if "ngày lẻ" in image_desc_norm:
        for label, text in norm_choice_map.items():
            if "ngày lẻ" in text:
                scores[label] += 3.0

    # =====================================================
    # 2. LAW-BASED SCORING
    # =====================================================
    for rank, hit in enumerate(retrieved_laws[:5], 1):
        payload = hit.get("payload", {})
        debug = hit.get("debug", {}) or {}

        rank_weight = 1.0 / rank
        fused_score = float(hit.get("score", 0.0))

        title_norm = normalize_vi_text(payload.get("title", ""))
        text_norm = normalize_vi_text(payload.get("text", "") or payload.get("full_text", ""))
        combined_norm = f"{title_norm} {text_norm}"

        matched_choices = set(debug.get("matched_choices", []) or [])

        # =================================================
        # 2.1 TEXT MATCH (yếu)
        # =================================================
        for label, choice_norm in norm_choice_map.items():
            if choice_norm and choice_norm in combined_norm:
                scores[label] += 0.3 * rank_weight

        # =================================================
        # 2.2 MATCH từ rerank debug
        # =================================================
        for label in matched_choices:
            if label in scores:
                scores[label] += 0.25 * rank_weight

        # =================================================
        # 2.3 SEMANTIC MATCH (QUAN TRỌNG)
        # =================================================
        variants = payload.get("semantics", {}).get("variants", [])

        for v in variants:
            constraints = v.get("constraints", {})

            # ---- ngày chẵn ----
            if constraints.get("day_parity") == "even":
                for label, text in norm_choice_map.items():
                    if "ngày chẵn" in text:
                        scores[label] += 1.5 * rank_weight
            # ---- ngày lẻ ----
            if constraints.get("day_parity") == "odd":
                for label, text in norm_choice_map.items():
                    if "ngày lẻ" in text:
                        scores[label] += 1.5 * rank_weight

        # =================================================
        # 2.4 BASE INTENT MATCH
        # =================================================
        base_intents = payload.get("semantics", {}).get("base_intents", [])

        for label, sem in choice_sem_map.items():
            if sem.get("base_intent") in base_intents:
                scores[label] += 0.5 * rank_weight

        # =================================================
        # 2.5 BOOST theo fused_score
        # =================================================
        for label in scores:
            scores[label] += min(fused_score, 2.0) * 0.1 * rank_weight

    # =====================================================
    # 3. PENALTY LOGIC (QUAN TRỌNG)
    # =====================================================

    # Nếu có lựa chọn ngày lẻ/chẵn → giảm điểm generic
    has_day_variant = any(
        "ngày chẵn" in normalize_vi_text(c) or "ngày lẻ" in normalize_vi_text(c)
        for c in choice_map.values()
    )

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
        support_true = 0.0
        support_false = 0.0

        for hit in retrieved_laws[:3]:
            dbg = hit.get("debug", {}) or {}
            fused_score = float(hit.get("score", 0.0))
            st = float(dbg.get("support_true", 0.0))
            sf = float(dbg.get("support_false", 0.0))
            overlap = float(dbg.get("overlap_score", 0.0))

            support_true += fused_score * max(st + 0.20 * overlap, 0.0)
            support_false += fused_score * max(sf + 0.20 * overlap, 0.0)

        debug = {
            "reason": "yesno_aggregate",
            "support_true": round(support_true, 6),
            "support_false": round(support_false, 6),
        }

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

    debug = {
        "choice_scores": choice_scores,
        "ranked_choice_scores": ranked,
        "top_gap": top_gap,
        "choice_gap": choice_gap,
        "exact_title_match": exact_title_match,
        "top1_title": top1.get("payload", {}).get("title", ""),
    }

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


# =========================================================
# Indexing
# =========================================================
# Indexing
# =========================================================
def index_examples(client: QdrantClient, dataset: QaDataset) -> None:
    ensure_collection(client, config.collection_examples)
    success_count = 0
    fail_count = 0

    for idx, item in enumerate(dataset):
        print(f"[BEGIN][EXAMPLE] {idx}")
        image_path = get_qa_image_path(config.train_image_dir, item)
        query_text = build_query_text(item)

        try:
            text_vec = embed_text_passage(query_text)
            print(f"[text_vec][EXAMPLE] {idx}: {text_vec}")

            image_vec = embed_image(image_path)
            print(f"[image_vec][EXAMPLE] {idx}: {image_vec}")

            object_labels_en = filter_detected_labels_by_intent(detect_objects(image_path, text_queries=get_question_guided_owl_queries(item)), item)
            object_labels_vi = translate_detected_labels_to_vi(object_labels_en)
            object_vec = embed_objects(image_path, item=item, labels=object_labels_vi)
            print(f"[object_vec][EXAMPLE] {idx}: {object_vec}")

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

            point = PointStruct(
                id=idx,
                vector={
                    "text": text_vec,
                    "image": image_vec,
                    "objects": object_vec,
                },
                payload=payload,
            )

            client.upsert(
                collection_name=config.collection_examples,
                points=[point],
            )

            success_count += 1
            print(f"[OK][INDEX_EXAMPLE] idx={idx} id={item.get('id')} upserted")

        except Exception as e:
            fail_count += 1
            print(f"[ERROR][INDEX_EXAMPLE] idx={idx} id={item.get('id')} error={e}")

    print(
        f"[DONE][EXAMPLE] Indexed success={success_count} | failed={fail_count}"
    )


def normalize_vi_text(s: str) -> str:
    s = str(s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def unique_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if not x:
            continue
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def extract_sign_codes_from_text(text: str) -> List[str]:
    if not text:
        return []
    codes = re.findall(r"\b([A-Z]\.\d{1,3}[a-z]?)\b", text)
    return unique_keep_order(codes)


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

    return {
        "applies_to": unique_keep_order(applies_to)
    }


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
        chunk = raw[start:end].strip(" ;,.")
        out[code] = chunk

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
        code_norm = code.lower()

        chunk = variant_chunks.get(code, "")
        chunk_norm = normalize_vi_text(chunk)

        if chunk_norm:
            constraints.update(infer_variant_constraints_from_text(code, chunk_norm))
        else:
            if code_norm.endswith("b") and "ngày lẻ" in full_norm and "ngày chẵn" not in full_norm:
                constraints["day_parity"] = "odd"
            elif code_norm.endswith("c") and "ngày chẵn" in full_norm and "ngày lẻ" not in full_norm:
                constraints["day_parity"] = "even"

        variants.append(
            {
                "variant_id": code,
                "constraints": constraints,
                "text": chunk if chunk else None,
            }
        )

    seen = set()
    deduped: List[Dict[str, Any]] = []
    for v in variants:
        vid = v["variant_id"]
        if vid in seen:
            continue
        seen.add(vid)
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

    semantics = {
        "base_intents": infer_base_intents(text_norm),
        "global_constraints": infer_global_constraints(text_norm),
        "variants": build_variants(sign_codes, title=title, text=full_text or text),
        "sign_codes": sign_codes,
        "applies_to": infer_entities(text_norm).get("applies_to", []),
    }
    return semantics


def truncate_for_embedding(text: str, max_chars: int = 3500) -> str:
    text = str(text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars]

def build_law_embedding_text(item: Dict[str, Any], semantics: Dict[str, Any]) -> str:
    law_id = str(item.get("law_id", "") or "")
    article_id = str(item.get("article_id", "") or "")
    law_title = str(item.get("law_title", "") or "")
    title = str(item.get("title", "") or "")
    text = str(item.get("text", "") or "")
    full_text = str(item.get("full_text", "") or "")

    sign_codes = ", ".join(semantics.get("sign_codes", []))
    base_intents = ", ".join(semantics.get("base_intents", []))
    applies_to = ", ".join(semantics.get("applies_to", []))

    variant_lines: List[str] = []
    for variant in semantics.get("variants", []):
        vid = variant.get("variant_id", "")
        constraints = variant.get("constraints", {}) or {}
        if constraints:
            constraint_str = ", ".join(f"{k}={v}" for k, v in constraints.items())
            variant_lines.append(f"{vid}: {constraint_str}")
        else:
            variant_lines.append(str(vid))

    variants_text = " | ".join(variant_lines)

    parts = [
        f"Law: {law_id}",
        f"Article: {article_id}",
        f"Law title: {law_title}",
        f"Article title: {title}",
        f"Sign codes: {sign_codes}",
        f"Base intents: {base_intents}",
        f"Applies to: {applies_to}",
        f"Variants: {variants_text}",
        f"Text: {full_text or text}",
    ]
    return "\n".join([p for p in parts if p.strip()])


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
            full_text = str(item.get("full_text", "")).strip()
            law_title = str(item.get("law_title", "")).strip()
            title = str(item.get("title", "")).strip()

            semantics = build_law_semantics(item)
            embedding_text = truncate_for_embedding(build_law_embedding_text(item, semantics), 3500)
            text_vec = embed_text_passage(embedding_text)

            payload_text = full_text#[:4000] if len(full_text) > 4000 else full_text
            if not payload_text:
                payload_text = text#[:4000] if len(text) > 4000 else text

            payload = {
                "kind": "law_article",
                "id": item.get("id"),
                "law_id": law_id,
                "article_id": article_id,
                "full_id": full_id,
                "law_title": law_title,
                "title": title,
                "text": payload_text,
                "full_text": payload_text,
                "sign_codes": semantics.get("sign_codes", []),
                "image_id": item.get("image_id"),
                "semantics": semantics,
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

            print("\n" + "-" * 80)
            print(f"[DEBUG][LAW] {law_id}::{article_id}")

            print(f"[TITLE] {title}")
            print(f"[LAW TITLE] {law_title}")

            print(f"[TEXT SAMPLE]")
            print((full_text or text)[:500])

            print(f"[SIGN_CODES RAW] {item.get('sign_codes', [])}")

            print(f"[SEMANTICS]")
            print(f"  article_type={semantics.get('article_type')}")
            print(f"  base_intents={semantics.get('base_intents')}")
            print(f"  global_constraints={semantics.get('global_constraints')}")
            print(f"  applies_to={semantics.get('applies_to')}")
            print(f"  sign_codes={semantics.get('sign_codes')}")

            print(f"[VARIANTS]")
            for v in semantics.get("variants", []):
                print(f"  - {v}")

            print("-" * 80 + "\n")


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
    question_intent = parse_question_intent(item)
    text_vec = embed_text_query(build_query_text(item))

    if len(text_vec) != TEXT_DIM:
        raise RuntimeError(
            f"Embedding dim sai khi retrieve query id={item.get('id')}: expected={TEXT_DIM}, got={len(text_vec)}"
        )

    has_image = bool(image_path and os.path.exists(image_path))
    guided_queries = get_question_guided_owl_queries(item)
    raw_detected_labels = detect_objects(image_path, text_queries=guided_queries) if has_image else []
    filtered_labels = filter_detected_labels_by_intent(raw_detected_labels, item) if has_image else []
    filtered_labels_vi = translate_detected_labels_to_vi(filtered_labels) if has_image else []

    image_vec = embed_image(image_path) if has_image else zero_vec(IMAGE_DIM)
    object_vec = embed_objects(image_path, item=item, labels=filtered_labels_vi) if has_image else zero_vec(OBJECT_DIM)
    image_desc = build_image_description(image_path, item=item, labels=filtered_labels_vi) if has_image else "Không tìm thấy ảnh để phân tích."

    example_weights = (0.70, 0.20, 0.10)
    if filtered_labels_vi:
        example_weights = (0.60, 0.20, 0.20)
        if question_intent.get("topic") == "parking_restriction":
            example_weights = (0.55, 0.15, 0.30)
        elif question_intent.get("topic") == "lane_assignment":
            example_weights = (0.60, 0.15, 0.25)

    example_hits = fuse_hits(
        search_named_vector(client, config.collection_examples, "text", text_vec, config.top_k_examples * 3),
        search_named_vector(client, config.collection_examples, "image", image_vec, config.top_k_examples * 3),
        search_named_vector(client, config.collection_examples, "objects", object_vec, config.top_k_examples * 3),
        config.top_k_examples,
        weights=example_weights,
    )

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
        print(f"[DEBUG][RETRIEVE] QUESTION INTENT: {question_intent}")
        print(f"[DEBUG][RETRIEVE] IMAGE PATH: {image_path}")
        print(f"[DEBUG][RETRIEVE] HAS IMAGE: {has_image}")
        print(f"[DEBUG][RETRIEVE] GUIDED OWL QUERIES: {guided_queries}")
        print(f"[DEBUG][RETRIEVE] RAW DETECTED LABELS: {raw_detected_labels}")
        print(f"[DEBUG][RETRIEVE] FILTERED LABELS EN: {filtered_labels}")
        print(f"[DEBUG][RETRIEVE] FILTERED LABELS VI: {filtered_labels_vi}")
        print(f"[DEBUG][RETRIEVE] EXAMPLE FUSION WEIGHTS: {example_weights}")
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
            if is_yes_no_question(item.get("question_type", "")):
                print(f"     matched_terms={dbg.get('matched_terms', [])}")
                print(f"     support_true={dbg.get('support_true', 0.0)}")
                print(f"     support_false={dbg.get('support_false', 0.0)}")
                print(f"     predicted_label={dbg.get('predicted_label')}")
                print(f"     sign_codes={dbg.get('sign_codes', [])}")
            else:
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

    def summarize_visual_cues(desc: str) -> List[str]:
        d = normalize_vi_text(desc)
        cues: List[str] = []

        priority_rules = [
            ("biển ngày chẵn", "Có dấu hiệu ngày chẵn"),
            ("biển ngày lẻ", "Có dấu hiệu ngày lẻ"),
            ("biển cấm dừng xe và đỗ xe", "Có dấu hiệu cấm dừng xe và đỗ xe"),
            ("biển cấm đỗ xe", "Có dấu hiệu cấm đỗ xe"),
            ("vạch chéo đỏ", "Có vạch chéo đỏ"),
            ("biển tròn xanh", "Có biển tròn xanh"),
            ("mũi tên", "Có mũi tên chỉ hướng"),
            ("làn dành cho ô tô", "Có dấu hiệu làn dành cho ô tô"),
            ("làn dành cho xe buýt", "Có dấu hiệu làn dành cho xe buýt"),
            ("làn dành cho xe mô tô", "Có dấu hiệu làn dành cho xe mô tô"),
        ]

        for key, msg in priority_rules:
            if key in d:
                cues.append(msg)

        return cues[:4]

    def extract_variant_hints(retrieved_laws: List[Dict[str, Any]]) -> List[str]:
        hints: List[str] = []

        for hit in retrieved_laws[:3]:
            p = hit.get("payload", {})
            text = str(p.get("text", "") or p.get("full_text", ""))
            text_norm = normalize_vi_text(text)
            title = str(p.get("title", "")).strip()

            if "p.131" in normalize_vi_text(title) or "cấm đỗ xe" in normalize_vi_text(title):
                if "ngày lẻ" in text_norm and "ngày chẵn" in text_norm:
                    hints.append("Trong luật nhóm biển P.131, biến thể ngày lẻ/ngày chẵn là các biến thể khác nhau, không được gộp chung.")
                if "p.131b" in text_norm and "ngày lẻ" in text_norm:
                    hints.append("P.131b tương ứng cấm đỗ xe vào ngày lẻ.")
                if "p.131c" in text_norm and "ngày chẵn" in text_norm:
                    hints.append("P.131c tương ứng cấm đỗ xe vào ngày chẵn.")

        seen = set()
        out = []
        for h in hints:
            if h not in seen:
                seen.add(h)
                out.append(h)
        return out[:4]

    question = str(item.get("question", "")).strip()
    qtype = str(item.get("question_type", "")).strip()
    choices = normalize_choices(item.get("choices", []))
    yes_no = is_yes_no_question(qtype)
    visual_cues = summarize_visual_cues(image_description)
    variant_hints = extract_variant_hints(retrieved_laws)

    lines: List[str] = []
    lines.append("Bạn là trợ lý giải bài MLQA-TSR về luật giao thông Việt Nam.")
    lines.append("Ưu tiên tuyệt đối điều luật và đặc trưng trực quan then chốt của ảnh.")
    lines.append(
        "Nếu luật mô tả một nhóm biển có nhiều biến thể (a/b/c...), phải chọn đúng biến thể cụ thể; "
        "không được chọn tên gọi chung nếu đáp án có phương án chi tiết hơn."
    )

    if yes_no:
        lines.append("Nhiệm vụ: xác định phát biểu là ĐÚNG hay SAI.")
        lines.append("Chỉ trả lời bằng đúng một từ: ĐÚNG hoặc SAI.")
    else:
        lines.append("Nhiệm vụ: chọn đúng một đáp án trong các lựa chọn.")
        lines.append("Chỉ trả lời bằng đúng một chữ cái: A, B, C hoặc D.")
        lines.append(
            "Quy tắc bắt buộc: nếu ảnh có tín hiệu đặc thù như 'ngày chẵn' hoặc 'ngày lẻ', "
            "phải ưu tiên đáp án chi tiết tương ứng, không chọn đáp án tổng quát."
        )

    lines.append("\n# DẤU HIỆU TRỰC QUAN CHÍNH")
    if visual_cues:
        for cue in visual_cues:
            lines.append(f"- {cue}")
    else:
        lines.append(f"- {image_description}")

    if variant_hints:
        lines.append("\n# GỢI Ý BIẾN THỂ QUAN TRỌNG")
        for hint in variant_hints:
            lines.append(f"- {hint}")

    if retrieved_laws:
        lines.append("\n# ĐIỀU LUẬT THAM KHẢO")
        for i, hit in enumerate(retrieved_laws[:5], 1):
            p = hit.get("payload", {})
            title = str(p.get("title", "")).strip()
            law_id = str(p.get("law_id", "")).strip()
            article_id = str(p.get("article_id", "")).strip()
            text = compact_text(p.get("text", "") or p.get("full_text", ""), 900)
            meta = " | ".join([x for x in [law_id, article_id, title] if x])

            lines.append(f"[LAW {i}] {meta}")
            if text:
                lines.append(text)

            dbg = hit.get("debug", {})
            matched_phrases = dbg.get("matched_phrases", []) or []
            if matched_phrases:
                lines.append(f"Gợi ý khớp lựa chọn: {', '.join(matched_phrases[:5])}")
            lines.append("")

    if retrieved_examples:
        lines.append("\n# VÍ DỤ THAM KHẢO")
        for i, hit in enumerate(retrieved_examples[:2], 1):
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

    lines.append("\n# QUY TẮC SUY LUẬN BẮT BUỘC")
    lines.append("- Đối chiếu từng lựa chọn với điều luật.")
    lines.append("- Nếu có đáp án tổng quát và đáp án cụ thể hơn, ưu tiên đáp án cụ thể đúng với đặc trưng ảnh.")
    lines.append("- Nếu có dấu hiệu 'ngày chẵn', không chọn đáp án 'Cấm' chung chung.")
    lines.append("- Nếu có dấu hiệu 'ngày lẻ', không chọn đáp án 'Cấm' chung chung.")
    lines.append("- Chỉ chọn đáp án tổng quát khi không có dấu hiệu nào đủ để xác định biến thể cụ thể.")

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


def compute_rule_confidence(
    item: Dict[str, Any],
    rule_prediction: Optional[str],
    rule_debug: Dict[str, Any],
    retrieved_laws: List[Dict[str, Any]],
) -> float:
    if not rule_prediction:
        return 0.0

    debug = rule_debug or {}
    reason = str(debug.get("reason", "") or "")
    if reason in {"not_applicable", "no_force", "no_choice_scores", "weak_gap"}:
        return 0.0

    conf = 0.0
    yes_no = is_yes_no_question(item.get("question_type", ""))

    if yes_no:
        support_true = float(debug.get("support_true", 0.0) or 0.0)
        support_false = float(debug.get("support_false", 0.0) or 0.0)
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


def compute_llm_confidence(
    item: Dict[str, Any],
    llm_prediction: Optional[str],
    raw_output: str,
    retrieved_laws: List[Dict[str, Any]],
) -> float:
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


def fuse_predictions(
    item: Dict[str, Any],
    rule_prediction: Optional[str],
    llm_prediction: Optional[str],
    rule_confidence: float,
    llm_confidence: float,
) -> Tuple[str, str, Dict[str, float], float]:
    yes_no = is_yes_no_question(item.get("question_type", ""))
    if yes_no:
        labels = ["ĐÚNG", "SAI"]
        fallback = "SAI"
    else:
        choice_map = build_choice_map(item)
        labels = list(choice_map.keys()) or ["A", "B", "C", "D"]
        fallback = labels[0]

    scores = {label: 0.0 for label in labels}

    if rule_prediction in scores and rule_confidence > 0.0:
        scores[rule_prediction] += rule_confidence

    if llm_prediction in scores and llm_confidence > 0.0:
        scores[llm_prediction] += llm_confidence

    if rule_prediction and llm_prediction and rule_prediction == llm_prediction and rule_prediction in scores:
        scores[rule_prediction] += 0.05

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_label, best_score = ranked[0] if ranked else (fallback, 0.0)
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = best_score - second_score

    decision_source = "llm"
    if rule_prediction and rule_confidence >= 0.80 and llm_confidence <0.90:
        best_label = rule_prediction
        decision_source = "rule_strong"
    elif rule_prediction and llm_prediction and rule_prediction == llm_prediction and rule_confidence >= 0.50:
        best_label = rule_prediction
        decision_source = "rule_llm_agree"
    elif best_score > 0.0:
        if rule_prediction == best_label and rule_confidence > llm_confidence:
            decision_source = "fusion_rule"
        elif llm_prediction == best_label:
            decision_source = "fusion_llm"
        else:
            decision_source = "fusion"
    elif llm_prediction:
        best_label = llm_prediction
        decision_source = "llm_fallback"
    elif rule_prediction:
        best_label = rule_prediction
        decision_source = "rule_fallback"
    else:
        best_label = fallback
        decision_source = "default_fallback"

    return best_label, decision_source, scores, margin


# =========================================================
# Evaluation run
# =========================================================
def run_eval(client: QdrantClient, dataset: QaDataset) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    with open(config.output_file, "w", encoding="utf-8") as f:
        f.write("[\n")

        for idx, item in enumerate(dataset, 1):
            print("=" * 100)
            print(f"[EVAL] {idx}/{len(dataset.items)} | id={item.get('id')} | image_id={item.get('image_id')}")

            retrieved_examples, retrieved_laws, image_description = retrieve_examples_and_laws(client, item)
            item["image_description"] = image_description
            rule_prediction, rule_debug = choose_by_law_priority(item, retrieved_laws)
            prompt = build_prompt(item, retrieved_examples, retrieved_laws, image_description)
            llm_prediction, raw_output = call_llm(prompt, question_type=item.get("question_type", ""))

            rule_confidence = compute_rule_confidence(
                item=item,
                rule_prediction=rule_prediction,
                rule_debug=rule_debug,
                retrieved_laws=retrieved_laws,
            )
            llm_confidence = compute_llm_confidence(
                item=item,
                llm_prediction=llm_prediction,
                raw_output=raw_output,
                retrieved_laws=retrieved_laws,
            )
            prediction, decision_source, fused_scores, fused_margin = fuse_predictions(
                item=item,
                rule_prediction=rule_prediction,
                llm_prediction=llm_prediction,
                rule_confidence=rule_confidence,
                llm_confidence=llm_confidence,
            )

            if config.debug_retrieval:
                print(f"[DEBUG][PROMPT]\n{prompt}")
                print(
                    f"[DEBUG][DECISION] rule_prediction={rule_prediction} | llm_prediction={llm_prediction} | "
                    f"rule_conf={rule_confidence:.4f} | llm_conf={llm_confidence:.4f} | "
                    f"final={prediction} | source={decision_source} | margin={fused_margin:.4f}"
                )
                print(f"[DEBUG][DECISION] fused_scores={fused_scores}")
                print(f"[DEBUG][DECISION] rule_debug={rule_debug}")

            result = {
                "id": item.get("id"),
                "image_id": item.get("image_id"),
                "question_type": item.get("question_type"),
                "prediction": prediction,
                "llm_prediction": llm_prediction,
                "rule_prediction": rule_prediction,
                "decision_source": decision_source,
                "rule_confidence": round(rule_confidence, 6),
                "llm_confidence": round(llm_confidence, 6),
                "fused_scores": {k: round(v, 6) for k, v in fused_scores.items()},
                "fused_margin": round(fused_margin, 6),
                "rule_debug": rule_debug,
                "llm_raw_output": raw_output,
                "retrieved_example_ids": [x["payload"].get("id") for x in retrieved_examples],
                "retrieved_law_ids": [x["payload"].get("article_id") or x["payload"].get("id") for x in retrieved_laws],
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

            results.append(result)
            json.dump(result, f, ensure_ascii=False)
            if idx < len(dataset.items):
                f.write(",\n")
            else:
                f.write("\n")
            f.flush()

        f.write("]\n")

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

        if has_module("scipy"):
            print("[INFO] Loading OWLv2...")
            OWL_PROCESSOR, OWL_MODEL = get_owl()
        else:
            print("[WARN][OBJECT] scipy chưa được cài -> bỏ qua OWLv2 object detection. Cài bằng: pip install scipy")
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
    print(f"[INFO] C-RADIO repo: {config.cradio_repo}")
    print(f"[INFO] C-RADIO local dir: {config.cradio_local_dir}")
    print(f"[INFO] OWLv2 repo: {config.owlv2_repo}")
    print(f"[INFO] OWLv2 local dir: {config.owlv2_local_dir}")
    print(f"[INFO] HF cache root: {HF_CACHE_ROOT}")

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
    print(f"[DONE] Saved predictions to {config.output_file} | total={len(results)}")

if __name__ == "__main__":
    main()
