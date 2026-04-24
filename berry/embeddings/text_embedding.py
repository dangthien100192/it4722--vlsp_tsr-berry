from typing import List
import torch
from sentence_transformers import SentenceTransformer

MODEL_NAME = "jinaai/jina-embeddings-v3"

_model = None
_device = "cuda" if torch.cuda.is_available() else "cpu"


def _load_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(
            MODEL_NAME,
            trust_remote_code=True,
            device=_device,
        )


def _embed(texts: List[str], task: str) -> List[List[float]]:
    _load_model()

    embeddings = _model.encode(
        texts,
        task=task,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    return embeddings.tolist()


def embed_text_query(text: str) -> List[float]:
    return _embed([text], task="retrieval.query")[0]


def embed_text_passage(text: str) -> List[float]:
    return _embed([text], task="retrieval.passage")[0]


def validate_embedding_dims() -> int:
    query_dim = len(embed_text_query("test dimension"))
    passage_dim = len(embed_text_passage("test dimension"))

    if query_dim != passage_dim:
        raise RuntimeError(
            f"Text embedding dim không khớp: "
            f"retrieval.query={query_dim}, retrieval.passage={passage_dim}"
        )

    return query_dim