from typing import Dict, List, Tuple
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from berry.config import config
from berry.runtime import TEXT_DIM, IMAGE_DIM, OBJECT_DIM

def collection_exists(client: QdrantClient, collection_name: str) -> bool:
    return any(c.name == collection_name for c in client.get_collections().collections)

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
        return
    info = client.get_collection(collection_name)
    current = getattr(info.config.params, "vectors", None)
    mismatch = False
    current_dict = current if isinstance(current, dict) else getattr(current, "params_map", {}) or {}
    for name, vp in expected_vectors.items():
        cur = current_dict.get(name)
        if getattr(cur, "size", None) != vp.size:
            mismatch = True
            print(f"[WARN][QDRANT] Vector dim mismatch in {collection_name} | name={name} expected={vp.size} got={getattr(cur, 'size', None)}")
    if mismatch:
        if config.recreate_on_dim_mismatch:
            client.delete_collection(collection_name=collection_name)
            client.create_collection(collection_name=collection_name, vectors_config=expected_vectors)
        else:
            raise RuntimeError(f"Collection {collection_name} có dim không khớp. Bật RECREATE_ON_DIM_MISMATCH=true hoặc đổi tên collection mới.")

def search_named_vector(client: QdrantClient, collection_name: str, vector_name: str, vector: List[float], limit: int) -> List[Dict]:
    if not vector or not any(abs(v) > 1e-12 for v in vector):
        return []
    try:
        response = client.query_points(collection_name=collection_name, query=vector, using=vector_name, with_payload=True, limit=limit)
        points = response.points
    except Exception:
        points = client.search(collection_name=collection_name, query_vector=(vector_name, vector), with_payload=True, limit=limit)
    hits = []
    for p in points:
        payload = getattr(p, "payload", None) or {}
        hits.append({"id": payload.get("id") or getattr(p, "id", None), "score": float(getattr(p, "score", 0.0) or 0.0), "payload": payload})
    return hits

def fuse_hits(text_hits: List[Dict], image_hits: List[Dict], object_hits: List[Dict], limit: int, weights: Tuple[float, float, float] = (0.60, 0.25, 0.15)) -> List[Dict]:
    fused: Dict[str, Dict] = {}
    def _add(hits: List[Dict], weight: float, channel: str):
        for h in hits:
            hid = str(h.get("id"))
            fused.setdefault(hid, {"id": hid, "score": 0.0, "payload": h.get("payload", {}), "channel_scores": {}})
            fused[hid]["score"] += float(h.get("score", 0.0)) * weight
            fused[hid]["channel_scores"][channel] = float(h.get("score", 0.0))
    _add(text_hits, weights[0], "text")
    _add(image_hits, weights[1], "image")
    _add(object_hits, weights[2], "objects")
    return sorted(fused.values(), key=lambda x: x["score"], reverse=True)[:limit]
