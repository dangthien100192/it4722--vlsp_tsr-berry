from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from berry.config import config
from berry.embeddings.text_embedding import embed_text_passage
from berry.retrieval.qdrant_utils import ensure_collection
from berry.runtime import IMAGE_DIM, OBJECT_DIM
from berry.semantics import build_law_semantics
from berry.utils.math_utils import zero_vec

def truncate_for_embedding(text: str, max_chars: int = 3500) -> str:
    text = str(text or "").strip()
    return text if len(text) <= max_chars else text[:max_chars]

def build_law_embedding_text(item, semantics):
    sign_codes = ", ".join(semantics.get("sign_codes", []))
    base_intents = ", ".join(semantics.get("base_intents", []))
    applies_to = ", ".join(semantics.get("applies_to", []))
    variant_lines = []
    for variant in semantics.get("variants", []):
        vid = variant.get("variant_id", "")
        constraints = variant.get("constraints", {}) or {}
        variant_lines.append(f"{vid}: " + ", ".join(f"{k}={v}" for k, v in constraints.items()) if constraints else str(vid))
    variants_text = " | ".join(variant_lines)
    parts = [
        f"Law: {item.get('law_id', '')}",
        f"Article: {item.get('article_id', '')}",
        f"Law title: {item.get('law_title', '')}",
        f"Article title: {item.get('title', '')}",
        f"Sign codes: {sign_codes}",
        f"Base intents: {base_intents}",
        f"Applies to: {applies_to}",
        f"Variants: {variants_text}",
        f"Text: {item.get('full_text', '') or item.get('text', '')}",
    ]
    return "\n".join([p for p in parts if p.strip()])

def index_laws(client: QdrantClient, dataset) -> None:
    ensure_collection(client, config.collection_law)
    points, error_count = [], 0
    for idx, item in enumerate(dataset):
        try:
            semantics = build_law_semantics(item)
            embedding_text = truncate_for_embedding(build_law_embedding_text(item, semantics), 3500)
            text_vec = embed_text_passage(embedding_text)
            payload_text = str(item.get("full_text", "")).strip() or str(item.get("text", "")).strip()
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
                "sign_codes": semantics.get("sign_codes", []),
                "image_id": item.get("image_id"),
                "semantics": semantics,
            }
            points.append(PointStruct(id=idx, vector={"text": text_vec, "image": zero_vec(IMAGE_DIM), "objects": zero_vec(OBJECT_DIM)}, payload=payload))
        except Exception as e:
            error_count += 1
            print(f"[ERROR][LAW] idx={idx} full_id={item.get('full_id')} error={e}")
    if points:
        client.upsert(collection_name=config.collection_law, points=points)
    print(f"[SUMMARY][LAW] Indexed={len(points)} | errors={error_count}")
