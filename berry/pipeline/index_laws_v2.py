from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from berry.config import config
from berry.embeddings.object_embedding import embed_objects
from berry.runtime import IMAGE_DIM, OBJECT_DIM
from berry.embeddings.text_embedding import embed_text_passage
from berry.embeddings.image_embedding import embed_image
from berry.vision import zero_vec
from berry.retrieval.qdrant_utils import ensure_collection
from berry.pipeline.law_index_parser import parse_law_item_for_index, stable_numeric_id
from berry.utils.text_utils import truncate_for_embedding

LAW_TEXT_COLLECTION = getattr(config, "collection_law_text", "berry_law_text")
LAW_ASSET_COLLECTION = getattr(config, "collection_law_asset", "berry_law_assets")


def resolve_asset_image_path(asset: Dict[str, Any], law_image_dir: Optional[str]) -> Optional[str]:
    if not law_image_dir:
        return None

    import os
    image_name = str(asset.get("image_name", "") or "").strip()
    if not image_name:
        return None

    path = os.path.join(law_image_dir, image_name)
    return path if os.path.exists(path) else None


def build_text_embedding_input(payload: Dict[str, Any]) -> str:
    kind = payload.get("kind", "")
    law_id = payload.get("law_id", "")
    article_id = payload.get("article_id", "")
    law_title = payload.get("law_title", "")
    title = payload.get("title", "")
    text = payload.get("text", "")
    asset_ref = payload.get("asset_ref", "")
    asset_type = payload.get("asset_type", "")

    lines: List[str] = [
        f"Law ID: {law_id}",
        f"Article ID: {article_id}",
        f"Law title: {law_title}",
        f"Title: {title}",
    ]

    if kind == "law_text_chunk":
        clause_id = payload.get("clause_id", "")
        if clause_id:
            lines.append(f"Clause: {clause_id}")
        lines.append(f"Content: {text}")

    elif kind in {"law_figure", "law_table"}:
        lines.append(f"Asset type: {asset_type}")
        lines.append(f"Asset ref: {asset_ref}")
        lines.append(f"Asset content: {text}")

    return truncate_for_embedding("\n".join(lines), 3500)


def index_laws_v2(client: QdrantClient, dataset) -> None:
    ensure_collection(client, LAW_TEXT_COLLECTION)
    ensure_collection(client, LAW_ASSET_COLLECTION)

    text_success = 0
    text_fail = 0
    asset_success = 0
    asset_fail = 0
    parse_error_count = 0

    for idx, item in enumerate(dataset):
        item_full_id = str(item.get("full_id") or item.get("id") or f"idx_{idx}")

        print("=" * 100)
        print(f"[LAW_V2][ITEM_BEGIN] idx={idx} full_id={item_full_id}")

        try:
            text_chunks, asset_chunks = parse_law_item_for_index(item)
            print(
                f"[LAW_V2][PARSED] idx={idx} "
                f"text_chunks={len(text_chunks)} asset_chunks={len(asset_chunks)}"
            )
        except Exception as e:
            parse_error_count += 1
            print(f"[ERROR][LAW_V2][PARSE] idx={idx} full_id={item_full_id} error={e}")
            continue

        # =========================
        # TEXT
        # =========================
        for chunk_idx, payload in enumerate(text_chunks):
            point_id = payload.get("id")

            try:
                emb_input = build_text_embedding_input(payload)
                text_vec = embed_text_passage(emb_input)
                pid = stable_numeric_id(str(point_id))

                point = PointStruct(
                    id=pid,
                    vector={
                        "text": text_vec,
                        "image": zero_vec(IMAGE_DIM),
                        "objects": zero_vec(OBJECT_DIM),
                    },
                    payload=payload,
                )

                client.upsert(
                    collection_name=LAW_TEXT_COLLECTION,
                    points=[point],
                )

                text_success += 1

                print(
                    f"[UPSERT][TEXT] "
                    f"idx={idx} chunk={chunk_idx} "
                    f"point_id={point_id} qdrant_id={pid}"
                )

            except Exception as e:
                text_fail += 1
                print(
                    f"[ERROR][TEXT] idx={idx} chunk={chunk_idx} "
                    f"point_id={point_id} error={e}"
                )

        # =========================
        # ASSET
        # =========================
        for asset_idx, payload in enumerate(asset_chunks):
            point_id = payload.get("id")

            try:
                emb_input = build_text_embedding_input(payload)
                text_vec = embed_text_passage(emb_input)

                image_vec = zero_vec(IMAGE_DIM)
                image_path = None

                if payload.get("kind") == "law_figure":
                    image_path = resolve_asset_image_path(
                        payload,
                        getattr(config, "law_image_dir", None)
                    )
                    if image_path:
                        image_vec = embed_image(image_path)
                        payload["image_path"] = image_path
                        object_vec = embed_objects(image_path)

                pid = stable_numeric_id(str(point_id))

                point = PointStruct(
                    id=pid,
                    vector={
                        "text": text_vec,
                        "image": image_vec,
                        "objects": object_vec,
                    },
                    payload=payload,
                )

                client.upsert(
                    collection_name=LAW_ASSET_COLLECTION,
                    points=[point],
                )

                asset_success += 1

                print(
                    f"[UPSERT][ASSET] "
                    f"idx={idx} asset={asset_idx} "
                    f"point_id={point_id} kind={payload.get('kind')} "
                    f"qdrant_id={pid}"
                )

            except Exception as e:
                asset_fail += 1
                print(
                    f"[ERROR][ASSET] idx={idx} asset={asset_idx} "
                    f"point_id={point_id} error={e}"
                )

        print(f"[LAW_V2][ITEM_DONE] idx={idx} full_id={item_full_id}")

    print("=" * 100)
    print(
        f"[SUMMARY][LAW_V2] "
        f"text_success={text_success} | text_fail={text_fail} | "
        f"asset_success={asset_success} | asset_fail={asset_fail} | "
        f"parse_errors={parse_error_count}"
    )