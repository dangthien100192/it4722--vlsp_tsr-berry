from typing import Dict, List, Tuple
from qdrant_client import QdrantClient
from berry.config import config
from berry.embeddings.text_embedding import embed_text_query
from berry.embeddings.image_embedding import embed_image
from berry.retrieval.build_query import build_query_text
from berry.retrieval.qdrant_utils import fuse_hits, search_named_vector
from berry.retrieval.rerank import rerank_law_hits
from berry.runtime import TEXT_DIM, IMAGE_DIM, OBJECT_DIM
from berry.semantics import parse_question_intent
from berry.utils.image_utils import get_qa_image_path
from berry.utils.math_utils import zero_vec
from berry.embeddings.object_embedding import embed_objects
from berry.vision import build_image_description, filter_detected_labels_by_intent, get_question_guided_owl_queries, translate_detected_labels_to_vi
from berry.models.owl import detect_objects
from berry.utils.text_utils import build_choice_map, compact_text, is_yes_no_question

def retrieve_examples_and_laws(client: QdrantClient, item: Dict) -> Tuple[List[Dict], List[Dict], str]:
    image_path = get_qa_image_path(config.test_image_dir, item)
    question_intent = parse_question_intent(item)

    query_text = build_query_text(item)
    text_vec = embed_text_query(query_text)

    if len(text_vec) != TEXT_DIM:
        raise RuntimeError(
            f"Embedding dim sai khi retrieve query id={item.get('id')}: "
            f"expected={TEXT_DIM}, got={len(text_vec)}"
        )

    has_image = bool(image_path)

    raw_detected_labels = detect_objects(image_path) if has_image else []
    labels = translate_detected_labels_to_vi(raw_detected_labels) if has_image else []

    image_vec = embed_image(image_path) if has_image else zero_vec(IMAGE_DIM)
    object_vec = embed_objects(image_path, item=item, labels=labels) if has_image else zero_vec(OBJECT_DIM)

    image_desc = (
        build_image_description(image_path, item=item, labels=labels)
        if has_image
        else "Không tìm thấy ảnh để phân tích."
    )

    # =========================================================
    # 1. RETRIEVE EXAMPLES
    # =========================================================
    example_weights = (0.70, 0.20, 0.10)

    if labels:
        example_weights = (0.60, 0.20, 0.20)

        if question_intent.get("topic") == "parking_restriction":
            example_weights = (0.55, 0.15, 0.30)

        elif question_intent.get("topic") == "lane_assignment":
            example_weights = (0.60, 0.15, 0.25)

    example_hits = fuse_hits(
        search_named_vector(
            client,
            config.collection_examples,
            "text",
            text_vec,
            config.top_k_examples * 3,
        ),
        search_named_vector(
            client,
            config.collection_examples,
            "image",
            image_vec,
            config.top_k_examples * 3,
        ),
        search_named_vector(
            client,
            config.collection_examples,
            "objects",
            object_vec,
            config.top_k_examples * 3,
        ),
        config.top_k_examples,
        weights=example_weights,
    )

    # =========================================================
    # 2. RETRIEVE LAW TEXT COLLECTION
    # berry_law_text: chỉ dùng vector "text"
    # =========================================================
    law_text_collection = getattr(
        config,
        "collection_law_text",
        "berry_law_text",
    )

    law_asset_collection = getattr(
        config,
        "collection_law_assets",
        "berry_law_assets",
    )

    law_text_hits = search_named_vector(
        client,
        law_text_collection,
        "text",
        text_vec,
        config.top_k_laws * 8,
    )

    # Gắn source để debug/rerank biết hit đến từ đâu
    for h in law_text_hits:
        h["source_collection"] = "law_text"
        h.setdefault("debug", {})
        h["debug"]["source_collection"] = "law_text"

    # =========================================================
    # 3. RETRIEVE LAW ASSET COLLECTION
    # berry_law_assets: dùng text + image + objects
    # =========================================================
    asset_text_hits = search_named_vector(
        client,
        law_asset_collection,
        "text",
        text_vec,
        config.top_k_laws * 8,
    )

    asset_image_hits = (
        search_named_vector(
            client,
            law_asset_collection,
            "image",
            image_vec,
            config.top_k_laws * 8,
        )
        if has_image
        else []
    )

    asset_object_hits = (
        search_named_vector(
            client,
            law_asset_collection,
            "objects",
            object_vec,
            config.top_k_laws * 8,
        )
        if has_image
        else []
    )

    # Asset nên ưu tiên image/object hơn text một chút vì đây là collection hình/bảng/phụ lục
    asset_hits = fuse_hits(
        asset_text_hits,
        asset_image_hits,
        asset_object_hits,
        config.top_k_laws * 8,
        weights=(0.45, 0.30, 0.25),
    )

    for h in asset_hits:
        h["source_collection"] = "law_asset"
        h.setdefault("debug", {})
        h["debug"]["source_collection"] = "law_asset"

    # =========================================================
    # 4. MERGE LAW TEXT + LAW ASSET
    # =========================================================
    raw_law_hits = fuse_hits(
        law_text_hits,
        asset_hits,
        [],
        config.top_k_laws * 10,
        weights=(0.70, 0.20, 0.10),
    )

    # Boost asset theo intent
    if question_intent.get("topic") in {
        "traffic_sign",
        "parking_restriction",
        "lane_assignment",
        "speed_limit",
    }:
        for h in raw_law_hits:
            p = h.get("payload", {}) or {}
            kind = p.get("kind", "")

            if kind in {"law_figure", "law_table"}:
                h["score"] = float(h.get("score", 0.0)) + 0.03
                h["choice_boost"] = float(h.get("choice_boost", 0.0)) + 0.03
                h.setdefault("debug", {})
                h["debug"]["asset_intent_boost"] = 0.03

    law_hits = rerank_law_hits(item, raw_law_hits)[: config.top_k_laws]

    # =========================================================
    # 5. DEBUG
    # =========================================================
    if config.debug_retrieval:
        print("=" * 100)
        print(f"[DEBUG][RETRIEVE] QUERY ID: {item.get('id')}")
        print(f"[DEBUG][RETRIEVE] QUESTION INTENT: {question_intent}")
        print(f"[DEBUG][RETRIEVE] RAW DETECTED LABELS: {raw_detected_labels}")
        print(f"[DEBUG][RETRIEVE] LABELS VI: {labels}")
        print(f"[DEBUG][RETRIEVE] IMAGE DESCRIPTION: {image_desc}")

        print("[DEBUG][RETRIEVE] TOP LAW TEXT HITS:")
        for i, hit in enumerate(law_text_hits[:5], 1):
            p = hit.get("payload", {})
            print(
                f"  {i}. score={hit.get('score', 0.0):.6f} "
                f"id={p.get('full_id')} "
                f"article={p.get('article_id')} "
                f"kind={p.get('kind')} "
                f"title={compact_text(p.get('title', ''), 120)}"
            )

        print("[DEBUG][RETRIEVE] TOP LAW ASSET HITS:")
        for i, hit in enumerate(asset_hits[:5], 1):
            p = hit.get("payload", {})
            print(
                f"  {i}. score={hit.get('score', 0.0):.6f} "
                f"id={p.get('full_id')} "
                f"article={p.get('article_id')} "
                f"kind={p.get('kind')} "
                f"asset_ref={p.get('asset_ref')} "
                f"title={compact_text(p.get('title', ''), 120)}"
            )

        print("[DEBUG][RETRIEVE] TOP LAW HITS AFTER MERGE + RERANK:")
        for i, hit in enumerate(law_hits, 1):
            p = hit.get("payload", {})
            dbg = hit.get("debug", {})
            print(
                f"  {i}. final={hit.get('score', 0.0):.6f} "
                f"base={hit.get('base_score', 0.0):.6f} "
                f"boost={hit.get('choice_boost', 0.0):.6f} "
                f"src={dbg.get('source_collection')} "
                f"kind={p.get('kind')} "
                f"id={p.get('full_id')} "
                f"article={p.get('article_id')} "
                f"title={compact_text(p.get('title', ''), 120)}"
            )

            if is_yes_no_question(item.get("question_type", "")):
                print(f"     matched_terms={dbg.get('matched_terms', [])}")
            else:
                print(f"     matched_choices={dbg.get('matched_choices', [])}")

        print("[DEBUG][RETRIEVE] TOP EXAMPLE HITS:")
        for i, hit in enumerate(example_hits[:5], 1):
            p = hit.get("payload", {})
            print(
                f"  {i}. score={hit.get('score', 0.0):.6f} "
                f"id={p.get('id')} "
                f"answer={p.get('answer')} "
                f"question={compact_text(p.get('question', ''), 100)}"
            )

    return example_hits, law_hits, image_desc
