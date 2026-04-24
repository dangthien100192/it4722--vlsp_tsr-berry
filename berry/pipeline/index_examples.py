from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from berry.config import config
from berry.embeddings.text_embedding import embed_text_passage
from berry.embeddings.image_embedding import embed_image
from berry.retrieval.build_query import build_query_text
from berry.retrieval.qdrant_utils import ensure_collection
from berry.utils.image_utils import get_qa_image_path
from berry.embeddings.object_embedding import embed_objects
from berry.vision import filter_detected_labels_by_intent, get_question_guided_owl_queries, translate_detected_labels_to_vi
from berry.models.owl import detect_objects
from berry.utils.math_utils import zero_vec

def index_examples(client: QdrantClient, dataset) -> None:
    ensure_collection(client, config.collection_examples)
    success_count = fail_count = 0
    for idx, item in enumerate(dataset):
        image_path = get_qa_image_path(config.train_image_dir, item)
        try:
            text_vec = embed_text_passage(build_query_text(item))
            image_vec = embed_image(image_path, zero_vec)
            object_labels_en = filter_detected_labels_by_intent(detect_objects(image_path, text_queries=get_question_guided_owl_queries(item)), item)
            object_labels_vi = translate_detected_labels_to_vi(object_labels_en)
            object_vec = embed_objects(image_path, item=item, labels=object_labels_vi)
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
            point = PointStruct(id=idx, vector={"text": text_vec, "image": image_vec, "objects": object_vec}, payload=payload)
            client.upsert(collection_name=config.collection_examples, points=[point])
            success_count += 1
        except Exception as e:
            fail_count += 1
            print(f"[ERROR][INDEX_EXAMPLE] idx={idx} id={item.get('id')} error={e}")
    print(f"[DONE][EXAMPLE] Indexed success={success_count} | failed={fail_count}")
