from qdrant_client import QdrantClient
from berry.config import config, HF_CACHE_ROOT
from berry.datasets.law_dataset import LawDataset
from berry.datasets.qa_dataset import QaDataset
from berry.embeddings.text_embedding import validate_embedding_dims
from berry.models.cradio import get_c_radio
from berry.models.owl import get_owl
from berry.pipeline.eval_pipeline import load_predicted_ids, run_eval
from berry.pipeline.index_examples import index_examples
from berry.pipeline.index_laws import index_laws
from berry.pipeline.index_laws_v2 import index_laws_v2
from berry.retrieval.qdrant_utils import collection_has_data
from berry.runtime import DEVICE, IMAGE_DIM, OBJECT_DIM, set_text_dim
from berry.utils.module_utils import has_module

def main() -> None:
    print("[INFO] Validating text embedding dimensions...")
    set_text_dim(validate_embedding_dims())
    from berry.runtime import TEXT_DIM  # import after set
    print(f"[INFO] TEXT_DIM={TEXT_DIM} | IMAGE_DIM={IMAGE_DIM} | OBJECT_DIM={OBJECT_DIM}")
    if config.use_image_models:
        print("[INFO] Loading C-RADIOv2...")
        get_c_radio()
        if has_module("scipy"):
            print("[INFO] Loading OWLv2...")
            get_owl()
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
    print(f"[INFO] Openai base URL: {config.openai_base_url}")
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

    # if not collection_has_data(client, config.collection_law):
    #     print("[INFO] Indexing laws...")
    #     index_laws(client, law_ds)
    # else:
    #     print("[SKIP] laws already indexed")

    # print("[INFO] Indexing laws v2...")
    # index_laws_v2(client, law_ds)

    print("[INFO] Running evaluation / prediction...")
    predicted_ids = load_predicted_ids(config.output_file)
    eval_ds.items = [item for item in eval_ds.items if str(item.get("id", "")).strip() not in predicted_ids]
    print(f"[INFO] Remaining eval items: {len(eval_ds.items)}")
    results = run_eval(client, eval_ds)
    print(f"[DONE] Saved predictions to {config.output_file} | total={len(results)}")

if __name__ == "__main__":
    main()
