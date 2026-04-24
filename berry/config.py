import os
from dataclasses import dataclass
from dotenv import load_dotenv

HF_CACHE_ROOT = os.getenv("HF_CACHE_ROOT", r"D:\hf_cache")
os.environ["HF_HOME"] = HF_CACHE_ROOT
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(HF_CACHE_ROOT, "hub")
os.environ["TRANSFORMERS_DYNAMIC_MODULE_NAME"] = "local"

load_dotenv()

@dataclass
class Config:
    train_json: str
    eval_json: str
    train_image_dir: str
    test_image_dir: str
    law_json: str

    qdrant_url: str = "http://localhost:6333"
    collection_examples: str = "berry_examples"
    collection_law: str = "berry_law"
    output_file: str = "predictions.json.ollama"
    collection_law_text: str = "berry_law_text"
    collection_law_asset: str = "berry_law_assets"
    law_image_dir: str = ""

    top_k_examples: int = 5
    top_k_laws: int = 5
    recreate_on_dim_mismatch: bool = False
    debug_retrieval: bool = False

    jina_api_key: str = ""
    jina_model: str = "jina-embeddings-v3"
    embed_url: str = "https://api.jina.ai/v1/embeddings"
    embed_timeout: int = 60

    ollama_base_url: str = "http://localhost:11434"
    openai_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"
    llm_timeout: int = 300
    llm_temperature: float = 0.0
    llm_num_predict: int = 16

    use_image_models: bool = False
    cradio_repo: str = "nvidia/C-RADIOv2-B"
    cradio_local_dir: str = r"D:\hf_models\C-RADIOv2-B"
    owlv2_repo: str = "google/owlv2-base-patch16-ensemble"
    owlv2_local_dir: str = r"D:\hf_models\owlv2-base-patch16-ensemble"

    owl_queries: tuple[str, ...] = (
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
            collection_law_text=os.getenv("LAW_TEXT_COLLECTION", "berry_law_text"),
            collection_law_asset=os.getenv("LAW_ASSET_COLLECTION", "berry_law_assets"),
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            collection_examples=os.getenv("EXAMPLE_COLLECTION", "berry_examples"),
            collection_law=os.getenv("LAW_COLLECTION", "berry_law"),
            output_file=os.getenv("OUTPUT_FILE", "predictions.json.ollama"),
            top_k_examples=int(os.getenv("TOP_K_EXAMPLES", "5")),
            top_k_laws=int(os.getenv("TOP_K_LAWS", "5")),
            recreate_on_dim_mismatch=os.getenv("RECREATE_ON_DIM_MISMATCH", "false").lower() == "true",
            debug_retrieval=os.getenv("DEBUG_RETRIEVAL", "false").lower() == "true",
            jina_api_key=os.getenv("JINA_API_KEY", ""),
            jina_model=os.getenv("JINA_MODEL", "jina-embeddings-v3"),
            embed_url=os.getenv("EMBED_URL", "https://api.jina.ai/v1/embeddings"),
            embed_timeout=int(os.getenv("EMBED_TIMEOUT", "60")),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            openai_base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:11434"),
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
