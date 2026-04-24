from functools import lru_cache
import torch
from transformers import Owlv2ForObjectDetection, Owlv2Processor
from berry.config import config
from berry.runtime import DEVICE
from berry.models.hf_common import ensure_hf_repo_local
from berry.utils.image_utils import load_image
from berry.utils.module_utils import has_module
from berry.utils.text_utils import unique_keep_order

@lru_cache(maxsize=1)
def get_owl():
    if not has_module("scipy"):
        raise RuntimeError("Thiếu scipy cho OWLv2. Cài bằng: pip install scipy")
    model_dir = ensure_hf_repo_local(config.owlv2_repo, config.owlv2_local_dir, "model")
    processor = Owlv2Processor.from_pretrained(model_dir, local_files_only=True)
    model = Owlv2ForObjectDetection.from_pretrained(model_dir, local_files_only=True).to(DEVICE)
    model.eval()
    return processor, model

def detect_objects(image_path: str | None, threshold: float = 0.10, text_queries: list[str] | None = None) -> list[str]:
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
        labels = []
        if results:
            for label_idx in results[0]["labels"].detach().cpu().tolist():
                if 0 <= label_idx < len(text_queries):
                    labels.append(text_queries[label_idx])
        return unique_keep_order(labels)
    except Exception as e:
        print(f"[WARN][OBJECT] detect_objects failed for {image_path}: {e}")
        return []
