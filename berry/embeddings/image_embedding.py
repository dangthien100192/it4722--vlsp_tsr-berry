import torch
import torchvision.transforms as T
from typing import Optional, List

from berry.runtime import DEVICE, IMAGE_DIM
from berry.models.cradio import get_c_radio
from berry.utils.image_utils import load_image
from berry.utils.math_utils import l2_normalize
from berry.utils.math_utils import zero_vec

# =========================
# Transform
# =========================

CRADIO_TRANSFORM = T.Compose(
    [
        T.Resize((384, 384)),
        T.ToTensor(),
        T.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
    ]
)


def preprocess_c_radio_image(image):
    return CRADIO_TRANSFORM(image).unsqueeze(0).to(DEVICE)


# =========================
# Extract vector
# =========================

def extract_cradio_vector(outputs) -> List[float]:
    tensor = None

    if isinstance(outputs, torch.Tensor):
        tensor = outputs

    elif hasattr(outputs, "summary") and outputs.summary is not None:
        tensor = outputs.summary

    elif hasattr(outputs, "features") and outputs.features is not None:
        feats = outputs.features
        tensor = feats.mean(dim=1) if feats.ndim >= 3 else feats

    elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        tensor = outputs.pooler_output

    elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        lhs = outputs.last_hidden_state
        tensor = lhs.mean(dim=1) if lhs.ndim >= 3 else lhs

    elif isinstance(outputs, dict):
        for key in ["summary", "pooler_output", "last_hidden_state", "features"]:
            val = outputs.get(key)
            if isinstance(val, torch.Tensor):
                tensor = val.mean(dim=1) if val.ndim >= 3 else val
                break

    elif isinstance(outputs, (tuple, list)) and len(outputs) > 0:
        if isinstance(outputs[0], torch.Tensor):
            tensor = outputs[0].mean(dim=1) if outputs[0].ndim >= 3 else outputs[0]

    if tensor is None:
        raise RuntimeError(f"Không extract được vector từ C-RADIO output: {type(outputs)}")

    vec = tensor.squeeze(0).detach().cpu().float().flatten().tolist()

    if len(vec) < IMAGE_DIM:
        vec += [0.0] * (IMAGE_DIM - len(vec))
    elif len(vec) > IMAGE_DIM:
        vec = vec[:IMAGE_DIM]

    return l2_normalize(vec)


# =========================
# Main API
# =========================

def embed_image(image_path: Optional[str]) -> List[float]:
    from berry.config import config

    if not config.use_image_models:
        return zero_vec(IMAGE_DIM)

    image = load_image(image_path)
    if image is None:
        return zero_vec(IMAGE_DIM)

    try:
        model = get_c_radio()

        with torch.no_grad():
            outputs = model(preprocess_c_radio_image(image))

        return extract_cradio_vector(outputs)

    except Exception as e:
        print(f"[WARN][IMAGE] C-RADIO failed for {image_path}: {e}")
        return zero_vec(IMAGE_DIM)