import os
import torch
from .config import config

DEVICE = torch.device(os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
MAX_JINA_TEXT_LENGTH = int(os.getenv("MAX_JINA_TEXT_LENGTH", "8192"))

TEXT_DIM = 1024
IMAGE_DIM = int(os.getenv("IMAGE_DIM", "1024"))
OBJECT_DIM = int(os.getenv("OBJECT_DIM", "1024"))

def set_text_dim(value: int) -> None:
    global TEXT_DIM
    TEXT_DIM = int(value)
