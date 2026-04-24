import os
from typing import Any, Dict, Optional
from PIL import Image

def get_qa_image_path(base_dir: str, item: Dict[str, Any]) -> Optional[str]:
    image_id = str(item.get("image_id") or "").strip()
    if not image_id:
        return None
    candidates = [
        os.path.join(base_dir, image_id),
        os.path.join(base_dir, f"{image_id}.jpg"),
        os.path.join(base_dir, f"{image_id}.jpeg"),
        os.path.join(base_dir, f"{image_id}.png"),
        os.path.join(base_dir, f"{image_id}.webp"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]

def load_image(image_path: Optional[str]) -> Optional[Image.Image]:
    if not image_path or not os.path.exists(image_path):
        return None
    try:
        return Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[WARN][IMAGE] Không mở được ảnh: {image_path} | error={e}")
        return None
