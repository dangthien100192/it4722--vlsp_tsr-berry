import os
import shutil
from functools import lru_cache
from pathlib import Path
from huggingface_hub import snapshot_download
from berry.config import config

def auto_clear_hf_dynamic_cache(repo_name: str = "C-RADIOv2-B") -> None:
    hf_modules = Path(os.environ["HF_HOME"]) / "modules" / "transformers_modules"
    if not hf_modules.exists():
        return
    repo_name_norm = repo_name.lower().replace("-", "").replace("_", "")
    for d in hf_modules.rglob("*"):
        if d.is_dir() and repo_name_norm in d.name.lower().replace("-", "").replace("_", ""):
            try:
                print(f"[AUTO-FIX][HF] Removing broken dynamic cache: {d}")
                shutil.rmtree(d, ignore_errors=True)
            except Exception as e:
                print(f"[WARN][HF] Cannot remove {d}: {e}")

def auto_fix_local_model_dir(model_dir: str) -> None:
    p = Path(model_dir).expanduser().resolve()
    required = ["config.json", "hf_model.py", "radio_model.py", "dual_hybrid_vit.py"]
    if p.exists():
        missing = [f for f in required if not (p / f).exists()]
        if missing:
            print(f"[AUTO-FIX][HF] Local model thiếu file {missing} -> removing {p}")
            shutil.rmtree(p, ignore_errors=True)

def ensure_cradio_dynamic_module(model_dir: str, repo_name: str = "C-RADIOv2-B") -> Path:
    src = Path(model_dir).expanduser().resolve()
    dst = Path(os.environ["HF_HOME"]) / "modules" / "transformers_modules" / repo_name
    dst.mkdir(parents=True, exist_ok=True)
    required_py = ["hf_model.py", "radio_model.py", "dual_hybrid_vit.py"]
    optional_py = ["__init__.py", "configuration_hf.py", "configuration_radio.py", "modeling_hf.py", "model.py"]
    for name in required_py + optional_py:
        s = src / name
        if s.exists():
            shutil.copy2(s, dst / name)
    init_file = dst / "__init__.py"
    if not init_file.exists():
        init_file.write_text("", encoding="utf-8")
    missing = [f for f in required_py if not (dst / f).exists()]
    if missing:
        raise RuntimeError(f"Dynamic cache của C-RADIO còn thiếu file: {missing} | dst={dst}")
    return dst

def _is_valid_hf_model_dir(model_dir: Path) -> bool:
    config_ok = (model_dir / "config.json").exists()
    weight_ok = (
        (model_dir / "model.safetensors").exists()
        or (model_dir / "pytorch_model.bin").exists()
        or (model_dir / "pytorch_model.bin.index.json").exists()
        or any(model_dir.glob("*.safetensors"))
    )
    return model_dir.exists() and model_dir.is_dir() and config_ok and weight_ok

def ensure_hf_repo_local(repo_id: str, local_dir: str, repo_type: str = "model") -> str:
    model_path = Path(local_dir).expanduser().resolve()
    if not _is_valid_hf_model_dir(model_path):
        print(f"[INFO][HF] Downloading repo '{repo_id}' to '{model_path}' ...")
        model_path.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=repo_id, repo_type=repo_type, local_dir=str(model_path), local_dir_use_symlinks=False)
        if not _is_valid_hf_model_dir(model_path):
            raise RuntimeError(f"Đã tải repo '{repo_id}' nhưng thư mục '{model_path}' vẫn không hợp lệ.")
    if repo_id == config.cradio_repo:
        required_code = ["config.json", "hf_model.py", "radio_model.py", "dual_hybrid_vit.py"]
        missing = [f for f in required_code if not (model_path / f).exists()]
        if missing:
            raise RuntimeError(f"Repo '{repo_id}' tại '{model_path}' còn thiếu file: {missing}")
    return str(model_path)
