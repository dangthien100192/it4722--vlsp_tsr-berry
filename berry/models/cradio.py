from functools import lru_cache
from pathlib import Path
import shutil

from transformers import AutoModel

from berry.config import config
from berry.runtime import DEVICE
from berry.models.hf_common import (
    auto_fix_local_model_dir,
    ensure_cradio_dynamic_module,
    ensure_hf_repo_local,
)


def _clear_cradio_dynamic_cache():
    hf_cache = Path("D:/hf_cache/modules/transformers_modules")

    for name in [
        "C-RADIOv2-B",
        "C_hyphen_RADIOv2_hyphen_B",
        "nvidia",
    ]:
        p = hf_cache / name
        if p.exists():
            print(f"[AUTO-FIX][HF] Removing broken dynamic cache: {p}")
            shutil.rmtree(p, ignore_errors=True)


def _patch_cradio_dynamic_cache(model_dir: Path):
    hf_cache = Path("D:/hf_cache/modules/transformers_modules")

    targets = [
        hf_cache / "C-RADIOv2-B",
        hf_cache / "C_hyphen_RADIOv2_hyphen_B",
    ]

    py_files = list(model_dir.glob("*.py"))

    for target in targets:
        target.mkdir(parents=True, exist_ok=True)

        for src in py_files:
            dst = target / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
                print(f"[AUTO-FIX][HF] Copied {src.name} -> {target}")


@lru_cache(maxsize=1)
def get_c_radio():
    last_error = None

    for attempt in range(1, 3):
        try:
            auto_fix_local_model_dir(config.cradio_local_dir)
            _clear_cradio_dynamic_cache()

            model_dir = ensure_hf_repo_local(
                config.cradio_repo,
                config.cradio_local_dir,
                "model"
            )

            model_dir = Path(model_dir).expanduser().resolve()

            ensure_cradio_dynamic_module(model_dir, "C-RADIOv2-B")
            _patch_cradio_dynamic_cache(model_dir)

            model = AutoModel.from_pretrained(
                str(model_dir),
                trust_remote_code=True,
                local_files_only=True,
            ).to(DEVICE)

            model.eval()
            return model

        except Exception as e:
            last_error = e
            print(f"[WARN][HF] get_c_radio attempt {attempt} failed: {e}")

            model_dir = Path(config.cradio_local_dir).expanduser().resolve()
            if model_dir.exists():
                shutil.rmtree(model_dir, ignore_errors=True)

            _clear_cradio_dynamic_cache()

    raise RuntimeError(
        f"Không thể load C-RADIO sau 2 lần thử. Last error: {last_error}"
    )