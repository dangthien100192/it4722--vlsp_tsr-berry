from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="nvidia/C-RADIOv2-B",
    local_dir=r"D:\hf_models\C-RADIOv2-B",
    local_dir_use_symlinks=False,
)