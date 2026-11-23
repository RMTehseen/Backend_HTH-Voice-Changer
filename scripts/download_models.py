#!/usr/bin/env python3
"""
Automated downloader for MeanVC2 model weights and upstream runtime.
Suitable for Google Colab, RunPod, Docker, and local setup.
"""
import os
import sys
import subprocess
from pathlib import Path
import urllib.request

HF_BASE_URL = "https://huggingface.co/ASLP-lab/MeanVC2/resolve/main"

# Asset definitions: (remote_url_or_filename, local_relative_path)
HF_ASSETS = [
    ("fastu2pp_80ms.pt", "preprocess/ckpts/fastu2pp_80ms.pt"),
    ("fastu2pp_160ms.pt", "preprocess/ckpts/fastu2pp_160ms.pt"),
    ("meanvc2_40ms_40ms.safetensors", "ckpts/pretrained_models/meanvc2_40ms_40ms.safetensors"),
    ("meanvc2_120ms_40ms.safetensors", "ckpts/pretrained_models/meanvc2_120ms_40ms.safetensors"),
    ("vocos.pt", "ckpts/vocos/vocos.pt"),
]

DIRECT_URLS = [
    (
        "https://huggingface.co/theoracleguy/MeanVC2-OpenVox-Quality/resolve/main/preprocess/ckpts/wavlm_large_cfg.pt",
        "preprocess/ckpts/wavlm_large_cfg.pt"
    ),
    (
        "https://huggingface.co/theoracleguy/MeanVC2-OpenVox-Quality/resolve/main/preprocess/ckpts/wavlm_large_finetune.pth",
        "preprocess/ckpts/wavlm_large_finetune.pth"
    ),
]


def download_file(url: str, dest_path: Path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 1024:
        print(f"  [EXISTS] {dest_path.name} ({dest_path.stat().st_size / 1e6:.1f} MB)")
        return

    print(f"  [DOWNLOADING] {dest_path.name} from {url}...")
    tmp_path = dest_path.with_suffix(".tmp")
    try:
        urllib.request.urlretrieve(url, str(tmp_path))
        tmp_path.replace(dest_path)
        print(f"  [DONE] {dest_path.name} saved ({dest_path.stat().st_size / 1e6:.1f} MB)")
    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink()
        print(f"  [ERROR] Failed to download {dest_path.name}: {e}")
        raise


def setup_meanvc2_code(target_dir: Path):
    """Clone ASLP-lab/MeanVC2 if source directory does not exist."""
    if (target_dir / "src" / "dit.py").exists():
        print(f"[FOUND] MeanVC2 source code present at: {target_dir}")
        return

    print(f"[CLONING] MeanVC2 upstream source code to: {target_dir}...")
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "https://github.com/ASLP-lab/MeanVC2.git", str(target_dir)],
        check=True
    )
    print("[DONE] MeanVC2 source code cloned.")


def main():
    print("==========================================================")
    print("  HTH Voice Changer — Cloud & Model Assets Setup")
    print("==========================================================")

    # 1. Determine Model Directory
    base_dir = Path.cwd()
    model_dir = Path(os.environ.get("HTH_MEANVC2_MODEL_DIR", base_dir / "models" / "MeanVC2Models"))
    source_dir = Path(os.environ.get("HTH_MEANVC2_SOURCE_DIR", base_dir / "MeanVC2-main"))

    print(f"Models Directory : {model_dir}")
    print(f"Source Directory : {source_dir}")
    print("----------------------------------------------------------")

    # 2. Clone MeanVC2 code
    setup_meanvc2_code(source_dir)

    # 3. Download HuggingFace MeanVC2 weights
    print("\nDownloading Model Weights (HuggingFace ASLP-lab/MeanVC2)...")
    for fname, rel_path in HF_ASSETS:
        url = f"{HF_BASE_URL}/{fname}"
        dest = model_dir / rel_path
        download_file(url, dest)

    # 4. Download Speaker Encoder weights
    print("\nDownloading Speaker Encoder Weights (WavLM)...")
    for url, rel_path in DIRECT_URLS:
        dest = model_dir / rel_path
        download_file(url, dest)

    print("\n==========================================================")
    print("  [SUCCESS] All MeanVC2 models & code successfully prepared!")
    print("==========================================================")


if __name__ == "__main__":
    main()
