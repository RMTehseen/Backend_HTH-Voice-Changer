"""MeanVC2 asset download and verification manager."""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from hth.core.logging import get_logger

logger = get_logger("hth.adapters.meanvc2.assets")

HF_REPO = "ASLP-lab/MeanVC2"
HF_BASE_URL = "https://huggingface.co/ASLP-lab/MeanVC2/resolve/main"

# (hf_filename, relative_local_path)
HF_ASSETS = [
    ("fastu2pp_80ms.pt",                "preprocess/ckpts/fastu2pp_80ms.pt"),
    ("fastu2pp_160ms.pt",               "preprocess/ckpts/fastu2pp_160ms.pt"),
    ("meanvc2_40ms_40ms.safetensors",   "ckpts/pretrained_models/meanvc2_40ms_40ms.safetensors"),
    ("meanvc2_120ms_40ms.safetensors",  "ckpts/pretrained_models/meanvc2_120ms_40ms.safetensors"),
    ("vocos.pt",                        "ckpts/vocos/vocos.pt"),
]

WAVLM_CFG_URL = "https://huggingface.co/theoracleguy/MeanVC2-OpenVox-Quality/resolve/main/preprocess/ckpts/wavlm_large_cfg.pt"
WAVLM_FINETUNE_URL = "https://huggingface.co/theoracleguy/MeanVC2-OpenVox-Quality/resolve/main/preprocess/ckpts/wavlm_large_finetune.pth"
WAVLM_LARGE_URL = "https://github.com/microsoft/unilm/releases/download/wavlm/WavLM-Large.pt"
WAVLM_FINETUNE_GDRIVE_URL = "https://drive.google.com/file/d/1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP/view"


@dataclass
class AssetStatus:
    name: str
    local_path: str
    exists: bool
    size_bytes: int = 0
    sha256: str = ""
    source_url: str = ""


def _sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk_size)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _download_with_progress(
    url: str,
    dest_path: str,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> None:
    """Download url to dest_path atomically, with progress callback(downloaded, total)."""
    import urllib.request

    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=dest.parent, prefix=".dl_")
    os.close(tmp_fd)
    try:
        def reporthook(block_num, block_size, total_size):
            if progress_cb and total_size > 0:
                downloaded = min(block_num * block_size, total_size)
                progress_cb(downloaded, total_size)

        urllib.request.urlretrieve(url, tmp_path, reporthook=reporthook)
        shutil.move(tmp_path, dest_path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def _download_hf_asset(hf_filename: str, local_path: str, progress_cb=None) -> None:
    """Download a file from Hugging Face model repo."""
    url = f"{HF_BASE_URL}/{hf_filename}"
    logger.info("Downloading HF asset", filename=hf_filename, url=url, dest=local_path)
    _download_with_progress(url, local_path, progress_cb)


class MeanVC2AssetManager:
    """Manages download and verification of MeanVC2 model assets."""

    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)

    def status(self) -> list[AssetStatus]:
        """Return status for all required assets."""
        statuses = []
        for hf_filename, rel_path in HF_ASSETS:
            local = self.model_dir / rel_path
            exists = local.exists()
            size = local.stat().st_size if exists else 0
            statuses.append(AssetStatus(
                name=hf_filename,
                local_path=str(local),
                exists=exists,
                size_bytes=size,
                source_url=f"{HF_BASE_URL}/{hf_filename}",
            ))
        # WavLM Large
        wavlm_path = self.model_dir / "preprocess" / "ckpts" / "wavlm_large.pt"
        wavlm_exists = wavlm_path.exists()
        statuses.append(AssetStatus(
            name="wavlm_large.pt",
            local_path=str(wavlm_path),
            exists=wavlm_exists,
            size_bytes=wavlm_path.stat().st_size if wavlm_exists else 0,
            source_url=WAVLM_LARGE_URL,
        ))
        # WavLM finetune
        ft_path = self.model_dir / "preprocess" / "ckpts" / "wavlm_large_finetune.pth"
        ft_exists = ft_path.exists()
        statuses.append(AssetStatus(
            name="wavlm_large_finetune.pth",
            local_path=str(ft_path),
            exists=ft_exists,
            size_bytes=ft_path.stat().st_size if ft_exists else 0,
            source_url=WAVLM_FINETUNE_GDRIVE_URL,
        ))
        return statuses

    def download_hf_assets(self, variant: str = "40ms", progress_cb=None) -> dict[str, bool]:
        """Download HF-hosted assets. When variant is 40ms, only downloads 40ms+80ms ASR models."""
        results = {}
        assets_to_download = [
            ("meanvc2_40ms_40ms.safetensors", "ckpts/pretrained_models/meanvc2_40ms_40ms.safetensors"),
            ("vocos.pt", "ckpts/vocos/vocos.pt"),
            ("fastu2pp_80ms.pt", "preprocess/ckpts/fastu2pp_80ms.pt"),
        ]
        if variant == "120ms":
            assets_to_download.extend([
                ("meanvc2_120ms_40ms.safetensors", "ckpts/pretrained_models/meanvc2_120ms_40ms.safetensors"),
                ("fastu2pp_160ms.pt", "preprocess/ckpts/fastu2pp_160ms.pt"),
            ])

        for hf_filename, rel_path in assets_to_download:
            local = self.model_dir / rel_path
            if local.exists() and local.stat().st_size > 0:
                logger.info("Asset already exists, skipping", asset=hf_filename)
                results[hf_filename] = True
                continue
            try:
                _download_hf_asset(hf_filename, str(local), progress_cb)
                results[hf_filename] = True
                logger.info("Downloaded", asset=hf_filename, path=str(local))
            except Exception as e:
                logger.error("Download failed", asset=hf_filename, error=str(e))
                results[hf_filename] = False
        return results

    def download_wavlm(self, progress_cb=None) -> dict[str, bool]:
        """Download WavLM-Large config and fine-tuned weights directly. Returns {name: success}."""
        results = {}
        cfg_path = self.model_dir / "preprocess" / "ckpts" / "wavlm_large_cfg.pt"
        ft_path = self.model_dir / "preprocess" / "ckpts" / "wavlm_large_finetune.pth"

        # Download WavLM config
        if not cfg_path.exists():
            try:
                logger.info("Downloading WavLM config", url=WAVLM_CFG_URL)
                _download_with_progress(WAVLM_CFG_URL, str(cfg_path), progress_cb)
                results["wavlm_large_cfg.pt"] = True
            except Exception as e:
                logger.error("WavLM config download failed", error=str(e))
                results["wavlm_large_cfg.pt"] = False
        else:
            results["wavlm_large_cfg.pt"] = True

        # Download WavLM finetuned weights
        if not ft_path.exists():
            try:
                logger.info("Downloading WavLM finetuned weights", url=WAVLM_FINETUNE_URL)
                _download_with_progress(WAVLM_FINETUNE_URL, str(ft_path), progress_cb)
                results["wavlm_large_finetune.pth"] = True
            except Exception as e:
                logger.error("WavLM finetune download failed", error=str(e))
                results["wavlm_large_finetune.pth"] = False
        else:
            results["wavlm_large_finetune.pth"] = True

        return results

    def verify_assets(self) -> dict[str, bool]:
        """Verify all required inference assets exist."""
        required = [rel for _, rel in HF_ASSETS]
        required += [
            "preprocess/ckpts/wavlm_large_finetune.pth",
            "preprocess/ckpts/wavlm_large_cfg.pt",
        ]
        result = {}
        for rel in required:
            p = self.model_dir / rel
            result[rel] = p.exists() and p.stat().st_size > 0
        return result
