"""Applio / RVC model format parser (Zero Unpickling in FastAPI)."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

from hth.domain.model_contracts import (
    AssetDescriptor,
    ModelFormat,
    ModelManifestMetadata,
)
from hth.services.model_parsers.base import BaseFormatParser, ParsedModelInfo


class ApplioRVCParser(BaseFormatParser):
    """Parses RVC/Applio weight packages (.pth + optional .index) WITHOUT unpickling bytecode."""

    def can_parse(self, directory_path: str) -> bool:
        p = Path(directory_path)
        if not p.is_dir():
            return False
        pths = list(p.rglob("*.pth"))
        return len(pths) > 0

    def parse(self, directory_path: str) -> ParsedModelInfo:
        p = Path(directory_path)
        pths = list(p.rglob("*.pth"))
        if not pths:
            raise ValueError("No .pth weights file found in directory")

        pth_file = pths[0]

        # Inspect ZIP container structure safely WITHOUT calling pickle or torch.load
        # PyTorch v1.6+ saves .pth as a standard uncompressed/compressed ZIP container
        try:
            with zipfile.ZipFile(pth_file, "r") as zf:
                namelist = zf.namelist()
                # Verify that it looks like a pytorch container (archive/data.pkl, etc.)
                has_data_pkl = any("data.pkl" in name for name in namelist)
                if not has_data_pkl and not any("data" in name for name in namelist):
                    raise ValueError(
                        "Provided .pth does not appear to be a valid PyTorch model container"
                    )
        except zipfile.BadZipFile as exc:
            raise ValueError(f"Invalid or corrupted .pth container: {exc}") from exc

        # Check for matching index file
        indices = list(p.rglob("*.index"))
        has_index = len(indices) > 0

        # Scan all assets and compute hashes
        assets: list[AssetDescriptor] = []
        for fpath in p.rglob("*"):
            if fpath.is_file():
                rel = str(fpath.relative_to(p)).replace("\\", "/")
                size = fpath.stat().st_size
                with open(fpath, "rb") as bf:
                    h = hashlib.sha256(bf.read()).hexdigest()
                assets.append(
                    AssetDescriptor(
                        relative_path=rel,
                        sha256=h,
                        size_bytes=size,
                        mime_type="application/octet-stream",
                        is_required=fpath == pth_file,
                    )
                )

        model_name = pth_file.stem
        manifest = ModelManifestMetadata(
            name=model_name,
            version="1.0.0",
            description=f"Applio/RVC model package (index: {'present' if has_index else 'none'})",
            author="User Imported",
            speaker_count=1,
            target_sample_rate=40000,
            extra_params={
                "has_index": has_index,
                "index_file": indices[0].name if has_index else None,
            },
        )

        return ParsedModelInfo(
            engine_id="applio",
            model_format=ModelFormat.RVC_PTH_INDEX,
            name=model_name,
            version="1.0.0",
            description="Applio / RVC weight checkpoint",
            sample_rate=40000,
            license_identifier="Custom-RVC-Requires-Review",
            license_source_url="",
            commercial_eligible=False,
            manifest=manifest,
            assets=assets,
        )
