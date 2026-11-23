"""Synthetic test format parser for FakeEngine testing and development."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from hth.domain.model_contracts import (
    AssetDescriptor,
    ModelFormat,
    ModelManifestMetadata,
)
from hth.services.model_parsers.base import BaseFormatParser, ParsedModelInfo


class SyntheticTestParser(BaseFormatParser):
    """Parses lightweight synthetic test models (manifest.json + dummy weights) for testing."""

    def can_parse(self, directory_path: str) -> bool:
        p = Path(directory_path)
        if not p.is_dir():
            return False
        return (p / "manifest.json").is_file()

    def parse(self, directory_path: str) -> ParsedModelInfo:
        p = Path(directory_path)
        manifest_file = p / "manifest.json"
        if not manifest_file.is_file():
            raise ValueError("manifest.json not found in synthetic model directory")

        with open(manifest_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        model_name = data.get("name", "Synthetic Model")
        version = data.get("version", "1.0.0")
        description = data.get("description", "Synthetic in-memory test model")
        sample_rate = int(data.get("sample_rate", 16000))
        license_id = data.get("license", "MIT")
        commercial_eligible = bool(data.get("commercial_eligible", True))
        speaker_count = int(data.get("speaker_count", 1))

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
                        mime_type="application/json"
                        if fpath.suffix == ".json"
                        else "application/octet-stream",
                        is_required=fpath == manifest_file,
                    )
                )

        manifest = ModelManifestMetadata(
            name=model_name,
            version=version,
            description=description,
            author=data.get("author", "HTH Internal"),
            speaker_count=speaker_count,
            target_sample_rate=sample_rate,
            extra_params=data.get("extra_params", {}),
        )

        return ParsedModelInfo(
            engine_id="fake",
            model_format=ModelFormat.SYNTHETIC_TEST,
            name=model_name,
            version=version,
            description=description,
            sample_rate=sample_rate,
            license_identifier=license_id,
            license_source_url="https://github.com/hth/voice-changer",
            commercial_eligible=commercial_eligible,
            manifest=manifest,
            assets=assets,
        )
