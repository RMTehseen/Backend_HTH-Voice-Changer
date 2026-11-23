"""Beatrice V2 paraphernalia model format parser."""

from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path
from typing import Any

from hth.domain.model_contracts import (
    AssetDescriptor,
    ModelFormat,
    ModelManifestMetadata,
)
from hth.services.model_parsers.base import BaseFormatParser, ParsedModelInfo


class BeatriceParaphernaliaParser(BaseFormatParser):
    """Parses Beatrice paraphernalia folders containing .toml manifests and .bin weights."""

    REQUIRED_BINS = {
        "waveform_generator.bin",
        "phone_extractor.bin",
        "pitch_estimator.bin",
    }

    def can_parse(self, directory_path: str) -> bool:
        p = Path(directory_path)
        if not p.is_dir():
            return False

        toml_files = list(p.glob("*.toml"))
        bin_files = {f.name for f in p.glob("*.bin")}

        if toml_files and self.REQUIRED_BINS.issubset(bin_files):
            return True

        # Check subdirectories if model was zipped with a root folder
        for sub in p.iterdir():
            if sub.is_dir():
                sub_tomls = list(sub.glob("*.toml"))
                sub_bins = {f.name for f in sub.glob("*.bin")}
                if sub_tomls and self.REQUIRED_BINS.issubset(sub_bins):
                    return True

        return False

    def parse(self, directory_path: str) -> ParsedModelInfo:
        p = Path(directory_path)
        target_dir = p

        toml_files = list(p.glob("*.toml"))
        if not toml_files:
            # Check nested directory
            for sub in p.iterdir():
                if sub.is_dir():
                    sub_tomls = list(sub.glob("*.toml"))
                    if sub_tomls:
                        toml_files = sub_tomls
                        target_dir = sub
                        break

        if not toml_files:
            raise ValueError("Beatrice model manifest (.toml) not found")

        toml_path = toml_files[0]
        with open(toml_path, "rb") as f:
            try:
                data = tomllib.load(f)
            except Exception as exc:
                raise ValueError(f"Failed to parse Beatrice TOML manifest: {exc}") from exc

        model_section = data.get("model", {})
        model_name = model_section.get("name", toml_path.stem)
        version = model_section.get("version", "2.0.0-rc.0")
        description = model_section.get("description", "").strip()

        # Parse voices
        speakers: list[dict[str, Any]] = []
        voice_data = data.get("voice", {})
        if isinstance(voice_data, dict):
            for spk_key, spk_val in voice_data.items():
                if isinstance(spk_val, dict):
                    speaker_name = spk_val.get("name", f"voice.{spk_key}")
                    avg_pitch = spk_val.get("average_pitch", 0.0)
                    speakers.append(
                        {
                            "key": f"voice.{spk_key}",
                            "name": speaker_name,
                            "average_pitch": avg_pitch,
                        }
                    )
        for key, val in data.items():
            if key.startswith("voice.") and isinstance(val, dict):
                speaker_name = val.get("name", key)
                avg_pitch = val.get("average_pitch", 0.0)
                speakers.append({"key": key, "name": speaker_name, "average_pitch": avg_pitch})

        # Scan all assets and compute hashes
        assets: list[AssetDescriptor] = []
        for fpath in target_dir.rglob("*"):
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
                        mime_type="application/toml"
                        if fpath.suffix == ".toml"
                        else "application/octet-stream",
                        is_required=fpath.name in self.REQUIRED_BINS or fpath.suffix == ".toml",
                    )
                )

        # Licensing heuristics
        is_jvs = (
            "jvs" in model_name.lower()
            or "jvs" in toml_path.name.lower()
            or "unauthorized commercial use is prohibited" in description.lower()
            or "許可なく営利目的で使用することが禁じられている" in description
        )

        if is_jvs:
            license_id = "JVS-NonCommercial"
            license_url = (
                "https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus"
            )
            commercial_eligible = False
        else:
            license_id = "Beatrice-Proprietary-Personal"
            license_url = "https://github.com/hth/beatrice"
            commercial_eligible = False  # Beatrice models are personal/non-commercial by default

        manifest = ModelManifestMetadata(
            name=model_name,
            version=version,
            description=description,
            author="Beatrice Upstream" if is_jvs else "Custom",
            speaker_count=max(len(speakers), 1),
            speakers=speakers,
            target_sample_rate=16000,
            extra_params={"beatrice_version": version, "toml_file": toml_path.name},
        )

        return ParsedModelInfo(
            engine_id="beatrice",
            model_format=ModelFormat.BEATRICE_PARAPHERNALIA,
            name=model_name,
            version=version,
            description=description,
            sample_rate=16000,
            license_identifier=license_id,
            license_source_url=license_url,
            commercial_eligible=commercial_eligible,
            manifest=manifest,
            assets=assets,
        )
