"""Unit tests for provider-neutral model format parsers."""

import zipfile

from hth.domain.model_contracts import ModelFormat
from hth.services.model_parsers.applio_rvc import ApplioRVCParser
from hth.services.model_parsers.beatrice import BeatriceParaphernaliaParser
from hth.services.model_parsers.synthetic import SyntheticTestParser


def test_beatrice_jvs_parser(tmp_path):
    """Test Beatrice paraphernalia parser detecting JVS non-commercial terms."""
    model_dir = tmp_path / "beatrice_jvs"
    model_dir.mkdir()

    toml_content = b"""
[model]
name = "JVS corpus"
version = "2.0.0-rc.0"
description = "The model was trained using the JVS corpus. Unauthorized commercial use is prohibited."

[voice.0]
name = "jvs001"
average_pitch = 50.0

[voice.1]
name = "jvs002"
average_pitch = 55.0
"""
    (model_dir / "beatrice_paraphernalia_jvs.toml").write_bytes(toml_content)
    (model_dir / "waveform_generator.bin").write_bytes(b"dummy waveform weights")
    (model_dir / "phone_extractor.bin").write_bytes(b"dummy phone weights")
    (model_dir / "pitch_estimator.bin").write_bytes(b"dummy pitch weights")

    parser = BeatriceParaphernaliaParser()
    assert parser.can_parse(str(model_dir)) is True

    info = parser.parse(str(model_dir))
    assert info.engine_id == "beatrice"
    assert info.model_format == ModelFormat.BEATRICE_PARAPHERNALIA
    assert info.name == "JVS corpus"
    assert info.license_identifier == "JVS-NonCommercial"
    assert info.commercial_eligible is False
    assert info.manifest.speaker_count == 2
    assert len(info.assets) == 4


def test_applio_rvc_parser_zero_unpickle(tmp_path):
    """Test Applio RVC parser inspects .pth ZIP container safely without unpickling bytecode."""
    model_dir = tmp_path / "rvc_model"
    model_dir.mkdir()

    # Create a synthetic PyTorch-like ZIP container (.pth)
    pth_path = model_dir / "my_voice.pth"
    with zipfile.ZipFile(pth_path, "w") as zf:
        zf.writestr("archive/data.pkl", b"simulated pickle binary")
        zf.writestr("archive/byteorder", b"little")

    # Add an index file
    index_path = model_dir / "added_IVF1024_Flat_Fast_my_voice.index"
    index_path.write_bytes(b"simulated faiss index file")

    parser = ApplioRVCParser()
    assert parser.can_parse(str(model_dir)) is True

    info = parser.parse(str(model_dir))
    assert info.engine_id == "applio"
    assert info.model_format == ModelFormat.RVC_PTH_INDEX
    assert info.name == "my_voice"
    assert info.commercial_eligible is False  # Requires explicit license review
    assert info.manifest.extra_params["has_index"] is True
    assert len(info.assets) == 2


def test_synthetic_parser(tmp_path):
    """Test Synthetic format parser for FakeEngine testing."""
    model_dir = tmp_path / "synthetic"
    model_dir.mkdir()

    manifest = b"""{
        "name": "Test Synthetic Voice",
        "version": "1.2.0",
        "description": "Test voice for CI",
        "sample_rate": 48000,
        "license": "MIT",
        "commercial_eligible": true,
        "speaker_count": 1
    }"""
    (model_dir / "manifest.json").write_bytes(manifest)

    parser = SyntheticTestParser()
    assert parser.can_parse(str(model_dir)) is True

    info = parser.parse(str(model_dir))
    assert info.engine_id == "fake"
    assert info.model_format == ModelFormat.SYNTHETIC_TEST
    assert info.name == "Test Synthetic Voice"
    assert info.sample_rate == 48000
    assert info.commercial_eligible is True
    assert info.license_identifier == "MIT"
