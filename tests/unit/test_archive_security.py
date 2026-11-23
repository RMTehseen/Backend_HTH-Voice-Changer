"""Unit tests for archive security validator: Zip Slip, bomb mitigation, and magic bytes."""

import io
import tarfile
import zipfile

import pytest

from hth.services.archive_validator import (
    ArchiveSecurityError,
    ArchiveSecurityValidator,
)


def test_magic_bytes_detection():
    # Valid ZIP header
    assert ArchiveSecurityValidator.verify_magic_bytes(b"PK\x03\x04\x14\x00\x00") == "zip"
    # GZIP header
    assert ArchiveSecurityValidator.verify_magic_bytes(b"\x1f\x8b\x08\x00") == "tar.gz"
    # Safetensors header
    safetensors_prefix = (1024).to_bytes(8, "little") + b'{"test":1}'
    assert ArchiveSecurityValidator.verify_magic_bytes(safetensors_prefix) == "safetensors"
    # Too small
    with pytest.raises(ArchiveSecurityError):
        ArchiveSecurityValidator.verify_magic_bytes(b"PK")


def test_zip_slip_path_traversal_rejected(tmp_path):
    """Ensure archives containing '../' path traversal entries are strictly rejected."""
    bad_zip = tmp_path / "evil.zip"
    quarantine = tmp_path / "quarantine"

    with zipfile.ZipFile(bad_zip, "w") as zf:
        zf.writestr("../escape.txt", b"malicious content outside target")

    with pytest.raises(ArchiveSecurityError, match="Path traversal"):
        ArchiveSecurityValidator.validate_and_extract_zip(bad_zip, quarantine)

    # Quarantine should be cleaned up
    assert not quarantine.exists()


def test_zip_absolute_path_rejected(tmp_path):
    """Ensure archives containing absolute paths are rejected."""
    bad_zip = tmp_path / "abs.zip"
    quarantine = tmp_path / "quarantine"

    with zipfile.ZipFile(bad_zip, "w") as zf:
        zf.writestr("/root/pwn.txt", b"malicious absolute path")

    with pytest.raises(ArchiveSecurityError, match="Absolute path"):
        ArchiveSecurityValidator.validate_and_extract_zip(bad_zip, quarantine)


def test_zip_drive_letter_rejected(tmp_path):
    """Ensure archives containing Windows drive letters are rejected."""
    bad_zip = tmp_path / "drive.zip"
    quarantine = tmp_path / "quarantine"

    with zipfile.ZipFile(bad_zip, "w") as zf:
        zf.writestr("C:/Windows/evil.dll", b"evil binary")

    with pytest.raises(ArchiveSecurityError, match="Drive letter"):
        ArchiveSecurityValidator.validate_and_extract_zip(bad_zip, quarantine)


def test_null_byte_path_rejected(tmp_path):
    """Ensure null bytes in paths are detected and rejected."""
    with pytest.raises(ArchiveSecurityError, match="Null byte"):
        ArchiveSecurityValidator._check_path_traversal("voice\0model.bin")


def test_valid_zip_extraction_and_hashes(tmp_path):
    """Ensure a well-formed ZIP archive extracts cleanly with correct hashes."""
    good_zip = tmp_path / "good.zip"
    quarantine = tmp_path / "quarantine"

    with zipfile.ZipFile(good_zip, "w") as zf:
        zf.writestr("manifest.json", b'{"name": "test_model"}')
        zf.writestr("weights/model.bin", b"binary weights simulation 12345")

    assets = ArchiveSecurityValidator.validate_and_extract_zip(good_zip, quarantine)

    assert len(assets) == 2
    paths = {a.relative_path for a in assets}
    assert "manifest.json" in paths
    assert "weights/model.bin" in paths
    assert all(len(a.sha256) == 64 for a in assets)
    assert (quarantine / "manifest.json").exists()
    assert (quarantine / "weights" / "model.bin").exists()


def test_tar_slip_traversal_rejected(tmp_path):
    """Ensure TAR archives with traversal paths are rejected."""
    bad_tar = tmp_path / "evil.tar"
    quarantine = tmp_path / "quarantine"

    with tarfile.open(bad_tar, "w") as tf:
        ti = tarfile.TarInfo(name="../escape.txt")
        ti.size = 12
        tf.addfile(ti, io.BytesIO(b"evil escape!"))

    with pytest.raises(ArchiveSecurityError, match="Path traversal"):
        ArchiveSecurityValidator.validate_and_extract_tar(bad_tar, quarantine)

    assert not quarantine.exists()
