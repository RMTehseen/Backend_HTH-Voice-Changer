"""Archive security validator — Zip Slip, archive bomb, and magic byte protections."""

from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from hth.core.logging import get_logger

logger = get_logger("hth.security.archive")

# Security ceilings for voice model archives
MAX_COMPRESSED_BYTES = 250 * 1024 * 1024  # 250 MB
MAX_UNCOMPRESSED_BYTES = 600 * 1024 * 1024  # 600 MB
MAX_MEMBER_COUNT = 150
MAX_COMPRESSION_RATIO = 10.0  # Max uncompressed / compressed ratio
CHUNK_SIZE = 64 * 1024  # 64 KB streaming chunks


class ArchiveSecurityError(Exception):
    """Raised when an archive violates safety or security policies."""


@dataclass
class ExtractedAsset:
    relative_path: str
    absolute_path: str
    sha256: str
    size_bytes: int


class ArchiveSecurityValidator:
    """Zero-trust archive security validator and safe quarantine extractor."""

    @staticmethod
    def verify_magic_bytes(data_prefix: bytes) -> str:
        """Identify format by magic bytes. Raises ArchiveSecurityError if unsupported."""
        if len(data_prefix) < 4:
            raise ArchiveSecurityError("Input file is too small to determine format")

        if data_prefix.startswith(b"PK\x03\x04") or data_prefix.startswith(b"PK\x05\x06"):
            return "zip"
        if len(data_prefix) >= 262 and data_prefix[257:262] == b"ustar":
            return "tar"
        if data_prefix.startswith(b"\x1f\x8b"):  # gzip / tar.gz
            return "tar.gz"
        if data_prefix.startswith(b"BZh"):  # bzip2 / tar.bz2
            return "tar.bz2"

        # Safetensors: first 8 bytes is an unsigned 64-bit little-endian integer for header length
        if len(data_prefix) >= 8:
            header_len = int.from_bytes(data_prefix[:8], "little")
            if 0 < header_len < 100 * 1024 * 1024:
                return "safetensors"

        # Raw ONNX / binary
        if data_prefix.startswith(b"\x08"):
            return "onnx"

        return "binary"

    @classmethod
    def validate_and_extract_zip(
        cls,
        zip_path: str | Path,
        quarantine_dir: str | Path,
    ) -> list[ExtractedAsset]:
        """Validate and safely extract a ZIP archive to a quarantine directory."""
        zip_path = Path(zip_path)
        quarantine_dir = Path(quarantine_dir).resolve()
        os.makedirs(quarantine_dir, exist_ok=True)

        compressed_size = zip_path.stat().st_size
        if compressed_size > MAX_COMPRESSED_BYTES:
            raise ArchiveSecurityError(
                f"Archive compressed size ({compressed_size} bytes) exceeds limit ({MAX_COMPRESSED_BYTES} bytes)"
            )

        # Verify magic bytes
        with open(zip_path, "rb") as f:
            header = f.read(16)
            fmt = cls.verify_magic_bytes(header)
            if fmt != "zip":
                raise ArchiveSecurityError(
                    f"File magic bytes do not match ZIP container: detected {fmt}"
                )

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                infolist = zf.infolist()
                if len(infolist) > MAX_MEMBER_COUNT:
                    raise ArchiveSecurityError(
                        f"Archive contains {len(infolist)} files, exceeding limit ({MAX_MEMBER_COUNT})"
                    )

                # Pre-scan headers for Zip Slip and total declared uncompressed size
                total_uncompressed = 0
                for info in infolist:
                    cls._check_path_traversal(info.filename)
                    total_uncompressed += info.file_size

                if total_uncompressed > MAX_UNCOMPRESSED_BYTES:
                    raise ArchiveSecurityError(
                        f"Total uncompressed size ({total_uncompressed} bytes) exceeds limit ({MAX_UNCOMPRESSED_BYTES} bytes)"
                    )

                if compressed_size > 0:
                    ratio = total_uncompressed / compressed_size
                    if ratio > MAX_COMPRESSION_RATIO and total_uncompressed > 5 * 1024 * 1024:
                        raise ArchiveSecurityError(
                            f"Suspicious compression ratio {ratio:.1f}:1 exceeds bomb threshold ({MAX_COMPRESSION_RATIO}:1)"
                        )

                # Streaming extraction with continuous size check
                extracted_assets: list[ExtractedAsset] = []
                extracted_bytes = 0

                for info in infolist:
                    # Ignore directory entries
                    if info.is_dir() or info.filename.endswith("/"):
                        continue

                    cls._check_path_traversal(info.filename)
                    # Normalize relative destination path
                    norm_parts = PurePosixPath(info.filename).parts
                    target_file = quarantine_dir.joinpath(*norm_parts).resolve()

                    # Re-verify path confinement
                    if not str(target_file).startswith(str(quarantine_dir)):
                        raise ArchiveSecurityError(
                            f"Zip Slip detected: target {target_file} escapes quarantine {quarantine_dir}"
                        )

                    target_file.parent.mkdir(parents=True, exist_ok=True)

                    hasher = hashlib.sha256()
                    current_file_size = 0

                    with zf.open(info, "r") as src, open(target_file, "wb") as dst:
                        while True:
                            chunk = src.read(CHUNK_SIZE)
                            if not chunk:
                                break
                            current_file_size += len(chunk)
                            extracted_bytes += len(chunk)

                            if extracted_bytes > MAX_UNCOMPRESSED_BYTES:
                                raise ArchiveSecurityError(
                                    f"Decompressed stream exceeded safe limit ({MAX_UNCOMPRESSED_BYTES} bytes)"
                                )

                            hasher.update(chunk)
                            dst.write(chunk)

                    rel_path = str(target_file.relative_to(quarantine_dir)).replace("\\", "/")
                    extracted_assets.append(
                        ExtractedAsset(
                            relative_path=rel_path,
                            absolute_path=str(target_file),
                            sha256=hasher.hexdigest(),
                            size_bytes=current_file_size,
                        )
                    )

                return extracted_assets

        except Exception as exc:
            # Atomic rollback: wipe quarantine directory on failure
            logger.error("Extraction validation failed, rolling back quarantine", error=str(exc))
            shutil.rmtree(quarantine_dir, ignore_errors=True)
            if isinstance(exc, ArchiveSecurityError):
                raise
            raise ArchiveSecurityError(f"Corrupt or invalid archive: {exc}") from exc

    @classmethod
    def validate_and_extract_tar(
        cls,
        tar_path: str | Path,
        quarantine_dir: str | Path,
    ) -> list[ExtractedAsset]:
        """Validate and safely extract a TAR / TAR.GZ archive to a quarantine directory."""
        tar_path = Path(tar_path)
        quarantine_dir = Path(quarantine_dir).resolve()
        os.makedirs(quarantine_dir, exist_ok=True)

        compressed_size = tar_path.stat().st_size
        if compressed_size > MAX_COMPRESSED_BYTES:
            raise ArchiveSecurityError(
                f"Archive compressed size ({compressed_size} bytes) exceeds limit ({MAX_COMPRESSED_BYTES} bytes)"
            )

        try:
            with tarfile.open(tar_path, "r:*") as tf:
                members = tf.getmembers()
                if len(members) > MAX_MEMBER_COUNT:
                    raise ArchiveSecurityError(
                        f"Archive contains {len(members)} files, exceeding limit ({MAX_MEMBER_COUNT})"
                    )

                total_uncompressed = 0
                for m in members:
                    cls._check_path_traversal(m.name)
                    if m.islnk() or m.issym():
                        raise ArchiveSecurityError(
                            f"Symbolic and hard links are prohibited: {m.name}"
                        )
                    total_uncompressed += m.size

                if total_uncompressed > MAX_UNCOMPRESSED_BYTES:
                    raise ArchiveSecurityError(
                        f"Total uncompressed size ({total_uncompressed} bytes) exceeds limit ({MAX_UNCOMPRESSED_BYTES} bytes)"
                    )

                extracted_assets: list[ExtractedAsset] = []
                extracted_bytes = 0

                for m in members:
                    if not m.isfile():
                        continue

                    cls._check_path_traversal(m.name)
                    norm_parts = PurePosixPath(m.name).parts
                    target_file = quarantine_dir.joinpath(*norm_parts).resolve()

                    if not str(target_file).startswith(str(quarantine_dir)):
                        raise ArchiveSecurityError(
                            f"Tar Slip detected: {m.name} escapes quarantine"
                        )

                    target_file.parent.mkdir(parents=True, exist_ok=True)

                    hasher = hashlib.sha256()
                    current_file_size = 0
                    src_f = tf.extractfile(m)
                    if src_f is None:
                        continue

                    with src_f, open(target_file, "wb") as dst:
                        while True:
                            chunk = src_f.read(CHUNK_SIZE)
                            if not chunk:
                                break
                            current_file_size += len(chunk)
                            extracted_bytes += len(chunk)

                            if extracted_bytes > MAX_UNCOMPRESSED_BYTES:
                                raise ArchiveSecurityError(
                                    f"Decompressed stream exceeded safe limit ({MAX_UNCOMPRESSED_BYTES} bytes)"
                                )

                            hasher.update(chunk)
                            dst.write(chunk)

                    rel_path = str(target_file.relative_to(quarantine_dir)).replace("\\", "/")
                    extracted_assets.append(
                        ExtractedAsset(
                            relative_path=rel_path,
                            absolute_path=str(target_file),
                            sha256=hasher.hexdigest(),
                            size_bytes=current_file_size,
                        )
                    )

                return extracted_assets

        except Exception as exc:
            shutil.rmtree(quarantine_dir, ignore_errors=True)
            if isinstance(exc, ArchiveSecurityError):
                raise
            raise ArchiveSecurityError(f"Corrupt or invalid tar archive: {exc}") from exc

    @staticmethod
    def _check_path_traversal(filename: str) -> None:
        """Detect Zip Slip / Tar Slip path traversal attacks."""
        if "\0" in filename:
            raise ArchiveSecurityError("Null byte in archive entry filename")

        norm = filename.replace("\\", "/")
        if norm.startswith("/"):
            raise ArchiveSecurityError(f"Absolute path in archive is prohibited: {filename}")

        if len(norm) > 1 and norm[1] == ":" and norm[0].isalpha():
            raise ArchiveSecurityError(f"Drive letter in archive is prohibited: {filename}")

        parts = PurePosixPath(norm).parts
        if ".." in parts:
            raise ArchiveSecurityError(f"Path traversal ('..') detected in archive: {filename}")
