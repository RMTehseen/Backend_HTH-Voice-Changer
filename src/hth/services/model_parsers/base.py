"""Base interface and data classes for format parsers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from hth.domain.model_contracts import (
    AssetDescriptor,
    ModelFormat,
    ModelManifestMetadata,
)


@dataclass
class ParsedModelInfo:
    engine_id: str
    model_format: ModelFormat
    name: str
    version: str
    description: str
    sample_rate: int
    license_identifier: str
    license_source_url: str
    commercial_eligible: bool
    manifest: ModelManifestMetadata
    assets: list[AssetDescriptor]


class BaseFormatParser(ABC):
    """Abstract interface for provider-neutral model format parsers."""

    @abstractmethod
    def can_parse(self, directory_path: str) -> bool:
        """Check whether the directory matches this format without deep mutation."""
        pass

    @abstractmethod
    def parse(self, directory_path: str) -> ParsedModelInfo:
        """Parse directory contents into ParsedModelInfo. Raises ValueError on schema error."""
        pass
