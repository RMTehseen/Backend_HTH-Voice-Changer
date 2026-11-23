"""Model format parsers package."""

from hth.services.model_parsers.applio_rvc import ApplioRVCParser
from hth.services.model_parsers.base import BaseFormatParser, ParsedModelInfo
from hth.services.model_parsers.beatrice import BeatriceParaphernaliaParser
from hth.services.model_parsers.synthetic import SyntheticTestParser

__all__ = [
    "BaseFormatParser",
    "ParsedModelInfo",
    "BeatriceParaphernaliaParser",
    "ApplioRVCParser",
    "SyntheticTestParser",
    "get_parser_for_extracted_directory",
]


def get_parser_for_extracted_directory(directory_path: str) -> BaseFormatParser | None:
    """Detect matching format parser based on directory contents."""
    parsers = [
        BeatriceParaphernaliaParser(),
        SyntheticTestParser(),
        ApplioRVCParser(),
    ]
    for p in parsers:
        if p.can_parse(directory_path):
            return p
    return None
