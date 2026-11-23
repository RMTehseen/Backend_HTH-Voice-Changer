"""Safe ctypes bindings to the Beatrice V2 VST dynamic library (C-ABI)."""

import ctypes
from pathlib import Path
from typing import Any, Optional

from hth.domain.errors import EngineError


class BeatriceNativeEngine:
    """Manages ctypes loading and lifetime of Beatrice V2 C-ABI components."""

    def __init__(self, lib_path: Path, paraphernalia_dir: Path) -> None:
        self.lib_path = Path(lib_path).resolve()
        self.paraphernalia_dir = Path(paraphernalia_dir).resolve()

        if not self.lib_path.exists():
            raise EngineError(f"Beatrice library not found at: {self.lib_path}")
        if not self.paraphernalia_dir.exists():
            raise EngineError(f"Paraphernalia directory not found at: {self.paraphernalia_dir}")

        self.lib = ctypes.CDLL(str(self.lib_path))
        self._setup_signatures()

        # Handle references
        self.phone_extractor: Optional[ctypes.c_void_p] = None
        self.phone_context: Optional[ctypes.c_void_p] = None
        self.pitch_estimator: Optional[ctypes.c_void_p] = None
        self.pitch_context: Optional[ctypes.c_void_p] = None
        self.waveform_generator: Optional[ctypes.c_void_p] = None
        self.waveform_context: Optional[ctypes.c_void_p] = None
        self.embedding_setter: Optional[ctypes.c_void_p] = None
        self.embedding_context: Optional[ctypes.c_void_p] = None

        self.num_speakers: int = 0
        self.codebooks: Any = None
        self.additive_embeddings: Any = None
        self.formant_shift_embeddings: Any = None
        self.kv_embeddings: Any = None

    def _setup_signatures(self) -> None:
        """Configures argument and return types for Beatrice C API functions."""
        # 1. Speaker Embeddings utility
        self.lib.Beatrice20rc0_ReadNSpeakers.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_int),
        ]
        self.lib.Beatrice20rc0_ReadNSpeakers.restype = ctypes.c_int

        # 2. Phone Extractor
        self.lib.Beatrice20rc0_CreatePhoneExtractor.restype = ctypes.c_void_p
        self.lib.Beatrice20rc0_ReadPhoneExtractorParameters.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
        ]
        self.lib.Beatrice20rc0_ReadPhoneExtractorParameters.restype = ctypes.c_int
        self.lib.Beatrice20rc0_DestroyPhoneExtractor.argtypes = [ctypes.c_void_p]

        self.lib.Beatrice20rc0_CreatePhoneContext1.restype = ctypes.c_void_p
        self.lib.Beatrice20rc0_DestroyPhoneContext1.argtypes = [ctypes.c_void_p]

        # 3. Pitch Estimator
        self.lib.Beatrice20rc0_CreatePitchEstimator.restype = ctypes.c_void_p
        self.lib.Beatrice20rc0_ReadPitchEstimatorParameters.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
        ]
        self.lib.Beatrice20rc0_ReadPitchEstimatorParameters.restype = ctypes.c_int
        self.lib.Beatrice20rc0_DestroyPitchEstimator.argtypes = [ctypes.c_void_p]

        self.lib.Beatrice20rc0_CreatePitchContext1.restype = ctypes.c_void_p
        self.lib.Beatrice20rc0_DestroyPitchContext1.argtypes = [ctypes.c_void_p]

        # 4. Waveform Generator
        self.lib.Beatrice20rc0_CreateWaveformGenerator.restype = ctypes.c_void_p
        self.lib.Beatrice20rc0_ReadWaveformGeneratorParameters.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
        ]
        self.lib.Beatrice20rc0_ReadWaveformGeneratorParameters.restype = ctypes.c_int
        self.lib.Beatrice20rc0_DestroyWaveformGenerator.argtypes = [ctypes.c_void_p]

        self.lib.Beatrice20rc0_CreateWaveformContext1.restype = ctypes.c_void_p
        self.lib.Beatrice20rc0_DestroyWaveformContext1.argtypes = [ctypes.c_void_p]

        # 5. Embedding Setter
        self.lib.Beatrice20rc0_CreateEmbeddingSetter.restype = ctypes.c_void_p
        self.lib.Beatrice20rc0_ReadEmbeddingSetterParameters.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
        ]
        self.lib.Beatrice20rc0_ReadEmbeddingSetterParameters.restype = ctypes.c_int
        self.lib.Beatrice20rc0_ReadSpeakerEmbeddings.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        self.lib.Beatrice20rc0_ReadSpeakerEmbeddings.restype = ctypes.c_int
        self.lib.Beatrice20rc0_DestroyEmbeddingSetter.argtypes = [ctypes.c_void_p]

        self.lib.Beatrice20rc0_CreateEmbeddingContext.restype = ctypes.c_void_p
        self.lib.Beatrice20rc0_DestroyEmbeddingContext.argtypes = [ctypes.c_void_p]

        # Real-time DSP inference signatures
        self.lib.Beatrice20rc0_ExtractPhone1.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p,
        ]
        self.lib.Beatrice20rc0_EstimatePitch1.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p,
        ]
        self.lib.Beatrice20rc0_GenerateWaveform1.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p,
        ]
        self.lib.Beatrice20rc0_SetFormantShiftEmbedding.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]

    def load_model(self) -> None:
        """Loads and initializes model parameters and allocates speaker buffers."""
        speaker_bin = str(self.paraphernalia_dir / "speaker_embeddings.bin").encode("utf-8")
        out_n = ctypes.c_int(0)
        res = self.lib.Beatrice20rc0_ReadNSpeakers(speaker_bin, ctypes.byref(out_n))
        if res != 0:
            raise EngineError(f"Failed to read speaker count (code {res})")
        self.num_speakers = out_n.value

        # Instantiate components
        self.phone_extractor = self.lib.Beatrice20rc0_CreatePhoneExtractor()
        phone_bin = str(self.paraphernalia_dir / "phone_extractor.bin").encode("utf-8")
        res = self.lib.Beatrice20rc0_ReadPhoneExtractorParameters(self.phone_extractor, phone_bin)
        if res != 0:
            raise EngineError(f"Failed to load phone extractor parameters (code {res})")
        self.phone_context = self.lib.Beatrice20rc0_CreatePhoneContext1()

        self.pitch_estimator = self.lib.Beatrice20rc0_CreatePitchEstimator()
        pitch_bin = str(self.paraphernalia_dir / "pitch_estimator.bin").encode("utf-8")
        res = self.lib.Beatrice20rc0_ReadPitchEstimatorParameters(self.pitch_estimator, pitch_bin)
        if res != 0:
            raise EngineError(f"Failed to load pitch estimator parameters (code {res})")
        self.pitch_context = self.lib.Beatrice20rc0_CreatePitchContext1()

        self.waveform_generator = self.lib.Beatrice20rc0_CreateWaveformGenerator()
        wave_bin = str(self.paraphernalia_dir / "waveform_generator.bin").encode("utf-8")
        res = self.lib.Beatrice20rc0_ReadWaveformGeneratorParameters(
            self.waveform_generator, wave_bin
        )
        if res != 0:
            raise EngineError(f"Failed to load waveform generator parameters (code {res})")
        self.waveform_context = self.lib.Beatrice20rc0_CreateWaveformContext1()

        self.embedding_setter = self.lib.Beatrice20rc0_CreateEmbeddingSetter()
        embed_bin = str(self.paraphernalia_dir / "embedding_setter.bin").encode("utf-8")
        res = self.lib.Beatrice20rc0_ReadEmbeddingSetterParameters(self.embedding_setter, embed_bin)
        if res != 0:
            raise EngineError(f"Failed to load embedding setter parameters (code {res})")
        self.embedding_context = self.lib.Beatrice20rc0_CreateEmbeddingContext()

        # Allocate embedding buffers
        self.codebooks = (ctypes.c_float * (self.num_speakers * 512 * 128))()
        self.additive_embeddings = (ctypes.c_float * (self.num_speakers * 256))()
        self.formant_shift_embeddings = (ctypes.c_float * (9 * 256))()
        self.kv_embeddings = (ctypes.c_float * (self.num_speakers * 384 * 128))()

        res = self.lib.Beatrice20rc0_ReadSpeakerEmbeddings(
            speaker_bin,
            ctypes.cast(self.codebooks, ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(self.additive_embeddings, ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(self.formant_shift_embeddings, ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(self.kv_embeddings, ctypes.POINTER(ctypes.c_float)),
        )
        if res != 0:
            raise EngineError(f"Failed to read speaker embeddings (code {res})")

    def unload(self) -> None:
        """Safely destroys all native Beatrice contexts and extractor handles."""
        if self.phone_context:
            self.lib.Beatrice20rc0_DestroyPhoneContext1(self.phone_context)
            self.phone_context = None
        if self.pitch_context:
            self.lib.Beatrice20rc0_DestroyPitchContext1(self.pitch_context)
            self.pitch_context = None
        if self.waveform_context:
            self.lib.Beatrice20rc0_DestroyWaveformContext1(self.waveform_context)
            self.waveform_context = None
        if self.embedding_context:
            self.lib.Beatrice20rc0_DestroyEmbeddingContext(self.embedding_context)
            self.embedding_context = None

        if self.phone_extractor:
            self.lib.Beatrice20rc0_DestroyPhoneExtractor(self.phone_extractor)
            self.phone_extractor = None
        if self.pitch_estimator:
            self.lib.Beatrice20rc0_DestroyPitchEstimator(self.pitch_estimator)
            self.pitch_estimator = None
        if self.waveform_generator:
            self.lib.Beatrice20rc0_DestroyWaveformGenerator(self.waveform_generator)
            self.waveform_generator = None
        if self.embedding_setter:
            self.lib.Beatrice20rc0_DestroyEmbeddingSetter(self.embedding_setter)
            self.embedding_setter = None
