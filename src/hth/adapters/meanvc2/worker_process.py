#!/usr/bin/env python3
"""
MeanVC2 inference worker process.

This runs in a SEPARATE Python process (Python 3.11 with PyTorch/s3prl).
Communication with HTH backend is via stdin/stdout JSON messages + shared memory.

Protocol:
  STDIN: JSON lines with commands
  STDOUT: JSON lines with responses/events

Commands (JSON):
  {"cmd": "load", "config": {...}}
  {"cmd": "set_target", "profile_id": "...", "embedding_path": "...", "wav_path": "..."}
  {"cmd": "process_chunk", "data_b64": "..."}  # base64 float32 LE PCM 16kHz
  {"cmd": "ping"}
  {"cmd": "shutdown"}

Events (JSON):
  {"event": "ready"}
  {"event": "loaded", "variant": "...", "device": "..."}
  {"event": "target_set", "profile_id": "..."}
  {"event": "audio_out", "data_b64": "..."}  # converted PCM
  {"event": "metrics", "inference_ms": ..., "rtf": ...}
  {"event": "error", "message": "..."}
  {"event": "pong"}
  {"event": "shutdown_ok"}
"""

from __future__ import annotations

import base64
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None


def _add_meanvc2_to_path(source_dir: str):
    runtime_path = str(Path(source_dir) / "runtime")
    if runtime_path not in sys.path:
        sys.path.insert(0, runtime_path)
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)


def _emit(obj: dict):
    line = json.dumps(obj)
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


class MeanVC2Worker:
    """Wraps VCRunner for use as a subprocess worker."""

    def __init__(self):
        self._runner = None
        self._lock = threading.Lock()
        self._out_buffer = []
        self._out_lock = threading.Lock()

    def load(self, config: dict) -> None:
        import torch
        source_dir = config["source_dir"]
        _add_meanvc2_to_path(source_dir)

        import json as _json

        from src.dit import DiT
        from src.speaker import extract_embedding, init_speaker_model

        variant = config.get("model_variant", "40ms")
        requested_device = config.get("device", "cpu")
        if requested_device == "cuda":
            if torch.cuda.is_available():
                device = "cuda"
            else:
                _emit({
                    "event": "warning",
                    "message": "CUDA requested but torch.cuda.is_available() is False. Falling back to CPU.",
                })
                device = "cpu"
        else:
            device = "cpu"

        num_threads = config.get("num_threads", 4)
        torch.set_num_threads(num_threads)
        if hasattr(torch, "set_num_interop_threads"):
            try:
                torch.set_num_interop_threads(1)
            except Exception:
                pass
        torch.backends.cudnn.enabled = False

        # Paths
        vc_ckpt = config["vc_model_path"]
        vc_config_path = config["vc_config_path"]
        asr_ckpt = config["asr_model_path"]
        vocoder_ckpt = config["vocoder_path"]
        spk_model_path = config["speaker_model_path"]
        wavlm_cfg_path = config.get("wavlm_config_path", "")

        # ASR (TorchScript traced model must stay on CPU — matches upstream run_rt.py)
        _emit({"event": "progress", "step": "loading_asr"})
        asr = torch.jit.load(asr_ckpt, map_location="cpu")
        asr.eval()

        # VC model
        _emit({"event": "progress", "step": "loading_vc"})
        with open(vc_config_path) as f:
            vc_config = _json.load(f)
        model = DiT(**vc_config["model"])
        if vc_ckpt.endswith(".safetensors"):
            from safetensors.torch import load_file
            weights = load_file(vc_ckpt)
            model.load_state_dict(weights, strict=False)
        else:
            ckpt = torch.load(vc_ckpt, weights_only=True, map_location=device)
            if "ema_model_state_dict" in ckpt:
                weights = {k.replace("ema_model.", ""): v
                           for k, v in ckpt["ema_model_state_dict"].items()
                           if k.replace("ema_model.", "") not in ("initted", "step")}
            elif "model_state_dict" in ckpt:
                weights = ckpt["model_state_dict"]
            else:
                weights = ckpt
            model.load_state_dict(weights, strict=False)
        model.float().eval().to(device)

        # Vocoder
        _emit({"event": "progress", "step": "loading_vocoder"})
        vocoder = torch.jit.load(vocoder_ckpt)
        if device == "cuda":
            vocoder = vocoder.to(device)

        # Speaker model
        _emit({"event": "progress", "step": "loading_speaker_model"})
        spk_model = init_speaker_model(
            spk_model_path, device,
            wavlm_config=wavlm_cfg_path if os.path.exists(wavlm_cfg_path) else None,
        )

        # Build runner-like object
        from src.dit import DiT

        # Store everything in a runner-compatible namespace
        class _Runner:
            pass

        runner = _Runner()
        runner.device = device
        runner.asr = asr
        runner.vc = model
        runner.vocoder = vocoder
        runner.spk_model = spk_model
        runner.vc_spk_emb = None
        runner.vc_gtm_kv = None
        runner.variant = variant
        runner.n_steps = int(config.get("cfm_steps", 1 if device == "cpu" else 2))

        # Streaming params from upstream run_rt.py
        if variant == "120ms":
            runner.chunk_size = 12
            runner._bn_window = 19
            runner._bn_stride = 16
            runner._required_cache_size = 8
            runner._asr_offset_init = 8
            runner._asr_offset_step = 4
        else:
            runner.chunk_size = 4
            runner._bn_window = 11
            runner._bn_stride = 8
            runner._required_cache_size = 4
            runner._asr_offset_init = 4
            runner._asr_offset_step = 2

        runner.block_size = 4
        runner.CHUNK = 2560

        # Vocoder overlap params
        runner.vocoder_overlap = 2
        runner.upsample_factor = 160
        runner.vocoder_wav_overlap = runner.upsample_factor
        runner.vocoder_wav_pad = (runner.vocoder_overlap - 1) * runner.upsample_factor - runner.vocoder_wav_overlap
        runner.down_linspace = torch.linspace(1, 0, steps=runner.vocoder_wav_overlap)
        runner.up_linspace = torch.linspace(0, 1, steps=runner.vocoder_wav_overlap)
        runner._min_bn_len = runner.chunk_size + runner.block_size

        self._runner = runner
        self._init_cache()
        self._extract_embedding = extract_embedding

        _emit({"event": "loaded", "variant": variant, "device": device, "cuda_available": torch.cuda.is_available()})

    def _init_cache(self):
        import torch
        r = self._runner
        r.samples_cache = np.zeros(0, dtype=np.float32)
        r.frame_cache = None
        r.encoder_output_cache = None
        r.asr_offset = r._asr_offset_init
        # ASR stays on CPU so its caches must remain CPU tensors
        r.asr_att_cache = torch.zeros(6, 4, r._required_cache_size, 128)
        r.asr_cnn_cache = torch.zeros(6, 1, 256, 8)
        r.vc_kv_cache = None
        r.vc_offset = 0
        r.noise_cache = None
        r.vocoder_cache = None
        r.last_wav = None
        r.bn_buffer = None

    def set_target(self, wav_path: str, profile_id: str, embedding_path: str = "") -> None:
        global torch, F
        if torch is None:
            import torch
            import torch.nn.functional as F
        r = self._runner
        if not r:
            raise RuntimeError("Worker not loaded")

        if embedding_path and Path(embedding_path).is_file():
            # Load cached embedding
            emb = torch.load(embedding_path, map_location="cpu", weights_only=True)
            r.vc_spk_emb = emb.to(r.device)
        elif wav_path and Path(wav_path).is_file():
            # Compute from wav
            from src.speaker import extract_embedding
            r.vc_spk_emb = extract_embedding(r.spk_model, wav_path, device=r.device)
            if embedding_path:
                Path(embedding_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(r.vc_spk_emb.cpu(), embedding_path)
        else:
            # Seeded cloud profile without local reference WAV — synthesize deterministic 256-dim embedding
            import hashlib
            seed_bytes = hashlib.sha256(profile_id.encode("utf-8")).digest()
            seed_int = int.from_bytes(seed_bytes[:4], "little")
            gen = torch.Generator(device="cpu").manual_seed(seed_int)
            vec = torch.randn(1, 256, generator=gen, dtype=torch.float32)
            vec = vec / (vec.norm(dim=-1, keepdim=True) + 1e-8) * 12.0
            r.vc_spk_emb = vec.to(r.device)
            _emit({
                "event": "warning",
                "message": f"Reference audio not present on server for {profile_id}; using deterministic fallback embedding. Upload a reference WAV for custom timbre.",
            })

        with torch.no_grad():
            r.vc_gtm_kv = r.vc.gtm(r.vc_spk_emb)

        # Reset streaming caches when target changes
        self._init_cache()
        _emit({"event": "target_set", "profile_id": profile_id})

    def process_chunk(self, samples: np.ndarray) -> Optional[np.ndarray]:
        """Process one audio chunk. Returns converted audio or None."""
        import torch.nn.functional as F
        r = self._runner
        if r is None or r.vc_spk_emb is None or r.vc_gtm_kv is None:
            return None

        # === ASR encode (from upstream run_rt.py _encode_chunk) ===
        inference_ctx = getattr(torch, "inference_mode", torch.no_grad)
        with inference_ctx():
            padded = (np.concatenate((r.samples_cache, samples))
                      if r.samples_cache.size else samples)

            if len(padded) < 400:
                r.samples_cache = padded
                return None

            wav_scaled = padded * (1 << 15)
            wav_t = torch.from_numpy(wav_scaled).unsqueeze(0)
            import torchaudio.compliance.kaldi as kaldi
            fbanks = kaldi.fbank(wav_t, frame_length=25, frame_shift=10,
                                 snip_edges=True, num_mel_bins=80,
                                 energy_floor=0.0, dither=0.0,
                                 sample_frequency=16000)

            r.samples_cache = padded[160 * fbanks.shape[0]:]

            if r.frame_cache is not None:
                fbanks = torch.cat([r.frame_cache, fbanks], dim=0)

            if r.asr_offset >= 4000:
                cache_t = r.asr_att_cache.size(2) if r.asr_att_cache.dim() > 0 else 0
                r.asr_offset = max(cache_t, r.asr_offset - 4000)

            bns = []
            i = 0
            while i + r._bn_window <= fbanks.shape[0]:
                fb = fbanks[i:i + r._bn_window].unsqueeze(0)
                out, r.asr_att_cache, r.asr_cnn_cache = r.asr(
                    fb,
                    torch.tensor(r.asr_offset, dtype=torch.int64),
                    torch.tensor(r._required_cache_size, dtype=torch.int64),
                    r.asr_att_cache, r.asr_cnn_cache,
                )
                bns.append(out.squeeze(0).detach())
                r.asr_offset += r._asr_offset_step
                i += r._bn_stride
            r.frame_cache = fbanks[i:]

            if not bns:
                return None

            bn = torch.cat(bns, dim=0).unsqueeze(0)  # [1, N, 256]

            if r.encoder_output_cache is not None:
                bn = torch.cat([r.encoder_output_cache, bn], dim=1)
            r.encoder_output_cache = bn[:, -1:, :]

            vc_chunk = 16
            if bn.shape[1] >= 2:
                bn_up = bn.transpose(1, 2)
                bn_up = F.interpolate(bn_up, size=vc_chunk + 1,
                                      mode='linear', align_corners=True)
                bn_up = bn_up.transpose(1, 2)[:, 1:, :]
            else:
                return None

            bn_new = bn_up

        # Build BN buffer
        if r.device == "cuda":
            bn_new = bn_new.to(r.device)

        if r.bn_buffer is None:
            r.bn_buffer = bn_new
        else:
            r.bn_buffer = torch.cat([r.bn_buffer, bn_new], dim=1)

        # Cap BN buffer to avoid memory buildup if input outpaces inference
        MAX_BN_FRAMES = 64
        if r.bn_buffer.shape[1] > MAX_BN_FRAMES:
            r.bn_buffer = r.bn_buffer[:, -MAX_BN_FRAMES:, :]

        if r.bn_buffer.shape[1] < r._min_bn_len:
            return None

        # VC + Vocoder
        wav_parts = []
        max_steps = 4
        for _ in range(max_steps):
            if r.bn_buffer.shape[1] < r._min_bn_len:
                break
            cond = r.bn_buffer[:, :r._min_bn_len, :]
            r.bn_buffer = r.bn_buffer[:, r.chunk_size:, :]

            # === VC step (optimized 1-step CFM on CPU cuts inference time in half) ===
            N = getattr(r, "n_steps", 1)
            dt = 1.0 / N
            inference_ctx = getattr(torch, "inference_mode", torch.no_grad)
            with inference_ctx():
                x = torch.randn(1, cond.shape[1], 80, device=r.device, dtype=cond.dtype)
                if r.noise_cache is not None:
                    x[:, :r.block_size, :] = r.noise_cache
                r.noise_cache = x[:, -r.block_size:, :].clone()

                if r.vc_offset >= 4000:
                    r.vc_offset = 0
                    r.vc_kv_cache = None

                pre_kv = r.vc_kv_cache
                for i in range(N):
                    t_val = 1.0 - i * dt
                    r_val = max(0.0, t_val - dt)
                    t_tensor = torch.full((1,), t_val, device=r.device)
                    r_tensor = torch.full((1,), r_val, device=r.device)
                    u, tmp_kv = r.vc(
                        x, t_tensor, r_tensor,
                        cache=None, cond=cond, spks=r.vc_spk_emb,
                        offset=r.vc_offset, is_inference=True,
                        kv_cache=pre_kv, gtm_kv=r.vc_gtm_kv,
                    )
                    x = x - dt * u
                    if i == N - 1:
                        r.vc_kv_cache = tmp_kv

                r.vc_offset += r.chunk_size
                mel = x[:, :r.chunk_size, :].transpose(1, 2)

            # === Vocoder ===
            with torch.no_grad():
                if r.vocoder_cache is not None:
                    mel = torch.cat([r.vocoder_cache, mel], dim=-1)
                r.vocoder_cache = mel[:, :, -r.vocoder_overlap:]
                mel_in = (mel + 1) / 2
                if r.device == "cuda":
                    mel_in = mel_in.to(r.device)
                wav = r.vocoder.decode(mel_in).squeeze().detach().cpu().numpy()
                if r.last_wav is not None:
                    wav = wav[r.vocoder_wav_pad:]
                    front = wav[:r.vocoder_wav_overlap]
                    smooth = (r.last_wav * r.down_linspace.numpy()
                              + front * r.up_linspace.numpy())
                    new_wav = np.concatenate([smooth, wav[r.vocoder_wav_overlap:-r.vocoder_wav_overlap]])
                else:
                    new_wav = wav[:-r.vocoder_wav_overlap]
                r.last_wav = wav[-r.vocoder_wav_overlap:]
            wav_parts.append(new_wav)

        if not wav_parts:
            return None
        return np.concatenate(wav_parts)

    def warmup(self) -> None:
        """Warmup the model, then reset caches."""
        r = self._runner
        if r is None or r.vc_spk_emb is None:
            return
        dummy = np.zeros(r.CHUNK, dtype=np.float32)
        for _ in range(3):
            self.process_chunk(dummy)
        self._init_cache()
        _emit({"event": "warmup_done"})


def main():
    worker = MeanVC2Worker()
    _emit({"event": "ready"})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            _emit({"event": "error", "message": f"Invalid JSON: {line[:80]}"})
            continue

        cmd = msg.get("cmd", "")

        try:
            if cmd == "ping":
                _emit({"event": "pong", "ts": time.time()})

            elif cmd == "load":
                worker.load(msg["config"])

            elif cmd == "warmup":
                worker.warmup()

            elif cmd == "set_target":
                worker.set_target(
                    wav_path=msg.get("wav_path", ""),
                    profile_id=msg.get("profile_id", ""),
                    embedding_path=msg.get("embedding_path", ""),
                )

            elif cmd == "init_cache":
                if worker._runner:
                    worker._init_cache()
                    _emit({"event": "cache_reset"})

            elif cmd == "process_chunk":
                t0 = time.perf_counter()
                raw = base64.b64decode(msg["data_b64"])
                samples = np.frombuffer(raw, dtype=np.float32).copy()
                out = worker.process_chunk(samples)
                t1 = time.perf_counter()
                inference_ms = (t1 - t0) * 1000
                chunk_ms = len(samples) / 16.0  # 16000 Hz => samples/16 = ms

                if out is not None:
                    out_b64 = base64.b64encode(out.astype(np.float32).tobytes()).decode()
                    _emit({
                        "event": "audio_out",
                        "data_b64": out_b64,
                        "n_samples": len(out),
                        "inference_ms": round(inference_ms, 2),
                        "chunk_ms": round(chunk_ms, 2),
                        "rtf": round(inference_ms / max(chunk_ms, 1.0), 4),
                    })
                else:
                    _emit({
                        "event": "audio_buffering",
                        "inference_ms": round(inference_ms, 2),
                    })

            elif cmd == "shutdown":
                _emit({"event": "shutdown_ok"})
                break

            else:
                _emit({"event": "error", "message": f"Unknown command: {cmd}"})

        except Exception as e:
            import traceback
            _emit({"event": "error", "message": str(e), "traceback": traceback.format_exc()})

    sys.exit(0)


if __name__ == "__main__":
    main()
