#!/usr/bin/env python3
"""
Colab Runner: Starts FastAPI backend with an automatic, free Cloudflare Tunnel.
Zero cost, zero registration, zero credit card required.
"""
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)


def check_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ NVIDIA GPU Detected: {gpu_name} (CUDA Active)")
        else:
            print("⚠️ No CUDA GPU detected. Running in CPU mode (slower).")
    except ImportError:
        print("⚠️ PyTorch not loaded yet.")


def setup_cloudflared() -> Path:
    """Download cloudflared binary for Linux if not present."""
    bin_path = Path("/tmp/cloudflared") if os.name != "nt" else Path("./cloudflared.exe")
    if bin_path.exists():
        return bin_path

    print("🌐 Setting up free Cloudflare Tunnel...")
    if os.name != "nt":
        url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
    else:
        url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"

    urllib.request.urlretrieve(url, str(bin_path))
    if os.name != "nt":
        bin_path.chmod(0o755)
    return bin_path


def _reset_stale_cloud_profiles() -> None:
    """
    Ensure Cloud profile store does not contain stale synthetic oscillator WAVs
    or legacy non-neutral post-DSP defaults (pitch/timbre/telephone/room).
    Real human reference WAVs are synced automatically from the desktop UI.
    """
    import json

    if os.name == "nt":
        profile_base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        profile_base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    profile_dir = profile_base / "HTH" / "MeanVC2Profiles"
    profiles_json = profile_dir / "profiles.json"
    if not profiles_json.exists():
        return

    try:
        with open(profiles_json, encoding="utf-8") as f:
            data = json.load(f)
        from hth.adapters.meanvc2.voice_profile import VoiceProfileStore
        curated_by_id = {
            p.profile_id: p
            for p in VoiceProfileStore(str(profile_dir), auto_seed=False)._build_default_profiles()
        }
        changed = False
        for pid, pdata in data.items():
            ref = pdata.get("reference_wav_path", "")
            dur = float(pdata.get("reference_duration_s") or 0.0)
            # Remove stale synthetic oscillator / ultra-short (< 5s) WAV references so natural human WAVs sync cleanly
            is_short_synth = (0.0 < dur < 5.0) or (
                ref and Path(ref).is_file() and Path(ref).stat().st_size in (128044, 160044, 288044)
            )
            if ref and ("default_references" in ref or is_short_synth):
                pdata["reference_wav_path"] = ""
                pdata["reference_duration_s"] = 0.0
                emb = profile_dir / f"emb_{pid}.pt"
                if emb.is_file():
                    emb.unlink()
                changed = True
            # Reset legacy room/telephone/radio FX, and upgrade flat untouched profiles to calibrated age/vocal presets
            if pdata.get("sound_effect") in ("room", "telephone", "radio"):
                pdata["pitch_shift"] = 0.0
                pdata["timbre_ratio"] = 1.0
                pdata["volume_gain"] = 1.0
                pdata["sound_effect"] = "none"
                changed = True
            curated = curated_by_id.get(pid)
            if (
                curated is not None
                and float(pdata.get("pitch_shift", 0.0)) == 0.0
                and float(pdata.get("timbre_ratio", 1.0)) == 1.0
                and pdata.get("sound_effect", "none") == "none"
                and (curated.pitch_shift != 0.0 or curated.timbre_ratio != 1.0)
            ):
                pdata["pitch_shift"] = curated.pitch_shift
                pdata["timbre_ratio"] = curated.timbre_ratio
                pdata["badge"] = curated.badge
                pdata["description"] = curated.description
                changed = True
        if changed:
            tmp = profiles_json.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            tmp.replace(profiles_json)
            print("🔄  Cleaned stale Cloud profile state for natural neural audio.", flush=True)
    except Exception as e:
        print(f"⚠️  Could not clean Cloud profiles.json: {e}", flush=True)


def ensure_meanvc2_runtime():
    """Ensure MeanVC2 runtime deps and model weights are installed on Colab."""
    import importlib.util

    unbuf_env = {**os.environ, "PYTHONUNBUFFERED": "1"}

    required_pkgs = {
        "einops": "einops>=0.8.0",
        "x_transformers": "x-transformers>=2.0.0",
        "safetensors": "safetensors>=0.4.0",
        "soundfile": "soundfile>=0.12.1",
        "huggingface_hub": "huggingface-hub>=0.20.0",
    }
    missing = [pkg for mod, pkg in required_pkgs.items() if importlib.util.find_spec(mod) is None]
    if missing:
        print(f"📦 Installing MeanVC2 runtime deps: {', '.join(missing)}...", flush=True)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", *missing],
            check=True,
            env=unbuf_env,
        )

    if importlib.util.find_spec("s3prl") is None:
        print("📦 Installing s3prl (--no-deps for fast WavLM support)...", flush=True)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "s3prl>=0.4"],
            check=True,
            env=unbuf_env,
        )

    model_dir = Path(os.environ.get("HTH_MEANVC2_MODEL_DIR", str(Path.cwd() / "models" / "MeanVC2Models")))
    source_dir = Path(os.environ.get("HTH_MEANVC2_SOURCE_DIR", str(Path.cwd() / "MeanVC2-main")))
    required_files = [
        source_dir / "src" / "config" / "config_40ms_40ms.json",
        model_dir / "ckpts" / "pretrained_models" / "meanvc2_40ms_40ms.safetensors",
        model_dir / "preprocess" / "ckpts" / "fastu2pp_80ms.pt",
        model_dir / "ckpts" / "vocos" / "vocos.pt",
        model_dir / "preprocess" / "ckpts" / "wavlm_large_finetune.pth",
        model_dir / "preprocess" / "ckpts" / "wavlm_large_cfg.pt",
    ]
    if not all(p.is_file() for p in required_files):
        print("⬇️ Downloading MeanVC2 source & neural weights (first run only)...", flush=True)
        dl_script = Path(__file__).resolve().parent / "download_models.py"
        if dl_script.is_file():
            subprocess.run([sys.executable, "-u", str(dl_script), "--40ms"], check=True, env=unbuf_env)

    # --- Copy bundled reference WAVs to Cloud profile store ---
    # This ensures all default profiles have real speaker embeddings (no "Fallback Timbre").
    _reset_stale_cloud_profiles()


def start_tunnel_and_server():
    check_gpu()

    # Set environment variables for headless cloud operation
    os.environ["HTH_COMMERCIAL_MODE"] = "false"
    os.environ["HTH_COMMERCIAL_BUILD"] = "false"
    os.environ["HTH_DEFAULT_ENGINE"] = "meanvc2"
    os.environ["HTH_ENGINE_ADAPTER"] = "meanvc2"
    os.environ["HTH_MEANVC2_DEVICE"] = "cuda"
    if "HTH_MEANVC2_MODEL_DIR" not in os.environ:
        os.environ["HTH_MEANVC2_MODEL_DIR"] = str(Path.cwd() / "models" / "MeanVC2Models")
    if "HTH_MEANVC2_SOURCE_DIR" not in os.environ:
        os.environ["HTH_MEANVC2_SOURCE_DIR"] = str(Path.cwd() / "MeanVC2-main")

    ensure_meanvc2_runtime()

    # Start Cloudflare Tunnel
    cloudflared = setup_cloudflared()
    print("🚀 Launching Cloudflare Tunnel on port 8765...")
    log_path = Path("/tmp/cloudflared.log") if os.name != "nt" else Path("./cloudflared.log")
    log_file = open(log_path, "w", encoding="utf-8")
    tunnel_cmd = [str(cloudflared), "tunnel", "--url", "http://127.0.0.1:8765"]
    tunnel_proc = subprocess.Popen(
        tunnel_cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )

    tunnel_url = None
    url_regex = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")

    # Read output until trycloudflare.com URL appears
    start_time = time.time()
    while time.time() - start_time < 30:
        time.sleep(0.5)
        if log_path.exists():
            content = log_path.read_text(encoding="utf-8", errors="ignore")
            match = url_regex.search(content)
            if match:
                tunnel_url = match.group(0)
                break

    if tunnel_url:
        print("\n" + "=" * 65)
        print("  🎉 HTH REAL VOICE CHANGER CLOUD BACKEND IS LIVE!")
        print("=" * 65)
        print(f"  👉 API Server URL : {tunnel_url}")
        print(f"  👉 WebSocket URL  : {tunnel_url.replace('https://', 'wss://')}/api/v1/events/ws")
        print("=" * 65)
        print("  Copy this URL into your Desktop App Settings to test real-time GPU voice conversion!")
        print("=" * 65 + "\n")
    else:
        print("⚠️ Could not parse tunnel URL automatically. Check cloudflared output.")

    # Start Uvicorn backend server
    print("Starting FastAPI Uvicorn engine on 0.0.0.0:8765...")
    cwd = Path.cwd()
    if (cwd / "backend" / "src").is_dir():
        src_dir = str(cwd / "backend" / "src")
        project_root = str(cwd / "backend")
    elif (cwd / "src").is_dir():
        src_dir = str(cwd / "src")
        project_root = str(cwd)
    else:
        src_dir = str(Path(__file__).resolve().parent.parent / "src")
        project_root = str(Path(__file__).resolve().parent.parent)

    sys.path.insert(0, src_dir)
    sys.path.insert(0, project_root)

    env = os.environ.copy()
    env["HTH_CLOUD_MODE"] = "true"
    env["PYTHONUNBUFFERED"] = "1"
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src_dir}:{project_root}:{existing_pythonpath}"

    stop_keepalive = False

    if tunnel_url:
        import threading

        def _tunnel_keepalive_loop():
            while not stop_keepalive:
                time.sleep(45)
                if stop_keepalive:
                    break
                try:
                    urllib.request.urlopen(f"{tunnel_url}/health", timeout=8).read()
                except Exception:
                    pass

        threading.Thread(target=_tunnel_keepalive_loop, daemon=True).start()

    try:
        subprocess.run([
            sys.executable, "-u", "-m", "uvicorn",
            "hth.core.app:create_app",
            "--host", "0.0.0.0",
            "--port", "8765",
            "--factory",
            "--app-dir", src_dir,
        ], env=env)
    finally:
        stop_keepalive = True
        tunnel_proc.terminate()
        log_file.close()


if __name__ == "__main__":
    start_tunnel_and_server()
