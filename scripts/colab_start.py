#!/usr/bin/env python3
"""
Colab Runner: Starts FastAPI backend with an automatic, free Cloudflare Tunnel.
Zero cost, zero registration, zero credit card required.
"""
import os
import re
import sys
import time
import subprocess
import urllib.request
from pathlib import Path


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


def start_tunnel_and_server():
    check_gpu()

    # Set environment variables for headless cloud operation
    os.environ["HTH_COMMERCIAL_MODE"] = "false"
    os.environ["HTH_DEFAULT_ENGINE"] = "meanvc2"
    if "HTH_MEANVC2_MODEL_DIR" not in os.environ:
        os.environ["HTH_MEANVC2_MODEL_DIR"] = str(Path.cwd() / "models" / "MeanVC2Models")
    if "HTH_MEANVC2_SOURCE_DIR" not in os.environ:
        os.environ["HTH_MEANVC2_SOURCE_DIR"] = str(Path.cwd() / "MeanVC2-main")

    # Start Cloudflare Tunnel
    cloudflared = setup_cloudflared()
    print("🚀 Launching Cloudflare Tunnel on port 8765...")
    tunnel_cmd = [str(cloudflared), "tunnel", "--url", "http://127.0.0.1:8765"]
    tunnel_proc = subprocess.Popen(
        tunnel_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    tunnel_url = None
    url_regex = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")

    # Read output until trycloudflare.com URL appears
    start_time = time.time()
    while time.time() - start_time < 30:
        line = tunnel_proc.stdout.readline()
        if not line:
            continue
        match = url_regex.search(line)
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
    sys.path.insert(0, str(Path.cwd() / "src"))
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "hth.core.app:create_app",
        "--host", "0.0.0.0",
        "--port", "8765",
        "--factory"
    ])


if __name__ == "__main__":
    start_tunnel_and_server()
