"""Download the speech models into models/ (setup step, not part of the running assistant).

Archives come from sherpa-onnx's official GitHub releases. The first download of each
pins its SHA-256 in config/models.lock.json; later downloads (another machine, a fresh
checkout) must match it or are rejected.

    poetry run python scripts/fetch_models.py            # everything
    poetry run python scripts/fetch_models.py kws vad    # just these
"""

from __future__ import annotations

import hashlib
import json
import sys
import tarfile
import tempfile
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
LOCK_FILE = ROOT / "config" / "models.lock.json"
BASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download"

MODELS = {
    "vad": f"{BASE}/asr-models/silero_vad.onnx",
    "kws": f"{BASE}/kws-models/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01.tar.bz2",
    "asr-moonshine": f"{BASE}/asr-models/sherpa-onnx-moonshine-base-en-quantized-2026-02-27.tar.bz2",
    "tts-kokoro": f"{BASE}/tts-models/kokoro-int8-en-v0_19.tar.bz2",
    "tts-piper": f"{BASE}/tts-models/vits-piper-en_US-lessac-medium.tar.bz2",
}


def main(names: list[str]) -> int:
    unknown = set(names) - set(MODELS)
    if unknown:
        print(f"unknown models {sorted(unknown)}; choose from {sorted(MODELS)}")
        return 2
    lock: dict[str, str] = json.loads(LOCK_FILE.read_text()) if LOCK_FILE.exists() else {}
    for name in names or list(MODELS):
        target = MODELS_DIR / name
        if target.exists():
            print(f"{name}: already in {target.relative_to(ROOT)}")
            continue
        url = MODELS[name]
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / url.rsplit("/", 1)[1]
            digest = download(url, archive)
            expected = lock.get(url)
            if expected and expected != digest:
                print(f"{name}: SHA-256 mismatch (expected {expected}, got {digest}); not installing")
                return 1
            lock[url] = digest
            install(archive, target)
        print(f"{name}: installed in {target.relative_to(ROOT)} (sha256 {digest[:12]}...)")
    LOCK_FILE.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    return 0


def download(url: str, dest: Path) -> str:
    sha = hashlib.sha256()
    with httpx.stream("GET", url, follow_redirects=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        done = 0
        with dest.open("wb") as handle:
            for chunk in response.iter_bytes(1 << 20):
                handle.write(chunk)
                sha.update(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {dest.name}: {done / 1e6:.0f}/{total / 1e6:.0f} MB", end="", flush=True)
    print()
    return sha.hexdigest()


def install(archive: Path, target: Path) -> None:
    """Each model gets its own new directory; tar members may not escape it."""
    target.mkdir(parents=True)
    if not archive.name.endswith(".tar.bz2"):
        archive.replace(target / archive.name)
        return
    with tarfile.open(archive, "r:bz2") as tar:
        tar.extractall(target, filter="data")
    # Archives hold one top-level folder; flatten it so paths are models/<name>/<file>.
    children = list(target.iterdir())
    if len(children) == 1 and children[0].is_dir():
        inner = children[0]
        for item in inner.iterdir():
            item.replace(target / item.name)
        inner.rmdir()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
