"""Download the card recognizer (embedder + reference database) into models/ (not versioned).

The files come from ``irfanhossainsust/playing-card-detector`` on Hugging Face (AGPL-3.0),
pinned to one commit, and every file is checked against its SHA-256. The ``.pt`` files are
pickles that run code when PyTorch loads them, so anything that does not match is deleted.
Only the recognizer is used. The repo's YOLO card detector is downloaded too, for comparison.

    python scripts/download_card_recognizer.py
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

REPO = "irfanhossainsust/playing-card-detector"
REVISION = "8d32fa35a3dd61696c7316ee157858dbf2ef8714"  # checked on 2026-11-18
FILES = {
    "models/card_embedder.pt": "3941abdef05ba32526c212d93b240505e12c31faf431202be512112a0fb77cf2",
    "models/card_detector.pt": "6a3357e0899e74ed8f3da2c3a56b8733f2790dba3920b4af81adef58261d57eb",
    "card_database/embeddings/card_embeddings.faiss": (
        "458be5cfffed34871d1d2b8566d9e67deeec5fa10f5e3fe0dbed882fdb4558e3"
    ),
    "card_database/embeddings/labels.json": (
        "ae2583a8b56cf2cc22b422223f2d124b22331f0199bf5e7fe1e678bb28d86632"
    ),
    "card_database/cards.json": "96dbade945fe8841f8ae767b7f0eb18a7ee6342b8a40e2b56aa7a27bce33639b",
    "README.md": None,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=Path("models/irfan-playing-card-detector"))
    parser.add_argument("--force", action="store_true", help="download again if present")
    args = parser.parse_args(argv)

    failed = False
    for name, expected in FILES.items():
        target = args.out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or args.force:
            url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}"
            print(f"Downloading {url}")
            urllib.request.urlretrieve(url, target)
        if expected is not None and sha256(target) != expected:
            target.unlink()
            print(f"SHA-256 mismatch for {target}, deleted", file=sys.stderr)
            failed = True
    if failed:
        return 1
    print(f"Done. Recognizer in {args.out}, all files verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
