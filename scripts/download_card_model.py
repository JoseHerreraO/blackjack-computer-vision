"""Download the pretrained playing-card detector into models/ (not versioned).

The default is ``lgd-cards-gen3`` from Hugging Face: YOLO11s fine-tuned on a public synthetic
card dataset (Roboflow ``playing-cards-ow27d``) plus real table video, 52 classes such as ``10H``
or ``QS``. It detects the corner index of each card. Its licence is AGPL-3.0, inherited from
Ultralytics, which is why the weights are downloaded instead of committed.

The download is pinned to one commit of the model repo, and ``best.pt`` is checked against its
SHA-256 before it is kept. A ``.pt`` file is a pickle that runs code when PyTorch loads it, so a
file that does not match is deleted instead of being used.

    python scripts/download_card_model.py
    python scripts/download_card_model.py --revision <commit> --sha256 <hash>  # another version
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

REPO = "sroot/lgd-cards-gen3"
# Commit of the model repo checked on 2026-11-18, and the SHA-256 of its best.pt (the LFS oid).
REVISION = "b2e9e89cc0138a70fc3ac5661922f99b4e3ae135"
WEIGHTS_SHA256 = "3513b77ab418ac2e86d3bf53be6566522666e656cbccb12678ba93feb6201004"
WEIGHTS = "best.pt"
EXTRA_FILES = ("model.classes.json", "README.md", "train-args.yaml")


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
    parser.add_argument("--repo", default=REPO, help="Hugging Face model repo")
    parser.add_argument("--revision", default=REVISION, help="commit to download (pinned)")
    parser.add_argument("--sha256", default=WEIGHTS_SHA256, help="expected SHA-256 of best.pt")
    parser.add_argument("--out", type=Path, default=Path("models/lgd-cards-gen3"))
    parser.add_argument("--force", action="store_true", help="download again if present")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    for name in (WEIGHTS, *EXTRA_FILES):
        target = args.out / name
        if target.exists() and not args.force:
            print(f"Already present: {target}")
        else:
            url = f"https://huggingface.co/{args.repo}/resolve/{args.revision}/{name}"
            print(f"Downloading {url}")
            urllib.request.urlretrieve(url, target)

    weights = args.out / WEIGHTS
    actual = sha256(weights)
    if actual != args.sha256.lower():
        weights.unlink()
        print(
            f"SHA-256 mismatch for {weights}: expected {args.sha256}, got {actual}. "
            "The file was deleted. Check --revision and --sha256.",
            file=sys.stderr,
        )
        return 1
    print(f"Done. {weights} verified (sha256 {actual[:12]}...)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
