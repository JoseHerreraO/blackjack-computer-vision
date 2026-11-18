"""Download the pretrained playing-card detector into models/ (not versioned).

The default is ``lgd-cards-gen3`` from Hugging Face: YOLO11s fine-tuned on a public synthetic
card dataset (Roboflow ``playing-cards-ow27d``) plus real table video, 52 classes such as ``10H``
or ``QS``. It detects the corner index of each card. Its licence is AGPL-3.0, inherited from
Ultralytics, which is why the weights are downloaded instead of committed.

    python scripts/download_card_model.py
    python scripts/download_card_model.py --repo sroot/lgd-cards-gen3 --out models/lgd-cards-gen3
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

FILES = ("best.pt", "model.classes.json", "README.md", "train-args.yaml")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repo", default="sroot/lgd-cards-gen3", help="Hugging Face model repo")
    parser.add_argument("--revision", default="main", help="branch, tag or commit")
    parser.add_argument("--out", type=Path, default=Path("models/lgd-cards-gen3"))
    parser.add_argument("--force", action="store_true", help="download again if present")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = args.out / name
        if target.exists() and not args.force:
            print(f"Already present: {target}")
            continue
        url = f"https://huggingface.co/{args.repo}/resolve/{args.revision}/{name}"
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, target)
    print(f"Done. Model in {args.out / 'best.pt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
