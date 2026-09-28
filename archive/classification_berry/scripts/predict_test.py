#!/usr/bin/env python
"""P2/P3 inference: multi-fold + TTA soft-voting over a test split.

Produces ``outputs/predictions/raw_predictions.csv`` in ``image_id,label,score``
form, which ``src/submit/prepare_submit.py`` consumes directly. Keeping the
model output and the submission format separate means the pack/verify stage can
be re-run without a GPU.

⚠️ STATUS: written and config-complete; not yet executed (no dataset / no
checkpoints). TTA settings come from ``configs/default.yaml`` and must be
*measured*, not assumed - flip-TTA was a negative result in the team's previous
(lane-detection) task, which says nothing about this classification task.

Usage
-----
    python scripts/predict_test.py --split testA
    python scripts/predict_test.py --split testB --no-tta
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from common.config import load_config  # noqa: E402
from common.io_utils import write_csv_rows  # noqa: E402


def _test_image_dir(cfg, split: str) -> Path:
    key = "data.test_a_dir" if split.lower() == "testa" else "data.test_b_dir"
    return cfg.path(key)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["testA", "testB"], required=True)
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    try:
        import torch
        import timm
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(f"inference requires torch + timm: {exc}") from exc

    cfg = load_config()
    out_path = args.out or cfg.path("inference.raw_predictions_path")
    image_dir = _test_image_dir(cfg, args.split)
    if not image_dir.is_dir():
        raise SystemExit(f"test image dir not found: {image_dir}")

    checkpoints = sorted(cfg.path("train.output_dir").glob("fold*.pt"))
    if not checkpoints:
        raise SystemExit(
            f"no fold checkpoints under {cfg.path('train.output_dir')} - train first"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    size = int(cfg.train.image_size)
    mean, std = list(cfg.data.imagenet_mean), list(cfg.data.imagenet_std)
    labels = list(cfg.competition.labels)

    from torchvision import transforms

    base = transforms.Compose([transforms.Resize(int(size * 1.14)),
                               transforms.CenterCrop(size), transforms.ToTensor()])
    normalise = transforms.Normalize(mean, std)

    scales = [1.0] if args.no_tta else list(cfg.tta.scales)
    use_hflip = (not args.no_tta) and bool(cfg.tta.hflip)

    images = sorted(
        path for path in image_dir.rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if not images:
        raise SystemExit(f"no images found under {image_dir}")

    # Ensemble = mean of softmax probabilities over folds x TTA views.
    ensemble_probs = torch.zeros(len(images), len(labels))
    for checkpoint in checkpoints:
        payload = torch.load(checkpoint, map_location=device)
        model = timm.create_model(
            payload.get("backbone", str(cfg.train.backbone)),
            pretrained=False, num_classes=len(labels),
        ).to(device)
        model.load_state_dict(payload["model"])
        model.eval()
        with torch.no_grad():
            for index, path in enumerate(images):
                with Image.open(path) as handle:
                    tensor = base(handle.convert("RGB"))
                views = []
                for scale in scales:
                    resized = transforms.functional.resize(
                        tensor, [int(size * scale), int(size * scale)]
                    )
                    views.append(resized)
                    if use_hflip:
                        views.append(transforms.functional.hflip(resized))
                batch = torch.stack([normalise(v) for v in views]).to(device)
                probs = torch.softmax(model(batch), dim=1).mean(dim=0).cpu()
                ensemble_probs[index] += probs
        print(f"  scored with {checkpoint.name}")

    ensemble_probs /= len(checkpoints)
    rows = []
    for index, path in enumerate(images):
        probability = ensemble_probs[index]
        label = int(labels[int(probability.argmax())])
        rows.append(
            {"image_id": path.stem, "label": label, "score": f"{float(probability.max()):.6f}"}
        )

    write_csv_rows(out_path, rows, ("image_id", "label", "score"))
    print(f"wrote {len(rows)} predictions to {out_path}")
    print("next: make submit-b   (runs prepare -> pack -> verify)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
