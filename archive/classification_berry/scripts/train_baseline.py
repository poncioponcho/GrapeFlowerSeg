#!/usr/bin/env python
"""P1 baseline training: grouped-K-fold fine-tuning of a timm classifier.

⚠️ STATUS: written and config-complete, but **not yet executed** - the official
dataset is not present locally (``data/raw/`` is empty) and torch/timm are not
installed. Once both are available this runs end to end and writes the OOF
artifacts that drive model selection.

Discipline encoded here (see configs/default.yaml):

* grouped K-fold by source image - the fold file comes from
  ``src/data/split_by_source.py`` and the leakage assertion already passed;
* AMP + cosine LR + label smoothing 0.1;
* conservative HSV jitter and **no** MixUp/CutMix - colour *is* the label
  signal, so blending images destroys exactly the fine ordinal distinction the
  task is about;
* one checkpoint per fold, and out-of-fold predictions on disk so the reported
  Macro-F1 is reproducible rather than retyped.

Usage
-----
    python scripts/train_baseline.py --backbone convnext_tiny --folds 0,1,2,3,4
    python scripts/train_baseline.py --smoke        # 1 epoch, tiny subset
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from common.config import load_config  # noqa: E402
from data.split_by_source import load_folds  # noqa: E402


def _require_training_stack():
    try:
        import torch  # noqa: F401
        import timm  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "P1 requires torch + timm. Install the training extras first:\n"
            "  pip install torch torchvision timm\n"
            f"(import error: {exc})"
        ) from exc


def build_transforms(cfg, *, train: bool):
    """Augmentation policy straight from config - nothing hardcoded."""
    from torchvision import transforms

    aug = cfg.train.aug
    size = int(cfg.train.image_size)
    mean, std = list(cfg.data.imagenet_mean), list(cfg.data.imagenet_std)
    if not train:
        return transforms.Compose(
            [transforms.Resize(int(size * 1.14)), transforms.CenterCrop(size),
             transforms.ToTensor(), transforms.Normalize(mean, std)]
        )

    ops = [
        transforms.RandomResizedCrop(
            size,
            scale=tuple(aug.random_resized_crop_scale),
            ratio=tuple(aug.random_resized_crop_ratio),
        ),
        transforms.RandomHorizontalFlip(p=float(aug.hflip_p)),
        transforms.RandomRotation(degrees=float(aug.rotation_deg)),
    ]
    jitter = aug.color_jitter
    # Deliberately gentle: strong colour jitter would erase the ordinal signal.
    ops.append(
        transforms.ColorJitter(
            brightness=float(jitter.brightness),
            contrast=float(jitter.contrast),
            saturation=float(jitter.saturation),
            hue=float(jitter.hue),
        )
    )
    if aug.randaugment.enabled:
        try:
            ops.append(
                transforms.RandAugment(
                    num_ops=int(aug.randaugment.num_ops),
                    magnitude=int(aug.randaugment.magnitude),
                )
            )
        except AttributeError:  # pragma: no cover - very old torchvision
            pass
    forbidden = {str(v).lower() for v in cfg.train.forbidden_augmentations}
    if {"mixup", "cutmix"} & forbidden:
        # Asserted rather than documented: a silent MixUp would invalidate the
        # fine-grained colour comparison this task depends on.
        assert not any(isinstance(op, (transforms.RandomMixUp,)) for op in ops) \
            if hasattr(transforms, "RandomMixUp") else True
    ops += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(ops)


class BerryDataset:
    """Reads a manifest and yields (image_tensor, label)."""

    def __init__(self, records, image_dir: Path, transform):
        self.records = list(records)
        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        from PIL import Image

        record = self.records[index]
        with Image.open(self.image_dir / record.file_name) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, record.label


def train_fold(cfg, records, train_indices, val_indices, *, fold: int, device: str, smoke: bool):
    """Train one fold and return (val_predictions, val_labels, checkpoint_path)."""
    import torch
    from torch.utils.data import DataLoader
    import timm

    from eval.macro_f1 import macro_f1

    image_dir = cfg.path("data.image_dir")
    size = int(cfg.train.image_size)
    train_ds = BerryDataset([records[i] for i in train_indices], image_dir,
                            build_transforms(cfg, train=True))
    val_ds = BerryDataset([records[i] for i in val_indices], image_dir,
                          build_transforms(cfg, train=False))

    batch_size = int(cfg.train.batch_size)
    workers = int(cfg.train.num_workers)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=int(cfg.inference.batch_size),
                            shuffle=False, num_workers=workers)

    model = timm.create_model(
        str(cfg.train.backbone), pretrained=True, num_classes=len(cfg.competition.labels)
    ).to(device)

    epochs = 1 if smoke else int(cfg.train.epochs)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg.train.lr), weight_decay=float(cfg.train.weight_decay)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler(device, enabled=bool(cfg.train.amp))
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=float(cfg.train.label_smoothing))

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device, enabled=bool(cfg.train.amp)):
                loss = criterion(model(images), targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.grad_clip))
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach()) * images.size(0)
        scheduler.step()
        print(f"    fold {fold} epoch {epoch + 1}/{epochs} loss={running / max(1, len(train_ds)):.4f}")

    model.eval()
    predictions: list[int] = []
    labels: list[int] = []
    with torch.no_grad():
        for images, targets in val_loader:
            logits = model(images.to(device))
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            labels.extend(targets.tolist())

    out_dir = cfg.path("train.output_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / str(cfg.train.checkpoint_name).format(fold=fold)
    torch.save({"model": model.state_dict(), "backbone": str(cfg.train.backbone),
                "image_size": size, "fold": fold}, checkpoint)
    print(f"    fold {fold} val Macro-F1 = {macro_f1(labels, predictions):.5f} -> {checkpoint.name}")
    return predictions, labels, checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", default=None)
    parser.add_argument("--folds", default=None, help="e.g. 0,1,2,3,4")
    parser.add_argument("--smoke", action="store_true", help="1 epoch, first fold only")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    _require_training_stack()
    import torch

    cfg = load_config()
    if args.backbone:
        cfg["train"]["backbone"] = args.backbone
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    manifest_path = cfg.path("data.manifest_path")
    if not manifest_path.is_file():
        raise SystemExit(
            f"manifest not found: {manifest_path}\n"
            "Run the data stage first (see README):\n"
            "  1. download the official data into data/raw/\n"
            "  2. python -m data.manifest\n"
            "  3. python -m data.split_by_source"
        )
    folds_path = cfg.path("split.output_dir") / "folds.json"
    if not folds_path.is_file():
        raise SystemExit(f"folds not found: {folds_path}\nRun: python -m data.split_by_source")

    split, records = load_folds(folds_path)
    requested = (
        [int(v) for v in args.folds.split(",")] if args.folds
        else ([0] if args.smoke else list(range(split.n_folds)))
    )
    print(f"training {len(requested)} fold(s) on {device} with {cfg.train.backbone}")

    oof: dict[int, int] = {}
    fold_scores: list[dict] = []
    fold_indices: dict[str, list[int]] = {}
    for fold in split.folds:
        if fold.fold not in requested:
            continue
        predictions, labels, _ = train_fold(
            cfg, records, list(fold.train_indices), list(fold.val_indices),
            fold=fold.fold, device=device, smoke=args.smoke,
        )
        for index, prediction in zip(fold.val_indices, predictions):
            oof[index] = int(prediction)
        from eval.macro_f1 import macro_f1

        fold_indices[str(fold.fold)] = list(fold.val_indices)
        fold_scores.append({"fold": fold.fold, "n": len(labels),
                            "macro_f1": macro_f1(labels, predictions)})

    oof_path = cfg.path("train.output_dir") / "oof_predictions.json"
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    oof_path.write_text(
        json.dumps(
            {
                "records": [r.to_row() for r in records],
                "oof": oof,
                "fold_indices": fold_indices,
                "folds": fold_scores,
                "backbone": str(cfg.train.backbone),
                "image_size": int(cfg.train.image_size),
                "seed": int(cfg.project.seed),
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"OOF predictions written to {oof_path}")
    print(f"Report: python scripts/make_oof_report.py --oof {oof_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
