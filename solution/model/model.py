"""Self-contained grape-panicle instance segmenter (ships inside solution.zip).

Reproduces, without any project-external imports, exactly the inference path of
``scripts/predict_test.py`` in the training repository:

* each checkpoint is a ``torchvision`` Mask R-CNN whose variant (v1/v2) is read
  from the checkpoint payload itself
* image resized so its short side equals ``short_side`` (BILINEAR), tensors via
  ``to_tensor`` only - no normalization (the model applies its own)
* detections with ``score >= 0.05`` are kept per model
* boxes/masks are mapped back to the original image grid; masks are binarized
  at 0.5, resized with NEAREST, then re-binarized
* multi-checkpoint ensembling: union of detections, per-class NMS at IoU 0.5
  (complete/incomplete panicles overlap heavily and must not suppress each
  other), sorted by score, truncated to 100 detections (official maxDets cap)
* ``category_id``: model label 1 -> submission 0 (完整花穗),
  model label 2 -> submission 1 (不完整花穗)
* masks encoded as COCO compressed RLE (Fortran order, counts as ``str``)

No network access at import or run time.
"""
from __future__ import annotations

from pathlib import Path

SCORE_THRESHOLD = 0.05
MASK_THRESHOLD = 0.5
NMS_IOU = 0.5
MAX_DETECTIONS = 100
NUM_CLASSES = 3  # background + 完整花穗 + 不完整花穗

# Inference resolution(s) used for the submitted ``result.json``.
#
# The checkpoints were *trained* at short side 640 for time-budget reasons, and
# the training resolution used to be reused at inference. Scoring the same
# weights on the same held-out folds (paired, one variable) showed:
#
#   * 1024 alone vs 640 alone: +0.18 pp mean over 5 folds, sd 0.69 pp - a null
#     result, not the +0.9 pp a single fold suggested
#   * running BOTH and fusing the detections: +0.52 pp mean, 4 of 5 folds up,
#     sd 0.39 pp - a real gain, because the two scales make different mistakes
#   * on the platform that fusion was worth +1.02 pp (0.31903 -> 0.32924), twice
#     what the holdout folds predicted
#
# A third scale (1152) was considered on top of that but could not be run before
# the submission deadline (a 3-scale pass over testB costs ~17 min, leaving no
# time to build, verify and upload), so the submitted package is 2-scale.
#
# This default must stay in sync with whatever the submission was produced with,
# otherwise the post-competition offline reproduction would score differently.
# Override with ``--short_side`` if you need to re-check that.
DEFAULT_INFER_SHORT_SIDES = (640, 1024)


def _as_resolutions(short_side, trained_at: int) -> list[int]:
    """Normalise the ``short_side`` argument to a non-empty list of ints.

    ``None`` keeps the historical behaviour (the checkpoint's training size);
    an int or a comma-separated string selects one or more inference scales.
    """
    if short_side is None:
        return [trained_at]
    if isinstance(short_side, (list, tuple)):
        values = [int(v) for v in short_side]
    else:
        values = [int(part) for part in str(short_side).split(",") if part.strip()]
    if not values:
        raise ValueError("short_side must select at least one resolution")
    return values


def _build_model(light: bool, num_classes: int):
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    factory = (
        torchvision.models.detection.maskrcnn_resnet50_fpn
        if light
        else torchvision.models.detection.maskrcnn_resnet50_fpn_v2
    )
    model = factory(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    hidden = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(hidden, hidden, num_classes)
    return model


class _SingleModel:
    """One checkpoint plus the input size(s) to run it at."""

    def __init__(self, weights_path: Path, device: str, short_side=None):
        import torch

        payload = torch.load(weights_path, map_location="cpu", weights_only=False)
        self.model = _build_model(
            bool(payload.get("light", False)),
            int(payload.get("num_classes", NUM_CLASSES)),
        )
        self.model.load_state_dict(payload["model"])
        self.model.eval()
        self.model.to(device)
        self.trained_at = int(payload.get("short_side", DEFAULT_INFER_SHORT_SIDES[0]))
        self.short_sides = _as_resolutions(short_side, self.trained_at)


class PanicleSegmenter:
    """One or more checkpoints fused by per-class NMS (the ensemble case).

    Each checkpoint may be run at several input scales; every (checkpoint, scale)
    pair contributes its detections to the union before fusion, which is how the
    submitted multi-scale ensemble is reproduced.
    """

    def __init__(self, members: list[_SingleModel]):
        self.members = members

    @classmethod
    def load(cls, weights, *, device: str = "cuda",
             short_side=None) -> "PanicleSegmenter":
        paths = weights if isinstance(weights, (list, tuple)) else [weights]
        paths = [Path(p) for p in paths]
        missing = [p for p in paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"missing weights: {missing}")
        if not paths:
            raise ValueError("no weights given")
        return cls([_SingleModel(p, device, short_side) for p in paths])

    def eval(self) -> None:
        for member in self.members:
            member.model.eval()

    def predict(self, image_path, width: int, height: int) -> list[dict]:
        import numpy as np
        import torch
        import torchvision.transforms.functional as TF
        from PIL import Image
        from torchvision.ops import nms

        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
        if image.size != (width, height):
            # Trust the caller's declared size (public manifest); the official
            # instructions require masks on the true original grid.
            image = image.resize((width, height), Image.BILINEAR)

        boxes_all, scores_all, labels_all, masks_all = [], [], [], []
        for member in self.members:
            for short_side in member.short_sides:
                scale = short_side / min(width, height)
                new_w, new_h = int(round(width * scale)), int(round(height * scale))
                resized = image.resize((new_w, new_h), Image.BILINEAR)
                tensor = TF.to_tensor(resized)

                device = next(member.model.parameters()).device
                with torch.no_grad():
                    output = member.model([tensor.to(device)])[0]

                keep = output["scores"] >= SCORE_THRESHOLD
                if not bool(keep.any()):
                    continue
                boxes = output["boxes"][keep].cpu()
                scores = output["scores"][keep].cpu()
                labels = output["labels"][keep].cpu()
                raw_masks = output["masks"][keep, 0].cpu()

                # Undo the input resize so detections land on the original grid.
                boxes = boxes * torch.tensor(
                    [width / new_w, height / new_h, width / new_w, height / new_h]
                )
                masks = torch.stack(
                    [
                        torch.from_numpy(
                            np.array(
                                Image.fromarray(
                                    (mask.numpy() > MASK_THRESHOLD).astype(np.uint8) * 255
                                ).resize((width, height), Image.NEAREST)
                            )
                            > 127
                        )
                        for mask in raw_masks
                    ]
                )
                boxes_all.append(boxes)
                scores_all.append(scores)
                labels_all.append(labels)
                masks_all.append(masks)

        if not boxes_all:
            return []

        boxes = torch.cat(boxes_all)
        scores = torch.cat(scores_all)
        labels = torch.cat(labels_all)
        masks = torch.cat(masks_all)

        # Per-class NMS across checkpoints: 完整/不完整花穗 overlap heavily and
        # must not suppress each other.
        keep_indices: list[int] = []
        for label in labels.unique():
            index = torch.nonzero(labels == label).flatten()
            kept = nms(boxes[index], scores[index], NMS_IOU)
            keep_indices.extend(index[kept].tolist())
        order = sorted(keep_indices, key=lambda i: -float(scores[i]))[:MAX_DETECTIONS]

        instances = []
        for index in order:
            mask = masks[index].numpy().astype("uint8")
            if not mask.any():
                continue
            rle = self._encode_rle(mask)
            if rle is None:
                continue
            instances.append(
                {
                    "category_id": int(labels[index]) - 1,  # 1/2 -> submission 0/1
                    "score": round(float(scores[index]), 6),
                    "segmentation": rle,
                }
            )
        return instances

    @staticmethod
    def _encode_rle(mask) -> dict | None:
        import numpy as np
        from pycocotools import mask as mask_utils

        if not mask.any():
            return None
        rle = mask_utils.encode(np.asfortranarray(mask))
        counts = rle["counts"]
        if isinstance(counts, bytes):
            counts = counts.decode("ascii")
        return {"size": [int(rle["size"][0]), int(rle["size"][1])], "counts": counts}
