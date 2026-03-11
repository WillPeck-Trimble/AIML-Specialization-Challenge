# I used this to test different models, left here as reference but unnecessary for final submission

import os
import sys

# Must be set before importing keras/torch
os.environ["KERAS_BACKEND"] = "torch"

import fiftyone as fo
import fiftyone.zoo as foz

print("Loading D-FINE small model...")
from dfine_s_coco import identify_targets, target_labels as _target_labels
print("Model loaded.\n")

# Build a lower-cased class list for matching against COCO ground-truth labels
TARGET_CLASSES = [name.lower() for name in _target_labels.values()]

CONFIDENCE_THRESHOLD = 0.5
MAX_SAMPLES = 50  # number of COCO validation images to download


def download_coco_samples(max_samples: int = MAX_SAMPLES) -> fo.Dataset:
    """
    Download COCO 2017 validation images that contain at least one
    person, cat, or dog annotation via the fiftyone model zoo.
    Returns a fiftyone Dataset.
    """
    print(f"Downloading up to {max_samples} COCO 2017 validation samples "
          f"containing: {TARGET_CLASSES}...")
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split="validation",
        label_types=["detections"],
        classes=TARGET_CLASSES,
        max_samples=max_samples,
        only_matching=True,          # only keep samples with at least one target class
    )
    print(f"Downloaded {len(dataset)} samples.\n")
    return dataset


def evaluate(dataset: fo.Dataset, threshold: float = CONFIDENCE_THRESHOLD) -> None:
    """
    Run the D-FINE model over every sample in the fiftyone dataset and
    compare image-level class presence against ground-truth annotations.

    Metrics reported per class:
        TP – model detected the class and it is present in ground truth
        FN – class is in ground truth but model missed it
        FP – model detected the class but it is NOT in ground truth
        Precision = TP / (TP + FP)
        Recall    = TP / (TP + FN)
    """
    stats = {cls: {"tp": 0, "fn": 0, "fp": 0} for cls in TARGET_CLASSES}

    print(f"Running inference on {len(dataset)} images (threshold={threshold})...\n")

    for idx, sample in enumerate(dataset, start=1):
        image_path = sample.filepath

        # ── Ground-truth classes present in this image ─────────────────────
        gt_classes: set[str] = set()
        if sample.ground_truth and sample.ground_truth.detections:
            for det in sample.ground_truth.detections:
                label = det.label.lower()
                if label in TARGET_CLASSES:
                    gt_classes.add(label)

        # ── Model predictions (delegated to dfine-s-coco.py) ───────────────
        detections = identify_targets(image_path, threshold=threshold)
        pred_classes: set[str] = {d["class"].lower() for d in detections}

        filename = os.path.basename(image_path)
        print(f"[{idx:>3}/{len(dataset)}] {filename}")
        print(f"         GT:   {sorted(gt_classes) or 'none'}")
        print(f"         Pred: {sorted(pred_classes) or 'none'}")
        for d in detections:
            print(f"               └ {d['class']} ({d['confidence']:.2f})")

        # ── Per-class image-level scoring ───────────────────────────────────
        for cls in TARGET_CLASSES:
            in_gt   = cls in gt_classes
            in_pred = cls in pred_classes
            if in_gt and in_pred:
                stats[cls]["tp"] += 1
            elif in_gt and not in_pred:
                stats[cls]["fn"] += 1
            elif not in_gt and in_pred:
                stats[cls]["fp"] += 1

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(f"{'CLASS':<10} {'TP':>4} {'FN':>4} {'FP':>4}  {'PRECISION':>10} {'RECALL':>8}")
    print("-" * 55)
    for cls in TARGET_CLASSES:
        tp = stats[cls]["tp"]
        fn = stats[cls]["fn"]
        fp = stats[cls]["fp"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall    = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        print(f"{cls:<10} {tp:>4} {fn:>4} {fp:>4}  {precision:>10.2f} {recall:>8.2f}")
    print("=" * 55)


if __name__ == "__main__":
    dataset = download_coco_samples(max_samples=MAX_SAMPLES)
    evaluate(dataset, threshold=CONFIDENCE_THRESHOLD)
