#!/usr/bin/env python3
"""
test.py
=======

Evaluate a trained Faster R-CNN / ODAM / DPGA-ODAM checkpoint on a held-out
COCO-format test split.

Typical single-GPU/CPU:
    python test.py \
        --checkpoint runs/dpga/best.pt \
        --test-images /data/test \
        --test-ann /data/annotations/test.json \
        --output runs/dpga_test

Kaggle T4x2:
    torchrun --standalone --nproc_per_node=2 test.py ...same args...
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from network import Network
from train import (
    CocoDetectionTrainDataset,
    DetectorConfig,
    DistributedEvalSampler,
    barrier,
    compute_generic_mr2,
    detection_collate,
    evaluate_coco,
    gather_objects,
    init_distributed,
    is_main_process,
    postprocess_single_image,
    set_seed,
    unwrap_model,
    validate_config,
)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained checkpoint on a COCO test split."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-images", required=True)
    parser.add_argument("--test-ann", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument(
        "--min-size",
        type=int,
        default=None,
        help="Defaults to the checkpoint training value, then 800.",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=None,
        help="Defaults to the checkpoint training value, then 1333.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument(
        "--eval-score-threshold",
        type=float,
        default=None,
        help="Defaults to the checkpoint training value, then 0.05.",
    )
    parser.add_argument(
        "--eval-nms",
        type=float,
        default=None,
        help="Defaults to the checkpoint training value, then 0.5.",
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=None,
        help="Defaults to the checkpoint training value, then 100.",
    )
    parser.add_argument(
        "--non-strict",
        action="store_true",
        help="Allow missing/unexpected checkpoint keys. Strict is safer.",
    )
    parser.add_argument(
        "--odam-inference",
        action="store_true",
        help=(
            "Return DAM columns during inference. Detection metrics use only "
            "box/score/class columns, so this is normally unnecessary."
        ),
    )

    return parser.parse_args()


def checkpoint_args(checkpoint: Dict) -> Dict:
    args = checkpoint.get("args", {})
    return args if isinstance(args, dict) else {}


def value_from_args_or_checkpoint(args, checkpoint: Dict, name: str, fallback):
    current = getattr(args, name)
    if current is not None:
        return current

    train_args = checkpoint_args(checkpoint)
    return train_args.get(name, fallback)


def make_detector_config(
    checkpoint: Dict,
    num_classes: int,
) -> DetectorConfig:
    saved = checkpoint.get("detector_config")

    if isinstance(saved, dict):
        config = DetectorConfig(**saved)
        if int(config.num_classes) != int(num_classes):
            raise ValueError(
                "Checkpoint num_classes does not match test annotations: "
                f"checkpoint={config.num_classes}, test={num_classes}"
            )
        return config

    return DetectorConfig(num_classes=num_classes)


def load_model(
    checkpoint: Dict,
    config: DetectorConfig,
    device: torch.device,
    strict: bool,
) -> Network:
    if "model" not in checkpoint:
        raise KeyError("Checkpoint does not contain a 'model' state_dict.")

    model = Network(config)
    missing, unexpected = model.load_state_dict(
        checkpoint["model"],
        strict=strict,
    )

    if missing or unexpected:
        print(
            "[checkpoint] "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )

    model.to(device)
    model.eval()
    model.set_odam_enabled(checkpoint.get("method") != "baseline")
    return model


@torch.no_grad()
def run_test(
    model,
    loader: DataLoader,
    dataset: CocoDetectionTrainDataset,
    device: torch.device,
    args,
    rank: int,
    world_size: int,
) -> List[Dict]:
    unwrap_model(model).set_odam_inference(args.odam_inference)

    predictions_local: List[Dict] = []
    image_ids_local: List[int] = []

    iterator = loader
    if tqdm is not None and rank == 0:
        iterator = tqdm(
            loader,
            desc="test",
            leave=False,
        )

    for image, im_info, _, metas in iterator:
        if len(metas) != 1:
            raise RuntimeError(
                "Test currently requires --batch-size 1 because RCNN output "
                "has no explicit batch column."
            )

        image = image.to(device, non_blocking=True)
        im_info = im_info.to(device, non_blocking=True)

        pred = model(image, im_info)
        meta = metas[0]

        predictions_local.extend(
            postprocess_single_image(
                pred,
                meta=meta,
                label_to_cat_id=dataset.label_to_cat_id,
                score_threshold=args.eval_score_threshold,
                nms_threshold=args.eval_nms,
                max_detections=args.max_detections,
            )
        )
        image_ids_local.append(int(meta["image_id"]))

    gathered_predictions = gather_objects(
        predictions_local,
        rank,
        world_size,
    )

    if rank != 0:
        return []

    predictions = []
    for part in gathered_predictions:
        predictions.extend(part)

    return predictions


def main():
    args = parse_args()
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
    )

    distributed, rank, world_size, local_rank = init_distributed()

    seed = value_from_args_or_checkpoint(
        args,
        checkpoint,
        "seed",
        42,
    )
    set_seed(int(seed), rank=rank)

    if torch.cuda.is_available():
        device = (
            torch.device("cuda", local_rank)
            if distributed
            else torch.device("cuda")
        )
    else:
        device = torch.device("cpu")

    args.min_size = int(
        value_from_args_or_checkpoint(
            args,
            checkpoint,
            "min_size",
            800,
        )
    )
    args.max_size = int(
        value_from_args_or_checkpoint(
            args,
            checkpoint,
            "max_size",
            1333,
        )
    )
    args.eval_score_threshold = float(
        value_from_args_or_checkpoint(
            args,
            checkpoint,
            "eval_score_threshold",
            0.05,
        )
    )
    args.eval_nms = float(
        value_from_args_or_checkpoint(
            args,
            checkpoint,
            "eval_nms",
            0.5,
        )
    )
    args.max_detections = int(
        value_from_args_or_checkpoint(
            args,
            checkpoint,
            "max_detections",
            100,
        )
    )

    if args.batch_size != 1:
        raise ValueError("Current test evaluator requires --batch-size 1.")

    output_dir = Path(args.output)
    if is_main_process(rank):
        output_dir.mkdir(parents=True, exist_ok=True)
    barrier()

    test_dataset = CocoDetectionTrainDataset(
        args.test_images,
        args.test_ann,
        min_size=args.min_size,
        max_size=args.max_size,
    )

    config = make_detector_config(
        checkpoint,
        num_classes=len(test_dataset.category_ids) + 1,
    )
    validate_config(config)

    sampler = (
        DistributedEvalSampler(
            test_dataset,
            rank=rank,
            world_size=world_size,
        )
        if distributed
        else None
    )

    loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=detection_collate,
    )

    model = load_model(
        checkpoint,
        config,
        device,
        strict=not args.non_strict,
    )

    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    predictions = run_test(
        model=model,
        loader=loader,
        dataset=test_dataset,
        device=device,
        args=args,
        rank=rank,
        world_size=world_size,
    )

    if is_main_process(rank):
        image_ids = sorted(map(int, test_dataset.image_ids))
        metrics = evaluate_coco(
            test_dataset.coco,
            predictions,
            image_ids,
        )

        if len(test_dataset.category_ids) == 1:
            metrics["MR-2_generic"] = compute_generic_mr2(
                test_dataset.coco,
                predictions,
                category_id=test_dataset.category_ids[0],
                image_ids=image_ids,
                iou_threshold=0.5,
            )

        (output_dir / "predictions_test.json").write_text(
            json.dumps(predictions),
            encoding="utf-8",
        )
        (output_dir / "metrics_test.json").write_text(
            json.dumps(metrics, indent=2),
            encoding="utf-8",
        )
        (output_dir / "test_config.json").write_text(
            json.dumps(
                {
                    "checkpoint": str(args.checkpoint),
                    "checkpoint_method": checkpoint.get("method"),
                    "checkpoint_epoch": checkpoint.get("epoch"),
                    "world_size": world_size,
                    "device": str(device),
                    "test_images": str(args.test_images),
                    "test_ann": str(args.test_ann),
                    "categories": test_dataset.category_ids,
                    "internal_label_mapping": test_dataset.cat_id_to_label,
                    "detector_config": asdict(config),
                    "args": vars(args),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        print("Test metrics:")
        for key, value in metrics.items():
            print(f"{key}={value:.6f}")
        print(f"Predictions: {output_dir / 'predictions_test.json'}")
        print(f"Metrics: {output_dir / 'metrics_test.json'}")

    barrier()

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
