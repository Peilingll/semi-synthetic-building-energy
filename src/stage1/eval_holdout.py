"""Hold-out test final evaluation for Stage 1 Phase B.

Picks the best checkpoint across 5 folds by val_macro_f1, then evaluates it
once on the frozen hold-out test set. Outputs the headline Table 1 numbers
that go into the paper (the 5-fold CV mean ± std is reported separately
from the aggregate evaluate.py call).
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.stage1.dataset import Stage1ImageDataset, collate_fn
from src.stage1.evaluate import (
    evaluate_predictions,
    per_city_breakdown,
    per_class_year_floor_breakdown,
)
from src.stage1.models import build_model
from src.stage1.train import predict

logger = logging.getLogger(__name__)


def find_best_fold_checkpoint(ckpt_dir: Path, model_name: str, n_folds: int = 5,
                              run_tag: str = "pooled") -> tuple[Path, float, int]:
    best_ckpt: Path | None = None
    best_f1 = -1.0
    best_fold = -1
    for f in range(n_folds):
        path = ckpt_dir / f"{run_tag}_{model_name}_fold{f}.pt"
        if not path.exists():
            logger.warning("missing checkpoint: %s", path)
            continue
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        f1 = ckpt.get("val_macro_f1", -1.0)
        logger.info("fold %d: val_macro_f1=%.4f from %s", f, f1, path.name)
        if f1 > best_f1:
            best_f1 = f1
            best_ckpt = path
            best_fold = f
    assert best_ckpt is not None, "no checkpoint found"
    logger.info("BEST: fold %d val_macro_f1=%.4f -> %s", best_fold, best_f1, best_ckpt)
    return best_ckpt, best_f1, best_fold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="dinov2_frozen")
    parser.add_argument("--ckpt", type=Path, default=None,
                        help="explicit checkpoint path; if omitted picks best across folds")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--holdout", type=Path, default=None,
                        help="override holdout_test_pand_ids.parquet path (e.g. LOCO splits)")
    parser.add_argument("--run-tag", default="pooled",
                        help="checkpoint prefix to load; non-pooled tags also prefix the outputs")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    repo = Path(__file__).resolve().parents[2]
    ckpt_dir = repo / "models" / "stage1"

    if args.ckpt:
        ckpt_path = args.ckpt
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        best_fold = ckpt.get("fold", -1)
        best_f1 = ckpt.get("val_macro_f1", -1.0)
    else:
        ckpt_path, best_f1, best_fold = find_best_fold_checkpoint(ckpt_dir, args.model,
                                                                  run_tag=args.run_tag)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(args.model).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    holdout_path = args.holdout if args.holdout else repo / "data/processed/holdout_test_pand_ids.parquet"
    holdout_ds = Stage1ImageDataset(
        manifest_path=repo / "data/processed/svi_manifest.parquet",
        gt_path=repo / "data/processed/stage1_gt.parquet",
        split="holdout",
        holdout_pand_ids_path=holdout_path,
        year_mean=ckpt["year_mean"], year_std=ckpt["year_std"],
        floors_mean=ckpt["floors_mean"], floors_std=ckpt["floors_std"],
    )
    loader = DataLoader(
        holdout_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True,
    )

    preds = predict(
        model, loader, device,
        ckpt["year_mean"], ckpt["year_std"],
        ckpt["floors_mean"], ckpt["floors_std"],
    )
    out_dir = (args.out.parent if args.out else repo / "reports" / "stage1" / args.model)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_prefix = "" if args.run_tag == "pooled" else f"{args.run_tag}_"
    preds_path = out_dir / f"{out_prefix}holdout_preds.parquet"
    preds.to_parquet(preds_path, index=False)
    logger.info("wrote %s (%d rows)", preds_path, len(preds))

    report = evaluate_predictions(preds, with_ci=True)
    report["per_city"] = per_city_breakdown(preds)
    report["per_class_year_floors"] = per_class_year_floor_breakdown(preds)
    report["source_checkpoint"] = {
        "path": str(ckpt_path),
        "fold": int(best_fold),
        "dev_val_macro_f1": float(best_f1),
    }

    out_path = args.out if args.out else out_dir / f"{out_prefix}holdout_metrics.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("wrote %s", out_path)

    print(json.dumps({k: v for k, v in report.items() if k != "type_confusion_matrix"},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
