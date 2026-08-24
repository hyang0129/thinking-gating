#!/usr/bin/env python3
"""
eval_transfer.py — zero-shot cross-task transfer of a trained probe.

    python scripts/eval_transfer.py \\
        --probe output/gsm8k_probe/seed_42/checkpoint.json \\
        --capture-dir shared/icr_capture/lsat_thinking_qwen3 \\
        --labels shared/lsat_thinking_labels.jsonl \\
        --source-metrics output/gsm8k_probe/aggregate_metrics.json \\
        --out-file output/gsm8k_to_lsat_transfer.json

No retraining and no refitting: the probe, its standardizer, and its routing
threshold all come from the source task. That is the point — a probe that has
learned something general about "this query needs reasoning" should survive the
move to a task it has never seen, and one that has memorized GSM8K's prompt
format should not.

Interpretation, per the experiment design:
    < 5pp AUROC drop    strong transfer
    5-15pp              moderate — likely label distribution shift
    > 15pp              transfer failure (format overfitting)

Passing several --probe checkpoints evaluates each and reports the spread, so a
single lucky seed cannot carry the conclusion.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.capture_io import (  # noqa: E402
    align_labels, load_capture, load_config, load_labels,
)
from scripts.run_experiment import (  # noqa: E402
    Standardizer, evaluate_split, mean_ci, select_layer,
)

logger = logging.getLogger("transfer")


def load_probe(path: Path):
    """Rebuild a probe from its checkpoint. Returns (score_fn, checkpoint)."""
    ckpt = json.loads(Path(path).read_text(encoding="utf-8"))

    if ckpt["method"] == "mlp":
        import torch
        from torch import nn

        state = {k: torch.tensor(v) for k, v in ckpt["state_dict"].items()}
        # Rebuild the architecture from the checkpoint's own shapes rather than
        # from defaults, so an older checkpoint with different widths still loads.
        linear_keys = sorted({k.rsplit(".", 1)[0] for k in state if k.endswith(".weight")},
                             key=lambda k: int(k.split(".")[0]))
        widths = [state[f"{k}.weight"].shape[0] for k in linear_keys]
        in_dim = state[f"{linear_keys[0]}.weight"].shape[1]
        layers = []
        for width in widths[:-1]:
            layers += [nn.Linear(in_dim, width), nn.ReLU(), nn.Dropout(0.1)]
            in_dim = width
        layers.append(nn.Linear(in_dim, widths[-1]))
        model = nn.Sequential(*layers)
        model.load_state_dict(state)
        model.eval()

        def score(X):
            with torch.no_grad():
                logits = model(torch.tensor(X, dtype=torch.float32)).squeeze(-1)
                return torch.sigmoid(logits).numpy()
    else:
        coef = np.asarray(ckpt["coef"], dtype=np.float32)
        intercept = np.asarray(ckpt["intercept"], dtype=np.float32)

        def score(X):
            z = X @ coef.T + intercept
            return 1.0 / (1.0 + np.exp(-z.squeeze(-1)))

    return score, ckpt


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--probe", nargs="+", required=True,
                   help="One or more checkpoint.json files from run_experiment.py")
    p.add_argument("--capture-dir", required=True, help="Target-task capture")
    p.add_argument("--labels", required=True, help="Target-task labels")
    p.add_argument("--out-file", required=True)
    p.add_argument("--source-metrics", default=None,
                   help="aggregate_metrics.json from the source task, for the drop")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(levelname)s %(message)s")

    meta, activations = load_capture(Path(args.capture_dir), mode="off")
    labels = load_labels(Path(args.labels))
    activations = activations[align_labels(meta, labels)]

    y = np.array([1 if lab["label"] == "helped" else 0 for lab in labels], dtype=np.int64)
    correct_off = np.array([bool(lab["correct_off"]) for lab in labels])
    correct_on = np.array([bool(lab["correct_on"]) for lab in labels])
    difficulties = np.array([str(lab.get("difficulty")) for lab in labels])

    target_config = load_config(Path(args.capture_dir))
    logger.info("target task %s: %d samples, %.1f%% helped",
                target_config.get("task"), len(y), 100 * y.mean())
    if y.sum() == 0:
        logger.error("target set has no positives — AUROC is undefined")
        return 1

    per_probe = []
    for probe_path in args.probe:
        score_fn, ckpt = load_probe(Path(probe_path))
        X = select_layer(activations, ckpt["layer"])
        if X.shape[1] != ckpt["input_dim"]:
            logger.error("probe expects %d features but target has %d — "
                         "different model or layer", ckpt["input_dim"], X.shape[1])
            return 1
        # Source-task standardizer and threshold, deliberately not refit.
        X = Standardizer.from_state(ckpt["scaler"]).transform(X)
        scores = score_fn(X)
        result = evaluate_split(scores, y, correct_off, correct_on,
                                difficulties, ckpt["threshold"])
        result["seed"] = ckpt.get("seed")
        result["probe"] = str(probe_path)
        per_probe.append(result)
        logger.info("probe seed %-4s transfer AUROC %.3f  routed acc %.3f",
                    ckpt.get("seed"), result["auroc"], result["routed_accuracy"])

    transfer_auroc = mean_ci([r["auroc"] for r in per_probe])
    summary = {
        "target_task": target_config.get("task"),
        "target_model": target_config.get("model_name"),
        "n_target": int(len(y)),
        "base_rate_helped": float(y.mean()),
        "transfer_auroc": transfer_auroc,
        "transfer_routed_accuracy": mean_ci([r["routed_accuracy"] for r in per_probe]),
        "target_baselines": per_probe[0]["baselines"],
        "per_probe": per_probe,
    }

    if args.source_metrics:
        source = json.loads(Path(args.source_metrics).read_text(encoding="utf-8"))
        source_auroc = source["aggregate"]["test_auroc"]["mean"]
        drop_pp = 100 * (source_auroc - transfer_auroc["mean"])
        verdict = ("strong transfer" if drop_pp < 5
                   else "moderate — likely label distribution shift" if drop_pp <= 15
                   else "transfer failure (format overfitting)")
        summary["source_task"] = source.get("task")
        summary["source_test_auroc"] = source_auroc
        summary["auroc_drop_pp"] = drop_pp
        summary["verdict"] = verdict
        logger.info("source %s AUROC %.3f -> target %s AUROC %.3f  "
                    "(drop %.1f pp: %s)", source.get("task"), source_auroc,
                    target_config.get("task"), transfer_auroc["mean"], drop_pp, verdict)

    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logger.info("wrote %s", out_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
