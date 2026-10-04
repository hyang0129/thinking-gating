#!/usr/bin/env python3
"""
run_experiment.py — train and evaluate thinking-gating probes.

    python scripts/run_experiment.py \\
        --capture-dir shared/icr_capture/gsm8k_thinking_qwen3 \\
        --labels shared/gsm8k_thinking_labels.jsonl \\
        --method mlp --seeds 42 1 2 3 4 \\
        --out-dir output/gsm8k_probe

Input is the **thinking-off** prefill state: the last-prompt-token activation
at one layer, captured before any token is generated. Training on thinking-on
activations would be circular — you cannot route a query using a state that
only exists after you already paid for thinking.

Reported per seed on a held-out test split, then aggregated over seeds. The
interval to quote is `aggregate.test_auroc_bootstrap.ci` (percentile
bootstrap over test rows); `aggregate.test_auroc.ci` is seed spread and too
narrow.

    AUROC / AUPRC          how well the probe separates helped from not_helped
    AUROC by difficulty    the confound check. If the probe is really a
                           difficulty detector, its AUROC collapses toward 0.5
                           within a difficulty stratum while looking strong
                           overall.
    routed accuracy        the metric that actually matters: send a query to
                           thinking only when the probe says so, and see what
                           task accuracy you end up with. The decision
                           threshold is chosen on validation, never on test.
                           Reported beside random escalation at the same rate
                           and the oracle at the same rate.
    routed curve / nAUC    the exact routed-accuracy curve over every
                           threshold (ties averaged, so it is deterministic)
                           and its area normalised so 0 = random escalation
                           and 1 = oracle, over 0-50% and 0-100% escalation.

Against three baselines: never think, always think, and the oracle that knows
the right answer per query. A probe is only interesting strictly between
`max(never, always)` and `oracle`.

Also writes `predictions.json` (per-row val/test scores by sample_id, every
seed) so compare_baselines.py can run a paired bootstrap against a baseline on
identical rows. The statistics themselves live in utils/metrics.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.capture_io import (  # noqa: E402
    align_labels, load_capture, load_config, load_labels,
)

logger = logging.getLogger("experiment")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def select_layers(activations: np.ndarray, layers: list[int]) -> np.ndarray:
    """(N, L+1, H) -> (N, len(layers)*H), layers concatenated.

    One layer throws away most of the network's state, but concatenating costs
    features linearly, and these probes are already sample-starved -- so this
    helps only where n is large enough to afford the dimensionality.
    """
    return np.concatenate([select_layer(activations, l) for l in layers], axis=1)


def select_layer(activations: np.ndarray, layer: int) -> np.ndarray:
    """(N, L+1, H) -> (N, H) for one layer. Negative indices count from the end."""
    n_layers = activations.shape[1]
    if not -n_layers <= layer < n_layers:
        raise ValueError(f"--layer {layer} out of range for {n_layers} layers")
    return activations[:, layer, :].astype(np.float32)


def stratified_split(y: np.ndarray, seed: int, fractions=(0.6, 0.2, 0.2)):
    """Train/val/test indices, preserving the positive rate in each split.

    With a ~20% positive rate and a few hundred samples, an unstratified split
    can hand a fold almost no positives and make AUROC undefined, so stratify
    even though the splits are simple.
    """
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(fractions[0] * n))
        n_val = int(round(fractions[1] * n))
        train_idx.append(idx[:n_train])
        val_idx.append(idx[n_train:n_train + n_val])
        test_idx.append(idx[n_train + n_val:])
    out = [np.concatenate(part) for part in (train_idx, val_idx, test_idx)]
    for part in out:
        rng.shuffle(part)
    return out


class Standardizer:
    """Fit on train only — validation and test must stay unseen."""

    def __init__(self) -> None:
        self.mean = None
        self.std = None

    def fit(self, X: np.ndarray) -> "Standardizer":
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std < 1e-6] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std

    def state_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_state(cls, state: dict) -> "Standardizer":
        obj = cls()
        obj.mean = np.asarray(state["mean"], dtype=np.float32)
        obj.std = np.asarray(state["std"], dtype=np.float32)
        return obj


# ---------------------------------------------------------------------------
# Metrics — implemented once in utils/metrics.py. Re-exported here because
# other scripts (and older notebooks) import them from run_experiment.
# ---------------------------------------------------------------------------

from utils.metrics import (  # noqa: E402,F401
    auprc, auroc, best_threshold, bootstrap_auroc_ci, cost_accuracy_curve,
    curve_accuracy_at, mean_bootstrap_ci, mean_ci, min_fraction_for_accuracy,
    oracle_curve, random_accuracy_at, routed_accuracy_curve, routed_nauc,
    routing_metrics,
)

# Escalation ranges the routed-accuracy area is reported over. [0, 0.5] is
# the pre-registered G2 range (paper/proposal_system_one_gating.md §3).
NAUC_RANGES = {"escalation_0_50": (0.0, 0.5), "escalation_0_100": (0.0, 1.0)}


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

def train_mlp(X_tr, y_tr, X_val, y_val, *, seed: int, hidden=(256, 64),
              dropout=0.1, lr=1e-3, batch_size=32, max_epochs=200, patience=20):
    """Small MLP, early-stopped on validation AUROC. Returns (model, history)."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    layers, in_dim = [], X_tr.shape[1]
    for width in hidden:
        layers += [nn.Linear(in_dim, width), nn.ReLU(), nn.Dropout(dropout)]
        in_dim = width
    layers.append(nn.Linear(in_dim, 1))
    model = nn.Sequential(*layers).to(device)

    Xtr = torch.tensor(X_tr, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_tr, dtype=torch.float32, device=device)
    Xva = torch.tensor(X_val, dtype=torch.float32, device=device)

    # Positives are the minority (~20%); without the reweighting the probe can
    # minimize loss by predicting "never helps" for everything.
    n_pos = max(float(y_tr.sum()), 1.0)
    pos_weight = torch.tensor([(len(y_tr) - n_pos) / n_pos], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_state, best_auc, best_epoch, history = None, -1.0, 0, []
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(len(Xtr), generator=generator).to(device)
        for start in range(0, len(perm), batch_size):
            idx = perm[start:start + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx]).squeeze(-1), ytr[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_scores = torch.sigmoid(model(Xva).squeeze(-1)).cpu().numpy()
        val_auc = auroc(y_val, val_scores)
        history.append({"epoch": epoch, "val_auroc": val_auc})
        if not math.isnan(val_auc) and val_auc > best_auc:
            best_auc, best_epoch = val_auc, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch - best_epoch >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, {"best_val_auroc": best_auc, "best_epoch": best_epoch,
                   "epochs_run": len(history)}


def predict(model, X) -> np.ndarray:
    import torch
    device = next(model.parameters()).device
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32, device=device)).squeeze(-1)
        return torch.sigmoid(logits).cpu().numpy()


def train_logreg(X_tr, y_tr, X_val, y_val, *, seed: int):
    """Logistic regression baseline — if the MLP cannot beat this, say so."""
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    clf.fit(X_tr, y_tr)
    val_scores = clf.predict_proba(X_val)[:, 1]
    return clf, {"best_val_auroc": auroc(y_val, val_scores), "epochs_run": 0}


def predict_sklearn(clf, X) -> np.ndarray:
    return clf.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def evaluate_split(scores, y, correct_off, correct_on, difficulties, threshold) -> dict:
    out = {
        "n": int(len(y)),
        "n_positive": int(y.sum()),
        "base_rate": float(y.mean()) if len(y) else float("nan"),
        "auroc": auroc(y, scores),
        "auroc_bootstrap": bootstrap_auroc_ci(y, scores),
        "auprc": auprc(y, scores),
    }
    out.update(routing_metrics(correct_off, correct_on, scores, threshold))
    out["baselines"] = {
        "never_think": float(np.mean(correct_off)),
        "always_think": float(np.mean(correct_on)),
        "oracle": float(np.mean(np.asarray(correct_off) | np.asarray(correct_on))),
    }
    by_difficulty = {}
    for level in sorted({str(d) for d in difficulties}):
        mask = np.array([str(d) == level for d in difficulties])
        if mask.sum() < 2:
            continue
        by_difficulty[level] = {
            "n": int(mask.sum()),
            "base_rate": float(y[mask].mean()),
            "auroc": auroc(y[mask], np.asarray(scores)[mask]),
            # Strata are small -- this is where the seed-spread CI misleads most.
            "auroc_bootstrap": bootstrap_auroc_ci(y[mask], np.asarray(scores)[mask]),
        }
    out["by_difficulty"] = by_difficulty

    # References at the probe's realised escalation rate: what escalating the
    # same fraction at random, or by oracle, would have scored.
    curve = routed_accuracy_curve(correct_off, correct_on, scores)
    oracle = oracle_curve(correct_off, correct_on)
    rate = out["fraction_routed_to_thinking"]
    out["random_at_matched_rate"] = float(random_accuracy_at(correct_off, correct_on, rate))
    out["oracle_at_matched_rate"] = float(curve_accuracy_at(oracle, rate))

    # The exact curve (its vertices, one per distinct score; linear between
    # them) and its normalised area. cost_curve is the old 21-point grid,
    # kept so existing readers of this JSON keep working.
    out["routed_curve"] = {
        "fraction": [round(float(v), 6) for v in curve["vertex_fraction"]],
        "accuracy": [round(float(v), 6) for v in curve["vertex_accuracy"]],
    }
    out["routed_nauc"] = {name: routed_nauc(correct_off, correct_on, scores, lo, hi)
                          for name, (lo, hi) in NAUC_RANGES.items()}
    out["cost_curve"] = cost_accuracy_curve(correct_off, correct_on, scores)

    # The headline cost number: the smallest routed fraction whose accuracy
    # still matches always-think, i.e. how much thinking is simply wasted.
    # Read off the exact curve, not the grid.
    always = float(np.mean(correct_on))
    out["min_routed_for_always_think_accuracy"] = min_fraction_for_accuracy(curve, always)
    return out


def run_seed(args, X, y, correct_off, correct_on, difficulties, seed: int,
             return_predictions: bool = False):
    """Train one seed. Returns (metrics, checkpoint), plus the per-row
    validation/test predictions when ``return_predictions`` is set."""
    train_idx, val_idx, test_idx = stratified_split(y, seed)
    scaler = Standardizer().fit(X[train_idx])
    Xtr, Xva, Xte = (scaler.transform(X[i]) for i in (train_idx, val_idx, test_idx))

    if args.method == "mlp":
        model, info = train_mlp(Xtr, y[train_idx], Xva, y[val_idx], seed=seed,
                                max_epochs=args.max_epochs, patience=args.patience)
        score_fn = predict
    else:
        model, info = train_logreg(Xtr, y[train_idx], Xva, y[val_idx], seed=seed)
        score_fn = predict_sklearn

    val_scores = score_fn(model, Xva)
    test_scores = score_fn(model, Xte)

    # Threshold picked on validation, then applied unchanged to test.
    threshold = best_threshold(correct_off[val_idx], correct_on[val_idx], val_scores)

    metrics = {
        "seed": seed,
        "train_info": info,
        "val": evaluate_split(val_scores, y[val_idx], correct_off[val_idx],
                              correct_on[val_idx], difficulties[val_idx], threshold),
        "test": evaluate_split(test_scores, y[test_idx], correct_off[test_idx],
                               correct_on[test_idx], difficulties[test_idx], threshold),
    }
    checkpoint = {
        "method": args.method, "target": args.target,
        "prefill_mode": args.prefill_mode,
        "layer": args.layer, "seed": seed,
        "threshold": threshold, "scaler": scaler.state_dict(),
        "input_dim": int(X.shape[1]),
    }
    if getattr(args, "layers", None):
        # Concatenated-layer probe: eval_transfer must rebuild the same
        # features, so record which layers, not just the unused --layer.
        checkpoint["layers"] = list(args.layers)
    if args.method == "mlp":
        checkpoint["state_dict"] = {k: v.cpu().tolist()
                                    for k, v in model.state_dict().items()}
    else:
        checkpoint["coef"] = model.coef_.tolist()
        checkpoint["intercept"] = model.intercept_.tolist()
    if not return_predictions:
        return metrics, checkpoint
    predictions = {"val": {"row": val_idx.tolist(), "score": val_scores.tolist()},
                   "test": {"row": test_idx.tolist(), "score": test_scores.tolist()}}
    return metrics, checkpoint, predictions


def predictions_record(rows: list[dict], y, correct_off, correct_on,
                       per_seed: dict[int, dict], *, target: str, source: str) -> dict:
    """Per-row predictions in the shape compare_baselines.py pairs on.

    ``per_seed[seed] = {"val": {"row": [...], "score": [...]}, "test": ...}``
    with row positions into ``rows``. Rows are written by sample_id so two
    methods' predictions are matched on identity, never on position.
    """
    out = {"target": target, "source": source, "seeds": {}}
    for seed, splits in per_seed.items():
        rec = {}
        for split, node in splits.items():
            idx = node["row"]
            rec[split] = {
                "sample_id": [str(rows[i]["sample_id"]) for i in idx],
                "y": [int(y[i]) for i in idx],
                "correct_off": [bool(correct_off[i]) for i in idx],
                "correct_on": [bool(correct_on[i]) for i in idx],
            }
            if "score" in node:       # one scorer (the probe)
                rec[split]["score"] = [float(s) for s in node["score"]]
            if "scores" in node:      # several named scorers (baselines)
                rec[split]["scores"] = {k: [float(s) for s in v]
                                        for k, v in node["scores"].items()}
        out["seeds"][str(seed)] = rec
    return out


def aggregate(per_seed: list[dict]) -> dict:
    def collect(path):
        out = []
        for m in per_seed:
            node = m
            for key in path:
                node = node[key]
            out.append(float(node))
        return out

    agg = {
        "test_auroc": mean_ci(collect(["test", "auroc"])),
        "test_auprc": mean_ci(collect(["test", "auprc"])),
        "test_routed_accuracy": mean_ci(collect(["test", "routed_accuracy"])),
        "fraction_routed": mean_ci(collect(["test", "fraction_routed_to_thinking"])),
        "baseline_never_think": mean_ci(collect(["test", "baselines", "never_think"])),
        "baseline_always_think": mean_ci(collect(["test", "baselines", "always_think"])),
        "oracle": mean_ci(collect(["test", "baselines", "oracle"])),
        "min_routed_for_always_think_accuracy": mean_ci(
            collect(["test", "min_routed_for_always_think_accuracy"])),
        "test_random_at_matched_rate": mean_ci(collect(["test", "random_at_matched_rate"])),
        "test_oracle_at_matched_rate": mean_ci(collect(["test", "oracle_at_matched_rate"])),
        "test_routed_nauc": {
            name: mean_ci(collect(["test", "routed_nauc", name, "nauc"]))
            for name in NAUC_RANGES},
    }
    # The honest interval. "test_auroc" above is seed-to-seed spread on a fixed
    # sample and is systematically too narrow; quote this one in the paper.
    agg["test_auroc_bootstrap"] = mean_bootstrap_ci(
        [m["test"]["auroc_bootstrap"] for m in per_seed])
    levels = {lvl for m in per_seed for lvl in m["test"]["by_difficulty"]}
    agg["test_auroc_by_difficulty"] = {
        lvl: mean_ci([m["test"]["by_difficulty"][lvl]["auroc"]
                      for m in per_seed if lvl in m["test"]["by_difficulty"]])
        for lvl in sorted(levels)
    }
    agg["test_auroc_by_difficulty_bootstrap"] = {
        lvl: mean_bootstrap_ci([m["test"]["by_difficulty"][lvl]["auroc_bootstrap"]
                                for m in per_seed
                                if lvl in m["test"]["by_difficulty"]])
        for lvl in sorted(levels)
    }
    return agg


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--capture-dir", required=True, nargs="+",
                   help="One or more capture dirs. Several are pooled into a "
                        "single training set, paired positionally with --labels.")
    p.add_argument("--labels", required=True, nargs="+")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--method", default="mlp", choices=["mlp", "logreg"])
    p.add_argument("--target", default="helped",
                   choices=["helped", "needs_thinking", "rescued"],
                   help="'helped' = thinking flipped wrong->right (the original "
                        "framing). 'needs_thinking' = the model is wrong WITHOUT "
                        "thinking, which is what a router actually has to decide "
                        "and is independent of the thinking token budget. "
                        "'rescued' = THE DECOMPOSITION: restrict to rows the "
                        "model already gets wrong without thinking, then predict "
                        "whether thinking rescues them. On that subset "
                        "'needs_thinking' is constant by construction, so any "
                        "signal here is about the marginal value of reasoning "
                        "rather than about difficulty.")
    p.add_argument("--prefill-mode", default="off", choices=["off", "on"],
                   help="Which prompt's prefill state to probe. 'off' is the "
                        "default and the safe one. 'on' probes the prefill of "
                        "the thinking-ENABLED prompt -- still captured before "
                        "any token is generated, so it sees no reasoning trace "
                        "and is not circular, but it differs by the template's "
                        "think tags and is only usable by a router that has "
                        "already committed to rendering that prompt.")
    p.add_argument("--layer", type=int, default=None,
                   help="Layer to probe (default: the middle layer)")
    p.add_argument("--layers", type=int, nargs="+", default=None,
                   help="Probe several layers concatenated. Overrides --layer. "
                        "Costs features linearly, so it needs a large n.")
    p.add_argument("--layer-stride", type=int, default=0,
                   help="With --layers unset, take every Nth layer (e.g. 4 -> "
                        "layers 0,4,8,...). Shorthand for a spread of depths.")
    p.add_argument("--layer-sweep", action="store_true",
                   help="Train on every layer with the first seed and report AUROC per layer")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2, 3, 4])
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(levelname)s %(message)s")

    if len(args.capture_dir) != len(args.labels):
        logger.error("got %d capture dir(s) but %d label file(s) — they pair "
                     "positionally", len(args.capture_dir), len(args.labels))
        return 1

    # Pool one or more tasks. Pooling happens before the split, so a pooled run
    # measures whether one probe works across tasks; holding a task out
    # entirely and scoring it with eval_transfer.py measures generalization to
    # an unseen task. They answer different questions — don't conflate them.
    act_parts, label_parts, task_parts = [], [], []
    for capture_dir, labels_file in zip(args.capture_dir, args.labels):
        meta_i, acts_i = load_capture(Path(capture_dir), mode=args.prefill_mode)
        labels_i = load_labels(Path(labels_file))
        acts_i = acts_i[align_labels(meta_i, labels_i)]
        task_name = load_config(Path(capture_dir)).get("task") or Path(capture_dir).name
        act_parts.append(acts_i)
        label_parts.extend(labels_i)
        task_parts.extend([task_name] * len(labels_i))
        logger.info("  %-28s %5d rows from %s", task_name, len(labels_i), capture_dir)

    shapes = {a.shape[1:] for a in act_parts}
    if len(shapes) > 1:
        logger.error("capture activation shapes differ %s — pooling would be "
                     "meaningless across different models", shapes)
        return 1
    activations = np.concatenate(act_parts, axis=0)
    labels = label_parts
    tasks = np.array(task_parts)

    correct_off = np.array([bool(lab["correct_off"]) for lab in labels])
    correct_on = np.array([bool(lab["correct_on"]) for lab in labels])
    difficulties = np.array([str(lab.get("difficulty")) for lab in labels])

    if args.target == "helped":
        y = np.array([1 if lab["label"] == "helped" else 0 for lab in labels],
                     dtype=np.int64)
    elif args.target == "needs_thinking":
        y = (~correct_off).astype(np.int64)
    else:
        # 'rescued'. Keep only the rows the model already fails without
        # thinking; among those, predict whether thinking saves it. Subsetting
        # happens BEFORE the split, so train/val/test all live on the
        # conditional distribution and no held-out row leaks in from the easy
        # half. Note the routing baselines below go degenerate on this subset
        # (never_think is 0 by construction) -- AUROC is the number that means
        # anything here, not routed accuracy.
        keep = np.flatnonzero(~correct_off)
        logger.info("target=rescued — restricting to %d/%d rows the model gets "
                    "wrong without thinking", len(keep), len(correct_off))
        if len(keep) < 40:
            logger.error("only %d rows with correct_off=False — too few to "
                         "condition on", len(keep))
            return 1
        activations = activations[keep]
        labels = [labels[i] for i in keep]
        tasks = tasks[keep]
        correct_off = correct_off[keep]
        correct_on = correct_on[keep]
        difficulties = difficulties[keep]
        y = correct_on.astype(np.int64)

    n_layers = activations.shape[1]
    if args.layer is None:
        args.layer = n_layers // 2
    logger.info("target=%s — %d samples, %d layers, %d positives (%.1f%% base rate)",
                args.target, len(y), n_layers, int(y.sum()), 100 * y.mean())
    if len(args.capture_dir) > 1:
        for name in sorted(set(tasks)):
            mask = tasks == name
            logger.info("    %-14s n=%-5d base rate %.1f%%",
                        name, int(mask.sum()), 100 * y[mask].mean())
    if y.sum() < 10:
        logger.error("only %d positive example(s) — not enough to train a probe",
                     int(y.sum()))
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.layer_sweep:
        seed = args.seeds[0]
        # The layer is chosen on VALIDATION AUROC. Picking it by test AUROC and
        # then reporting that same test AUROC would be selection leakage: with
        # ~37 layers to choose from, the best test score is partly luck, and
        # the reported number would be biased upward by exactly that luck.
        sweep_val, sweep_test = {}, {}
        for layer in range(n_layers):
            args.layer = layer
            X = select_layer(activations, layer)
            metrics, _ = run_seed(args, X, y, correct_off, correct_on, difficulties, seed)
            sweep_val[layer] = metrics["val"]["auroc"]
            sweep_test[layer] = metrics["test"]["auroc"]
            logger.info("layer %2d: val AUROC %.3f  (test %.3f)",
                        layer, sweep_val[layer], sweep_test[layer])
        best_layer = max(sweep_val,
                         key=lambda k: (sweep_val[k] if not math.isnan(sweep_val[k]) else -1))
        (out_dir / "layer_sweep.json").write_text(
            json.dumps({"val_auroc_by_layer": sweep_val,
                        "test_auroc_by_layer": sweep_test,
                        "best_layer": best_layer,
                        "selected_on": "validation", "seed": seed}, indent=2) + "\n")
        logger.info("best layer: %d (val AUROC %.3f, test %.3f)",
                    best_layer, sweep_val[best_layer], sweep_test[best_layer])
        args.layer = best_layer

    if args.layer_stride and not args.layers:
        args.layers = list(range(0, n_layers, args.layer_stride))
    if args.layers:
        for l in args.layers:
            if not -n_layers <= l < n_layers:
                logger.error("--layers %d out of range for %d layers", l, n_layers)
                return 1
        X = select_layers(activations, args.layers)
        logger.info("probing layers %s — features %s, method=%s",
                    args.layers, X.shape, args.method)
    else:
        X = select_layer(activations, args.layer)
        logger.info("probing layer %d — features %s, method=%s",
                    args.layer, X.shape, args.method)

    per_seed, seed_predictions = [], {}
    for seed in args.seeds:
        metrics, checkpoint, preds = run_seed(args, X, y, correct_off, correct_on,
                                              difficulties, seed, return_predictions=True)
        per_seed.append(metrics)
        seed_predictions[seed] = preds
        seed_dir = out_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        (seed_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        (seed_dir / "checkpoint.json").write_text(json.dumps(checkpoint) + "\n")
        logger.info("seed %-3d test AUROC %.3f  routed acc %.3f  "
                    "(never %.3f / always %.3f / oracle %.3f)",
                    seed, metrics["test"]["auroc"], metrics["test"]["routed_accuracy"],
                    metrics["test"]["baselines"]["never_think"],
                    metrics["test"]["baselines"]["always_think"],
                    metrics["test"]["baselines"]["oracle"])

    agg = aggregate(per_seed)
    first_config = load_config(Path(args.capture_dir[0]))
    summary = {
        "task": (first_config.get("task") if len(args.capture_dir) == 1
                 else sorted(set(tasks))),
        "model": first_config.get("model_name"),
        "capture_dirs": list(args.capture_dir),
        "method": args.method, "target": args.target,
        "layer": args.layer, "seeds": args.seeds,
        "n_samples": int(len(y)), "base_rate_helped": float(y.mean()),
        "base_rate": float(y.mean()), "prefill_mode": args.prefill_mode,
        "aggregate": agg, "per_seed": per_seed,
    }
    (out_dir / "aggregate_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    # Per-row validation/test scores, keyed by sample_id: what
    # compare_baselines.py needs for a paired probe-vs-baseline bootstrap.
    (out_dir / "predictions.json").write_text(json.dumps(predictions_record(
        labels, y, correct_off, correct_on, seed_predictions,
        target=args.target, source="prefill_probe")) + "\n")

    auc = agg["test_auroc"]
    boot = agg["test_auroc_bootstrap"]["ci"]
    logger.info("=" * 64)
    logger.info("test AUROC        %.3f  [%.3f, %.3f] bootstrap 95%% CI  over %d seeds",
                auc["mean"], boot[0], boot[1], auc["n"])
    nauc = agg["test_routed_nauc"]["escalation_0_50"]
    logger.info("routed nAUC 0-50%% %.3f  (0 = random escalation, 1 = oracle)",
                nauc["mean"])
    logger.info("routed accuracy   %.3f  [%.3f, %.3f]",
                agg["test_routed_accuracy"]["mean"], *agg["test_routed_accuracy"]["ci"])
    logger.info("  never think     %.3f", agg["baseline_never_think"]["mean"])
    logger.info("  always think    %.3f", agg["baseline_always_think"]["mean"])
    logger.info("  oracle          %.3f", agg["oracle"]["mean"])
    saved = agg["min_routed_for_always_think_accuracy"]
    logger.info("thinking needed for always-think accuracy: %.0f%% of queries "
                "(so %.0f%% of thinking is wasted)",
                100 * saved["mean"], 100 * (1 - saved["mean"]))
    for level, stats in agg["test_auroc_by_difficulty"].items():
        logger.info("  AUROC %-6s    %.3f  [%.3f, %.3f]",
                    level, stats["mean"], *stats["ci"])
    logger.info("wrote %s", out_dir / "aggregate_metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
