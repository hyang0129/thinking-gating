#!/usr/bin/env bash
# run_full_analysis.sh — captures in, results table out.
#
# Labels every capture, trains a probe per (task x objective), evaluates every
# ordered cross-task pair with no retraining, and renders one table.
#
#   bash scripts/run_full_analysis.sh                  # everything
#   TASKS="math500 bbh" bash scripts/run_full_analysis.sh
#
# Idempotent: each step skips when its output already exists, so it can be
# re-run after adding a task without repeating finished work. Pass FORCE=1 to
# recompute.
#
# CPU only — no GPU needed. Threads are bounded because the login node has 192
# cores and torch will otherwise thrash across all of them.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY="${PY:-$REPO_ROOT/.venv/bin/python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

TASKS="${TASKS:-gsm8k lsat math500 mmlu_pro bbh}"
TARGETS="${TARGETS:-needs_thinking helped}"
LAYER="${LAYER:-18}"          # a priori middle layer; never swept, to avoid selection effects
METHOD="${METHOD:-logreg}"
SEEDS="${SEEDS:-42 1 2 3 4}"
METRICS_DIR="paper/results/metrics"

# task -> capture directory. Captures are named {task}_thinking_{CAPTURE_SLUG}
# with {task} the task module name exactly, so this is one substitution rather
# than a table to keep in sync.
#
#   CAPTURE_SLUG=qwen3v3 bash scripts/run_full_analysis.sh
#
# The legacy branch covers the pre-v3 captures, which spelled three tasks
# differently: gsm8k_full (all 1319 rows, as against a 500-row pilot),
# lsat_long (8192 thinking budget, as against 3072) and mmlupro. Those
# qualifiers marked capture parameters that varied then and do not now -- v3
# captures each task once, gsm8k at full size and lsat at the long budget --
# so they are history to read, not a convention to extend. Do not add to this
# branch; name new captures after the task.
CAPTURE_SLUG="${CAPTURE_SLUG:-qwen3}"

# The alias is consulted BEFORE the plain name, and the order is load-bearing:
# shared/icr_capture/gsm8k_thinking_qwen3 also exists and is a 500-row pilot,
# so probing for the plain name first would silently swap the legacy analysis
# from 1319 rows to 500 and report it as the same run.
capture_dir() {
  local task="$1"
  if [ "$CAPTURE_SLUG" = "qwen3" ]; then
    case "$task" in
      gsm8k)    echo "shared/icr_capture/gsm8k_full_thinking_qwen3" ; return ;;
      lsat)     echo "shared/icr_capture/lsat_long_thinking_qwen3"  ; return ;;
      mmlu_pro) echo "shared/icr_capture/mmlupro_thinking_qwen3"    ; return ;;
    esac
  fi
  local dir="shared/icr_capture/${task}_thinking_${CAPTURE_SLUG}"
  if [ -d "$dir" ]; then echo "$dir"; else echo ""; fi
}
labels_file() { echo "shared/${1}_labels.jsonl"; }
probe_dir()   { echo "output/probe_${1}_${2}"; }

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

mkdir -p "$METRICS_DIR" paper/results/labels

# --- 1. labels -------------------------------------------------------------
AVAILABLE=""
for task in $TASKS; do
  cap="$(capture_dir "$task")"
  if [ -z "$cap" ] || ! ls "$cap"/meta.shard*.jsonl >/dev/null 2>&1; then
    log "SKIP $task — no capture at ${cap:-<unmapped>}"
    continue
  fi
  AVAILABLE="$AVAILABLE $task"
  lab="$(labels_file "$task")"
  if [ -s "$lab" ] && [ -z "${FORCE:-}" ]; then
    log "labels $task — already present"
  else
    log "labels $task"
    "$PY" scripts/generate_labels.py --capture-dir "$cap" --out-file "$lab" \
      2>&1 | grep -E "base rate|accuracy :|truncated" | sed "s/^/    /"
  fi
  cp -f "${lab%.jsonl}.summary.json" "paper/results/labels/${task}.json" 2>/dev/null
done

log "tasks with captures:${AVAILABLE:- none}"
[ -z "$AVAILABLE" ] && { log "nothing to analyze"; exit 1; }

# --- 2. one probe per (task, objective) ------------------------------------
for task in $AVAILABLE; do
  for target in $TARGETS; do
    out="$(probe_dir "$task" "$target")"
    if [ -s "$out/aggregate_metrics.json" ] && [ -z "${FORCE:-}" ]; then
      log "probe $task/$target — already present"
    else
      log "probe $task/$target"
      "$PY" scripts/run_experiment.py \
        --capture-dir "$(capture_dir "$task")" --labels "$(labels_file "$task")" \
        --out-dir "$out" --method "$METHOD" --target "$target" \
        --layer "$LAYER" --seeds $SEEDS \
        2>&1 | grep -E "test AUROC  |AUROC (easy|medium|hard)|thinking needed" | sed "s/^/    /"
    fi
    cp -f "$out/aggregate_metrics.json" "$METRICS_DIR/${task}__${target}.json" 2>/dev/null
  done
done

# --- 3. every ordered cross-task pair, no retraining -----------------------
for target in $TARGETS; do
  for src in $AVAILABLE; do
    ckpts=""
    for seed in $SEEDS; do
      c="$(probe_dir "$src" "$target")/seed_${seed}/checkpoint.json"
      [ -s "$c" ] && ckpts="$ckpts $c"
    done
    [ -z "$ckpts" ] && { log "SKIP transfer from $src/$target — no checkpoints"; continue; }
    for tgt in $AVAILABLE; do
      [ "$src" = "$tgt" ] && continue
      out="output/transfer_${src}_to_${tgt}_${target}.json"
      if [ -s "$out" ] && [ -z "${FORCE:-}" ]; then
        log "transfer $src -> $tgt ($target) — already present"
      else
        log "transfer $src -> $tgt ($target)"
        "$PY" scripts/eval_transfer.py --probe $ckpts \
          --capture-dir "$(capture_dir "$tgt")" --labels "$(labels_file "$tgt")" \
          --source-metrics "$(probe_dir "$src" "$target")/aggregate_metrics.json" \
          --out-file "$out" 2>&1 | grep -E "source .* -> target" | sed "s/^/    /"
      fi
      cp -f "$out" "$METRICS_DIR/transfer__${src}_to_${tgt}__${target}.json" 2>/dev/null
    done
  done
done

# --- 4. table --------------------------------------------------------------
log "rendering results table"
"$PY" scripts/results_table.py --metrics-dir "$METRICS_DIR" \
  --out paper/results/results_table.txt
"$PY" scripts/results_table.py --metrics-dir "$METRICS_DIR" \
  --format csv --out paper/results/results_table.csv
echo
cat paper/results/results_table.txt
