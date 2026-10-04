#!/usr/bin/env bash
# run_full_analysis.sh — captures in, results table out.
#
# Labels every capture, trains a probe per (task x objective), runs the text
# and confidence baselines on the identical splits, evaluates every ordered
# cross-task pair with no retraining, runs the pooled within-group AUROC where
# a group key is configured, and renders the tables.
#
#   CAPTURE_SLUG=qwen3v3 bash scripts/run_full_analysis.sh
#   TASKS="math500 bbh" CAPTURE_SLUG=qwen3v3 bash scripts/run_full_analysis.sh
#   CAPTURE_SLUG=qwen3v3 PROMOTE=1 bash scripts/run_full_analysis.sh
#
# Outputs go to output/<slug>/ ONLY. paper/results/ is the provenance record
# and is written by one explicit step:
#
#   PROMOTE=1   after the analysis, copy output/<slug>/metrics/ into
#               paper/results/metrics/<slug>/. Refuses (and copies nothing)
#               if any file there would change, unless FORCE=1. Identical
#               files are skipped. Write the README entry when you promote.
#
# Incremental: a step re-runs when its output is missing OR older than any of
# its inputs (capture meta -> labels -> probes/baselines/transfer), so
# re-graded labels propagate instead of being silently skipped.
#   FORCE=1     recompute everything (and, with PROMOTE=1, allow overwriting
#               files already in paper/results/).
#
# Fails fast: `set -euo pipefail`, every step's full output goes to
# output/<slug>/logs/<step>.log, and a failing step prints its log tail and
# stops the run. Nothing is filtered through grep on the way to a decision.
#
# Layer: unset LAYER (the default) lets run_experiment.py / within_group_auroc.py
# take the middle layer from the activation shape — 18 for Qwen3-8B (37 hidden
# states), 16 for Nemotron-8B (33). The published Nemotron numbers used a
# hard-coded 18; set LAYER=18 to reproduce them. Never swept, to avoid
# selection effects.
#
# CPU only — no GPU needed. Threads are bounded because the login node has 192
# cores and torch will otherwise thrash across all of them.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY="${PY:-$REPO_ROOT/.venv/bin/python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

TASKS="${TASKS:-gsm8k lsat math500 mmlu_pro bbh}"
TARGETS="${TARGETS:-rescued needs_thinking helped}"
LAYER="${LAYER:-}"            # empty = middle layer from the activation shape
METHOD="${METHOD:-logreg}"
SEEDS="${SEEDS:-42 1 2 3 4}"
FORCE="${FORCE:-}"
PROMOTE="${PROMOTE:-}"
# Pooled within-group AUROC: space-separated task=key pairs, key as
# within_group_auroc.py --group-key takes it. Only tasks whose key genuinely
# partitions them belong here (sample_id:middle is the BBH subtask; for the
# other tasks it is the split name and the script refuses).
WITHIN_GROUP="${WITHIN_GROUP:-bbh=sample_id:middle}"
WITHIN_TARGETS="${WITHIN_TARGETS:-rescued needs_thinking}"

# task -> capture directory. Captures are named {task}_thinking_{CAPTURE_SLUG}
# with {task} the task module name exactly, so this is one substitution rather
# than a table to keep in sync.
#
# The legacy branch covers the pre-v3 captures, which spelled three tasks
# differently: gsm8k_full (all 1319 rows, as against a 500-row pilot),
# lsat_long (8192 thinking budget, as against 3072) and mmlupro. Those
# qualifiers marked capture parameters that varied then and do not now -- v3
# captures each task once, gsm8k at full size and lsat at the long budget --
# so they are history to read, not a convention to extend. Do not add to this
# branch; name new captures after the task.
CAPTURE_SLUG="${CAPTURE_SLUG:-qwen3}"

# Every derived artifact is qualified by the capture slug, so two models'
# results can never land on the same path.
OUT_ROOT="output/${CAPTURE_SLUG}"
METRICS_DIR="${METRICS_DIR:-${OUT_ROOT}/metrics}"
LOG_DIR="${OUT_ROOT}/logs"
PROMOTE_DIR="${PROMOTE_DIR:-paper/results/metrics/${CAPTURE_SLUG}}"

case "$METRICS_DIR" in
  paper/*|./paper/*|"$REPO_ROOT"/paper/*)
    echo "METRICS_DIR=$METRICS_DIR is under paper/ — analysis writes to output/; use PROMOTE=1" >&2
    exit 2 ;;
esac

# The alias is consulted BEFORE the plain name, and the order is load-bearing:
# shared/icr_capture/gsm8k_thinking_qwen3 also exists and is a 500-row pilot,
# so probing for the plain name first would silently swap the legacy analysis
# from 1319 rows to 500 and report it as the same run.
capture_dir() {
  local task="$1"
  if [ "$CAPTURE_SLUG" = "qwen3" ]; then
    case "$task" in
      gsm8k)    echo "shared/icr_capture/gsm8k_full_thinking_qwen3" ; return 0 ;;
      lsat)     echo "shared/icr_capture/lsat_long_thinking_qwen3"  ; return 0 ;;
      mmlu_pro) echo "shared/icr_capture/mmlupro_thinking_qwen3"    ; return 0 ;;
    esac
  fi
  local dir="shared/icr_capture/${task}_thinking_${CAPTURE_SLUG}"
  if [ -d "$dir" ]; then echo "$dir"; else echo ""; fi
}
labels_file() { echo "shared/labels/${CAPTURE_SLUG}/${1}_labels.jsonl"; }
probe_dir()   { echo "${OUT_ROOT}/probe_${1}_${2}"; }

log() { echo "[$(date -u +%H:%M:%S)] $*"; }
die() { log "ERROR: $*"; exit 1; }

# stale OUT IN... — true (0) if OUT must be (re)built: FORCE set, OUT missing
# or empty, or any IN newer than OUT.
stale() {
  local out="$1"; shift
  if [ -n "$FORCE" ] || [ ! -s "$out" ]; then return 0; fi
  local in
  for in in "$@"; do
    if [ "$in" -nt "$out" ]; then return 0; fi
  done
  return 1
}

# run_step NAME SUMMARY_REGEX CMD... — run CMD with its full output in
# $LOG_DIR/NAME.log. On failure print the log tail and stop. On success echo
# the lines matching SUMMARY_REGEX (display only; never a gate; "" = none).
run_step() {
  local name="$1" pattern="$2"; shift 2
  local logf="$LOG_DIR/${name}.log"
  if ! "$@" >"$logf" 2>&1; then
    log "FAILED $name — last lines of $logf:"
    tail -n 40 "$logf" | sed "s/^/    /" >&2
    exit 1
  fi
  if [ -n "$pattern" ]; then
    grep -E "$pattern" "$logf" | sed "s/^/    /" || true
  fi
}

# Copy SRC to DST (inside METRICS_DIR), creating the directory.
publish() { mkdir -p "$(dirname "$2")"; cp -f "$1" "$2"; }

mkdir -p "$METRICS_DIR" "$METRICS_DIR/labels" "$METRICS_DIR/baselines" \
         "$LOG_DIR" "shared/labels/${CAPTURE_SLUG}"

LAYER_ARGS=()
if [ -n "$LAYER" ]; then LAYER_ARGS=(--layer "$LAYER"); fi

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
  # Labels are regraded from the stored responses with the current grader, so
  # a grader fix re-labels old captures; the task module is an input for that.
  # REGRADE=0 keeps the grades stored at capture time (synthetic test captures
  # carry placeholder responses that cannot be regraded).
  if [ "${REGRADE:-1}" = "1" ]; then regrade=(--regrade); else regrade=(); fi
  if stale "$lab" "$cap"/meta.shard*.jsonl "tasks/${task}.py"; then
    log "labels $task"
    run_step "labels_${task}" "base rate|accuracy :|truncated" \
      "$PY" scripts/generate_labels.py ${regrade[@]+"${regrade[@]}"} --capture-dir "$cap" --out-file "$lab"
  else
    log "labels $task — up to date"
  fi
  summary="${lab%.jsonl}.summary.json"
  [ -s "$summary" ] || die "generate_labels.py wrote no $summary"
  publish "$summary" "$METRICS_DIR/labels/${task}.json"
done

log "tasks with captures:${AVAILABLE:- none}"
if [ -z "$AVAILABLE" ]; then log "nothing to analyze"; exit 1; fi

# --- 2. one probe per (task, objective) ------------------------------------
for task in $AVAILABLE; do
  lab="$(labels_file "$task")"
  for target in $TARGETS; do
    out="$(probe_dir "$task" "$target")"
    if stale "$out/aggregate_metrics.json" "$lab"; then
      log "probe $task/$target"
      run_step "probe_${task}_${target}" "test AUROC  |routed nAUC|AUROC (easy|medium|hard)|thinking needed" \
        "$PY" scripts/run_experiment.py \
          --capture-dir "$(capture_dir "$task")" --labels "$lab" \
          --out-dir "$out" --method "$METHOD" --target "$target" \
          ${LAYER_ARGS[@]+"${LAYER_ARGS[@]}"} --seeds $SEEDS
    else
      log "probe $task/$target — up to date"
    fi
    [ -s "$out/aggregate_metrics.json" ] || die "no $out/aggregate_metrics.json"
    publish "$out/aggregate_metrics.json" "$METRICS_DIR/${task}__${target}.json"
    if [ -s "$out/predictions.json" ]; then
      publish "$out/predictions.json" "$METRICS_DIR/${task}__${target}.predictions.json"
    fi
  done
done

# --- 2b. baselines on the identical splits ---------------------------------
# An 8B forward pass has to beat TF-IDF on the raw question, and it has to be
# compared against the model's own thinking-off confidence. Both run on the
# same splits/seeds/target as the probe so the numbers sit in one table.
# baseline_confidence needs --capture-logprobs captures (all of v3).
for task in $AVAILABLE; do
  lab="$(labels_file "$task")"
  for target in $TARGETS; do
    if [ "$target" = "helped" ]; then continue; fi
    for kind in text confidence; do
      out="$METRICS_DIR/baselines/${kind}__${task}__${target}.json"
      if stale "$out" "$lab"; then
        log "baseline $kind $task/$target"
        run_step "baseline_${kind}_${task}_${target}" "AUROC" \
          "$PY" "scripts/baseline_${kind}.py" \
            --capture-dir "$(capture_dir "$task")" --labels "$lab" \
            --target "$target" --seeds $SEEDS --out-file "$out"
      else
        log "baseline $kind $task/$target — up to date"
      fi
      [ -s "$out" ] || die "baseline_${kind}.py wrote no $out"
    done
  done
done

# --- 2c. pooled within-group AUROC -----------------------------------------
for pair in $WITHIN_GROUP; do
  task="${pair%%=*}"; key="${pair#*=}"
  case " $AVAILABLE " in *" $task "*) ;; *) continue ;; esac
  lab="$(labels_file "$task")"
  for target in $WITHIN_TARGETS; do
    out="$METRICS_DIR/within_group__${task}__${target}.json"
    if stale "$out" "$lab"; then
      log "within-group $task/$target (key $key)"
      run_step "within_group_${task}_${target}" "overall|within-group" \
        "$PY" scripts/within_group_auroc.py \
          --capture-dir "$(capture_dir "$task")" --labels "$lab" \
          --group-key "$key" --target "$target" \
          ${LAYER_ARGS[@]+"${LAYER_ARGS[@]}"} --seeds $SEEDS --out-file "$out"
    else
      log "within-group $task/$target — up to date"
    fi
  done
done

# --- 3. every ordered cross-task pair, no retraining -----------------------
for target in $TARGETS; do
  for src in $AVAILABLE; do
    src_agg="$(probe_dir "$src" "$target")/aggregate_metrics.json"
    ckpts=()
    for seed in $SEEDS; do
      c="$(probe_dir "$src" "$target")/seed_${seed}/checkpoint.json"
      if [ -s "$c" ]; then ckpts+=("$c"); fi
    done
    if [ ${#ckpts[@]} -eq 0 ]; then
      log "SKIP transfer from $src/$target — no checkpoints"
      continue
    fi
    for tgt in $AVAILABLE; do
      if [ "$src" = "$tgt" ]; then continue; fi
      tgt_lab="$(labels_file "$tgt")"
      out="${OUT_ROOT}/transfer_${src}_to_${tgt}_${target}.json"
      if stale "$out" "$src_agg" "$tgt_lab" "${ckpts[@]}"; then
        log "transfer $src -> $tgt ($target)"
        run_step "transfer_${src}_${tgt}_${target}" "source .* -> target" \
          "$PY" scripts/eval_transfer.py --probe "${ckpts[@]}" \
            --capture-dir "$(capture_dir "$tgt")" --labels "$tgt_lab" \
            --source-metrics "$src_agg" --out-file "$out"
      else
        log "transfer $src -> $tgt ($target) — up to date"
      fi
      [ -s "$out" ] || die "eval_transfer.py wrote no $out"
      publish "$out" "$METRICS_DIR/transfer__${src}_to_${tgt}__${target}.json"
    done
  done
done

# --- 4. tables -------------------------------------------------------------
log "rendering results table"
run_step "results_table" "" \
  "$PY" scripts/results_table.py --metrics-dir "$METRICS_DIR" \
    --out "$METRICS_DIR/results_table.txt"
run_step "results_table_csv" "" \
  "$PY" scripts/results_table.py --metrics-dir "$METRICS_DIR" \
    --format csv --out "$METRICS_DIR/results_table.csv"
log "rendering baseline comparison"
run_step "baseline_comparison" "" \
  "$PY" scripts/compare_baselines.py --metrics-dir "$METRICS_DIR" \
    --out "$METRICS_DIR/baseline_comparison.txt"
run_step "baseline_comparison_csv" "" \
  "$PY" scripts/compare_baselines.py --metrics-dir "$METRICS_DIR" \
    --format csv --out "$METRICS_DIR/baseline_comparison.csv"
echo
cat "$METRICS_DIR/results_table.txt"
echo
cat "$METRICS_DIR/baseline_comparison.txt"

# --- 5. promote (explicit only) --------------------------------------------
if [ -z "$PROMOTE" ]; then
  log "results in $METRICS_DIR (not promoted; PROMOTE=1 copies them to $PROMOTE_DIR)"
  exit 0
fi

log "promoting $METRICS_DIR -> $PROMOTE_DIR"
conflicts=()
while IFS= read -r -d '' f; do
  rel="${f#"$METRICS_DIR"/}"
  dst="$PROMOTE_DIR/$rel"
  if [ -e "$dst" ] && ! cmp -s "$f" "$dst"; then conflicts+=("$rel"); fi
done < <(find "$METRICS_DIR" -type f -print0)

if [ ${#conflicts[@]} -gt 0 ] && [ -z "$FORCE" ]; then
  log "REFUSING to promote: ${#conflicts[@]} file(s) in $PROMOTE_DIR would change:"
  printf '    %s\n' "${conflicts[@]}" >&2
  log "nothing was copied. Re-run with FORCE=1 PROMOTE=1 to overwrite, and add errata to the README."
  exit 3
fi

copied=0
while IFS= read -r -d '' f; do
  rel="${f#"$METRICS_DIR"/}"
  dst="$PROMOTE_DIR/$rel"
  if [ -e "$dst" ] && cmp -s "$f" "$dst"; then continue; fi
  mkdir -p "$(dirname "$dst")"
  cp "$f" "$dst"
  copied=$((copied + 1))
done < <(find "$METRICS_DIR" -type f -print0)
log "promoted $copied file(s) to $PROMOTE_DIR (${#conflicts[@]} of them overwrote existing files)"
log "write the README entry in $PROMOTE_DIR — the caveats live there"
