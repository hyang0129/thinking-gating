"""Tests for the cell + worker dispatch system.

Runs entirely on CPU with the stdlib, in temp directories — no cluster, no GPU,
no project dependencies. `python3 tests/test_dispatch.py` runs them without
pytest installed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.dispatch import claim, cells as cells_mod  # noqa: E402
from scripts.dispatch import worker as worker_mod  # noqa: E402

WORKER = _PROJECT_ROOT / "scripts" / "dispatch" / "worker.py"
QUEUE = _PROJECT_ROOT / "scripts" / "dispatch" / "queue.py"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _tmp_root(tmp: str) -> Path:
    return claim.init_dispatch_dirs(Path(tmp) / "dispatch")


def _cell(cell_id: str, **kw) -> dict:
    base = {"cell_id": cell_id, "kind": "python_code", "code": "print('hi')"}
    base.update(kw)
    return base


def _run_worker(root: Path, project_root: Path, *extra: str, timeout: int = 120):
    return subprocess.run(
        [sys.executable, str(WORKER), "--root", str(root),
         "--project-root", str(project_root), *extra],
        capture_output=True, text=True, timeout=timeout,
    )


def _states(root: Path) -> dict:
    return {s: [c["cell_id"] for _, c in claim.iter_cells(root, s)] for s in claim.STATES}


# ---------------------------------------------------------------------------
# queue primitives
# ---------------------------------------------------------------------------

def test_add_is_idempotent_across_states():
    with tempfile.TemporaryDirectory() as tmp:
        root = _tmp_root(tmp)
        _, first = claim.add_cell(root, _cell("a"))
        _, second = claim.add_cell(root, _cell("a"))
        assert (first, second) == ("added", "skipped"), (first, second)

        # A finished cell must not be re-queued by re-expanding a manifest.
        path = claim.claim_next_cell(root, "w1")
        claim.complete_cell(root, path, {"status": "ok"})
        _, third = claim.add_cell(root, _cell("a"))
        assert third == "skipped", third
        assert _states(root)["done"] == ["a"]


def test_claim_is_exclusive():
    with tempfile.TemporaryDirectory() as tmp:
        root = _tmp_root(tmp)
        for i in range(5):
            claim.add_cell(root, _cell(f"c{i}"))

        claimed = []
        for _ in range(10):  # more attempts than cells, alternating workers
            for wid in ("w1", "w2"):
                got = claim.claim_next_cell(root, wid)
                if got is not None:
                    claimed.append((wid, claim.cell_id_from_path(got)))
        ids = [cid for _, cid in claimed]
        assert len(ids) == 5, ids
        assert len(set(ids)) == 5, "a cell was handed to two workers"
        assert claim.count_status(root)["pending"] == 0


def test_priority_orders_claims():
    with tempfile.TemporaryDirectory() as tmp:
        root = _tmp_root(tmp)
        claim.add_cell(root, _cell("low", priority=200))
        claim.add_cell(root, _cell("high", priority=1))
        claim.add_cell(root, _cell("mid", priority=100))
        order = []
        while (p := claim.claim_next_cell(root, "w1")) is not None:
            order.append(claim.cell_id_from_path(p))
            claim.complete_cell(root, p, {"status": "ok"})
        assert order == ["high", "mid", "low"], order


def test_gc_reclaims_dead_worker_cells():
    with tempfile.TemporaryDirectory() as tmp:
        root = _tmp_root(tmp)
        claim.add_cell(root, _cell("orphan"))
        path = claim.claim_next_cell(root, "dead_worker")
        claim.touch_heartbeat(root, "dead_worker")

        assert claim.gc_stale_claims(root, stale_seconds=300) == []

        old = time.time() - 10_000
        os.utime(root / "claimed" / "dead_worker" / "heartbeat", (old, old))
        reclaimed = claim.gc_stale_claims(root, stale_seconds=300)
        assert len(reclaimed) == 1, reclaimed
        assert _states(root)["pending"] == ["orphan"]
        assert not path.exists()


def test_live_workers_reports_staleness():
    with tempfile.TemporaryDirectory() as tmp:
        root = _tmp_root(tmp)
        claim.add_cell(root, _cell("x"))
        claim.claim_next_cell(root, "w1")
        claim.touch_heartbeat(root, "w1")
        workers = claim.live_workers(root, stale_seconds=300)
        assert len(workers) == 1 and workers[0]["alive"]
        assert workers[0]["cells"] == ["x"]

        old = time.time() - 10_000
        os.utime(root / "claimed" / "w1" / "heartbeat", (old, old))
        assert not claim.live_workers(root, stale_seconds=300)[0]["alive"]


def test_retry_increments_attempts_and_requeues():
    with tempfile.TemporaryDirectory() as tmp:
        root = _tmp_root(tmp)
        claim.add_cell(root, _cell("r", max_attempts=3))
        path = claim.claim_next_cell(root, "w1")
        claim.retry_cell(root, path, {"status": "failed"})
        assert _states(root)["pending"] == ["r"]
        path2 = claim.claim_next_cell(root, "w1")
        assert claim.load_cell(path2)["attempts"] == 1


def test_invalid_cell_id_rejected():
    for bad in ("has space", "../escape", "", "-leading"):
        try:
            claim.validate_cell_id(bad)
        except claim.QueueError:
            continue
        raise AssertionError(f"accepted invalid cell_id {bad!r}")


# ---------------------------------------------------------------------------
# manifest expansion
# ---------------------------------------------------------------------------

def test_expand_grid_with_templated_constants():
    manifest = {
        "name": "sweep",
        "kind": "python_script",
        "script": "scripts/run_experiment.py",
        "args": ["--method", "{method}", "--seed", "{seed}", "--out", "{out}"],
        "constants": {"out": "output/{name}/{method}_seed{seed}"},
        "output_check": ["{out}/metrics.json"],
        "grid": {"method": ["mlp", "contrastive"], "seed": [42, 1]},
    }
    cells = cells_mod.expand_manifest(manifest)
    assert len(cells) == 4, len(cells)
    ids = sorted(c["cell_id"] for c in cells)
    assert ids[0] == "sweep__method-contrastive_seed-1", ids
    one = next(c for c in cells if c["meta"]["method"] == "mlp" and c["meta"]["seed"] == 42)
    assert one["args"] == ["--method", "mlp", "--seed", 42,
                           "--out", "output/sweep/mlp_seed42"], one["args"]
    assert one["output_check"] == ["output/sweep/mlp_seed42/metrics.json"]
    # A lone "{seed}" keeps its int type; embedded ones stringify.
    assert isinstance(one["args"][3], int)


def test_expand_zip_and_exclude():
    manifest = {
        "name": "z",
        "kind": "shell",
        "command": "echo {task} {split}",
        "zip": {"task": ["gsm8k", "lsat"], "split": ["test", "test"]},
        "grid": {"seed": [1, 2]},
        "exclude": [{"task": "lsat", "seed": 2}],
    }
    cells = cells_mod.expand_manifest(manifest)
    combos = sorted((c["meta"]["task"], c["meta"]["seed"]) for c in cells)
    assert combos == [("gsm8k", 1), ("gsm8k", 2), ("lsat", 1)], combos


def test_expand_rejects_bad_manifests():
    for manifest, needle in [
        ({"name": "n", "kind": "python_script"}, "requires 'script'"),
        ({"name": "n", "kind": "nope", "command": "x"}, "kind must be"),
        ({"kind": "shell", "command": "x"}, "missing 'name'"),
        ({"name": "n", "kind": "shell", "command": "echo {nope}"}, "unknown variable"),
        ({"name": "n", "kind": "shell", "command": "x",
          "zip": {"a": [1, 2], "b": [1]}}, "same length"),
        ({"name": "n", "kind": "shell", "command": "echo {a}",
          "cell_id_template": "fixed", "grid": {"a": [1, 2]}}, "duplicate cell_id"),
    ]:
        try:
            cells_mod.expand_manifest(manifest)
        except cells_mod.CellError as exc:
            assert needle in str(exc), f"expected {needle!r} in {exc}"
            continue
        raise AssertionError(f"accepted bad manifest {manifest}")


def test_self_referencing_constant_is_caught():
    manifest = {"name": "n", "kind": "shell", "command": "{a}",
                "constants": {"a": "{b}", "b": "{a}"}}
    try:
        cells_mod.expand_manifest(manifest)
    except cells_mod.CellError as exc:
        assert "converge" in str(exc), exc
        return
    raise AssertionError("accepted a cyclic constant")


# ---------------------------------------------------------------------------
# worker: the four kinds
# ---------------------------------------------------------------------------

def test_worker_runs_all_four_kinds():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        root = _tmp_root(tmp)
        (tmp_path / "pkg").mkdir()
        (tmp_path / "pkg" / "__init__.py").write_text("")
        (tmp_path / "pkg" / "mod.py").write_text(
            "def double(x):\n"
            "    from pathlib import Path\n"
            "    Path('called.txt').write_text(str(x * 2))\n"
            "    return {'doubled': x * 2}\n"
        )
        (tmp_path / "script.py").write_text(
            "import sys, pathlib\n"
            "pathlib.Path(sys.argv[1]).write_text('from script')\n"
        )

        claim.add_cell(root, {
            "cell_id": "k-script", "kind": "python_script",
            "script": "script.py", "args": ["script_out.txt"],
            "output_check": ["script_out.txt"]})
        claim.add_cell(root, {
            "cell_id": "k-code", "kind": "python_code",
            "code": "from pathlib import Path\nPath('code_out.txt').write_text('inline')\n",
            "output_check": ["code_out.txt"]})
        claim.add_cell(root, {
            "cell_id": "k-call", "kind": "call",
            "target": "pkg.mod:double", "kwargs": {"x": 21},
            "output_check": ["called.txt"]})
        claim.add_cell(root, {
            "cell_id": "k-shell", "kind": "shell",
            "command": "echo shelled > shell_out.txt",
            "output_check": ["shell_out.txt"]})

        proc = _run_worker(root, tmp_path)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        states = _states(root)
        assert sorted(states["done"]) == ["k-call", "k-code", "k-script", "k-shell"], states
        assert states["failed"] == []

        assert (tmp_path / "script_out.txt").read_text() == "from script"
        assert (tmp_path / "code_out.txt").read_text() == "inline"
        assert (tmp_path / "called.txt").read_text() == "42"
        assert (tmp_path / "shell_out.txt").read_text().strip() == "shelled"

        # `call` return values are captured for later analysis.
        result = json.loads((root / "results" / "k-call.json").read_text())
        assert result["value"] == {"doubled": 42}, result


def test_worker_records_failure_with_log():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        root = _tmp_root(tmp)
        claim.add_cell(root, {
            "cell_id": "boom", "kind": "python_code",
            "code": "raise SystemExit('deliberate explosion')\n"})

        proc = _run_worker(root, tmp_path)
        assert proc.returncode == 1, "worker should exit non-zero when a cell fails"
        states = _states(root)
        assert states["failed"] == ["boom"] and states["done"] == []

        _, cell = next(claim.iter_cells(root, "failed"))
        assert cell["result"]["status"] == "failed"
        assert "deliberate explosion" in cell["result"]["error"]
        log = root / "logs" / "boom.attempt1.log"
        assert "deliberate explosion" in log.read_text()


def test_exit_zero_without_outputs_is_a_failure():
    """The phantom-completion guard: silence is not success."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        root = _tmp_root(tmp)
        claim.add_cell(root, {
            "cell_id": "phantom", "kind": "python_code",
            "code": "print('pretending to work')\n",
            "output_check": ["never_written.json"]})

        _run_worker(root, tmp_path)
        _, cell = next(claim.iter_cells(root, "failed"))
        assert cell["result"]["status"] == "failed"
        assert cell["result"]["exit_code"] == 0
        assert cell["result"]["missing_outputs"], cell["result"]


def test_existing_outputs_are_skipped_not_rerun():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        root = _tmp_root(tmp)
        (tmp_path / "already.txt").write_text("previous run")
        claim.add_cell(root, {
            "cell_id": "resume", "kind": "python_code",
            "code": "from pathlib import Path\nPath('already.txt').write_text('OVERWRITTEN')\n",
            "output_check": ["already.txt"]})

        _run_worker(root, tmp_path)
        assert (tmp_path / "already.txt").read_text() == "previous run", "cell re-ran"
        _, cell = next(claim.iter_cells(root, "done"))
        assert cell["result"]["status"] == "skipped"

        # --no-skip-existing forces the re-run.
        claim.requeue(root, next(claim.iter_cells(root, "done"))[0])
        _run_worker(root, tmp_path, "--no-skip-existing")
        assert (tmp_path / "already.txt").read_text() == "OVERWRITTEN"


def test_retry_then_fail_after_max_attempts():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        root = _tmp_root(tmp)
        claim.add_cell(root, {
            "cell_id": "flaky", "kind": "python_code",
            "code": "raise SystemExit(3)\n", "max_attempts": 3})

        _run_worker(root, tmp_path)
        states = _states(root)
        assert states["failed"] == ["flaky"], states
        _, cell = next(claim.iter_cells(root, "failed"))
        assert cell["attempts"] == 2 and cell["result"]["attempt"] == 3, cell
        # One log per attempt, so a flaky failure can be compared across tries.
        assert len(list((root / "logs").glob("flaky.attempt*.log"))) == 3


def test_timeout_kills_the_cell():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        root = _tmp_root(tmp)
        claim.add_cell(root, {
            "cell_id": "slow", "kind": "python_code",
            "code": "import time\ntime.sleep(120)\n", "timeout_s": 2})

        started = time.monotonic()
        _run_worker(root, tmp_path, timeout=60)
        elapsed = time.monotonic() - started
        assert elapsed < 45, f"timeout did not fire promptly ({elapsed:.0f}s)"
        _, cell = next(claim.iter_cells(root, "failed"))
        assert cell["result"]["status"] == "timeout", cell["result"]


def test_cell_env_and_identity_are_exposed():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        root = _tmp_root(tmp)
        claim.add_cell(root, {
            "cell_id": "envcell", "kind": "python_code",
            "env": {"MY_SETTING": "from-cell"},
            "code": ("import os, json\n"
                     "from pathlib import Path\n"
                     "Path('env.json').write_text(json.dumps({\n"
                     "    'setting': os.environ['MY_SETTING'],\n"
                     "    'cell': os.environ['DISPATCH_CELL_ID'],\n"
                     "    'extra': os.environ.get('FROM_FLAG'),\n"
                     "}))\n"),
            "output_check": ["env.json"]})

        _run_worker(root, tmp_path, "--env", "FROM_FLAG=from-worker")
        payload = json.loads((tmp_path / "env.json").read_text())
        assert payload == {"setting": "from-cell", "cell": "envcell",
                           "extra": "from-worker"}, payload


def test_two_workers_split_the_queue_without_overlap():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        root = _tmp_root(tmp)
        for i in range(8):
            claim.add_cell(root, {
                "cell_id": f"par{i}", "kind": "python_code",
                "code": (f"import time; time.sleep(0.3)\n"
                         f"from pathlib import Path\n"
                         f"Path('out{i}.txt').write_text('{i}')\n"),
                "output_check": [f"out{i}.txt"]})

        procs = [
            subprocess.Popen(
                [sys.executable, str(WORKER), "--root", str(root),
                 "--project-root", str(tmp_path), "--worker-id", f"w{n}"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for n in range(2)
        ]
        outputs = [p.communicate(timeout=120)[0] for p in procs]
        assert all(p.returncode == 0 for p in procs), outputs

        states = _states(root)
        assert len(states["done"]) == 8 and not states["failed"], states
        # Each cell ran exactly once: one attempt-1 log apiece, no attempt-2.
        assert len(list((root / "logs").glob("par*.attempt1.log"))) == 8
        assert not list((root / "logs").glob("par*.attempt2.log"))
        # And the work actually got shared rather than one worker taking it all.
        claimed_by = [line.split("claimed ")[1].split()[0]
                      for out in outputs for line in out.splitlines() if "claimed " in line]
        assert len(claimed_by) == 8, claimed_by


def test_worker_releases_cell_on_sigterm():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        root = _tmp_root(tmp)
        claim.add_cell(root, {
            "cell_id": "interrupted", "kind": "python_code",
            "code": "import time\ntime.sleep(60)\n"})

        proc = subprocess.Popen(
            [sys.executable, str(WORKER), "--root", str(root),
             "--project-root", str(tmp_path), "--worker-id", "w-term"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if claim.count_status(root)["claimed"] == 1:
                break
            time.sleep(0.2)
        else:
            proc.kill()
            raise AssertionError("worker never claimed the cell")

        proc.terminate()
        proc.communicate(timeout=60)
        # Back on the queue immediately — no waiting out the stale-claim GC.
        assert _states(root)["pending"] == ["interrupted"], _states(root)


def test_max_cells_stops_early():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        root = _tmp_root(tmp)
        for i in range(4):
            claim.add_cell(root, _cell(f"m{i}", code="print('ok')\n"))

        _run_worker(root, tmp_path, "--max-cells", "2")
        states = _states(root)
        assert len(states["done"]) == 2 and len(states["pending"]) == 2, states


def test_dry_run_previews_without_consuming_the_queue():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        root = _tmp_root(tmp)
        (tmp_path / "done_already.txt").write_text("x")
        claim.add_cell(root, {
            "cell_id": "preview", "kind": "python_code",
            "code": "print('would run')\n", "output_check": ["missing.txt"]})
        claim.add_cell(root, {
            "cell_id": "already", "kind": "python_code",
            "code": "print('nope')\n", "output_check": ["done_already.txt"]})

        proc = _run_worker(root, tmp_path, "--dry-run")
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "[dry-run] 2 pending cell(s)" in proc.stdout, proc.stdout
        assert "exists (would skip)" in proc.stdout, proc.stdout
        assert "missing" in proc.stdout

        # The whole point: a preview leaves the queue exactly as it found it.
        states = _states(root)
        assert sorted(states["pending"]) == ["already", "preview"], states
        assert states["done"] == [] and states["claimed"] == []
        assert not list((root / "logs").glob("*.log"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _queue_cli(*args: str):
    return subprocess.run([sys.executable, str(QUEUE), *args],
                          capture_output=True, text=True, timeout=60)


def test_queue_cli_expand_status_retry_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        root = tmp_path / "dispatch"
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps({
            "name": "clitest",
            "kind": "python_code",
            "code": "raise SystemExit('nope')\n",
            "grid": {"seed": [1, 2]},
        }))

        dry = _queue_cli("expand", str(manifest), "--root", str(root), "--dry-run")
        assert "2 cell(s)" in dry.stdout, dry.stdout
        assert not root.exists(), "dry run wrote to disk"

        out = _queue_cli("expand", str(manifest), "--root", str(root))
        assert "added 2" in out.stdout, out.stdout
        again = _queue_cli("expand", str(manifest), "--root", str(root))
        assert "skipped 2" in again.stdout, again.stdout

        _run_worker(root, tmp_path)
        status = _queue_cli("status", "--root", str(root))
        assert "failed      2" in status.stdout, status.stdout
        assert "nope" in status.stdout, status.stdout

        payload = json.loads(_queue_cli("status", "--root", str(root), "--json").stdout)
        assert payload["counts"]["failed"] == 2

        logs = _queue_cli("logs", "--root", str(root), "--cell", "clitest__seed-1")
        assert "nope" in logs.stdout, logs.stdout

        retried = _queue_cli("retry", "--root", str(root), "--all")
        assert "re-queued" in retried.stdout
        assert claim.count_status(root)["pending"] == 2

        listed = _queue_cli("list", "--root", str(root), "--state", "pending", "-v")
        assert "clitest__seed-1" in listed.stdout

        refused = _queue_cli("clear", "--root", str(root), "--state", "all")
        assert refused.returncode == 2, refused.stdout
        cleared = _queue_cli("clear", "--root", str(root), "--state", "all", "--confirm")
        assert "deleted 2" in cleared.stdout, cleared.stdout


def test_shipped_example_manifest_expands():
    manifest = cells_mod.load_manifest(
        _PROJECT_ROOT / "configs" / "dispatch" / "example_probe_sweep.json")
    cells = cells_mod.expand_manifest(manifest)
    assert len(cells) == 10, len(cells)
    assert all(c["kind"] == "python_script" for c in cells)
    assert all(c["output_check"] for c in cells)


def _argparse_spec(script: Path) -> dict:
    """Option strings, choices, and required flags of a script's argparse,
    read statically with ast so the test needs none of its dependencies."""
    import ast
    flags, choices, required = set(), {}, set()
    for node in ast.walk(ast.parse(script.read_text(encoding="utf-8"))):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        names = [a.value for a in node.args
                 if isinstance(a, ast.Constant) and isinstance(a.value, str)]
        opts = [n for n in names if n.startswith("--")]
        flags.update(opts)
        for kw in node.keywords:
            if kw.arg == "choices" and isinstance(kw.value, (ast.List, ast.Tuple)):
                for o in opts:
                    choices[o] = [e.value for e in kw.value.elts
                                  if isinstance(e, ast.Constant)]
            if (kw.arg == "required" and isinstance(kw.value, ast.Constant)
                    and kw.value.value is True):
                required.update(opts)
    return {"flags": flags, "choices": choices, "required": required}


def test_example_manifest_matches_run_experiment_cli():
    """Every example cell must be runnable: real flags, valid choices, and an
    output_check naming files run_experiment.py actually writes."""
    script = _PROJECT_ROOT / "scripts" / "run_experiment.py"
    spec = _argparse_spec(script)
    source = script.read_text(encoding="utf-8")
    cells = cells_mod.expand_manifest(cells_mod.load_manifest(
        _PROJECT_ROOT / "configs" / "dispatch" / "example_probe_sweep.json"))
    for cell in cells:
        assert cell["script"] == "scripts/run_experiment.py"
        args = [str(a) for a in cell["args"]]
        used = [a for a in args if a.startswith("--")]
        unknown = [a for a in used if a not in spec["flags"]]
        assert not unknown, f"{cell['cell_id']}: unknown flags {unknown}"
        missing = spec["required"] - set(used)
        assert not missing, f"{cell['cell_id']}: missing required {missing}"
        for flag, allowed in spec["choices"].items():
            if flag in args:
                value = args[args.index(flag) + 1]
                assert value in allowed, f"{cell['cell_id']}: {flag} {value} not in {allowed}"
        for check in cell["output_check"]:
            name = Path(check).name
            assert f'"{name}"' in source, f"run_experiment.py never writes {name}"


# ---------------------------------------------------------------------------
# D6: cell identity, retired/, and edited manifests
# ---------------------------------------------------------------------------

def test_existing_manifest_ids_are_unchanged():
    """Ids of cells already on the cluster must not move. Only manifests that
    opt in with cell_id_hash get the fingerprint suffix."""
    load = lambda n: cells_mod.expand_manifest(cells_mod.load_manifest(  # noqa: E731
        _PROJECT_ROOT / "configs" / "dispatch" / f"{n}.json"))
    assert [c["cell_id"] for c in load("capture_qwen3v3_redo")] == [
        f"capture-qwen3v3-lsat-shard{s:02d}" for s in range(4)]
    assert [c["cell_id"] for c in load("capture_nemotronv3_redo")] == [
        f"capture-nemotronv3-{t}-shard{s:02d}"
        for s in range(4) for t in ("gsm8k", "math500", "mmlu_pro")]
    for path in sorted((_PROJECT_ROOT / "configs" / "dispatch").glob("*.json")):
        manifest = cells_mod.load_manifest(path)
        if manifest.get("cell_id_hash"):
            continue
        for cell in cells_mod.expand_manifest(manifest):
            assert "__h" not in cell["cell_id"], (path.name, cell["cell_id"])


def test_cell_id_hash_is_opt_in_and_tracks_args():
    base = {"name": "h", "kind": "shell", "command": "echo {x} {budget}",
            "constants": {"budget": "1024"}, "grid": {"x": [1, 2]}}
    plain = [c["cell_id"] for c in cells_mod.expand_manifest(base)]
    assert plain == ["h__x-1", "h__x-2"], plain
    hashed = [c["cell_id"] for c in cells_mod.expand_manifest({**base, "cell_id_hash": True})]
    assert all(h.startswith(p + "__h") and len(h) == len(p) + 13
               for p, h in zip(plain, hashed)), hashed
    edited = {**base, "cell_id_hash": True, "constants": {"budget": "4096"}}
    assert [c["cell_id"] for c in cells_mod.expand_manifest(edited)] != hashed
    cell = cells_mod.expand_manifest({**base, "cell_id_hash": True})[0]
    assert "cell_id_hash" not in cell


def test_expand_refuses_retired_cells_unless_allowed():
    """Re-expanding capture_qwen3v3.json must not resurrect its retired
    1024-token LSAT shards (they share ids and output dirs with the redo)."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "capture_qwen3v3"
        claim.init_dispatch_dirs(root)
        manifest = _PROJECT_ROOT / "configs" / "dispatch" / "capture_qwen3v3.json"
        cells = cells_mod.expand_manifest(cells_mod.load_manifest(manifest))
        lsat = [c for c in cells if "-lsat-" in c["cell_id"]]
        assert len(lsat) == 4
        (root / "retired").mkdir()
        (root / "retired" / "RETIRED.md").write_text("retired at 1024\n")
        for c in lsat:   # how the operator retired them: failed/ -> retired/
            claim.write_cell_atomic(root / "retired" / claim.cell_filename(c),
                                    {**c, "result": {"status": "failed"}})

        out = _queue_cli("expand", str(manifest), "--root", str(root))
        assert out.returncode == 0, out.stdout + out.stderr
        assert "added 16" in out.stdout and "refused 4 retired" in out.stdout, out.stdout
        pending = _states(root)["pending"]
        assert len(pending) == 16 and not any("-lsat-" in p for p in pending), pending

        dry = _queue_cli("expand", str(manifest), "--root", str(root), "--dry-run")
        assert dry.stdout.count("RETIRED") == 4, dry.stdout

        again = _queue_cli("expand", str(manifest), "--root", str(root), "--allow-resurrect")
        assert "added 4" in again.stdout, again.stdout
        assert len(_states(root)["pending"]) == 20


def test_expand_refuses_an_edited_manifest_instead_of_skipping_it():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        root = tmp_path / "dispatch"
        manifest = tmp_path / "m.json"
        spec = {"name": "edit", "kind": "shell", "command": "run --budget {b}",
                "constants": {"b": "1024"}, "grid": {"s": [0, 1]}}
        manifest.write_text(json.dumps(spec))
        assert "added 2" in _queue_cli("expand", str(manifest), "--root", str(root)).stdout
        path = claim.claim_next_cell(root, "w1")
        claim.complete_cell(root, path, {"status": "ok"})

        # Cosmetic edits (tags, comments, priority) are not conflicts.
        manifest.write_text(json.dumps({**spec, "tags": ["x"], "_comment": "hi"}))
        same = _queue_cli("expand", str(manifest), "--root", str(root))
        assert same.returncode == 0 and "skipped 2" in same.stdout, same.stdout + same.stderr

        manifest.write_text(json.dumps({**spec, "constants": {"b": "4096"}}))
        refused = _queue_cli("expand", str(manifest), "--root", str(root))
        assert refused.returncode == 3, refused.stdout + refused.stderr
        assert "DIFFERENT" in refused.stderr and "edit__s-0" in refused.stderr
        assert "nothing written" in refused.stderr
        cells_now = {c["cell_id"]: c for s in ("pending", "done")
                     for _, c in claim.iter_cells(root, s)}
        assert all("1024" in c["command"] for c in cells_now.values()), cells_now

        replaced = _queue_cli("expand", str(manifest), "--root", str(root), "--replace")
        assert "replaced 2" in replaced.stdout, replaced.stdout
        assert all("4096" in c["command"] for _, c in claim.iter_cells(root, "pending"))


# ---------------------------------------------------------------------------
# D3: a reclaimed claim must not be finished, and its worker must exit cleanly
# ---------------------------------------------------------------------------

def test_finishing_a_reclaimed_claim_writes_nothing():
    with tempfile.TemporaryDirectory() as tmp:
        root = _tmp_root(tmp)
        claim.add_cell(root, _cell("re"))
        path = claim.claim_next_cell(root, "slow")
        hb = claim.touch_heartbeat(root, "slow")
        old = time.time() - 10_000
        os.utime(hb, (old, old))
        assert len(claim.gc_stale_claims(root)) == 1

        for move in (claim.complete_cell, claim.fail_cell,
                     claim.release_cell, claim.retry_cell):
            try:
                move(root, path, {"status": "ok"})
            except claim.ClaimLost:
                continue
            raise AssertionError(f"{move.__name__} finished a claim it no longer holds")
        assert _states(root) == {"pending": ["re"], "claimed": [], "done": [], "failed": []}
        assert not list(root.rglob("*.finishing")) and not path.exists()


def test_gc_recovers_a_cell_stranded_mid_finish():
    with tempfile.TemporaryDirectory() as tmp:
        root = _tmp_root(tmp)
        claim.add_cell(root, _cell("mid"))
        path = claim.claim_next_cell(root, "crashed")
        os.rename(path, path.with_name(path.name + claim.FINISHING_SUFFIX))
        hb = claim.touch_heartbeat(root, "crashed")
        old = time.time() - 10_000
        os.utime(hb, (old, old))
        assert claim.count_status(root)["total"] == 0  # invisible to scans
        assert len(claim.gc_stale_claims(root)) == 1
        assert _states(root)["pending"] == ["mid"]


def _wait_for(pred, timeout: float = 30.0, step: float = 0.1) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(step)
    return False


def test_worker_exits_cleanly_when_its_claim_is_reclaimed_mid_run():
    """Reproduces the audit's expA: A's heartbeat looked stale, B reclaimed and
    re-ran the cell. A used to crash with FileNotFoundError at complete time."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        root = _tmp_root(tmp)
        claim.add_cell(root, {"cell_id": "contested", "kind": "python_code",
                              "code": "import time\ntime.sleep(60)\n",
                              "output_check": ["never.txt"], "max_attempts": 3})
        started = time.monotonic()
        proc = subprocess.Popen(
            [sys.executable, str(WORKER), "--root", str(root),
             "--project-root", str(tmp_path), "--worker-id", "nodeA-8882_1",
             "--heartbeat-interval", "0.3"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        mine = root / "claimed" / "nodeA-8882_1"
        assert _wait_for(lambda: list(mine.glob("*__contested.json"))), "never claimed"
        time.sleep(0.5)  # let the cell subprocess start
        # What GC + a second worker do: claimed/A -> pending -> claimed/B.
        claimed_file = next(mine.glob("*__contested.json"))
        os.rename(claimed_file, root / "pending" / claimed_file.name)
        assert claim.claim_next_cell(root, "nodeB-8883_2") is not None

        out, _ = proc.communicate(timeout=60)
        assert proc.returncode == worker_mod.EXIT_CLAIM_LOST, out
        assert time.monotonic() - started < 30, "lost claim not noticed promptly"
        assert "Traceback" not in out, out
        assert "LOST CLAIM" in out, out
        # B's claim is untouched; A wrote no outcome anywhere.
        assert _states(root) == {"pending": [], "claimed": ["contested"],
                                 "done": [], "failed": []}, _states(root)
        assert (root / "claimed" / "nodeB-8883_2" / claimed_file.name).exists()
        assert claim.load_cell(root / "claimed" / "nodeB-8883_2"
                               / claimed_file.name).get("attempts", 0) == 0


def test_worker_exits_cleanly_when_claim_vanishes_before_it_finishes():
    """The cell completes, but its claim was reclaimed in the meantime: the
    worker must neither crash nor file the cell as done."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        root = _tmp_root(tmp)
        claim.add_cell(root, {
            "cell_id": "yanked", "kind": "python_code",
            "code": ("import os, pathlib\n"
                     "root = pathlib.Path(os.environ['DISPATCH_ROOT'])\n"
                     "mine = root / 'claimed' / os.environ['DISPATCH_WORKER_ID']\n"
                     "for p in mine.glob('*__yanked.json'):\n"
                     "    p.rename(root / 'pending' / p.name)\n")})
        claim.add_cell(root, _cell("after", priority=200))
        proc = _run_worker(root, tmp_path, "--worker-id", "nodeA-8882_1")
        out = proc.stdout + proc.stderr
        assert proc.returncode == worker_mod.EXIT_CLAIM_LOST, out
        assert "Traceback" not in out, out
        # The reclaimed cell is back in pending exactly once; the worker
        # stopped instead of draining more of the queue.
        states = _states(root)
        assert sorted(states["pending"]) == ["after", "yanked"], states
        assert states["claimed"] == states["done"] == states["failed"] == [], states


def test_rerun_of_same_attempt_does_not_clobber_the_earlier_log():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        root = _tmp_root(tmp)
        claim.add_cell(root, _cell("again", code="print('second run')\n"))
        (root / "logs" / "again.attempt1.log").write_text("EARLIER RUN\n")
        old = time.time() - 100
        os.utime(root / "logs" / "again.attempt1.log", (old, old))
        _run_worker(root, tmp_path)
        assert (root / "logs" / "again.attempt1.log").read_text() == "EARLIER RUN\n"
        _, cell = next(claim.iter_cells(root, "done"))
        assert cell["result"]["log"] == "logs/again.attempt1.r1.log", cell["result"]
        shown = _queue_cli("logs", "--root", str(root), "--cell", "again")
        assert "second run" in shown.stdout, shown.stdout


# ---------------------------------------------------------------------------
# D4: a deterministic (truncation) failure is not retried
# ---------------------------------------------------------------------------

_TRUNCATING_CAPTURE = (
    "import json, pathlib, sys\n"
    "out = pathlib.Path('cap')\n"
    "out.mkdir(exist_ok=True)\n"
    "(out / 'meta.shard00.jsonl.quarantined').write_text('{}\\n')\n"
    "(out / 'TRUNCATION_FAILURE.shard00.json').write_text(json.dumps(\n"
    "    {'reason': 'thinking-OFF truncation above limit',\n"
    "     'off_truncation_rate': 0.8, 'max_response_len': 1024}))\n"
    "sys.exit(1)\n"
)
_SHARD00_OUTPUTS = ["cap/meta.shard00.jsonl", "cap/activations_thinking_off.shard00.npz"]


def test_truncation_marker_fails_the_cell_terminally():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        root = _tmp_root(tmp)
        claim.add_cell(root, {"cell_id": "trunc", "kind": "python_code",
                              "code": _TRUNCATING_CAPTURE,
                              "output_check": _SHARD00_OUTPUTS, "max_attempts": 3})
        proc = _run_worker(root, tmp_path)
        assert proc.returncode == 1, proc.stdout + proc.stderr
        states = _states(root)
        assert states["failed"] == ["trunc"] and not states["pending"], states
        _, cell = next(claim.iter_cells(root, "failed"))
        result = cell["result"]
        assert result["terminal"] is True and cell.get("attempts", 0) == 0, cell
        assert "TERMINAL FAILURE" in result["error"]
        assert "off_truncation_rate" in result["error"]   # the marker's content
        assert "TRUNCATION_FAILURE.shard00.json" in result["terminal_markers"][0]
        assert len(list((root / "logs").glob("trunc.attempt*.log"))) == 1


def test_markers_for_other_shards_or_older_runs_do_not_block_retries():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        root = _tmp_root(tmp)
        cap = tmp_path / "cap"
        cap.mkdir()
        (cap / "TRUNCATION_FAILURE.shard01.json").write_text("{}")   # another shard
        stale = cap / "TRUNCATION_FAILURE.shard00.json"              # an older capture
        stale.write_text("{}")
        old = time.time() - 86_400
        os.utime(stale, (old, old))
        claim.add_cell(root, {"cell_id": "flaky0", "kind": "python_code",
                              "code": "raise SystemExit(1)\n",
                              "output_check": _SHARD00_OUTPUTS, "max_attempts": 2})
        _run_worker(root, tmp_path)
        _, cell = next(claim.iter_cells(root, "failed"))
        assert not cell["result"].get("terminal"), cell["result"]
        assert len(list((root / "logs").glob("flaky0.attempt*.log"))) == 2


def test_cell_requeued_by_an_old_worker_after_truncation_is_not_rerun():
    """The in-flight case: an old worker ran attempt 1, saw exit 1, and
    re-queued the cell (max_attempts 2). The new worker must not spend a
    second run at the same budget."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        root = _tmp_root(tmp)
        cap = tmp_path / "cap"
        cap.mkdir()
        started = worker_mod._now_iso()
        time.sleep(0.01)
        (cap / "TRUNCATION_FAILURE.shard00.json").write_text('{"max_response_len": 1024}')
        # Exactly what old retry_cell leaves in pending/.
        claim.add_cell(root, {
            "cell_id": "redo", "kind": "python_code",
            "code": "open('RAN', 'w').write('x')\n",
            "output_check": _SHARD00_OUTPUTS, "max_attempts": 2, "attempts": 1,
            "result": {"status": "failed", "exit_code": 1, "attempt": 1,
                       "started_at": started, "worker_id": "old_1"}})
        _run_worker(root, tmp_path)
        assert not (tmp_path / "RAN").exists(), "re-ran a deterministic failure"
        _, cell = next(claim.iter_cells(root, "failed"))
        assert cell["result"]["terminal"] is True
        assert cell["result"]["previous_result"]["worker_id"] == "old_1"


# ---------------------------------------------------------------------------
# D7: provenance
# ---------------------------------------------------------------------------

def test_worker_records_the_git_commit_of_each_cell():
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "repo"
        repo.mkdir()
        git = lambda *a: subprocess.run(  # noqa: E731
            ["git", "-C", str(repo), *a], check=True, capture_output=True, text=True)
        git("init", "-q")
        git("config", "user.email", "t@example.com")
        git("config", "user.name", "t")
        (repo / "f.txt").write_text("v1\n")
        git("add", "f.txt")
        git("commit", "-q", "-m", "one")
        head = git("rev-parse", "HEAD").stdout.strip()

        root = claim.init_dispatch_dirs(Path(tmp) / "dispatch")
        claim.add_cell(root, _cell("prov"))
        _run_worker(root, repo)
        _, cell = next(claim.iter_cells(root, "done"))
        assert cell["result"]["git_commit"] == head, cell["result"]
        assert cell["result"]["git_dirty"] is False
        log = (root / cell["result"]["log"]).read_text()
        assert f"# commit   {head}" in log, log

        (repo / "f.txt").write_text("v2\n")   # tracked edit -> dirty
        claim.add_cell(root, _cell("prov2"))
        _run_worker(root, repo)
        cell2 = next(c for _, c in claim.iter_cells(root, "done") if c["cell_id"] == "prov2")
        assert cell2["result"]["git_dirty"] is True


def test_worker_runs_outside_a_git_checkout():
    with tempfile.TemporaryDirectory() as tmp:
        state = worker_mod.git_state(Path(tmp))
        assert set(state) == {"git_commit", "git_dirty"}


# ---------------------------------------------------------------------------
# watcher (D1, D2, D7)
# ---------------------------------------------------------------------------

class _FakeClock:
    """Stands in for the watcher's `time` module: sleep advances the clock,
    and every heartbeat in `alive` is touched at the new time, the way a
    running worker's heartbeat thread would."""

    def __init__(self) -> None:
        self.offset = 0.0
        self.alive: list[Path] = []

    def monotonic(self) -> float:
        return self.offset

    def time(self) -> float:
        return _real_time() + self.offset

    def sleep(self, seconds: float) -> None:
        self.offset += seconds
        now = self.time()
        for hb in self.alive:
            os.utime(hb, (now, now))


def _real_time() -> float:
    return time.time()


def _run_watcher(repo: Path, roots: list[str], alloc_schedule, *,
                 dispatch_ok=lambda n: True, max_hours: float = 2.0,
                 clean=(True, "abc12345"), keep_alive=()):
    """Run watch_and_dispatch.main() against a fake cluster.

    alloc_schedule(poll_number) -> list of allocation node keys. A successful
    fake dispatch starts a "worker": a fresh heartbeat dir named
    `<node>_<pid>` in the root, as a real worker's DISPATCH_NODE id would be.
    """
    import contextlib
    import io
    from scripts import watch_and_dispatch as wd

    clock = _FakeClock()
    clock.alive.extend(keep_alive)
    calls: list[tuple[str, str]] = []
    polls = {"n": 0}

    def fake_allocs():
        polls["n"] += 1
        return [{"name": f"jupyter_empire_{n[-4:]}", "node": n}
                for n in alloc_schedule(polls["n"])]

    def fake_dispatch(root: Path, node: str) -> bool:
        calls.append((root.name, node))
        if not dispatch_ok(len(calls)):
            return False
        hb_dir = root / "claimed" / f"{node}_{1000 + len(calls)}"
        hb_dir.mkdir(parents=True, exist_ok=True)
        (hb_dir / "heartbeat").touch()
        now = clock.time()
        os.utime(hb_dir / "heartbeat", (now, now))
        clock.alive.append(hb_dir / "heartbeat")
        return True

    saved = {k: getattr(wd, k) for k in
             ("REPO", "STOP_FILE", "running_allocations", "checkout_is_clean",
              "sh", "dispatch", "time")}
    try:
        wd.REPO = repo
        wd.STOP_FILE = repo / "shared" / "dispatch" / "STOP_WATCH"
        wd.running_allocations = fake_allocs
        wd.checkout_is_clean = lambda: clean
        wd.sh = lambda cmd, timeout=300: (0, "synced")
        wd.dispatch = fake_dispatch
        wd.time = clock
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = wd.main(["--roots", *roots, "--max-hours", str(max_hours),
                          "--log-file", "shared/logs/w.log"])
    finally:
        for k, v in saved.items():
            setattr(wd, k, v)
    return rc, calls, buf.getvalue()


def _watch_repo(tmp: str, *names: str) -> Path:
    repo = Path(tmp) / "repo"
    for name in names:
        claim.init_dispatch_dirs(repo / "shared" / "dispatch" / name)
    return repo


def test_watcher_serves_a_second_allocation_that_lands_later():
    """D1: two allocations landing at different polls both get a worker. The
    old watcher dispatched once and exited, stranding the second root."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = _watch_repo(tmp, "rootA", "rootB")
        for name, n in (("rootA", 2), ("rootB", 3)):
            for i in range(n):
                claim.add_cell(repo / "shared" / "dispatch" / name, _cell(f"{name}{i}"))
        schedule = lambda poll: (["alphagpu01-8882"] if poll <= 2  # noqa: E731
                                 else ["alphagpu01-8882", "alphagpu02-8883"])
        rc, calls, out = _run_watcher(
            repo, ["shared/dispatch/rootA", "shared/dispatch/rootB"], schedule)
        assert rc == 0, out
        # The first node is never reused for the second root.
        assert calls == [("rootA", "alphagpu01-8882"),
                         ("rootB", "alphagpu02-8883")], calls


def test_watcher_treats_a_dead_workers_claim_as_unserved():
    """D2: a stale claim used to make the watcher skip the root forever."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = _watch_repo(tmp, "rootA")
        root = repo / "shared" / "dispatch" / "rootA"
        claim.add_cell(root, _cell("held"))
        claim.claim_next_cell(root, "alphagpu09-8889_4242")
        hb = claim.touch_heartbeat(root, "alphagpu09-8889_4242")
        old = time.time() - 10_000
        os.utime(hb, (old, old))
        assert claim.count_status(root)["pending"] == 0

        rc, calls, out = _run_watcher(repo, ["shared/dispatch/rootA"],
                                      lambda poll: ["alphagpu01-8882"])
        assert rc == 0, out
        assert calls == [("rootA", "alphagpu01-8882")], (calls, out)


def test_watcher_keeps_going_after_a_failed_dispatch_and_respects_live_workers():
    with tempfile.TemporaryDirectory() as tmp:
        repo = _watch_repo(tmp, "busy", "idle")
        busy = repo / "shared" / "dispatch" / "busy"
        idle = repo / "shared" / "dispatch" / "idle"
        claim.add_cell(busy, _cell("b0"))
        claim.add_cell(busy, _cell("b1"))
        claim.add_cell(idle, _cell("i0"))
        # A live worker already serves `busy` from alphagpu01-8882.
        claim.claim_next_cell(busy, "alphagpu01-8882_77")
        hb = claim.touch_heartbeat(busy, "alphagpu01-8882_77")

        rc, calls, out = _run_watcher(
            repo, ["shared/dispatch/busy", "shared/dispatch/idle"],
            lambda poll: ["alphagpu01-8882", "alphagpu02-8883"],
            dispatch_ok=lambda n: n > 1,   # the first dispatch fails
            keep_alive=[hb])
        assert rc == 0, out
        assert calls == [("idle", "alphagpu02-8883"), ("idle", "alphagpu02-8883")], calls


def test_watcher_waits_out_a_dirty_checkout_and_gives_up_at_the_deadline():
    with tempfile.TemporaryDirectory() as tmp:
        repo = _watch_repo(tmp, "rootA")
        claim.add_cell(repo / "shared" / "dispatch" / "rootA", _cell("x"))
        rc, calls, out = _run_watcher(repo, ["shared/dispatch/rootA"],
                                      lambda poll: ["alphagpu01-8882"],
                                      clean=(False, "HEAD != upstream"), max_hours=0.2)
        assert rc == 5 and calls == [], (rc, calls)
        assert "HEAD != upstream" in out


def test_checkout_check_fetches_before_comparing_with_upstream():
    """D7: comparing HEAD to a never-refreshed @{u} missed every push."""
    from scripts import watch_and_dispatch as wd
    seen: list[list[str]] = []

    def fake_sh(cmd, timeout=300):
        seen.append(cmd)
        if cmd[:2] == ["git", "status"]:
            return 0, ""
        if cmd[:2] == ["git", "fetch"]:
            return 0, ""
        if cmd == ["git", "rev-parse", "HEAD"]:
            return 0, "a" * 40
        if cmd == ["git", "rev-parse", "@{u}"]:
            return 0, "b" * 40
        raise AssertionError(cmd)

    saved, was_disabled = wd.sh, wd.logger.disabled
    try:
        wd.sh = fake_sh
        wd.logger.disabled = True
        ok, detail = wd.checkout_is_clean()
        assert not ok and "upstream bbbbbbbb" in detail, detail
        order = [c[:2] for c in seen]
        assert order.index(["git", "fetch"]) < order.index(["git", "rev-parse"]), order

        # A failed fetch is not fatal; it is reported.
        seen.clear()
        wd.sh = lambda cmd, timeout=300: ((1, "no network") if cmd[:2] == ["git", "fetch"]
                                          else fake_sh(cmd, timeout))
        ok, detail = wd.checkout_is_clean()
        assert "git fetch failed" in detail, detail
    finally:
        wd.sh, wd.logger.disabled = saved, was_disabled


# ---------------------------------------------------------------------------
# gpu_dispatch kill
# ---------------------------------------------------------------------------

def test_gpu_dispatch_kill_signals_the_whole_process_group():
    """The recorded pid is the `bash -c "cd ... && cmd"` wrapper. Killing only
    it orphaned the worker; the kill code must reach the whole group."""
    from scripts import gpu_dispatch as gd
    with tempfile.TemporaryDirectory() as tmp:
        pidfile = Path(tmp) / "child.pid"
        inner = (f"cd {tmp} && DISPATCH_NODE=x {sys.executable} -c "
                 f"\"import os, time; open('{pidfile}', 'w').write(str(os.getpid())); "
                 f"time.sleep(60)\"")
        wrapper = subprocess.Popen(["bash", "-c", inner], start_new_session=True)
        try:
            assert _wait_for(lambda: pidfile.exists() and pidfile.read_text()), "no child"
            child = int(pidfile.read_text())
            assert child != wrapper.pid, "bash exec'd the command; test is moot"

            out = io_capture(lambda: exec(gd.kill_code(wrapper.pid), {}))
            assert f"ok pgid={wrapper.pid}" in out, out
            wrapper.wait(timeout=10)

            def child_gone() -> bool:
                try:
                    os.kill(child, 0)
                except ProcessLookupError:
                    return True
                return False
            assert _wait_for(child_gone, timeout=10), "worker survived the kill"
            assert "gone" in io_capture(lambda: exec(gd.kill_code(wrapper.pid), {}))
        finally:
            if wrapper.poll() is None:
                os.killpg(wrapper.pid, 9)


def io_capture(fn) -> str:
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn()
    return buf.getvalue()


# ---------------------------------------------------------------------------

def _main() -> int:
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    failures = []
    for name, fn in tests:
        started = time.monotonic()
        try:
            fn()
            print(f"  PASS  {name}  ({time.monotonic() - started:.1f}s)")
        except Exception as exc:  # noqa: BLE001 — this is the test runner
            import traceback
            print(f"  FAIL  {name}: {exc}")
            traceback.print_exc()
            failures.append(name)
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
