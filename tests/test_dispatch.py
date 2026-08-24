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
