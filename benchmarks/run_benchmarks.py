"""Run the envelope benchmarks and write ``benchmarks/results/<date>.json``.

Plain script, run by hand (spec 15 §5.5); not pytest, not packaged, not CI.

    python benchmarks/run_benchmarks.py                 # every case
    python benchmarks/run_benchmarks.py --quick         # skip the long edge-of-envelope cases
    python benchmarks/run_benchmarks.py --only geometry # cases whose name contains a substring
    python benchmarks/run_benchmarks.py --list

Each case runs in its own child process with the BLAS/OpenMP thread variables
set to 1, as in CI, so timings are single-threaded and the memory figures of one
case do not include another's. See ``benchmarks/README.md``.
"""

# ruff: noqa: T201  (a command-line script: it reports progress on stdout)
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import resource
import subprocess
import sys
import time
import tracemalloc
import warnings
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
THREAD_VARS = {
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",  # Apple Accelerate
    "MPLBACKEND": "agg",
}
TRACE_LIMIT_S = 30.0  # rerun under tracemalloc only when one run is at most this long


def _import_path(installed: bool) -> None:
    """Benchmark the checkout's ``src/`` unless ``--installed`` is given."""
    if not installed:
        sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(HERE))


def _maxrss_bytes() -> int:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r if sys.platform == "darwin" else r * 1024)  # bytes on macOS, KiB on Linux


# ---------------------------------------------------------------------------
# Environment (spec 15 §5.13 fields, plus CPU, memory and threads)
# ---------------------------------------------------------------------------


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def _loadavg() -> list[float] | None:
    """1, 5 and 15 minute load averages: other work on the machine inflates the timings."""
    try:
        return [round(x, 2) for x in os.getloadavg()]
    except (AttributeError, OSError):
        return None


def _cpu() -> dict[str, Any]:
    info: dict[str, Any] = {
        "processor": platform.processor() or None,
        "logical_cpus": os.cpu_count(),
    }
    if sys.platform == "darwin":
        info["model"] = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        info["physical_cpus"] = _run(["sysctl", "-n", "hw.physicalcpu"])
        mem = _run(["sysctl", "-n", "hw.memsize"])
        info["memory_bytes"] = int(mem) if mem else None
    elif Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                info["model"] = line.split(":", 1)[1].strip()
                break
        try:
            pages = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
            info["memory_bytes"] = int(pages)
        except (ValueError, OSError):
            info["memory_bytes"] = None
    return info


def _blas() -> dict[str, Any]:
    import numpy as np

    try:
        deps = np.show_config(mode="dicts")["Build Dependencies"]
        return {
            k: {"name": deps[k].get("name"), "version": deps[k].get("version")}
            for k in ("blas", "lapack")
        }
    except Exception:  # noqa: BLE001 - informational only
        return {"blas": None, "lapack": None}


def _threadpools() -> list[dict[str, Any]] | None:
    try:
        from threadpoolctl import threadpool_info
    except ImportError:
        return None
    return [
        {k: d.get(k) for k in ("user_api", "internal_api", "num_threads", "version")}
        for d in threadpool_info()
    ]


def environment(installed: bool) -> dict[str, Any]:
    """Describe the interpreter, libraries, machine and thread settings of this run."""
    import numpy as np
    import scipy

    import active_subspaces as ac

    head = _run(["git", "-C", str(REPO), "rev-parse", "HEAD"])
    dirty = _run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"])
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "active_subspaces": ac.__version__,
        "source": "installed" if installed else "checkout src/",
        "git_commit": head,
        "git_dirty": None if head is None else bool(dirty),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": _cpu(),
        "blas": _blas(),
        "thread_env": {k: os.environ.get(k) for k in THREAD_VARS},
        "threadpoolctl": _threadpools(),
        "load_average_at_end": _loadavg(),
    }


# ---------------------------------------------------------------------------
# Child: one case
# ---------------------------------------------------------------------------


def _child(name: str) -> dict[str, Any]:
    from cases import CASES

    case = next(c for c in CASES if c.name == name)
    rec: dict[str, Any] = {"name": case.name, "group": case.group, "params": case.params}
    rec["rss_after_setup_bytes"] = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t0 = time.perf_counter()
        state = case.setup(case.params)
        rec["setup_s"] = time.perf_counter() - t0
        rec["rss_after_setup_bytes"] = _maxrss_bytes()
        times = []
        info: dict[str, Any] = {}
        for _ in range(case.repeat):
            t0 = time.perf_counter()
            info = case.run(state)
            times.append(time.perf_counter() - t0)
        rec["rss_peak_bytes"] = _maxrss_bytes()
        rec["wall_s"] = min(times)
        rec["wall_s_all"] = times
        if min(times) <= TRACE_LIMIT_S:
            tracemalloc.start()
            case.run(state)
            rec["tracemalloc_peak_bytes"] = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
        else:
            rec["tracemalloc_peak_bytes"] = None
    codes = sorted({str(getattr(w.message, "code", type(w.message).__name__)) for w in caught})
    rec["warnings"] = codes
    rec["info"] = info
    return rec


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _output_path(arg: str | None, date: str) -> Path:
    if arg:
        return Path(arg)
    out = HERE / "results" / f"{date}.json"
    k = 2
    while out.exists():
        out = HERE / "results" / f"{date}-{k}.json"
        k += 1
    return out


def main(argv: list[str] | None = None) -> int:
    """Run the selected cases, one child process each, and write the results file."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--only",
        action="append",
        default=[],
        help="run cases whose name contains this (repeatable)",
    )
    ap.add_argument("--quick", action="store_true", help="skip the long edge-of-envelope cases")
    ap.add_argument("--list", action="store_true", help="list the cases and exit")
    ap.add_argument(
        "--installed", action="store_true", help="benchmark the installed package, not src/"
    )
    ap.add_argument("--timeout", type=float, default=900.0, help="per-case timeout in seconds")
    ap.add_argument("--output", help="results file (default results/<date>.json)")
    ap.add_argument(
        "--runner", default="", help="free-text description of the machine, stored in the results"
    )
    ap.add_argument("--child", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    _import_path(args.installed)
    if args.child:
        print(json.dumps(_child(args.child)))
        return 0

    from cases import CASES

    cases = [
        c
        for c in CASES
        if (not args.only or any(s in c.name for s in args.only))
        and not (args.quick and c.boundary)
    ]
    if args.list:
        for c in cases:
            flag = " [boundary]" if c.boundary else ""
            print(f"{c.name}{flag}")
        return 0

    env = {**os.environ, **THREAD_VARS}
    os.environ.update(THREAD_VARS)  # so environment() records them
    date = _dt.date.today().isoformat()
    started = _dt.datetime.now().astimezone().isoformat(timespec="seconds")
    results = []
    t_all = time.perf_counter()
    for c in cases:
        cmd = [sys.executable, str(Path(__file__).resolve()), "--child", c.name]
        if args.installed:
            cmd.append("--installed")
        load = _loadavg()
        print(f"{c.name} ...", end=" ", flush=True)
        limit = min(args.timeout, c.timeout_s or args.timeout)
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=limit)
        except subprocess.TimeoutExpired:
            rec = {"name": c.name, "group": c.group, "params": c.params, "status": "timeout"}
            rec["timeout_s"] = limit
            print(f"timeout after {limit:.0f} s")
        else:
            if proc.returncode == 0:
                rec = {**json.loads(proc.stdout.strip().splitlines()[-1]), "status": "ok"}
                print(f"{rec['wall_s']:.3f} s")
            else:
                tail = proc.stderr.strip().splitlines()[-5:]
                rec = {
                    "name": c.name,
                    "group": c.group,
                    "params": c.params,
                    "status": "error",
                    "stderr": tail,
                }
                print("ERROR\n  " + "\n  ".join(tail))
        rec["process_s"] = time.perf_counter() - t0
        rec["load_average_before"] = load
        rec["repeat"] = c.repeat
        rec["boundary"] = c.boundary
        rec["note"] = c.note
        results.append(rec)

    doc = {
        "schema": "active_subspaces.benchmarks/1",
        "date": date,
        "started": started,
        "total_s": time.perf_counter() - t_all,
        "runner": args.runner,
        "selection": {"only": args.only, "quick": args.quick},
        "environment": environment(args.installed),
        "method": {
            "wall_s": "minimum over `repeat` timed runs of the call alone (time.perf_counter); "
            "setup is not timed",
            "tracemalloc_peak_bytes": "peak Python/NumPy allocation during one extra run under "
            f"tracemalloc; only when wall_s <= {TRACE_LIMIT_S:.0f} s, else null",
            "rss_peak_bytes": "process high-water RSS (getrusage ru_maxrss) after the timed runs; "
            "includes the interpreter, imports and the setup data",
            "rss_after_setup_bytes": "high-water RSS before the first timed run",
            "threads": "every child runs with the thread variables of thread_env",
            "load_average_before": "os.getloadavg() just before the case's child starts; "
            "a value above the core count means the timings shared the CPU",
        },
        "cases": results,
    }
    out = _output_path(args.output, date)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {out} ({doc['total_s']:.0f} s)")
    return 0 if all(r["status"] != "error" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
