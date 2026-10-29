"""Recapture the ``v1_reference`` regression baselines [D-163] (spec 15 §5.13, spec 17 §5.5).

The D-37 baselines in ``tests/data/baselines/<area>/<case>.npz`` (inputs ``in/<name>``,
outputs ``out/<name>``) and ``<case>.json`` (environment, legacy call, per-output status)
were captured from the Python 3 reference ports of the 2016 code. The ports and the legacy
capture mode of this script were removed before v1.0; both remain at the git tag
``legacy-py3`` (spec 15 §5.13). What is left is the ``--v1`` mode: it reads the stored inputs
of a case, runs ``src/active_subspaces`` and writes the ``v1_reference`` outputs, which have
no 2016 counterpart and no closed form, into the same files. The files are byte-identical
from run to run on one platform.

Usage (from the repository root)::

    python tools/capture_baselines.py --v1             # recapture the v1_reference outputs
    python tools/capture_baselines.py --v1 --check     # recapture into a temporary directory and compare bytes
"""

from __future__ import annotations

import argparse
import dataclasses
import filecmp
import io
import json
import os
import platform
import shutil
import sys
import tempfile
import warnings
import zipfile
from typing import Any, Callable

for _var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"  # one BLAS thread: summation order, and so the bytes, do not depend on scheduling

import numpy as np  # noqa: E402
import scipy  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "tests", "data", "baselines")
_FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


# ---------------------------------------------------------------------------
# Status records (spec 17 §5.5 step 4)
# ---------------------------------------------------------------------------


def v1_reference(tier: str, scale: str = "max", **extra: Any) -> dict:
    return {"status": "v1_reference", "tolerance": {"tier": tier, "scale": scale}, **extra}


# ---------------------------------------------------------------------------
# Cases with v1_reference outputs
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Case:
    area: str
    name: str
    v1: Callable[[dict[str, np.ndarray], dict], dict[str, tuple[Any, dict]]]


CASES: list[Case] = []


def v1_capture(area: str, name: str):
    def deco(fn):
        CASES.append(Case(area, name, fn))
        return fn

    return deco


def c16_function(X: np.ndarray, A: np.ndarray, x0: np.ndarray) -> np.ndarray:
    D = np.atleast_2d(X) - x0
    return np.einsum("ij,jk,ik->i", D, A, D)


@v1_capture("optimization", "lift")
def _lift_v1(inp: dict[str, np.ndarray], meta: dict) -> dict[str, tuple[Any, dict]]:
    import active_subspaces as asub

    A, x0 = inp["A"], inp["x0"]
    out = {}
    for name, space in (("gaussian", asub.Gaussian.reference(4)), ("uniform", asub.Uniform.reference(4))):
        sub = asub.active_subspace(grads=inp["grads"], space=space, n=1)
        d = asub.active_domain(sub)
        y = np.asarray(sub.W1).T @ x0
        res = asub.lift(y, domain=d, X=inp["X"], f=inp["f"])
        out[f"v1_f_xstar_{name}"] = (
            np.float64(c16_function(np.asarray(res.x)[None, :], A, x0)[0]),
            v1_reference("F8", decision="D-163", call=f"lift(W1^T x0, domain=active_domain(active_subspace(grads, {type(space).__name__}.reference(4), n=1)), X, f)  # default lift_dim = 3"),
        )
    return out


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


def _write_npz(path: str, arrays: dict[str, np.ndarray]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name in sorted(arrays):
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.asarray(arrays[name], order="C"), allow_pickle=False)  # keeps 0-d arrays 0-d
            info = zipfile.ZipInfo(name + ".npy", date_time=_FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, buf.getvalue())


def _read_npz(path: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _write_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(obj, indent=1, sort_keys=True, allow_nan=False) + "\n")


def _blas() -> str:
    try:
        info = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
        return f"{info.get('name')} {info.get('version')}"
    except Exception:  # noqa: BLE001
        return "unknown"


def _environment() -> dict:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "blas": _blas(),
    }


def _array_meta(arr: np.ndarray) -> dict:
    return {"shape": list(arr.shape), "dtype": str(arr.dtype)}


def _paths(out: str, c: Case) -> tuple[str, str]:
    d = os.path.join(out, c.area)
    return os.path.join(d, c.name + ".npz"), os.path.join(d, c.name + ".json")


def write_v1(out: str, c: Case) -> None:
    npz_path, json_path = _paths(out, c)
    with open(json_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    arrays = _read_npz(npz_path)
    inputs = {k[3:]: v for k, v in arrays.items() if k.startswith("in/")}
    import active_subspaces

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        produced = c.v1(inputs, meta)
    for k, (v, rec) in produced.items():
        arr = np.asarray(v)
        arrays["out/" + k] = arr
        meta["outputs"][k] = {**rec, **_array_meta(arr)}
    meta["v1_environment"] = {**_environment(), "active_subspaces": active_subspaces.__version__}
    _write_npz(npz_path, arrays)
    _write_json(json_path, meta)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _selected(names: list[str]) -> list[Case]:
    if not names:
        return list(CASES)
    keep = [c for c in CASES if f"{c.area}/{c.name}" in names or c.area in names]
    missing = set(names) - {f"{c.area}/{c.name}" for c in keep} - {c.area for c in keep}
    if missing:
        raise SystemExit(f"capture_baselines: unknown case(s) {sorted(missing)}")
    return keep


def capture_v1(out: str, names: list[str]) -> None:
    sys.path.insert(0, os.path.join(ROOT, "src"))
    for c in _selected(names):
        write_v1(out, c)
        print(f"v1: {c.area}/{c.name}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cases", nargs="*", help="area or area/case names (default: all)")
    ap.add_argument("--v1", action="store_true", help="capture the v1_reference outputs from src/ [D-163] (the only mode)")
    ap.add_argument("--check", action="store_true", help="recapture into a temporary directory and compare bytes with the stored files")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    if not a.v1:
        ap.error("only --v1 remains; the legacy capture is at the git tag legacy-py3 (spec 15 §5.13)")
    if a.check:
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copytree(a.out, tmp, dirs_exist_ok=True)
            capture_v1(tmp, a.cases)
            bad = []
            for c in _selected(a.cases):
                for p_new, p_old in zip(_paths(tmp, c), _paths(a.out, c)):
                    if not filecmp.cmp(p_new, p_old, shallow=False):
                        bad.append(os.path.relpath(p_old, ROOT))
            for p in bad:
                print(f"differs: {p}")
            print("baselines identical" if not bad else f"{len(bad)} file(s) differ")
            return 1 if bad else 0
    capture_v1(a.out, a.cases)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
