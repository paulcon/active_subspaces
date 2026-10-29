"""Envelope benchmark cases (spec 15 §5.5; target/00 §2.3).

Each case is a ``Case``: ``setup(params)`` builds the inputs (not timed) and
``run(state)`` performs the timed call and returns a dict of facts about the
output (sizes, counts, checks). ``run_benchmarks.py`` runs every case in a fresh
process, so the peak-memory figures of one case never include another's.

All data are synthetic and seeded. Physical and normalized coordinates coincide
here (``Uniform(-1, 1)`` boxes and ``Gaussian(0, I)``), so ``m``, ``n`` and ``M``
are the only sizes that matter.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

import active_subspaces as ac

SEED = 20261029


@dataclass(frozen=True)
class Case:
    """One benchmark: untimed ``setup(params)``, then timed ``run(state)`` calls."""

    name: str
    group: str
    params: dict[str, Any]
    setup: Callable[[dict[str, Any]], Any]
    run: Callable[[Any], dict[str, Any]]
    repeat: int = 1  # timed repetitions; the minimum and all values are reported
    boundary: bool = False  # a long case at the edge of the envelope; skipped by --quick
    note: str = ""
    timeout_s: float | None = None  # a per-case limit; reaching it is recorded, not an error


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def _uniform(m: int) -> ac.Uniform:
    return ac.Uniform(-np.ones(m), np.ones(m))


def _ridge_data(m: int, M: int, seed: int = SEED) -> dict[str, Any]:
    """Points in [-1, 1]^m, f = sin(a1.x) + (a2.x)^2 / 2, exact gradients plus 1% noise."""
    rng = np.random.default_rng(seed)
    A = np.linalg.qr(rng.standard_normal((m, 2)))[0]
    X = rng.uniform(-1.0, 1.0, size=(M, m))
    Y = X @ A
    f = np.sin(Y[:, 0]) + 0.5 * Y[:, 1] ** 2
    G = np.cos(Y[:, 0])[:, None] * A[:, 0] + Y[:, 1][:, None] * A[:, 1]
    G += 0.01 * rng.standard_normal((M, m))
    return {"space": _uniform(m), "X": X, "f": f, "grads": G}


def _basis_subspace(m: int, n: int, *, gaussian: bool = False, seed: int = SEED):
    """Return a subspace result from a random orthonormal basis (generic position)."""
    rng = np.random.default_rng(seed + 1000 * n + m)
    B = np.linalg.qr(rng.standard_normal((m, n)))[0]
    space = ac.Gaussian(np.zeros(m), np.ones(m)) if gaussian else _uniform(m)
    return ac.subspace_from_basis(B, space=space)


def _interior_points(sub, k: int, seed: int = SEED) -> np.ndarray:
    """Active coordinates of k uniform points of the cube (interior of the domain)."""
    rng = np.random.default_rng(seed + 7)
    X = rng.uniform(-1.0, 1.0, size=(k, sub.m))
    return ac.forward_map(sub, X=X).Y


def _quadratic_model(X: np.ndarray) -> np.ndarray:
    return 0.5 * np.sum(X * X, axis=1)


def _nbytes(*arrays: Any) -> int:
    return int(sum(np.asarray(a).nbytes for a in arrays if a is not None))


# ---------------------------------------------------------------------------
# Subspace estimation, bootstrap, gradients  (target/00 §2.3 row 1; [R-10])
# ---------------------------------------------------------------------------


def _setup_ridge(p: dict[str, Any]) -> dict[str, Any]:
    return {**_ridge_data(p["m"], p["M"]), "p": p}


def _run_estimator(s: dict[str, Any]) -> dict[str, Any]:
    p = s["p"]
    data = {k: s[k] for k in ("X", "f", "grads") if k in p["uses"]}
    r = ac.active_subspace(space=s["space"], estimator=p["estimator"], n=2, **data)
    return {"n": r.n, "eigenvectors_bytes": _nbytes(r.eigenvectors)}


def _run_bootstrap(s: dict[str, Any]) -> dict[str, Any]:
    p = s["p"]
    r = ac.active_subspace(space=s["space"], grads=s["grads"], n=2, nboot=p["nboot"], rng=SEED)
    return {
        "nboot": r.nboot,
        "bootstrap_arrays_bytes": _nbytes(
            r.bootstrap_eigenvalues, r.bootstrap_distances, r.bootstrap_dets
        ),
    }


def _run_llg(s: dict[str, Any]) -> dict[str, Any]:
    p = s["p"]
    centers = None if p["centers"] is None else np.arange(p["centers"])
    g = ac.local_linear_gradients(X=s["X"], f=s["f"], space=s["space"], centers=centers)
    return {"p_used": g.p, "n_centers": int(g.grads.shape[0])}


def _estimator_case(est: str, m: int, M: int, uses: tuple[str, ...], note: str = "") -> Case:
    return Case(
        name=f"subspace.{est}.m{m}.M{M}",
        group="subspace_estimation",
        params={"estimator": est, "m": m, "M": M, "n": 2, "uses": list(uses)},
        setup=_setup_ridge,
        run=_run_estimator,
        repeat=3 if m * M <= 2_000_000 else 1,
        note=note,
    )


SUBSPACE_CASES = [
    _estimator_case("as", 1000, 2000, ("grads",)),
    _estimator_case("as", 1000, 10000, ("grads",)),
    _estimator_case("nas", 1000, 2000, ("grads",)),
    _estimator_case("asx", 1000, 2000, ("X", "grads")),
    _estimator_case("ols", 1000, 2000, ("X", "f")),
    _estimator_case("sir", 1000, 2000, ("X", "f")),
    _estimator_case("phd", 1000, 2000, ("X", "f")),
    _estimator_case("save", 1000, 2000, ("X", "f")),
    _estimator_case(
        "qphd",
        30,
        1000,
        ("X", "f"),
        "QPHD needs M >= (m+1)(m+2)/2 samples (501 501 at m = 1000), so it is run at m = 30.",
    ),
    _estimator_case(
        "ss",
        100,
        1000,
        ("X", "f"),
        "SS forms all M^2 pairwise difference quotients: O(M^2 m^2); run at m = 100.",
    ),
    _estimator_case(
        "opg",
        100,
        1000,
        ("X", "f"),
        "OPG is local linear gradients at every sample; see the gradients.local_linear cases.",
    ),
]

BOOTSTRAP_CASES = [
    Case(
        name=f"bootstrap.as.m{m}.M{M}.nboot{B}",
        group="bootstrap",
        params={"estimator": "as", "m": m, "M": M, "n": 2, "nboot": B},
        setup=_setup_ridge,
        run=_run_bootstrap,
        boundary=(m == 1000),
        note=note,
    )
    for m, M, B, note in [
        (100, 1000, 100, ""),
        (300, 1000, 20, ""),
        (
            1000,
            2000,
            3,
            "Scaled down from nboot = 100: three replicates give the per-replicate cost.",
        ),
    ]
]

GRADIENT_CASES = [
    Case(
        name=f"gradients.local_linear.m{m}.M{M}.centers{c if c is not None else 'all'}",
        group="local_linear_gradients",
        params={"m": m, "M": M, "centers": c},
        setup=_setup_ridge,
        run=_run_llg,
        note=note,
    )
    for m, M, c, note in [
        (100, 1000, None, ""),
        (300, 1000, None, ""),
        (
            1000,
            2000,
            10,
            "Scaled down from all 2000 centers: 10 centers give the per-center cost "
            "(p = 1700 neighbours, a 1700 x 1001 least-squares fit each).",
        ),
    ]
]


# ---------------------------------------------------------------------------
# Geometry  (target/00 §2.3 row 2 and the vertex table; spec 08 §8)
# ---------------------------------------------------------------------------


def _setup_domain(p: dict[str, Any]) -> dict[str, Any]:
    return {"sub": _basis_subspace(p["m"], p["n"]), "p": p}


def _run_domain(s: dict[str, Any]) -> dict[str, Any]:
    p = s["p"]
    m, n = p["m"], p["n"]
    d = ac.active_domain(s["sub"])
    V = np.asarray(d.vertices)
    out: dict[str, Any] = {
        "N_v": int(V.shape[0]),
        "N_v_formula": ac.zonotope_vertex_count(m, n),
        "N_f": int(np.asarray(d.facet_normals).shape[0]),
        "general_position": d.general_position,
        "domain_bytes": _nbytes(d.vertices, d.vertex_preimages, d.facet_normals, d.facet_offsets),
        "predicted_bytes_8Nv(m+n)": 8 * ac.zonotope_vertex_count(m, n) * (m + n),
        "Nv_times_m_plus_n": ac.zonotope_vertex_count(m, n) * (m + n),
    }
    if n == 2:
        x, y = V[:, 0], V[:, 1]
        out["signed_area"] = float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
        out["counter_clockwise"] = out["signed_area"] > 0
    return out


def _domain_case(m: int, n: int, *, boundary: bool = False, note: str = "") -> Case:
    Nv = ac.zonotope_vertex_count(m, n)
    return Case(
        name=f"geometry.active_domain.n{n}.m{m}",
        group="active_domain",
        params={"m": m, "n": n},
        setup=_setup_domain,
        run=_run_domain,
        repeat=3 if Nv * (m + n) <= 1_000_000 else 1,
        boundary=boundary,
        note=note,
    )


def _setup_counts(p: dict[str, Any]) -> dict[str, Any]:
    return {"p": p}


def _run_counts(s: dict[str, Any]) -> dict[str, Any]:
    table = {}
    for n in s["p"]["n"]:
        for m in s["p"]["m"]:
            Nv = ac.zonotope_vertex_count(m, n)
            table[f"n={n},m={m}"] = {
                "N_v": Nv,
                "Nv_times_m_plus_n": Nv * (m + n),
                "within_1e8": Nv * (m + n) <= 10**8,
            }
    return {"table": table}


GEOMETRY_CASES = [
    Case(
        name="geometry.zonotope_vertex_count.table",
        group="zonotope_vertex_count",
        params={"n": [2, 3, 4], "m": [30, 100, 300, 1000]},
        setup=_setup_counts,
        run=_run_counts,
        repeat=3,
        note="The target/00 §2.3 vertex table, from the public count function.",
    ),
    _domain_case(30, 2),
    _domain_case(100, 2),
    _domain_case(300, 2),
    _domain_case(1000, 2, note="spec 08 §8: 2m vertices, counter-clockwise, in < 1 s."),
    _domain_case(30, 3),
    _domain_case(100, 3),
    _domain_case(300, 3),
    _domain_case(
        450,
        3,
        boundary=True,
        note="Just inside N_v (m + n) <= 1e8 (9.2e7); the n = 3 edge of the guidance.",
    ),
    _domain_case(30, 4),
    _domain_case(60, 4),
    _domain_case(100, 4),
    _domain_case(
        130,
        4,
        boundary=True,
        note="Just inside N_v (m + n) <= 1e8 (9.6e7); the n = 4 edge of the guidance.",
    ),
]


# ---------------------------------------------------------------------------
# Integration, quadrature and the inverse map at the default n_mc = 50
# ---------------------------------------------------------------------------


def _setup_inverse(p: dict[str, Any]) -> dict[str, Any]:
    sub = _basis_subspace(p["m"], p["n"], gaussian=p["space"] == "gaussian")
    return {"sub": sub, "Y": _interior_points(sub, p["points"]), "p": p}


def _run_inverse(s: dict[str, Any]) -> dict[str, Any]:
    r = ac.inverse_map(s["sub"], Y=s["Y"], n_draws=s["p"]["n_draws"], rng=SEED)
    return {
        "sampler": r.sampler,
        "burn_in": r.burn_in,
        "thin": r.thin,
        "X_bytes": _nbytes(r.X),
        "steps_per_chain": None if r.burn_in is None else r.burn_in + r.n_draws * r.thin,
    }


def _setup_integrate(p: dict[str, Any]) -> dict[str, Any]:
    sub = _basis_subspace(p["m"], p["n"], gaussian=p["space"] == "gaussian")
    return {"domain": ac.active_domain(sub), "p": p}


def _run_integrate(s: dict[str, Any]) -> dict[str, Any]:
    p = s["p"]
    r = ac.integrate(
        _quadratic_model, domain=s["domain"], n_nodes=p["n_nodes"], n_mc=p["n_mc"], rng=SEED
    )
    Q = int(np.asarray(r.design.nodes).shape[0])
    return {
        "nodes_Q": Q,
        "model_evaluations": int(np.asarray(r.f).shape[0]),
        "sampler": r.design.inactive_sampler,
        "estimate": float(r.estimate),
        "exact": p["m"] / 6.0 if p["space"] == "uniform" else p["m"] / 2.0,
    }


def _run_rule(s: dict[str, Any]) -> dict[str, Any]:
    p = s["p"]
    r = ac.active_quadrature_rule(s["domain"], n_nodes=p["n_nodes"], rng=SEED)
    return {
        "N_v": int(np.asarray(s["domain"].vertices).shape[0]),
        "nodes_Q": int(np.asarray(r.nodes).shape[0]),
        "weight_deficit": float(r.weight_deficit),
    }


def _run_design(s: dict[str, Any]) -> dict[str, Any]:
    p = s["p"]
    r = ac.active_design(domain=s["domain"], n_points=p["n_points"], rng=SEED)
    return {
        "N_v": int(np.asarray(s["domain"].vertices).shape[0]),
        "N_f": int(np.asarray(s["domain"].facet_normals).shape[0]),
        "method": r.method,
        "min_distance": r.min_distance,
    }


INTEGRATION_CASES = (
    [
        Case(
            name=f"geometry.inverse_map.{sp}.n2.m{m}.points{k}.draws50",
            group="inverse_map",
            params={"space": sp, "m": m, "n": 2, "points": k, "n_draws": 50},
            setup=_setup_inverse,
            run=_run_inverse,
            repeat=3 if sp == "gaussian" else 1,
            note=note,
        )
        for sp, m, k, note in [
            (
                "uniform",
                100,
                64,
                "Hit-and-run from the Chebyshev centre; one lockstep block of 64 chains.",
            ),
            ("uniform", 1000, 10, "Hit-and-run; burn-in 10(m-n) and thinning m-n by default."),
            ("gaussian", 1000, 64, "Exact Gaussian draws."),
        ]
    ]
    + [
        Case(
            name=f"designs.active_design.uniform.n{n}.m{m}.N{N}",
            group="active_design",
            params={"space": "uniform", "m": m, "n": n, "n_points": N},
            setup=_setup_integrate,
            run=_run_design,
            boundary=boundary,
            note=note,
            timeout_s=300.0 if (n, m, N) in ((3, 100, 10), (2, 1000, 25)) else None,
        )
        for n, m, N, boundary, note in [
            (2, 100, 10, False, "Maximin by SLSQP (N(N-1)/2 <= 2000 pairs), 3 starts."),
            (
                2,
                1000,
                10,
                True,
                "Maximin by SLSQP; the facet constraints number N N_f = 20 000 here.",
            ),
            (
                2,
                100,
                25,
                False,
                "N = 5**n = 25 is the default design of build_reduced_model (model path).",
            ),
            (
                2,
                1000,
                25,
                True,
                "The reduced-model default design at m = 1000: N N_f = 50 000. Capped at 300 s.",
            ),
            (3, 30, 10, False, "Maximin by SLSQP; N N_f = 8 700 facet constraints."),
            (
                3,
                100,
                10,
                True,
                "Maximin by SLSQP; N N_f = 99 020 facet constraints. Capped at 300 s.",
            ),
            (3, 100, 100, False, "Greedy farthest-point (N(N-1)/2 = 4 950 > 2000 pairs)."),
            (
                3,
                100,
                125,
                True,
                "Greedy; N = 5**3 is the reduced-model default design at n = 3.",
            ),
            (
                4,
                20,
                625,
                True,
                "Greedy; N = 5**4 is the reduced-model default design at n = 4. Its candidate-to-"
                "vertex distance array holds 50 N N_v n doubles (2.3 GB here).",
            ),
        ]
    ]
    + [
        Case(
            name=f"integration.quadrature_rule.uniform.n{n}.m{m}.Nd0",
            group="active_quadrature_rule",
            params={"space": "uniform", "m": m, "n": n, "n_nodes": 0},
            setup=_setup_integrate,
            run=_run_rule,
            note="Delaunay triangulation of the vertices (N_d = 0) and 100 000 weight samples; "
            "the maximin points are timed in the designs cases.",
        )
        for n, m in [(2, 100), (2, 1000), (3, 30), (3, 100)]
    ]
    + [
        Case(
            name=f"integration.integrate.{sp}.n{n}.m{m}.nodes{nn}.nmc50",
            group="integrate",
            params={"space": sp, "m": m, "n": n, "n_nodes": nn, "n_mc": 50},
            setup=_setup_integrate,
            run=_run_integrate,
            note=note,
        )
        for sp, n, m, nn, note in [
            ("uniform", 2, 10, 10, "Includes the maximin design of the 10 interior points."),
            ("uniform", 2, 30, 10, "Includes the maximin design of the 10 interior points."),
            ("uniform", 2, 100, 0, "Vertices only (N_d = 0); hit-and-run inactive draws."),
            ("uniform", 2, 1000, 0, "Vertices only (N_d = 0); hit-and-run inactive draws."),
            ("gaussian", 2, 1000, 25, "5 x 5 Gauss-Hermite nodes; exact Gaussian inactive draws."),
        ]
    ]
)


CASES: list[Case] = (
    SUBSPACE_CASES + BOOTSTRAP_CASES + GRADIENT_CASES + GEOMETRY_CASES + INTEGRATION_CASES
)

_names = [c.name for c in CASES]
assert len(_names) == len(set(_names)), "duplicate case names"
