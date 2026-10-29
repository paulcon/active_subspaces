"""Private randomness helpers (spec 05).

Every stochastic public function calls :func:`generator` exactly once, at
entry, and records the returned description in provenance. Independent tasks
get child streams from :func:`spawn` (called before any draw from the parent),
with the number of children fixed by the problem, never by the worker count.
:func:`replay` rebuilds the entry generator from a description (tests only).

Foundation helper index
-----------------------
::

    # type alias for `rng` parameters
    RngLike = None | int | np.random.SeedSequence | np.random.Generator
    # normalize once at entry; dict -> Provenance.rng
    generator(rng) -> tuple[np.random.Generator, dict]
    # k child streams; call before drawing from gen
    spawn(gen, k: int) -> list[np.random.Generator]
    # source: description dict, Provenance or result
    replay(source) -> np.random.Generator

Typical use::

    gen, rng_desc = _random.generator(rng)               # InputError input.parameter on bad rng
    children = _random.spawn(gen, nboot)                 # one child per task, in task order
    ... results._make_provenance("active_subspace", params, space=space, rng_state=rng_desc)

The description's ``entropy`` is a decimal string, or a list of decimal strings
for a ``SeedSequence`` built from a list; ``pool_size`` is recorded only when it
is not 4 (spec 05 §5.3).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np

from active_subspaces.errors import InputError

__all__ = ()

RngLike = None | int | np.random.SeedSequence | np.random.Generator


def _seed_seq_description(ss: Any) -> dict[str, Any] | None:
    if not isinstance(ss, np.random.SeedSequence):
        return None
    entropy = ss.entropy
    if isinstance(entropy, (int, np.integer)):
        ent: Any = str(int(entropy))
    else:
        ent = [str(int(e)) for e in cast("Sequence[int]", entropy)]
    d: dict[str, Any] = {
        "entropy": ent,
        "spawn_key": [int(k) for k in cast("Sequence[int]", ss.spawn_key)],
        "n_children_spawned": int(ss.n_children_spawned),
    }
    if ss.pool_size != 4:
        d["pool_size"] = int(ss.pool_size)
    return d


def _stringify(obj: Any) -> Any:
    """Integers (and integer arrays) to decimal strings, recursively, so they survive JSON."""
    if isinstance(obj, Mapping):
        return {str(k): _stringify(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return {"__uint_array__": str(obj.dtype), "data": [str(int(v)) for v in obj.ravel()]}
    if isinstance(obj, (list, tuple)):
        return [_stringify(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return str(int(obj))
    if isinstance(obj, (float, np.floating)):
        return float(obj)
    return obj


def _destringify(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        if "__uint_array__" in obj:
            return np.array([int(v) for v in obj["data"]], dtype=np.dtype(obj["__uint_array__"]))
        return {k: (v if k == "bit_generator" else _destringify(v)) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_destringify(v) for v in obj]
    if isinstance(obj, str) and obj.lstrip("-").isdigit():
        return int(obj)
    return obj


def _bad_rng(rng: Any) -> InputError:
    return InputError(
        "rng: expected None, a non-negative int, a numpy SeedSequence or a numpy Generator, got "
        f"{type(rng).__name__}",
        code="input.parameter",
        details={"param": "rng", "got": type(rng).__name__},
    )


def generator(rng: Any) -> tuple[np.random.Generator, dict[str, Any]]:
    """Normalize a public ``rng`` argument (SPEC 7) to ``(Generator, provenance description)``,
    before any draw.
    """
    if rng is None:
        ss = np.random.SeedSequence()  # fresh OS entropy, recorded [D-72]
    elif isinstance(rng, (bool, np.bool_)):
        raise _bad_rng(rng)
    elif isinstance(rng, (int, np.integer)):
        if int(rng) < 0:
            raise InputError(
                f"rng: integer seeds must be non-negative, got {int(rng)}",
                code="input.parameter",
                details={"param": "rng", "got": int(rng)},
            )
        ss = np.random.SeedSequence(int(rng))
    elif isinstance(rng, np.random.SeedSequence):
        ss = rng
    elif isinstance(rng, np.random.Generator):
        bg = rng.bit_generator
        desc: dict[str, Any] = {
            "kind": "generator",
            "bit_generator": type(bg).__name__,
            "state": _stringify(bg.state),
            "seed_sequence": _seed_seq_description(getattr(bg, "seed_seq", None)),
        }
        return rng, desc
    else:
        raise _bad_rng(rng)
    ss_desc = cast("dict[str, Any]", _seed_seq_description(ss))  # never None for a SeedSequence
    desc = {"kind": "seed_sequence", "bit_generator": "PCG64", **ss_desc}
    return np.random.Generator(np.random.PCG64(ss)), desc


def spawn(gen: np.random.Generator, k: int) -> list[np.random.Generator]:
    """Return ``k`` independent child generators; call before any draw from ``gen``."""
    return gen.spawn(int(k))


def _seed_sequence_from(d: Mapping[str, Any]) -> np.random.SeedSequence:
    ent = d["entropy"]
    entropy = int(ent) if isinstance(ent, str) else [int(e) for e in ent]
    kwargs: dict[str, Any] = {
        "spawn_key": tuple(int(k) for k in d["spawn_key"]),
        "n_children_spawned": int(d["n_children_spawned"]),
    }
    if "pool_size" in d:
        kwargs["pool_size"] = int(d["pool_size"])
    return np.random.SeedSequence(entropy, **kwargs)


def replay(source: Any) -> np.random.Generator:
    """Rebuild the generator a function saw on entry from a description, a ``Provenance`` or a
    result (tests only).
    """
    d = source
    if hasattr(d, "provenance"):
        d = d.provenance
    if hasattr(d, "rng") and not isinstance(d, Mapping):
        d = d.rng
    if d is None:
        raise ValueError("no rng description recorded (deterministic function)")
    bit_gen_cls = getattr(np.random, d["bit_generator"])
    if d["kind"] == "seed_sequence":
        return np.random.Generator(bit_gen_cls(_seed_sequence_from(d)))
    if d["kind"] == "generator":
        ss_desc = d.get("seed_sequence")
        bg = bit_gen_cls(_seed_sequence_from(ss_desc)) if ss_desc is not None else bit_gen_cls()
        bg.state = _destringify(d["state"])
        return np.random.Generator(bg)
    raise ValueError(f"unknown rng description kind {d['kind']!r}")
