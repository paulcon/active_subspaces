"""Fixed seeds for statistical checks (spec 17 §5.2, [D-78], [D-159]).

Tests do not call these directly: they take the ``stat_rng`` fixture of
``tests/conftest.py``, which honours ``--stat-seeds=fixed|random|<int>``::

    def test_x(stat_rng):
        X = space.sample(100, rng=stat_rng("data"))          # check_seed(nodeid, "data")

    def test_uc01(stat_rng):
        # ACCEPTANCE_SEEDS["UC-01/sample"] == 101
        X = U5.sample(1000, rng=stat_rng("UC-01/sample"))

``stat_rng(label)`` returns a fresh ``np.random.SeedSequence`` whose entropy is a
single integer on every call (pass ``int(ss.entropy)`` where an ``int`` seed is
wanted). A seed is never changed to make a check pass [D-78].
"""

from __future__ import annotations

import zlib
from collections.abc import Mapping
from types import MappingProxyType

import numpy as np

__all__ = ("ACCEPTANCE_SEEDS", "check_seed", "stat_seed")

ACCEPTANCE_SEEDS: Mapping[str, int] = MappingProxyType(
    {
        "UC-01/sample": 101,
        "UC-01/bootstrap": 102,
        "UC-02/sample": 201,
        "UC-02/bootstrap/0": 0,
        "UC-02/bootstrap/1": 1,
        "UC-03/ols": 301,
        "UC-03/qphd": 302,
        "UC-03/sir": 303,
        "UC-03/phd": 304,
        "UC-04/sample": 401,
        "UC-06/sample": 601,
        "UC-06/new_points": 602,
        "UC-06/noise": 603,
        "UC-07/bounded": 701,
        "UC-07/gaussian": 702,
        "UC-07/replicates": 703,
        "UC-08/sample": 801,
        "UC-08/bootstrap": 802,
        "UC-08/minimize": 803,
        "UC-09/build": 901,
        "UC-09/average": 902,
        "UC-09/probability": 903,
        "UC-09/minimum": 904,
        "UC-09/test_points": 905,
        "UC-09/data_path": 906,
        "UC-11/sample": 1101,
        "UC-11/bootstrap": 1102,
        "UC-11/gradient_points": 1103,
    }
)


def check_seed(nodeid: str, label: str = "") -> int:
    """``crc32(f"{nodeid}|{label}") & 0x7FFFFFFF``; ``nodeid`` without parameter brackets
    (§5.2).
    """
    return zlib.crc32(f"{nodeid.split('[')[0]}|{label}".encode()) & 0x7FFFFFFF


def stat_seed(nodeid: str, label: str = "", entropy: int | None = None) -> int:
    """The integer seed of one ``stat_rng`` stream.

    Fixed mode (``entropy is None``): ``ACCEPTANCE_SEEDS[label]`` when ``label`` is a key, else
    ``check_seed(nodeid, label)``. Random mode: a 63-bit integer derived from the run's entropy
    and the stream's name, so ``--stat-seeds=<entropy>`` replays the whole run.
    """
    if entropy is None:
        return ACCEPTANCE_SEEDS[label] if label in ACCEPTANCE_SEEDS else check_seed(nodeid, label)
    words = np.random.SeedSequence([int(entropy), check_seed(nodeid, label)]).generate_state(
        2, np.uint32
    )
    return (int(words[0]) << 31) ^ int(words[1])
