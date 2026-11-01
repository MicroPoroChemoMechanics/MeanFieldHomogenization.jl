"""The Echoes references of the test suite for the ALV schemes.

Writes `test/Viscoelasticity/echoes_alv_cells.json`, read by
`test/Viscoelasticity/test_alv_echoes_cells.jl`: every case of `cases.py` whose name
starts with `g` (the asymmetric self-consistent scheme) or `h` (every scheme of Echoes
on complex cells, at orders 4 and 2), run with Echoes on 4 instants of the span of the
case so that the file stays small. The test rebuilds each cell with `cells.jl`, as
`run_mfh.jl` does.

Run with the Python of the environment where echoes is installed:

    python3 -I make_test_references.py
"""

import json
import os
import pathlib
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import run_echoes as R  # noqa: E402
from cases import CASES  # noqa: E402

N = 4
OUT = pathlib.Path(HERE).parents[2] / "test" / "Viscoelasticity" / "echoes_alv_cells.json"

rows = []
for c in CASES:
    if not (c["name"].startswith("g") or c["name"].startswith("h_")):
        continue
    t = c["times"]
    c = dict(c, times=[float(x) for x in np.geomspace(t[0], t[-1], N)])
    ver, prop, u, times, _, keep = R.make_rve(c, {})
    eff = R.homogenize(c, ver, prop, times, u, c["maxnb"])
    if c["scheme"] == "differential":
        ver2, *_ = R.make_rve(c, {})
        eff = 2 * R.homogenize(c, ver2, prop, times, u, 2 * c["maxnb"]) - eff
    rows.append(dict(case=c, effective=eff.tolist()))
    print(f"{c['name']:40s} {eff.shape}", flush=True)

OUT.write_text(json.dumps(dict(generator="scripts/bench_echoes/alv_harness/make_test_references.py",
                               instants=N, cases=rows)))
print(f"wrote {OUT} ({len(rows)} cases)")
