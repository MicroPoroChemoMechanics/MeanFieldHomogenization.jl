# `scripts/bench_echoes/` — MeanFieldHomogenization.jl vs Echoes (C++) benchmark

Side-by-side numerical + timing comparison between the Julia implementation
and the Echoes C++ reference called from Python via `PyCall.jl`.

## Prerequisites

Echoes must already be importable from a Python interpreter on this
machine (the Boost.Python build works; no recompilation needed here).

```python
# Must succeed in the target interpreter:
>>> import echoes
```

The first time you run this benchmark, point PyCall at that interpreter
and rebuild:

```julia
julia> ENV["PYTHON"] = raw"C:\path\to\python.exe"   # or /usr/bin/python3 on Linux
julia> using Pkg; Pkg.activate("."); Pkg.build("PyCall")
```

## Run

From the package root `MeanFieldHomogenization.jl/`:

```bash
julia --project=scripts/bench_echoes scripts/bench_echoes/benchmark.jl
```

Two summary tables are printed at the end:

- Hill tensor **P** — 5 cases (iso sphere / prolate / oblate, cubic prolate,
  triclinic triaxial).
- Crack compliance **ΔS** (ε = 1) — 3 cases (penny in iso / cubic /
  triclinic matrix).

Each Echoes reference is computed with both `RESIDUES` and `NUMINT3D`
(DECUHR) algorithms.  `RESIDUES` occasionally fails (Masson polynomial
degenerates on perfectly isotropic stiffness) — such cases are reported
as `FAIL` in the summary, and `NUMINT3D` is the robust fallback.

Timings are collected with `BenchmarkTools.@belapsed`.

## Notes

- This subfolder has its **own Project.toml** — it does not pollute the
  main `MeanFieldHomogenization.jl` environment with PyCall / BenchmarkTools.
- `:residues` is used on both sides for the cubic and triclinic cases
  (same mathematical algorithm → expected agreement at machine
  precision).
- For the isotropic cases, the Julia side uses the analytic path
  (`hill_tensor` with `:auto`) and the Python side uses `RESIDUES` +
  `NUMINT3D` — a small discrepancy (~1e-6 to 1e-8) is expected.

## Benchmark & cross-validation scripts (keep, run any time)

The four production cross-checks against echoes:

| Script | echoes reference | Validated quantities | Tolerance |
|---|---|---|---|
| `benchmark.jl` | Hill P / crack ΔS (residue, DECUHR) | P, ΔS on 5+3 cases | ~1e-6 (iso), machine (aniso) |
| `benchmark_nlayers.jl` | n-layer sphere reference | n-layer sphere averages / local fields | 1e-6 |
| `benchmark_porous.jl` | porous benchmark | porous SC / MT moduli | 1e-6 |
| `benchmark_strength.jl` | Pichler et al. (CCR 2011) mortar model | k, μ, E (mortar) + strength fc | **moduli 1 %, fc 2 %** |
| `benchmark_hill_derivative.jl` | Hill-derivative reference | ∂P/∂C (ISO analytical, ORTHO NUMINT3D) | ISO ~1e-15, ORTHO ~1e-6 |

`benchmark_hill_derivative.jl` compares the reference analytical / numerical
`hill_derivative` against MeanFieldHomogenization's ForwardDiff through `hill_tensor`,
across ellipsoid shapes, for ISO and ORTHO reference media, plus a fully
triclinic reference that has no counterpart there (MFH ForwardDiff only). The
standalone numbered demo `scripts/08_hill_derivatives.jl` shows the same MFH
capability without PyCall (validated against finite differences).

`benchmark_strength.jl` is built on the shared model `../common/quasibrittle_strength.jl`
(public MeanFieldHomogenization API only — multi-bin Self-Consistent hydrate foam with
several `TISymmetrize` needle families whose EXACT azimuthal average
(`TensTI{4,T,8}`) flows through the generic SC kernel; strength sensitivity by
a single ForwardDiff pass through the whole three-scale chain). The former
hand-rolled Mandel/IFT bypass is gone, and the fc tolerance tightened from
15 % to 2 %.

The `bench_layered_alv*` jl/py/json triads are cross-validation assets for the
ageing-viscoelastic (ALV) layered-sphere recurrences; the committed
`*_python.json` dumps let the Julia side be checked WITHOUT a live echoes /
PyCall install.

## Debug / development scripts (kept, documented, re-runnable)

These are the narrowly-scoped scripts written while settling the Volterra /
ALV numerics. They are intentionally kept — each answers a specific "why does
X behave this way?" question and can be re-run to reproduce the finding.

| Script | Question it answers | Re-run | Needs |
|---|---|---|---|
| `debug_n4_elastic.jl` | Does the N=4 layered-sphere elastic limit match Eshelby? | `julia --project=. debug_n4_elastic.jl` | — |
| `debug_n_sweep.jl` (+ `.py`, `_python.json`) | How does the layered-ALV result converge with the number of layers N? | `julia --project=. debug_n_sweep.jl` | json (or PyCall for regen) |
| `debug_volterra_divide.jl` | Is `volterra_divide` consistent with `volterra_inverse ∘ product`? | `julia --project=. debug_volterra_divide.jl` | — |
| `bench_step_trap.jl` | Trapezoidal-step convergence of a Heaviside-loaded creep kernel | `julia --project=. bench_step_trap.jl` | — |
| `bench_step_*` / `bench_step_n2.*` | Single-step & N=2 kernel cross-checks vs echoes Python dumps | `julia --project=. bench_step_n2.jl` | `*_python.json` |
| `bench_volterra_lapack.jl` | LAPACK vs block-forward-substitution Volterra inverse timing/accuracy | `julia --project=. bench_volterra_lapack.jl` | — |
| `bench_iso_fastpath.jl` | Structured-ISO ALV fast path vs dense 6n×6n | `julia --project=. bench_iso_fastpath.jl` | — |
| `bench_ellipsoid2.jl`, `bench_order2_alv.jl`, `derive_shear_transition.py` | ellipsoid-2 / order-2 ALV and the shear-transition-matrix derivation | see file header | some need PyCall |

To regenerate a `*_python.json` reference dump, run the paired `.py` file in a
Python interpreter where `echoes` imports (see Prerequisites).
