# `scripts/bench` — optimization-campaign benchmark suite

Two things live here:

* **`bench_alv.jl`, `bench_sc_solvers.jl`** — the pre-existing hand-rolled
  quick-look scripts (`@elapsed` / `@allocated`, 3 warm-ups + 5 samples, min
  reported). Unchanged, still useful for a one-off glance.
* **`bench_suite.jl` + `harness.jl` + `cases_*.jl`** — the campaign suite:
  a registry of cases, three independent measurement channels, a committed
  baseline, and a gated diff.

## Why a new harness

The hand-rolled convention cannot support this campaign, for three reasons —
all of them visible in `bench_sc_solvers.jl`:

1. **It cannot resolve the microsecond tier.** `@elapsed f()` measures exactly
   one call. `_sym3_inv_acoustic` and a `TensISO ⊡ TensISO` run in
   10–200 ns, below clock granularity. `BenchmarkTools` auto-tunes the number
   of evaluations per sample so such a kernel is measured against a window
   several orders of magnitude above the timer's resolution.
2. **It reports allocations from a different call than the one it timed.**
   The helper does `s_alloc = @allocated f()` and then `t = @elapsed f()` —
   two separate invocations — and pairs the two numbers in its output. That
   pairing is meaningless for an allocation investigation.
3. **It reports only the minimum.** The failure mode this campaign has to
   guard against is precisely a min/total confusion (see below).

## Why `BenchmarkTools` is not enough on its own

`@benchmark` requires `$`-interpolation of its arguments to avoid measuring
global-lookup cost. But `$`-interpolation of a pre-built closure **hoists the
capture out of the measured region** — which is exactly what would hide a
closure-boxing regression.

This is not hypothetical. Earlier in this project, replacing a
`Matrix`+`inv` by twelve local scalars inside a nested quadrature closure made
the per-node computation ~10× cheaper in isolation and **increased total
allocation by 24 %** (10.22 MB → 12.69 MB), because a deeply nested closure
boxed differently depending on the number of its own locals. The fix was to
extract the computation into a top-level function with no captures
(→ 8.34 MB). A single `@allocated` on a fresh process reported the *opposite*
of the truth.

## The three channels

| Channel | How | Answers |
|---|---|---|
| `time` | `BenchmarkTools.@benchmark`, min + median, auto-tuned `evals` | how fast |
| `alloc` | hand-rolled loop of `@allocated` on the **verbatim** thunk, no `$`, ≥2 warm-ups + N samples in the *same* process, **min and max** | how much garbage, and is it stable |
| `counters` | instrumented `Ref{Int}` counters (`src/Core/counters.jl`) read around a *separate* clean call | how much **work** |

Reporting allocation **max as well as min** is the cheap upgrade that catches
the failure above: `max ≠ min` signals a type instability, and it immediately
invalidates any "one measurement on a fresh process" reading.

The `counters` channel distinguishes *faster* from *did less work*. A drop in
adaptive-quadrature node count is a change of behavior (and of accuracy), not
a speed-up. A tier that claims a 2× win must show the node count unchanged and
the Hill-call count halved — not the other way round.

## Instrumentation is free

Counting integrand evaluations would normally perturb the innermost loop.
`Core._counted_quadgk` branches on `COUNT_INTEGRAND[]` **once per `quadgk`
call**, and each side passes a concretely typed callable (`_CountingFn{F}`, a
struct — not a closure returned from a branch, which would make the branch's
return type a small `Union` and cost a dynamic dispatch per node).

Measured against the uninstrumented tree (a `git worktree` at the pre-campaign
commit), same workloads, same machine:

| case | time (before → after) | allocations |
|---|---|---|
| `hill_tensor` triaxial/triclinic, `:nestedquadgk` | 78.601 → 78.572 ms | 103 363 248 B → **identical** |
| `hill_tensor` triaxial/triclinic, `:residues` | 4.278 → 4.296 ms | 7 377 152 B → **identical** |
| `cod_tensor` penny/triclinic, `:residues` | 2.729 → 2.716 ms | 3 577 360 → 3 577 280 B (−0.002 %) |
| `hill_tensor` iso sphere (analytical) | 0 → 0 | 0 → 0 |

The timed pass asserts `!COUNT_INTEGRAND[]`, so it can never see the wrapper.

## The noise floor

`--repeat-suite=2` runs the whole registry twice and computes, **over the
control cases only**, `noise = p90(|Δt|/t)`. A case is reported as MOVED only
beyond `max(3·noise, 3 %)`; a *control* case moving that far makes the whole
run invalid (exit 2).

Control cases are chosen so that no planned change can touch them: analytical
Hill branches, the elliptic integrals, Voigt/Reuss, and the schemes that call
only one of the two de-duplicated helpers (`Dilute`, `DiluteDual`, `Maxwell`,
`DifferentialScheme`).

This is what makes the comparison a measurement rather than a table of
numbers: it has a calibrated null hypothesis.

## Numerical gate

Every case carries a `checksum` closure evaluated on **the same call that is
timed**, so non-regression is asserted on the real code path.

* `--gate=bitwise` — sha256 of the canonical `%.17g` rendering. `%.17g` is
  exactly round-trippable for `Float64`, so this is a true bit-identity test
  *and* the values stored in the report are exact (old reports can be
  re-diffed without re-running).
* `--gate=1e-14` — scale-relative bound
  `max|new−ref| ≤ tol · max(1, max|ref|)`. The relative form is required:
  component magnitudes across the suite span `C ~ O(200)`, `A_εε ~ O(1)` and
  a stiff-matrix `H ~ O(1e-3)`.

Iterative schemes are pinned (`abstol = reltol = 1e-10`, `maxiters = 300`,
`select_best = true`) so the iteration count is part of the contract via the
`sc_iterations` counter.

## Running

```shell
julia --project=scripts/bench -e 'using Pkg; Pkg.instantiate()'

# capture the reference (refuses a dirty tree unless --force)
JULIA_NUM_THREADS=1 julia --project=scripts/bench scripts/bench/bench_suite.jl \
    --record-baseline --label=P0-baseline --repeat-suite=2

# after an optimization tier
JULIA_NUM_THREADS=1 julia --project=scripts/bench scripts/bench/bench_suite.jl \
    --label=P1-dedup --out=scripts/bench/results/P1.json \
    --baseline=scripts/bench/baseline.json --gate=bitwise --repeat-suite=2

# correctness only, no timing (fast)
julia --project=scripts/bench scripts/bench/bench_suite.jl --verify-only

# re-diff two stored reports
julia --project=scripts/bench scripts/bench/bench_suite.jl \
    --diff scripts/bench/baseline.json scripts/bench/results/P1.json
```

Other flags: `--filter=<group|tag|id-substring>`, `--samples=N`,
`--seconds=S`, `--warmups=N`, `--shuffle`, `--tol=1e-14`, `--force`.

Exit codes: `0` clean · `1` checksum-gate failure · `2` control regression or
invalid run · `3` setup error.

**Run conditions.** `JULIA_NUM_THREADS=1`, no other load on the machine, and
never concurrently with a test suite or another Julia process — the timing
channel is wall-clock and will happily measure your other work.

## Files

| File | Contents |
|---|---|
| `harness.jl` | `BCase`, the 5-step runner, checksums, JSON I/O, diff, noise floor |
| `fixtures.jl` | shared stiffnesses and RVE builders, copied verbatim from `test/` and `scripts/` |
| `cases_kernels.jl` | Hill / COD back-ends and per-node primitives |
| `cases_schemes.jl` | `homogenize` for every scheme + ForwardDiff sensitivities |
| `cases_alv.jl` | ageing linear viscoelasticity (O(n²) in time steps) |
| `cases_tensnd.jl` | TensND primitives — every MFH case is downstream of these |
| `baseline.json` | **committed** reference report |
| `results/` | per-tier reports (gitignored) |

Fixtures are copied verbatim rather than simplified on purpose: a simplified
extract can take a different branch from the real path, and then the
before/after comparison measures something that does not exist.
