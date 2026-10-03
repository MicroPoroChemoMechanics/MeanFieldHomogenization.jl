# [Performance notes](@id dev-performance-notes)

!!! info "Before this page"
    [Adding a new algorithm](@ref dev-adding-algorithm), for the dispatch
    between the `Analytical`, `Residue` and cubature paths compared below.

This page describes what keeps the hot paths of the package fast, how their
cost is measured, and which path dominates a computation.

## What the hot paths do

- Small-matrix helpers (`_inv3` in `src/Core/green_kernel.jl`,
  `_sym3_inv_acoustic` in `src/Elasticity/hill_3d_aniso_nestedquadgk.jl`,
  `_A_and_Tn` / `_phi_cache` / `_qnn_pair_components` in
  `src/Core/green_helpers.jl`) are **pure functions returning `StaticArrays`**,
  and invert ``3\times 3`` matrices in closed form rather than through an LU
  factorization. They run once per quadrature node, so allocation there
  dominates everything else: keep new helpers on this path allocation-free and
  non-mutating.
- The Hill-tensor builders return the **most specific** TensND type they can —
  `TensISO`, `TensTI{4}`, `TensOrtho` rather than a generic `Tens`. This is not
  cosmetic: the specificity propagates into the homogenization schemes, where it
  removes redundant symmetry checks and lets the structured storage classes take
  over. Returning a generic tensor from a new builder silently deoptimizes every
  scheme downstream.
- `ForwardDiff.Dual` propagates through the nested `QuadGK` paths. The AD path
  must stay clear of `PolynomialRoots`, which is why `Residue` is `Float64`-only
  and `:auto` routes `Dual` inputs elsewhere.

## Measuring

- `scripts/bench_echoes/` — timings against Echoes:
  [Performance vs Echoes](@ref dev-benchmarks); accuracy:
  [Cross-validation](@ref dev-validation).
- `scripts/bench/` — Julia-only regression gating: 67 cases, committed
  baseline, bitwise checksum gate (`bench_suite.jl --help`).

!!! warning "The Echoes benchmark needs its extension imports"
    `scripts/bench_echoes/benchmark.jl` exercises `jl_method = :decuhr`, which
    lives in a package extension. Without `import DECUHR, Integrals` in scope
    the run dies with *"the `:decuhr` backend requires the DECUHR extension"* —
    even though both packages are in that environment's Manifest. Same
    requirement as `test/runtests.jl`.

## A pitfall worth knowing: closure boxing

Extracting a few scalars into a local variable used inside a nested closure —
the usual "hoist it out of the loop" reflex — can **increase** allocations in
this codebase rather than reduce them. Julia may box a local captured by an inner
closure when it cannot prove the binding is never reassigned, and the box is a
heap allocation per call.

The reliable fix is to extract the inner computation into a **top-level
function** taking the scalars as arguments, instead of hoisting them into a
closure's captured scope.

The practical consequence for anyone optimizing here: **do not trust a single
measurement**. Cross-check with more than one method — `@allocated` on a warmed
call, `@benchmark` from BenchmarkTools, and a before/after diff on a realistic
workload — because the effect is counter-intuitive and easy to attribute to the
wrong change.

## Where the cost actually is

Rough ordering, useful for deciding what to optimize:

| Path | Cost |
| :--- | :--- |
| `Analytical` (isotropic, conduction, coaxial TI) | ``O(1)``, no quadrature |
| `Residue` (3-D anisotropic, `:residues` only) | ~4 ms; one polynomial solve + 1-D quadrature per call. The fastest path, but not the default: it degenerates on a reference anisotropic in type and isotropic in value |
| `DECUHR` / `NestedQuadGK` (3-D anisotropic `:auto`) | ~11 ms / ~31 ms; adaptive cubature, AD-safe, robust everywhere. `DECUHR` is picked when its extension is loaded |
| ALV (ageing viscoelasticity) | dominates everything: operators are ``(B n)\times(B n)`` block-triangular matrices, so cost grows with the *square* of the time-grid length. The structured kernel classes (`ALVKernelISO`, `ALVKernelTI`, `ALVKernelOrtho`) exist precisely to cut the storage and the Volterra algebra from ``36n^2`` down to ``2n^2``–``12n^2``; see [Ageing linear viscoelasticity](@ref th-viscoelasticity). |
| Spheroid coupling matrices | quadrature backend is `Float64` and converges to machine precision; the `:series` BigFloat backend is an *oracle*, orders of magnitude slower, and should never be a default |

## Where to go next

[Performance vs Echoes](@ref dev-benchmarks) gives the measured timings of
these paths against the C++ reference.
