# [Performance vs Echoes](@id dev-benchmarks)

!!! info "Before this page"
    [Performance notes](@ref dev-performance-notes), for the paths timed below,
    and [Cross-validation against Echoes](@ref dev-validation), for the accuracy
    of the same runs.

Timings from `scripts/bench_echoes/benchmark.jl`: `@belapsed` on both sides,
same machine, Echoes called through its Python API. `t(E)/t(J)` > 1 means
`MeanFieldHomogenization` is faster. Accuracy for the same runs is in
[Cross-validation](@ref dev-validation).

## Hill tensor ``\mathbb P``

| Case | Julia back-end | Echoes | ``t(J)`` | ``t(E)`` | ``t(E)/t(J)`` |
| :--- | :--- | :--- | ---: | ---: | ---: |
| sphere / ISO | analytic | analytic | 109 ns | 72.9 ms | 666 883× |
| prolate 5:1:1 / ISO | analytic | analytic | 308 ns | 108 ms | 351 014× |
| prolate 5:1:1 / ISO, rotated | analytic | analytic | 740 ns | 104 ms | 141 083× |
| prolate 3:1:1 / cubic | `:residues` | NUMINT3D | 4.41 ms | 95.8 ms | 21.7× |
| triaxial 2:1:0.5 / triclinic | `:residues` | RESIDUES | 4.40 ms | 5.01 ms | **1.14×** |
| triaxial 2:1:0.5 / triclinic | `:nestedquadgk` | NUMINT3D | 77.3 ms | 85.9 ms | **1.11×** |
| triaxial 2:1:0.5 / triclinic | `:decuhr` | RESIDUES | 17.9 ms | 5.00 ms | 0.28× |

## Crack compliance ``\mathbb H``

| Case | Julia back-end | Echoes | ``t(J)`` | ``t(E)`` | ``t(E)/t(J)`` |
| :--- | :--- | :--- | ---: | ---: | ---: |
| penny / ISO | analytic | analytic | 26.2 µs | 55.7 ms | 2124× |
| penny / cubic | `:residues` | NUMINT3D | 3.74 ms | 61.4 ms | 16.4× |
| penny / triclinic | `:residues` | RESIDUES | 3.77 ms | 3.67 ms | **0.97×** |

## Conductivity ``\mathbb P^{(2)}`` and derivative ``\partial\mathbb P/\partial\mathbb C``

| Case | ``t(J)`` | ``t(E)`` | ``t(E)/t(J)`` |
| :--- | ---: | ---: | ---: |
| sphere / iso ``\mathbb K`` | 73.4 ns | 10.1 µs | 138× |
| triaxial 2:1:0.5 / aniso ``\mathbb K`` | 6.50 µs | 14.8 µs | 2.27× |
| ``\partial\mathbb P/\partial\mathbb C``, sphere / ISO (ForwardDiff vs closed form) | 904 ns | 12.5 µs | 13.8× |
| ``\partial\mathbb P/\partial\mathbb C``, ORTHO, per component (NUMINT3D) | 5.8 ms | 94–141 ms | 16–24× |

## What the numbers say

- **Same algorithm ⇒ same speed.** Residues against residues on a triclinic
  matrix: 1.14×, 0.97×, 0.98×. Neither implementation has a systematic edge.
- **The orders of magnitude come from dispatch, not from Julia.** Isotropic and
  transversely-isotropic matrices take a closed-form branch
  ([`hill_tensor`](@ref) with `method = :auto`), where Echoes still integrates.
- **`:decuhr` is slower by design** — an adaptive cubature kept as the
  type-generic, `ForwardDiff`-safe fallback, not a default.
- **AD costs nothing over a hand-coded derivative.** `ForwardDiff` through
  `hill_tensor` beats Echoes' closed-form ``\partial\mathbb P/\partial\mathbb C``
  by 13.8× on the isotropic sphere.

## Reproducing

```julia
import DECUHR, Integrals      # the :decuhr back-end lives in an extension
include("scripts/bench_echoes/benchmark.jl")
```

Internal (Julia-only) regression benchmarking is separate: `scripts/bench/`
holds a 67-case gated suite with a committed baseline and a bitwise checksum
gate, described in its own `README.md`.

## Where to go next

[Roadmap](@ref dev-roadmap) lists what remains to be implemented.
