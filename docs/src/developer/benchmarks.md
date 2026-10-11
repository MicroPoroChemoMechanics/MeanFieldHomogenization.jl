# [Performance vs Echoes](@id dev-benchmarks)

!!! info "Before this page"
    [Performance notes](@ref dev-performance-notes), for the paths timed below,
    and [Cross-validation against Echoes](@ref dev-validation), for the accuracy
    of the same runs.

Timings from `scripts/bench_echoes/benchmark.jl`: `@belapsed` (the minimum of
20 samples) on both sides, same machine, one thread each, Echoes called through
its Python API. `t(E)/t(J)` > 1 means `MeanFieldHomogenization` is faster.
Measured with MeanFieldHomogenization 0.16.0, TensND 0.6.0 and DECUHR 0.3.0 on
Julia 1.13. Accuracy for the same runs is in [Cross-validation](@ref dev-validation).

## Hill tensor ``\mathbb P``

| Case | Julia back-end | Echoes | ``t(J)`` | ``t(E)`` | ``t(E)/t(J)`` |
| :--- | :--- | :--- | ---: | ---: | ---: |
| sphere / ISO | analytic | analytic | 210 ns | 71.1 ms | 337 943× |
| prolate 5:1:1 / ISO | analytic | analytic | 316 ns | 106 ms | 334 238× |
| prolate 5:1:1 / ISO, rotated | analytic | analytic | 190 ns | 106 ms | 559 565× |
| prolate 3:1:1 / cubic | `:residues` | NUMINT3D | 4.20 ms | 96.9 ms | 23.1× |
| prolate 3:1:1 / cubic | `:decuhr` | NUMINT3D | 1.23 ms | 94.9 ms | 77.4× |
| prolate 3:1:1 / cubic | `:nestedquadgk` | NUMINT3D | 130 µs | 96.8 ms | 745× |
| triaxial 2:1:0.5 / triclinic | `:residues` | RESIDUES | 4.21 ms | 5.02 ms | **1.19×** |
| triaxial 2:1:0.5 / triclinic | `:decuhr` | RESIDUES | 4.51 ms | 5.01 ms | **1.11×** |
| triaxial 2:1:0.5 / triclinic | `:nestedquadgk` | RESIDUES | 936 µs | 5.02 ms | 5.36× |
| triaxial 2:1:0.5 / triclinic | `:nestedquadgk` | NUMINT3D | 936 µs | 88.7 ms | 94.7× |
| triaxial 2:1:0.5 / triclinic, rotated | `:residues` | RESIDUES | 3.00 ms | 3.13 ms | **1.04×** |

## Crack compliance ``\mathbb H``

| Case | Julia back-end | Echoes | ``t(J)`` | ``t(E)`` | ``t(E)/t(J)`` |
| :--- | :--- | :--- | ---: | ---: | ---: |
| penny / ISO | analytic | analytic | 2.65 µs | 56.2 ms | 21 248× |
| penny / cubic | `:residues` | NUMINT3D | 2.82 ms | 62.0 ms | 22.0× |
| penny / triclinic | `:residues` | RESIDUES | 2.95 ms | 3.68 ms | **1.24×** |

## Conductivity ``\mathbb P^{(2)}`` and derivative ``\partial\mathbb P/\partial\mathbb C``

| Case | ``t(J)`` | ``t(E)`` | ``t(E)/t(J)`` |
| :--- | ---: | ---: | ---: |
| sphere / iso ``\mathbb K`` | 80.1 ns | 9.77 µs | 122× |
| prolate 5:1:1 / iso ``\mathbb K`` | 1.73 µs | 10.0 µs | 5.79× |
| triaxial 2:1:0.5 / aniso ``\mathbb K`` | 4.36 µs | 14.8 µs | 3.39× |
| ``\partial\mathbb P/\partial\mathbb C``, sphere / ISO (ForwardDiff vs closed form) | 1.15 µs | 12.1 µs | 10.6× |
| ``\partial\mathbb P/\partial\mathbb C``, ORTHO, per component (NUMINT3D) | 3.6–3.7 ms | 79–125 ms | 21–34× |

## What the numbers say

- **Same algorithm ⇒ same speed.** Residues against residues on a triclinic
  matrix: 1.19×, 1.04× rotated, 1.24× for the crack. Neither implementation
  has a systematic edge.
- **The orders of magnitude come from dispatch, not from Julia.** Isotropic and
  transversely-isotropic matrices take a closed-form branch
  ([`hill_tensor`](@ref) with `method = :auto`), where Echoes still integrates.
- **The cubatures are the fastest general back-ends.** `:nestedquadgk` agrees
  with NUMINT3D to 8e-9 on the triclinic case and runs 95 times faster, and
  4.5 to 5.4 times faster than the two residue implementations; `:decuhr`
  matches the residues and runs 20 to 77 times faster than NUMINT3D.
- **AD costs nothing over a hand-coded derivative.** `ForwardDiff` through
  `hill_tensor` beats Echoes' closed-form ``\partial\mathbb P/\partial\mathbb C``
  by 10.6× on the isotropic sphere.
- The script passes `method` as a keyword holding a run-time `Symbol`, which
  costs a dynamic dispatch: about 110 ns on the closed forms, the same in 0.15.
  Called without it, the sphere takes 124 ns.

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
