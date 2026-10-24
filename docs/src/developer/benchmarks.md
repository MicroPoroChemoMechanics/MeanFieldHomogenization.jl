# [Performance vs Echoes](@id dev-benchmarks)

!!! info "Before this page"
    [Performance notes](@ref dev-performance-notes), for the paths timed below,
    and [Cross-validation against Echoes](@ref dev-validation), for the accuracy
    of the same runs.

Every number on this page is read from a JSON file written by a script of the
repository, and can be measured again by anyone with the commands of the last
section. Both codes run on the same machine, one thread each, timed by
BenchmarkTools' `@belapsed` (the minimum of the samples); Echoes is called
through its Python API, from Python modules that also run on their own.
``t(E)/t(J) > 1`` means that `MeanFieldHomogenization` is faster; a ratio below
1 is in bold. The *gap* is the largest relative difference between the two
results, so that each ratio compares two computations of the same number.

## Kernels

```@eval
using MeanFieldHomogenization
include(joinpath(pkgdir(MeanFieldHomogenization), "docs", "benchmark_tables.jl"))
machine_line("benchmark_kernels.json")
```

### Hill tensor ``\mathbb{P}``

The Julia back-end is the default one unless the case names another.

```@eval
using MeanFieldHomogenization
include(joinpath(pkgdir(MeanFieldHomogenization), "docs", "benchmark_tables.jl"))
kernel_table("hill")
```

### Crack compliance ``\mathbb{H}``

```@eval
using MeanFieldHomogenization
include(joinpath(pkgdir(MeanFieldHomogenization), "docs", "benchmark_tables.jl"))
kernel_table("crack")
```

### Conductivity ``\mathbb{P}^{(2)}``

```@eval
using MeanFieldHomogenization
include(joinpath(pkgdir(MeanFieldHomogenization), "docs", "benchmark_tables.jl"))
kernel_table("conduction")
```

### Derivative ``\partial\mathbb{P}/\partial\mathbb{C}``

ForwardDiff through [`hill_tensor`](@ref) on the Julia side, Echoes'
`hill_derivative` on the other (a closed form for an isotropic medium).

```@eval
using MeanFieldHomogenization
include(joinpath(pkgdir(MeanFieldHomogenization), "docs", "benchmark_tables.jl"))
kernel_table("derivative")
```

## Schemes

Whole homogenizations of the same RVEs, with the same tolerances on both sides:
numerical Hill tensors to a relative ``10^{-6}`` (absolute ``10^{-8}``), fixed
points to a relative ``10^{-8}``. The aging viscoelastic cases discretize a
Maxwell matrix on 11 instants.

```@eval
using MeanFieldHomogenization
include(joinpath(pkgdir(MeanFieldHomogenization), "docs", "benchmark_tables.jl"))
machine_line("benchmark_schemes_speed.json")
```

```@eval
using MeanFieldHomogenization
include(joinpath(pkgdir(MeanFieldHomogenization), "docs", "benchmark_tables.jl"))
scheme_table()
```

The differential scheme is not the same algorithm on both sides: Echoes takes
1000 explicit first-order steps, `MeanFieldHomogenization` integrates adaptively
to a relative ``10^{-8}``, hence a gap of a few ``10^{-5}``, the error of the
first-order steps.

### The three-scale strength model: not a like-for-like timing

The model of [Quasi-brittle strength of cement paste and mortar](@ref app-strength) computes a strength and its
derivative with respect to the shear modulus of one family of hydrate needles.
The two codes do not take the same route, and the ratio below compares routes
rather than codes. Echoes converges the self-consistent estimate with the
needles averaged isotropically (closed-form Hill tensors), then obtains the
derivative from a single step over the 20 transversely isotropic families, with
numerical Hill tensors to ``10^{-3}``. `MeanFieldHomogenization` converges the
self-consistent estimate over the 20 families themselves, with dual numbers,
every Hill tensor to ``10^{-6}``.

```@eval
using MeanFieldHomogenization
include(joinpath(pkgdir(MeanFieldHomogenization), "docs", "benchmark_tables.jl"))
scheme_table(; strength = true)
```

## What the numbers say

- **Closed forms against Echoes' calls.** On an isotropic medium both codes
  evaluate a closed form, and a Hill tensor costs tens of nanoseconds in Julia
  against tens of milliseconds through Echoes' Python API; the isotropic
  schemes inherit the factor.
- **Same algorithm, comparable speed.** Residues against residues on a
  triclinic medium, and the anisotropic conductivity, differ by a factor 2 to
  2.5. The one case where Echoes is faster, `:decuhr` on the rotated triclinic
  ellipsoid against Echoes' residues, is a cubature against residues.
- **The cubatures are the fastest general back-ends.** `:nestedquadgk` agrees
  with NUMINT3D to ``10^{-9}`` and is 80 to 400 times faster.
- **Derivatives by automatic differentiation cost no more than closed forms.**
  ForwardDiff through `hill_tensor` beats Echoes' closed-form derivative on the
  isotropic cases.
- **Aging viscoelasticity**: 3.6 to 10 times faster on 11 instants, the
  self-consistent schemes in the cheapest algebra of the reference kernel.

## Reproducing

The Julia scripts call the Echoes side through PyCall, from Python modules
next to them; Echoes must be importable as `import echoes` from the
interpreter PyCall uses (see `scripts/bench_echoes/README.md`). From the root
of the repository, alone on the machine:

```bash
julia --project=scripts/bench_echoes scripts/bench_echoes/benchmark.jl
julia --project=scripts/bench_echoes scripts/bench_echoes/benchmark_schemes_speed.jl
```

They write `scripts/bench_echoes/figures/benchmark_kernels.json` and
`benchmark_schemes_speed.json`, which this page reads. The Echoes side runs on
its own as well, with the cases defined once in these modules:

```bash
python3 -I scripts/bench_echoes/kernels_echoes.py
python3 -I scripts/bench_echoes/schemes_speed_echoes.py
```

with `python3` the interpreter in which `import echoes` works.

Internal (Julia-only) regression benchmarking is separate: `scripts/bench/`
holds a gated suite with a committed baseline and a bitwise checksum gate,
described in its own `README.md`.

## Where to go next

[Roadmap](@ref dev-roadmap) lists what remains to be implemented.
