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
  must stay clear of polynomial root finding (the eigenvalues of a companion
  matrix, by LAPACK), which is why `Residue` is `Float64`-only and `:auto`
  routes `Dual` inputs elsewhere.

## Measuring

- `scripts/bench_echoes/` — timings against Echoes:
  [Performance vs Echoes](@ref dev-benchmarks); accuracy:
  [Cross-validation](@ref dev-validation).
- `scripts/bench/` — Julia-only regression gating: 77 cases, committed
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

## Fast paths of the shapes and the self-consistent schemes

Two paths ran far slower than the arithmetic they perform: building a shape
whose semi-axes have to be sorted, and the step of a self-consistent iteration.
In both, the computation itself took a fraction of the time; the rest went to
lookups, allocations and dispatches decided at run time. Both were rewritten
in version 0.20.2.

!!! tip "The same numbers, to the bit"
    Both rewrites perform the same arithmetic operations in the same order as
    before. The benchmark suite checks it: every case of `scripts/bench`, the
    common cases below included, returns a checksum identical to the bit with
    the code of version 0.20.1. Nothing a user computes has moved, and neither
    has the work done: the counters of Hill tensors evaluated and of
    self-consistent iterations are the same.

The gains, on one thread of the same machine (minimum of the samples), with
the RVE or the shape built inside the timed call as a user builds it:

| Case of `scripts/bench` | 0.20.1 | 0.20.2 |
| :--- | ---: | ---: |
| `common/ctor.spheroid5`: `Spheroid(5.0)` | 1.48 µs | 0.45 µs |
| `common/hill.spheroid5.iso`: its Hill tensor, isotropic matrix | 1.41 µs | 0.57 µs |
| `common/hill.ellipsoid.iso`: triaxial ellipsoid, isotropic matrix | 0.85 µs | 0.38 µs |
| `common/rve.build`: a two-phase RVE | 0.85 µs | 0.46 µs |
| `common/mt.spheres`: Mori–Tanaka, spheres | 1.95 µs | 1.44 µs |
| `common/mt.spheroids5`: Mori–Tanaka, isotropic spheroids | 3.72 µs | 2.38 µs |
| `common/asc.spheres`: asymmetric self-consistent, spheres | 20.6 µs | 5.7 µs |
| `common/asc.spheroids5`: the same, isotropic spheroids | 23.1 µs | 8.6 µs |
| `schemes/sc.porous.sphere.phi30`: self-consistent, porous, spheres | 37.7 µs | 20.9 µs |

```julia
julia --project=scripts/bench scripts/bench/bench_suite.jl --filter=common
```

The anisotropic cubatures (`common/hill.ellipsoid.ortho`, the crack cases) are
unchanged: their cost is the integration, which these rewrites do not touch.

### Building a shape

`Ellipsoid(a, b, c)` stores its semi-axes in decreasing order and permutes the
vectors of its basis to match, so a prolate `Spheroid(5.0)`, declared with its
axis along ``\underline{e}_3``, is built on a permuted basis.

| | up to 0.20.1 | from 0.20.2 |
| :--- | :--- | :--- |
| order of the semi-axes | `sortperm` on a vector collected from the tuple | `Core._descending_order`, an insertion sort on the tuple; the same permutation, ties kept in order |
| permuted basis | `RotatedBasis` built from a matrix whose size, hence the dimension, was read at run time; orthogonality checked again | with TensND ≥ 0.8.3, built with its dimension as a type parameter, without re-checking the orthogonality of an orthonormal basis |
| type of the shape's basis | not inferred: every computation on the shape started with a dynamic dispatch | a union of the canonical and the rotated basis of dimension 3 |

!!! note "The gain needs TensND 0.8.3"
    The sort is in this package and works with any TensND it supports. The
    permuted basis is built by TensND: with an earlier version the result is
    the same, the constructor only slower.

### A self-consistent iteration

[`SelfConsistent`](@ref) and [`AsymmetricSelfConsistent`](@ref) solve a fixed
point whose step is a sum over the phases of their dilute concentration
tensors in the current estimate. Everything the step reads of the RVE except
that estimate is the same at every iteration.

| | up to 0.20.1 | from 0.20.2 |
| :--- | :--- | :--- |
| what a step read of the RVE | at each step, the list of phase names (a new vector), then the fraction, property and orientation average of each phase from dictionaries holding `Any` | a view of each phase (`Schemes._PhaseView`), gathered once per solve by `Schemes._phase_views` |
| call of the concentration kernel | a dynamic dispatch per phase and per step | static, in an isotropic reference; dynamic otherwise (see below) |
| specialization of the step | none: the step saw abstract types | the solver loop specializes on the views, behind one function barrier (`Schemes._solve_sc_on`) |
| conductivity and stiffness | two copies of each step, one per tensor order | one step per scheme form; `_compose` is the contraction of the order |

!!! warning "Compiled once per kind of RVE"
    Up to eight phases, the views go in a tuple, so that the step compiles for
    the shapes and properties of that RVE. The first estimate of a new kind of
    RVE therefore costs a compilation, which later ones reuse. Beyond eight
    phases (a discretized distribution of orientations) the views stay a
    vector, so that one method serves every length.

!!! warning "Why the kernel is called dynamically in an anisotropic reference"
    A fully inferable step lets inference reach down through the
    concentration kernel, the anisotropic Hill cubature included. With a crack
    family next to an inclusion, the second step has a generic anisotropic
    reference, and inferring that step took more than 4.8 GB, against 1.6 GB
    for the same kernel compiled on its own. In a reference that is not
    isotropic, the kernel is therefore called through an inference barrier,
    as it was before; its cost there, a cubature, dwarfs that of the dispatch.
    In an isotropic reference, the common case, the call is static and the
    step allocates nothing for the phases. In an anisotropic reference the
    dynamic call boxes its result: a porous self-consistent estimate with
    oblate spheroids (`schemes/sc.porous.oblate.phi15`) allocates 11 % more
    bytes than before, in the same time.

### The solver stays a damped Picard iteration

!!! info "Measured, not retained: Newton as the default"
    [`NewtonDefault`](@ref) converges in fewer iterations, but each of them
    computes a Jacobian by automatic differentiation. On the asymmetric
    self-consistent cases above, measured with version 0.20.1 (23 against
    20 µs), it was slower than the default damped Picard iteration
    ([`AndersonDefault`](@ref)), and the default is the solver that runs every
    case of the tests, tutorials and applications, the bifurcation of a porous
    medium at percolation included. The default is unchanged.

## Cracks in the schemes: checks that cost more than the computation

Until version 0.23.1, a Mori–Tanaka step with a penny crack in an isotropic
matrix spent most of its time checking symmetries it already had. The crack
compliance ``\mathbb{H} = k\,\underline{n}\stackrel{s}{\otimes}\boldsymbol{B}\stackrel{s}{\otimes}\underline{n}``
was formed as a product of components, through an array of 81 entries, and the
operations of the scheme with the isotropic matrix did the same; each time,
the result was checked at run time for the minor symmetries before being
stored. Two changes remove them:

- ``\mathbb{H}`` is read off the out-of-plane block of ``\boldsymbol{B}`` in
  the frame of the crack (`compliance_op_block`), whose Kelvin–Mandel matrix it
  is, and stored directly with its symmetries;
- TensND 0.8.5 keeps that storage through the sum, the difference and the
  double contraction with an isotropic tensor, and through the negation.

On one thread of the same machine (minimum of the samples), values unchanged to
rounding:

| Case of `scripts/bench` | 0.23.0 | 0.23.1 |
| :--- | ---: | ---: |
| `schemes/mt.crack.penny`: Mori–Tanaka, penny crack, isotropic matrix | 96 µs, 1.26 MB | 1.45 µs, 3.6 kB |
| `schemes/mt.crack.penny.tri`: the same, triclinic matrix | 369 µs, 680 kB | 302 µs, 58 kB |

```julia
julia --project=scripts/bench scripts/bench/bench_suite.jl --filter=mt.crack
```

With a triclinic matrix the opening displacement tensor, an integral over the
crack front, is what remains of the cost.

## Where the cost actually is

Rough ordering, useful for deciding what to optimize:

| Path | Cost |
| :--- | :--- |
| `Analytical` (isotropic, conduction, coaxial TI) | ``O(1)``, no quadrature |
| `Residue` (3-D anisotropic, `:residues` only) | ~3 ms; one eigenvalue solve and a residue sum per node of a 1-D quadrature. Not the default: it takes `Float64` coefficients only |
| `NestedQuadGK` / `DECUHR` (3-D anisotropic; `:auto` is `NestedQuadGK`) | ~1 ms / ~4 ms on a triaxial ellipsoid in a triclinic matrix ([Performance vs Echoes](@ref dev-benchmarks)); adaptive cubature, AD-safe, robust everywhere; `NestedQuadGK` is also the more accurate, ~1e-13 against 1e-11 to 1e-8 at the default tolerances |
| ALV (aging linear Volterra) | dominates everything: operators are ``(B n)\times(B n)`` block-triangular matrices, so cost grows with the *square* of the time-grid length. The structured kernel classes (`ALVKernelISO`, `ALVKernelTI`, `ALVKernelOrtho`) exist precisely to cut the storage and the Volterra algebra from ``36n^2`` down to ``2n^2``–``12n^2``; see [Aging linear Volterra behavior](@ref th-viscoelasticity). |
| Spheroid coupling matrices | quadrature backend is `Float64` and converges to machine precision; the `:series` BigFloat backend is an *oracle*, orders of magnitude slower, and should never be a default |

## Where to go next

[Performance vs Echoes](@ref dev-benchmarks) gives the measured timings of
these paths against the C++ reference.
