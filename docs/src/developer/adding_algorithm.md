# [Adding a new algorithm](@id dev-adding-algorithm)

!!! info "Before this page"
    [Architecture](@ref dev-architecture), for the `_resolve_algo` / `_kernel`
    tables extended below, and [Hill polarization tensors](@ref th-hill-tensors)
    for the algorithms already in them.

An *algorithm* here means a way of evaluating the Hill tensor, or a crack
quantity, for a given inclusion and a given matrix symmetry — closed form,
residue reduction, adaptive cubature, and so on. Adding one has two halves:
write the kernel, then teach the dispatcher when to pick it.

## 1. Declare the trait

Algorithms are singleton types subtyping [`MeanFieldHomogenization.AbstractAlgorithm`](@ref),
declared in `src/Core/traits.jl`. The five that exist:

| Trait | Selected by | Nature |
| :---- | :---------- | :----- |
| `Analytical` | `:auto` on isotropic, 2-D, conduction, coaxial TI | closed form |
| `Residue` | `:residues` only (explicit) on 3-D anisotropic elasticity | Cauchy residues then 1-D quadrature; fastest, but `Float64` only and degenerate on an anisotropically-typed isotropic-valued reference |
| `DECUHR` | `:auto` on 3-D anisotropic elasticity **when its extension is loaded**, or `:decuhr` | adaptive cubature for singular integrands; AD-safe |
| `NestedQuadGK` | `:auto` on 3-D anisotropic elasticity otherwise, or `:nestedquadgk` | nested 1-D quadrature; type-generic, so also the `:auto` route for `Dual` / `Complex` / symbolic coefficients |
| `CylinderQuadrature` | `Cylinder` with an anisotropic matrix | 1-D quadrature over the transverse circle |

```julia
struct MyAlgorithm <: AbstractAlgorithm end
```

## 2. Implement the kernel

Add the `_kernel(inclusion, C₀, ::MyAlgorithm; kwargs...)` methods in the
sub-module the algorithm belongs to (`src/Elasticity/`, `src/Cracks/`,
`src/Conductivity/`). Keep the trait in the last positional argument so the
method table stays readable.

## 3. Register the dispatch rule

`src/Core/dispatch.jl` is the **single** place where a user-facing symbol
(`:auto`, `:residues`, `:decuhr`, …) becomes an algorithm instance. Every method
carries the same three-axis signature:

```julia
_resolve_algo(::Val{method}, ::InclusionClass, ::MatrixType) = MyAlgorithm()
```

matching simultaneously on

1. the **method symbol**, as `Val{:auto}`, `Val{:residues}`, … — a bare `::Val`
   catches every symbol not explicitly listed;
2. the **inclusion class** — `AbstractInclusion`,
   `AbstractEllipsoidalInclusion`, `AbstractCrack`, `Cylinder`, …;
3. the **matrix symmetry**, through the TensND type of ``\mathbb{C}_0``:
   `TensND.TensISO`, `TensND.TensTI`, `TensND.AbstractTens{4,3}`,
   `TensND.AbstractTens{2,3}`, `TensND.AbstractTens{4,2}`.

!!! warning "Match all three axes in every signature"
    It is tempting to write one method refined on the symmetry and another
    refined on the inclusion class. Don't: that produces the classic *diamond*
    ambiguity, where neither method is more specific than the other and Julia
    raises an ambiguity error. This is why `dispatch.jl` contains seemingly
    redundant methods — the `TensISO → Analytical` rule is stated once for
    `AbstractInclusion`, once for `AbstractEllipsoidalInclusion` and once for
    `AbstractCrack`, all returning the same thing. Follow that pattern rather
    than trying to compress it.

!!! note "Ordering inside the file"
    Rules that depend on the inclusion class are injected at the **end** of
    `dispatch.jl`, once the abstract types of `abstractions.jl` are available.
    Putting an inclusion-dependent rule near the top introduces a cycle.

The rules currently in force:

- isotropic matrix (`TensISO`), any order, any dimension → `Analytical`;
- 2-D stiffness or conductivity, any inclusion → `Analytical`;
- 3-D conductivity (order-2 tensor) → `Analytical`, since the square-root
  transformation is closed-form for *any* anisotropy;
- 3-D anisotropic elasticity → a **cubature** by default (`DECUHR` when its
  extension is loaded, else `NestedQuadGK`); `Residue` is reachable on an
  explicit `:residues` only, because it degenerates on a reference that is
  anisotropic in type and isotropic in value;
  a crack in a TI matrix aligned with ``\underline{n}`` → `Analytical`.

## 4. Document and test the AD story

State in the docstring whether the algorithm is `ForwardDiff`-compatible, and
**back the claim with a test**: the dispatch tables in
[Hill polarization tensors](@ref th-hill-tensors) are written from those
claims, so an unverified one propagates straight into the manual. A polynomial
root finder or an in-place `Float64` buffer usually makes an algorithm
`Float64`-only — that is acceptable, but it must be stated, and `:auto` must not
route `Dual` inputs to it.

If the algorithm sits behind a weak dependency, as `DECUHR` does, remember it
only exists once the extension loads: `import DECUHR, Integrals` is required in
`test/runtests.jl` and in any benchmark script that exercises it.

## Where to go next

[Adding a homogenization scheme](@ref dev-adding-scheme) moves from the tensors
of one inclusion to the schemes that consume them.
