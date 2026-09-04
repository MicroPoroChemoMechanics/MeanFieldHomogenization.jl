# [Sensitivities — autodiff via ForwardDiff](@id man-sensitivities)

`MeanFieldHomogenization` differentiates `homogenize(rve, scheme)` with respect to any
scalar input — physical (stiffness coefficients, conductivities), geometric
(radii, aspect ratios, volume fractions, crack densities, distribution-shape
envelopes), including fields of user-defined inclusion types.

The whole machinery is a thin convenience layer on top of [ForwardDiff.jl];
ForwardDiff is shipped as a [weak dependency](https://pkgdocs.julialang.org/v1/creating-packages/#Conditional-loading-of-code-in-packages-(Extensions))
so the API only activates when you `using ForwardDiff` alongside `MeanFieldHomogenization`.

## Why autodiff

Every kernel (`hill_tensor`, `eshelby_tensor`, the ten schemes, the
SC/ASC/Differential solvers) is `ForwardDiff.Dual`-friendly, pinned by
`test/Schemes/test_dual_compat.jl`. The sensitivity API lifts that property
into a user-facing form: no manual RVE rebuild with `Dual` values, no `Tag`
plumbing, no scalar extraction by hand.

Practical consequences:

- **Geometry parameters are first-class.** Differentiate w.r.t. semi-axes,
  crack opening, distribution-shape envelopes, anything stored as a `Number`
  field on an inclusion.
- **User-defined inclusions just work.** Define your own
  `<: AbstractEllipsoidalInclusion` (or any subtype of `AbstractInclusion`)
  with `Number` fields, register `hill_tensor` and friends — and the
  sensitivity API differentiates them with no further code change.
- **Multi-scale chain rule is automatic.** Compose two or three calls to
  `homogenize` in a closure; ForwardDiff propagates partial derivatives
  through all of them in a single Dual sweep — no manual chain rule
  required.

## Quick start

```julia
using MeanFieldHomogenization, ForwardDiff, TensND

rve = RVE()
add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(30.0, 10.0)); fraction = :rest)
add_phase!(rve, :I, Ellipsoid(1.0),
            Dict(:C => TensISO{3}(60.0, 20.0)); fraction = 0.2)

# Sensitivity of C_eff[1,1,1,1] w.r.t. the inclusion volume fraction
∂_f = derivative(rve, MoriTanaka(), amount(:I);
                 indexer = C -> get_array(C)[1, 1, 1, 1])
```

That's it — no closure to write, no manual `set_param`/`homogenize`
plumbing.

## Parameter lenses

The first argument to `derivative`/`gradient`/`jacobian` is an
`AbstractParameter` *lens* describing the scalar input you want to
differentiate against. Four concrete lens kinds are shipped, plus a
`sensitivity(f, x₀)` closure fallback for everything else.

| Helper                          | Kind                                | Underlying type             |
| :------------------------------ | :---------------------------------- | :-------------------------- |
| `amount(:I)`                    | volume fraction or crack density    | `AmountParameter`           |
| `property(:I, :C, :bulk)`       | scalar coefficient of a tensor      | `PropertyParameter`         |
| `geometry(:I, :semi_axes, 3)`   | scalar geometry field               | `GeometryParameter`         |
| `shape_param(:semi_axes, 1)`    | distribution-shape geometry field   | `DistributionShapeParameter`|

Named selectors recognized by `property` (other symbols fall back to a
positional `Int` index into `get_data(tensor)`):

| Tensor type    | Named selectors                                      |
| :------------- | :--------------------------------------------------- |
| `TensISO{2}`   | `:scalar`, `:λ`                                      |
| `TensISO{4,3}` | `:bulk`, `:K`, `:α` ; `:shear`, `:μ`, `:β`           |
| `TensTI{2}`    | `:transverse`, `:a` ; `:axial`, `:b`                 |
| `TensTI{4}`    | `:ℓ₁` … `:ℓ₆` (with `ℓ₃ = ℓ₄` in the major-symmetric case) |

## Single derivative, gradient, full Jacobian

```julia
# scalar in / scalar out
∂_K = derivative(rve, MoriTanaka(), property(:I, :C, :bulk);
                 indexer = C -> get_array(C)[1, 1, 1, 1])

# vector in / scalar out
ps = [amount(:I), property(:I, :C, :bulk), property(:M, :C, :shear)]
∇  = gradient(rve, MoriTanaka(), ps;
              indexer = C -> get_array(C)[1, 1, 1, 1])

# vector in / tensor out → full Jacobian (flattened to 81 × N for a 4-tensor)
J = jacobian(rve, MoriTanaka(), ps)         # size(J) == (81, 3)
```

`gradient` and `jacobian` accept an optional `chunk = ForwardDiff.Chunk(N)`
kwarg; without it ForwardDiff picks a chunk size automatically.

## Closure fallback for arbitrary parameterizations

Anything that can't be expressed as a single lens (composite parameters,
parameters of a user inclusion that don't map to a `Number` field, etc.)
can be differentiated through a user-supplied closure:

```julia
∂α = sensitivity(0.3) do α
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(30.0, 10.0)); fraction = :rest)
    add_phase!(rve, :I, Ellipsoid(1.0, α, α^2),
                Dict(:C => TensISO{3}(60.0, 20.0)); fraction = 0.2)
    return get_array(homogenize(rve, MoriTanaka(), :C))[1, 1, 1, 1]
end
```

`sensitivity(f, x₀)` auto-detects the derivative / gradient / Jacobian flavor
from the shape of `x₀` and the return type of `f(x₀)`. Pass
`kind = :derivative | :gradient | :jacobian` to force a specific mode.

## User-defined inclusions

```julia
struct MyBlob{T <: Number, B <: TensND.AbstractBasis} <:
       MeanFieldHomogenization.AbstractEllipsoidalInclusion{3, T}
    radius::T
    eccentricity::T
    basis::B
end
# Register hill_tensor / eshelby_tensor (delegate to an equivalent Ellipsoid)
MeanFieldHomogenization.hill_tensor(b::MyBlob, C₀::TensND.AbstractTens; kw...) =
    MeanFieldHomogenization.hill_tensor(Ellipsoid(b.radius, b.radius*(1-b.eccentricity),
                                        b.radius*(1-b.eccentricity)^2), C₀; kw...)

# Differentiate w.r.t. any scalar field — no library change required.
∂_e = derivative(rve, Dilute(), geometry(:B, :eccentricity);
                 indexer = C -> get_array(C)[1,1,1,1])
```

The generic `_replace_geom_field` reflects on the struct's fieldnames and
promotes any sibling `<:Number` field to the new `Dual` element type, so
the parametric inner constructor of `MyBlob{T,B}` resolves cleanly.

## Multi-scale chain rule

Composing several `homogenize` calls in one closure applies the chain rule
automatically.
[`scripts/28_multiscale_strength.jl`](https://github.com/MicroPoroChemoMechanics/MeanFieldHomogenization.jl/blob/main/scripts/28_multiscale_strength.jl)
follows the quasi-brittle strength model of [pichler2011](@cite) — a three-scale cement-paste / mortar
upscaling (SC hydrate foam + two Mori-Tanaka stages) requiring
`∂C_mortar / ∂μ_hyd` through the full chain — in two styles:

- **Manual chain rule** — one `jacobian` per scale, then a tensor product of
  the partial Jacobians; use it when the intermediate partials matter.
- **End-to-end autodiff** — a single `ForwardDiff.derivative` on the nested
  closure; shorter and scheme-agnostic.

Both approaches agree to the floor of ForwardDiff itself, and the
end-to-end approach is the recommended default unless you specifically
need the intermediate Jacobians.

## Symmetrize and orientation distributions

A thin oblate spheroid with a *uniform spatial distribution of orientations*
is, on average, isotropic. The `symmetrize` keyword on [`add_phase!`](@ref)
declares such a distribution, so the kernel projects the phase
localization tensor onto the corresponding symmetry class:

| Symmetrize value           | Meaning                                                     | Result class |
| :------------------------- | :---------------------------------------------------------- | :----------- |
| `:none` (default)          | inclusion at its declared orientation                       | as input     |
| `:iso`                     | uniform spatial distribution (Reynolds avg over `SO(3)`)    | `TensISO`    |
| `:ti` / `TISymmetrize(n)`  | uniform azimuthal distribution around axis `n` (default ez) | `TensTI(n)`  |

```julia
rve = RVE()
add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(216.0, 64.0)); fraction = :rest)
# Oblate inclusions with a uniform-in-orientation distribution: the
# effective stiffness is iso even though each individual inclusion is TI.
add_phase!(rve, :I, Ellipsoid(1.0, 1.0, 0.2),
            Dict(:C => TensISO{3}(3.0e-6, 2.0e-6));
            fraction = 0.3, symmetrize = :iso)
```

Sensitivities on RVEs that carry a `symmetrize` keyword work the same way:
`derivative` / `gradient` / `jacobian` propagate Duals through the
projection automatically.

## Limitations

- **No `Complex{T}` autodiff.** ForwardDiff doesn't mix Dual + Complex
  cleanly. Use closure-style `sensitivity` only when the input is real.
  Frequency-domain viscoelastic computations remain available via
  `Complex{Float64}` moduli (independent of the autodiff API).
- **Symbolic differentiation is not exposed via this API.** Use SymPy
  directly through the closed-form schemes (Voigt, Reuss, Dilute, MT,
  Maxwell, PCW) when symbolic derivatives are needed.
- **Geometry derivative across shape categories.** Perturbing axes of an
  `Ellipsoid{Spherical}` inclusion preserves the `Spherical` shape trait
  in the parametric type, so the symmetry-imposed derivative is `0`. To
  get a non-trivial geometric sensitivity, perturb around an already
  non-degenerate (`Triaxial` / `Prolate` / `Oblate`) configuration.
- **TI symmetrize on non-coaxial inclusions** routes the matrix used for
  the localization-tensor computation through an isotropic projection —
  the result still satisfies the outer `TI(axis)` projection, and is
  exact at the iso fixed-point of the SC iteration. This is documented
  in [`src/Schemes/symmetrize.jl`](https://github.com/MicroPoroChemoMechanics/MeanFieldHomogenization.jl/blob/main/src/Schemes/symmetrize.jl).

## Validation

[`test/Schemes/test_sensitivities.jl`](https://github.com/MicroPoroChemoMechanics/MeanFieldHomogenization.jl/blob/main/test/Schemes/test_sensitivities.jl)
compares every sensitivity against centered finite differences on every scheme,
plus the Christensen 1990 closed form for `∂k_MT/∂f`: `rtol ≈ 1e-6`
(closed-form schemes), `rtol ≈ 1e-4` (iterative, limited by the fixed-point
tolerance rather than the autodiff).
