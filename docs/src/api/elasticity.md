# [API — Elasticity](@id api-elasticity)

```@docs
MeanFieldHomogenization.Elasticity
MeanFieldHomogenization.EllipsoidShape
MeanFieldHomogenization.Spherical
MeanFieldHomogenization.Prolate
MeanFieldHomogenization.Oblate
MeanFieldHomogenization.Triaxial
MeanFieldHomogenization.Circular
MeanFieldHomogenization.Elasticity.Elliptic
MeanFieldHomogenization.Ellipsoid
MeanFieldHomogenization.Spheroid
MeanFieldHomogenization.Cylinder
MeanFieldHomogenization.CylindricalShape
MeanFieldHomogenization.CircularCylindrical
MeanFieldHomogenization.EllipticCylindrical
MeanFieldHomogenization.tens_IA
MeanFieldHomogenization.tens_UA
MeanFieldHomogenization.tens_VA
MeanFieldHomogenization.hill_tensor
MeanFieldHomogenization.HillCache
MeanFieldHomogenization.with_hill_cache
MeanFieldHomogenization.Core.eshelby_tensor
MeanFieldHomogenization.Core.second_hill_tensor
```

## Nanoinclusion interfaces

```@docs
surface_stiffness
equivalent_particle
```

## Parameter conversions

The isotropic moduli are TensND's, re-exported by the package unchanged:
`iso_stiffness(k, μ)`, `k_mu(ℂ)`, `E_nu(ℂ)` and `iso_stiffness_E_nu(E, ν)`
([TensND, isotropic elastic moduli](https://MicroPoroChemoMechanics.github.io/TensND.jl/stable/api/structured)).
The Hoenig parameters of a transversely isotropic stiffness are read and built here,
on TensND's compliance forms `ti_args_hoenig` and `tens_ti_hoenig`.

```@docs
MeanFieldHomogenization.Elasticity.hoenig_params
MeanFieldHomogenization.Elasticity.hoenig_stiffness
```

## [The cubic symmetry class](@id api-cubic)

The class itself lives in **TensND**: `TensCubic` stores the three constants and
a cube frame, its products and inverses are componentwise, `tens_cubic` /
`arg_cubic` convert to and from ``(C_{11}, C_{12}, C_{44})``,
`cubic_anisotropy` is the Zener-type departure from isotropy, and
`proj_tens(Val(:CUBIC), t, frame)` — or `best_fit_cubic` — projects onto the
class. None of that is reimplemented here.

What this package adds is a name for the projection **residual**, which is the
quantity a solver reads, and the symmetry trait. The trait is honest only now
that a storage type exists: `material_symmetry` answers from the *container*,
and before TensND had `TensCubic` a `CubicSym` would have let dispatch claim a
structure the object did not carry.

Read `cubic_residual` and `cubic_anisotropy` **together**. The first is the
distance to a class the answer belongs to by group theory, so it is
discretization error and nothing else; the second lives *inside* the class. An
artifact breaks the symmetry, a real morphological anisotropy does not.

```@docs
cubic_residual
CubicSym
```
