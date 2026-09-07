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
MeanFieldHomogenization.Core.eshelby_tensor
MeanFieldHomogenization.Elasticity.iso_stiffness
```

## Nanoinclusion interfaces

```@docs
surface_stiffness
equivalent_particle
```

## Parameter conversions

```@docs
k_mu
E_nu
iso_stiffness_E_nu
MeanFieldHomogenization.Elasticity.hoenig_params
MeanFieldHomogenization.Elasticity.hoenig_stiffness
```

## [The cubic symmetry class](@id api-cubic)

A fourth-order tensor with the minor symmetries and cubic symmetry has **three**
independent constants. There is no TensND storage type for the class, so
`material_symmetry` still answers `GeneralAnisotropicSym` for such a tensor —
inventing a trait with no type behind it would let dispatch claim a structure
the object does not carry. What is provided is the algebra, which is what a
solver needs: the projection, its residual, the constants, and a constructor.

The cube axes are the **canonical basis** vectors. A cubic tensor expressed in
any other frame is not cubic in this sense, and `cubic_residual` says so.

```@docs
best_fit_cubic
cubic_parameters
cubic_residual
cubic_anisotropy
cubic_stiffness
```
