# [API — Schemes](@id api-schemes)

```@docs
MeanFieldHomogenization.Schemes
```

## RVE / Phase / Amount

```@docs
RVE
Phase
AbstractAmount
VolumeFraction
CrackDensity
AbstractDistributionShape
UniformDistribution
AbstractSymmetrize
NoSymmetrize
IsoSymmetrize
TISymmetrize
phase_symmetrize
add_matrix!
add_phase!
matrix_phase
inclusion_phase_names
phase_property
matrix_property
volume_fraction
crack_density
matrix_volume_fraction
phase_property_raw
validate_rve
promote_rve
```

## Schemes

```@docs
HomogenizationScheme
Voigt
Reuss
Laminated
Dilute
DiluteDual
MoriTanaka
Maxwell
PonteCastanedaWillis
SelfConsistent
AsymmetricSelfConsistent
DifferentialScheme
DifferentialTrajectory
Proportional
Sequential
CustomPath
Path
AndersonDefault
NewtonDefault
AutoNonlinear
```

## Entry point

```@docs
homogenize
crack_family_compliances
crack_family_residual
fracture_permeability
differential_path
MeanFieldHomogenization.Schemes.SCHEME_ALIAS
```

## Symmetry projections

Best-fit projection of a tensor onto a symmetry class. These force major
symmetry, unlike the exact rotation-group averages
`isotropify` / `transverse_isotropify` (re-exported from `TensND`); the two differ whenever the input is not
major-symmetric, and the difference is worked through in
[Symmetrization showcase](../tutorials/generated/symmetrization.md).

`best_fit_iso(t)`, `best_fit_ti(t, axis)` and `best_fit_ortho(t, frame)` live
in **`TensND`**, next to the `proj_tens` machinery they wrap, and are
re-exported here because turning a scheme result into publishable parameters is
part of this package's surface. See the TensND documentation for their
docstrings.
