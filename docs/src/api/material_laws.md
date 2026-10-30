# [API — MaterialLaws](@id api-material-laws)

```@docs
MeanFieldHomogenization.MaterialLaws
```

## Writing a law

```@docs
AbstractPointLaw
ResidualLaw
PotentialLaw
PointState
kinematics
SmallStrain
state_layout
gradient_layout
flux_layout
law_parameters
```

## Kelvin–Mandel algebra

The laws are written on the Kelvin–Mandel vectors of TensND, which the package
re-exports: `Mandel`, `identity_mandel`, `trace_mandel`, `deviator`,
`sqnorm_deviator`, `von_mises`, `equivalent_strain`, documented with TensND
([Symmetric tensors as Kelvin-Mandel vectors](https://MicroPoroChemoMechanics.github.io/TensND.jl/stable/api/tensors)).

## Integrating a step

```@docs
AbstractLocalAlgorithm
LocalNewton
LocalIntegrationError
point_update
```

## Laws shipped

```@docs
LinearElasticity
PowerLawElasticity
MaxwellViscoelasticity
BiotPoroelasticity
NortonViscoplasticity
PerzynaViscoplasticity
J2Plasticity
CrystalViscoplasticity
slip_systems
```

## Nonlinear homogenization

```@docs
NonlinearHomogenized
SecantMethod
ClassicalSecant
ModifiedSecant
secant_moduli
```

## Damage by the growth of microcracks

```@docs
CrackGrowthDamage
CrackGrowthResistance
```

## Uniform fields

```@docs
AbstractPhaseComposite
UniformStrain
UniformStress
```

## The incremental tangent approach

```@docs
IncrementalTangent
```

## The affine formulation

```@docs
AffineComposite
SecondOrderComposite
AffineEVPComposite
AffineEVPState
QuasiElasticComposite
QuasiElasticState
```

## Polycrystals

```@docs
PolycrystalModel
TaylorModel
KronerModel
BerveillerZaouiModel
HillModel
TangentModel
AnisotropicTangentModel
Polycrystal
PolycrystalState
grain_states
PowerLawCrystal
crystal_strain_rate
ViscoplasticPolycrystal
flow_stress
```

## Strength of a frictional matrix

```@docs
DruckerPrager
Heterogeneities
Pores
RigidInclusions
FrictionalComposite
limit_stress
```

## Nonlinear poroelasticity of cracked rocks

```@docs
ClosingCrackPoroelasticity
ExponentialClosure
```

## Driving a point

```@docs
LoadingPath
point_response
PointResponse
PointDriverError
```

## Coupling to a finite-element code

```@docs
QuadratureStates
commit!
rollback!
ModelingHypothesis
ThreeD
PlaneStrain
Axisymmetric
mandel_strain
volume_weight
quasistatic!
GlobalNewtonError
material_element!
material_system
update_points!
write_material_vtk
```

