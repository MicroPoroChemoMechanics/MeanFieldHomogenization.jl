# [API — MaterialPoints](@id api-material-points)

```@docs
MeanFieldHomogenization.MaterialPoints
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

```@docs
Mandel
identity_mandel
trace_mandel
deviator
von_mises
equivalent_strain
```

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
J2Plasticity
```

## Driving a point

```@docs
LoadingPath
point_response
PointResponse
PointDriverError
```
