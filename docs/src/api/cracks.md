# [API — Cracks](@id api-cracks)

`compliance_contribution`, `delta_compliance` and `delta_resistivity` are
`Core`-level generics shared with every other inclusion family; they are
documented under
[API — Localization & contribution](@ref api-localization).

```@docs
MeanFieldHomogenization.Cracks
MeanFieldHomogenization.EllipticCrack
MeanFieldHomogenization.RibbonCrack
MeanFieldHomogenization.PennyCrack
MeanFieldHomogenization.ConductiveCrack
MeanFieldHomogenization.Cracks.fracture_conductivity
MeanFieldHomogenization.Cracks.with_conductivity
MeanFieldHomogenization.cod_tensor
MeanFieldHomogenization.B_tensor
MeanFieldHomogenization.Cracks.crack_density_factor
MeanFieldHomogenization.Cracks.CrackShape
MeanFieldHomogenization.Cracks.EllipticShape
MeanFieldHomogenization.Cracks.Penny
MeanFieldHomogenization.Cracks.Ribbon
MeanFieldHomogenization.crack_basis
MeanFieldHomogenization.crack_normal
MeanFieldHomogenization.semi_major
MeanFieldHomogenization.semi_minor
MeanFieldHomogenization.aspect_ratio
MeanFieldHomogenization.sif
MeanFieldHomogenization.dif
MeanFieldHomogenization.compliance_from_cod
MeanFieldHomogenization.cod_from_compliance
```
