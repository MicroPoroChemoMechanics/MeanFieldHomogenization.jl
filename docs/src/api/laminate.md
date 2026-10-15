# [API — Laminate](@id api-laminate)

## The cell

```@docs
MeanFieldHomogenization.Laminates
MeanFieldHomogenization.Laminates.Laminate
MeanFieldHomogenization.Laminates.Layer
MeanFieldHomogenization.Laminates.add_layer!
MeanFieldHomogenization.Laminates.layer_names
MeanFieldHomogenization.Laminates.layer_property
MeanFieldHomogenization.Laminates.layer_property_raw
MeanFieldHomogenization.Laminates.layer_thickness
MeanFieldHomogenization.Laminates.laminate_period
MeanFieldHomogenization.Laminates.laminate_basis
MeanFieldHomogenization.Laminates.laminate_normal
MeanFieldHomogenization.Laminates.validate_laminate
```

## Anisotropic interfaces

A plane, unlike a sphere, imposes no symmetry on the interface, so a laminate
also accepts tensor-valued interface properties.

```@docs
MeanFieldHomogenization.Laminates.AnisotropicSpringInterface
MeanFieldHomogenization.Laminates.AnisotropicMembraneInterface
MeanFieldHomogenization.Laminates.AnisotropicSurfaceConductiveInterface
```

## Fields and per-layer tensors

```@docs
MeanFieldHomogenization.Laminates.laminate_hill
MeanFieldHomogenization.Laminates.layer_strain_concentration
MeanFieldHomogenization.Laminates.layer_stress_concentration
MeanFieldHomogenization.Laminates.layer_gradient_concentration
MeanFieldHomogenization.Laminates.layer_flux_concentration
MeanFieldHomogenization.Laminates.interface_jump
```

A layer also answers the package-wide concentration generics —
[`strain_strain_concentration`](@ref MeanFieldHomogenization.strain_strain_concentration),
[`stress_strain_concentration`](@ref MeanFieldHomogenization.stress_strain_concentration),
[`strain_stress_concentration`](@ref MeanFieldHomogenization.strain_stress_concentration),
[`stress_stress_concentration`](@ref MeanFieldHomogenization.stress_stress_concentration) and their
four transport twins — called as `strain_strain_concentration(lam, :LAYER)`. Since a
laminate has neither a matrix nor a reference medium, the layer *name* takes
the place of the `(ℂ₁, ℂ₀)` pair of the inclusion signature. Two of the eight
(`𝔸^{σε}` and `𝔸^{εσ}`) exist only under those names; the other six are
synonyms of the `layer_*` functions above. See
[API — Concentration & contribution](@ref api-concentration).

## Parameter lenses

```@docs
MeanFieldHomogenization.Laminates.ThicknessParameter
MeanFieldHomogenization.Laminates.thickness
MeanFieldHomogenization.Laminates.InterfaceParameter
MeanFieldHomogenization.Laminates.interface_param
```

## Aging viscoelasticity

```@docs
MeanFieldHomogenization.Viscoelasticity.laminate_alv
```

## The block algebra

The kernel behind the cell, in `Core`, so that the aging-viscoelastic
laminate reuses it with the Volterra inversion substituted.

```@docs
MeanFieldHomogenization.Core.KM_IP
MeanFieldHomogenization.Core.KM_OP
MeanFieldHomogenization.Core.plane_pinv
MeanFieldHomogenization.Core.plane_pinv2
MeanFieldHomogenization.Core._inv_km6
MeanFieldHomogenization.Core.flat_hill
MeanFieldHomogenization.Core.acoustic_tensor
MeanFieldHomogenization.Core.compliance_op_block
MeanFieldHomogenization.Core.laminate_stiffness
MeanFieldHomogenization.Core.laminate_conductivity
MeanFieldHomogenization.Core.laminate_strain_concentration
MeanFieldHomogenization.Core.laminate_stress_concentration
MeanFieldHomogenization.Core.laminate_stress_strain_concentration
MeanFieldHomogenization.Core.laminate_strain_stress_concentration
MeanFieldHomogenization.Core.laminate_gradient_concentration
MeanFieldHomogenization.Core.laminate_flux_concentration
MeanFieldHomogenization.Core.laminate_flux_gradient_concentration
MeanFieldHomogenization.Core.laminate_gradient_flux_concentration
```

## The cell contract and declarative nesting

```@docs
MeanFieldHomogenization.Core.AbstractHomogenizationCell
MeanFieldHomogenization.Core.validate_cell
MeanFieldHomogenization.Core.cell_member_names
MeanFieldHomogenization.Core.cell_container_property
MeanFieldHomogenization.Core.cell_set_property
MeanFieldHomogenization.Core.Homogenized
MeanFieldHomogenization.Core.NestedParameter
MeanFieldHomogenization.Core.nested
MeanFieldHomogenization.Core.resolve_property
MeanFieldHomogenization.Core.has_nested_property
MeanFieldHomogenization.Core.MAX_NESTING
```
