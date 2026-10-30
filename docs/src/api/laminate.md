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
MeanFieldHomogenization.Laminates.layer_gradient_concentration
MeanFieldHomogenization.Laminates.layer_flux_concentration
MeanFieldHomogenization.Laminates.interface_jump
```

`layer_gradient_concentration` and `layer_flux_concentration` serve elasticity
(`property = :C`, the default) and conduction (`property = :K`). Their names in
elasticity accept a fourth-order layer property only:

```@docs
MeanFieldHomogenization.Laminates.layer_strain_concentration
MeanFieldHomogenization.Laminates.layer_stress_concentration
```

A layer also answers the package-wide concentration generics —
[`gradient_gradient_concentration`](@ref MeanFieldHomogenization.gradient_gradient_concentration),
[`flux_gradient_concentration`](@ref MeanFieldHomogenization.flux_gradient_concentration),
[`gradient_flux_concentration`](@ref MeanFieldHomogenization.gradient_flux_concentration) and
[`flux_flux_concentration`](@ref MeanFieldHomogenization.flux_flux_concentration) — called as
`gradient_gradient_concentration(lam, :LAYER; property = :C)`. Since a laminate
has neither a matrix nor a reference medium, the layer *name* takes the place
of the property pair of the inclusion signature. Two of the four (the
flux-gradient and the gradient-flux tensors) exist only under those names; the
other two are synonyms of the `layer_*` functions above. See
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
laminate reuses it with the Volterra inversion substituted. It is written on the
Kelvin–Mandel block algebra about the layer normal, which is TensND's and is
re-exported from `Core`: the slots [`MANDEL_IP`](@extref TensND TensND.MANDEL_IP) and [`MANDEL_OP`](@extref TensND TensND.MANDEL_OP), the blocks and their
embeddings, the Schur inverse [`mandel_inv_schur`](@extref TensND TensND.mandel_inv_schur), the plane pseudo-inverses
[`plane_pinv`](@extref TensND TensND.plane_pinv) and [`plane_pinv2`](@extref TensND TensND.plane_pinv2), [`acoustic_tensor`](@extref TensND TensND.acoustic_tensor) and [`compliance_op_block`](@extref TensND TensND.compliance_op_block) (see
[Kelvin-Mandel blocks about a normal](@extref TensND Kelvin-Mandel-blocks-about-a-normal)
in the TensND documentation).

```@docs
MeanFieldHomogenization.Core.flat_hill
MeanFieldHomogenization.Core.laminate_stiffness
MeanFieldHomogenization.Core.laminate_conductivity
MeanFieldHomogenization.Core.laminate_gradient_concentration
MeanFieldHomogenization.Core.laminate_flux_concentration
MeanFieldHomogenization.Core.laminate_flux_gradient_concentration
MeanFieldHomogenization.Core.laminate_gradient_flux_concentration
```

The four kernels serve the ``6\times 6`` Kelvin–Mandel blocks of elasticity and
the ``3\times 3`` blocks of conduction. Their names in elasticity accept the
former only:

```@docs
MeanFieldHomogenization.Core.laminate_strain_concentration
MeanFieldHomogenization.Core.laminate_stress_concentration
MeanFieldHomogenization.Core.laminate_stress_strain_concentration
MeanFieldHomogenization.Core.laminate_strain_stress_concentration
```

## The cell contract and declarative nesting

```@docs
MeanFieldHomogenization.Core.AbstractHomogenizationCell
MeanFieldHomogenization.Core.validate_cell
MeanFieldHomogenization.Core.cell_member_names
MeanFieldHomogenization.Core.cell_container_property
MeanFieldHomogenization.Core.cell_has_property
MeanFieldHomogenization.Core.cell_set_property
MeanFieldHomogenization.Core.Homogenized
MeanFieldHomogenization.Core.NestedParameter
MeanFieldHomogenization.Core.nested
MeanFieldHomogenization.Core.resolve_property
MeanFieldHomogenization.Core.has_nested_property
MeanFieldHomogenization.Core.MAX_NESTING
```
