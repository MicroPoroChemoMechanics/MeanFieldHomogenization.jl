# [API — Viscoelasticity](@id api-viscoelasticity)

```@docs
MeanFieldHomogenization.Viscoelasticity
```

```@meta
CurrentModule = MeanFieldHomogenization.Viscoelasticity
```

## Constitutive laws

```@docs
AbstractViscoLaw
ViscoLaw
visco_mode
VALID_VISCO_MODES
visco_eval
maxwell_relaxation
kelvin_creep
maxwell_iso
kelvin_iso
heaviside_law
```

## Trapezoidal discretization

```@docs
trapezoidal_matrix
_trapezoidal_relaxation
```

## Volterra algebra

```@docs
volterra_inverse
volterra_product
volterra_divide
volterra_left_divide
```

## Symmetry-class conversions

```@docs
iso_params_from_blocks
iso_blocks_from_params
ti_params_from_blocks
ti_blocks_from_params
ortho_params_from_blocks
ortho_blocks_from_params
```

## Hill ALV kernel

```@docs
hill_kernel
```

## Schemes — generic (6n × 6n) algebra

```@docs
voigt_alv
reuss_alv
dilute_alv
dilute_dual_alv
mori_tanaka_alv
maxwell_alv
self_consistent_alv
asymmetric_self_consistent_alv
pcw_alv
differential_alv
dilute_concentration_alv
dilute_contribution_alv
```

### Internal guards

```@docs
_alv_diff_keeps_iso
```

## Schemes — iso fast path

```@docs
voigt_alv_iso
reuss_alv_iso
dilute_alv_iso
dilute_dual_alv_iso
mori_tanaka_alv_iso
maxwell_alv_iso
dilute_concentration_alv_iso
dilute_contribution_alv_iso
```

## Schemes — TI fast path

```@docs
voigt_alv_ti
reuss_alv_ti
dilute_alv_ti
dilute_dual_alv_ti
mori_tanaka_alv_ti
maxwell_alv_ti
dilute_concentration_alv_ti
dilute_contribution_alv_ti
```

## Schemes — ortho fast path

```@docs
voigt_alv_ortho
reuss_alv_ortho
dilute_alv_ortho
dilute_dual_alv_ortho
mori_tanaka_alv_ortho
maxwell_alv_ortho
dilute_concentration_alv_ortho
dilute_contribution_alv_ortho
```

## Cracks

```@docs
cod_kernel_alv
compliance_contribution_alv
delta_compliance_alv
stiffness_contribution_alv
stiffness_contribution_alv_at
delta_stiffness_alv
```

## Layered spheres

```@docs
bulk_localization_alv
bulk_state_seq_alv
bulk_amplitude_seq_alv
shear_localization_alv
strain_strain_loc_alv
```

## Order-2 ALV (conductivity / diffusion)

```@docs
hill_kernel_order2
voigt_alv_order2
reuss_alv_order2
dilute_alv_order2
dilute_dual_alv_order2
mori_tanaka_alv_order2
maxwell_alv_order2
homogenize_alv_order2
dilute_concentration_alv_order2
dilute_contribution_alv_order2
iso_order2_params_from_blocks
iso_order2_blocks_from_params
```

## Structured ALV kernel types

```@docs
AbstractALVKernel
ALVKernelISO
ALVKernelTI
ALVKernelOrtho
```

## Orientation symmetrization

```@docs
_maybe_symmetrize_alv
_maybe_symmetrize_alv2
```

## Public dispatcher

```@docs
homogenize_alv
has_visco_property
```
