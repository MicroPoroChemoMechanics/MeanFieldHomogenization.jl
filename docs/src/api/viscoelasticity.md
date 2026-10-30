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
SampledViscoLaw
operator_law
precipitated
dissolved
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
second_hill_kernel
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

The stiffness contribution of a crack is a method of
[`property_contribution_alv`](@ref), documented with the layered spheres below.

```@docs
cod_kernel_alv
inverse_property_contribution_alv
delta_inverse_property_alv
property_contribution_alv_at
delta_property_alv
```

## Layered spheres

One function per notion serves elasticity and conduction, the order of the
reference law selecting the method: [`gradient_gradient_concentration_alv`](@ref)
and [`property_contribution_alv`](@ref) return a ``6n\times 6n`` matrix for a
fourth-order reference and a ``3n\times 3n`` one for a second-order reference.

```@docs
gradient_gradient_concentration_alv
property_contribution_alv
```

The per-layer kernels behind them, of the bulk and shear modes in elasticity
and of the single mode of conduction:

```@docs
bulk_concentration_alv
bulk_state_seq_alv
bulk_amplitude_seq_alv
shear_concentration_alv
gradient_concentration_alv
```

## The names of elasticity and of conduction

Each generic function also answers to the name it has in elasticity, and the
contribution to the name it has in conduction. Such a name forwards to the
generic function, accepts one order only, and refuses the other with an error
that names the generic function. The crack functions are elastic only, and
their names in elasticity are plain synonyms.

| Generic function | Elasticity (order 4) | Conduction (order 2) |
|:-----------------|:---------------------|:---------------------|
| `gradient_gradient_concentration_alv` | `strain_strain_concentration_alv` | |
| `property_contribution_alv` | `stiffness_contribution_alv` | `conductivity_contribution_alv` |
| `property_contribution_alv_at` | `stiffness_contribution_alv_at` | |
| `inverse_property_contribution_alv` | `compliance_contribution_alv` | |
| `delta_property_alv` | `delta_stiffness_alv` | |
| `delta_inverse_property_alv` | `delta_compliance_alv` | |

```@docs
strain_strain_concentration_alv
stiffness_contribution_alv
conductivity_contribution_alv
stiffness_contribution_alv_at
compliance_contribution_alv
delta_stiffness_alv
delta_compliance_alv
```

## Order-2 ALV (conductivity / diffusion)

The kernels below act on Volterra matrices, which do not carry their order in
their type: a ``3n\times 3n`` matrix with ``n`` even also has ``6m`` rows. They
keep the order in their names, as the iso, TI and ortho fast paths keep their
symmetry class; [`homogenize_alv`](@ref) chooses among them from the property.

```@docs
hill_kernel_order2
second_hill_kernel_order2
voigt_alv_order2
reuss_alv_order2
dilute_alv_order2
dilute_dual_alv_order2
mori_tanaka_alv_order2
maxwell_alv_order2
self_consistent_alv_order2
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
