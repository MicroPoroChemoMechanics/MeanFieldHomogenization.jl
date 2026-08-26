# [API — Laplace-Carson viscoelasticity](@id api-laplace-carson)

The non-ageing half of `MeanFieldHomogenization.Viscoelasticity`: numerical
Laplace inversion, the rheological model catalog, the exact Kelvin ⇄ Maxwell
conversion, and the homogenization driver that ties them together.

See [the theory](@ref th-laplace-carson), the
[model manual](@ref man-rheological-models) and the
[inversion manual](@ref man-laplace-inversion).

```@meta
CurrentModule = MeanFieldHomogenization.Viscoelasticity
```

## Numerical inversion

```@docs
AbstractLaplaceInversion
GaverStehfest
FixedTalbot
TalbotTrefethen
DeHoog
DEFAULT_INVERSION
inverse_laplace
inverse_carson
inverse_carson_rate
```

### Internals

```@docs
_accumulate
_realpart
_decompose
_gs_exact_weights
_scalar_float
_plain_value
_dehoog_from_values
_dehoog_blocks
_DEHOOG_MIN_RATIO
```

## The model interface

```@docs
AbstractRheology
AbstractTensorRheology
carson_relaxation
carson_creep
relaxation
creep
complex_modulus
storage_modulus
loss_modulus
loss_factor
glassy_modulus
equilibrium_modulus
is_fluid
default_inversion
```

## Discrete spectra, and the exact conversion

```@docs
PronyRelaxation
PronyCreep
PRONY_MERGE_TOL
PRONY_FLUID_TOL
maxwell_to_kelvin
kelvin_to_maxwell
prony_fit_relaxation
prony_fit_creep
```

### Internals

```@docs
_phi_maxwell
_dphi_maxwell
_psi_kelvin
_dpsi_kelvin
_bracketed_root
_expand_bracket
_shrink_bracket
_root_with_ad
_sort_and_merge
_default_collocation
_lsq
_nnls
```

## The model catalog

### Elementary elements

```@docs
Spring
Dashpot
MaxwellUnit
KelvinUnit
relaxation_time
retardation_time
```

### Named chains

```@docs
zener_maxwell
zener_kelvin
burgers
```

### Fractional elements

```@docs
ScottBlair
FractionalMaxwell
FractionalKelvin
FractionalZener
Rabotnov
```

### Bituminous binders and long-term creep

```@docs
HuetSayegh
Model2S2P1D
creep_kernel
carson_creep_kernel
creep_kernel_law
LogarithmicCreep
```

## Lifting to tensors, and homogenizing

```@docs
AbstractIsoPairing
BulkShear
YoungPoisson
IsoRheology
iso_rheology
iso_rheology_E_nu
homogenize_lc
```

## Optional Mittag-Leffler support

```@docs
_mittag_leffler
```
