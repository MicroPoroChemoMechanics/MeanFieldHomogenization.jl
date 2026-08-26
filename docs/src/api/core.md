# [API — Core](@id api-core)

```@docs
MeanFieldHomogenization.Core
MeanFieldHomogenization.AbstractInclusion
MeanFieldHomogenization.AbstractEllipsoidalInclusion
MeanFieldHomogenization.AbstractCrack
MeanFieldHomogenization.AbstractLayeredInclusion
MeanFieldHomogenization.AbstractAlgorithm
MeanFieldHomogenization.Analytical
MeanFieldHomogenization.Residue
MeanFieldHomogenization.DECUHR
MeanFieldHomogenization.CylinderQuadrature
MeanFieldHomogenization.MaterialSymmetry
MeanFieldHomogenization.material_symmetry
MeanFieldHomogenization.Core.extract_iso_moduli
MeanFieldHomogenization.Core.extract_ti_moduli
MeanFieldHomogenization.Core.newton_potential_3d
MeanFieldHomogenization.Core.newton_potential_2d
MeanFieldHomogenization.Core.newton_potential_3d_cylinder
MeanFieldHomogenization.Core.dimension
MeanFieldHomogenization.Core.element_type
MeanFieldHomogenization.Core.inclusion_basis
MeanFieldHomogenization.Core.shape_trait
MeanFieldHomogenization.Core.shape_tensor
```

## Exact rotation-group averages

Exact averages of a tensor over a rotation group — the *exact* counterpart of the
best-fit projections in [API — Schemes](schemes.md). See
[Symmetrization showcase](../tutorials/generated/symmetrization.md) for
the comparison between the two.

These live in **`TensND`** — they are pure tensor algebra, with nothing
homogenization-specific about them — and are re-exported by this package, so
`MeanFieldHomogenization.isotropify` and `Core.isotropify` both resolve to
them:

| Name | What it returns |
| :--- | :--- |
| `isotropify(t)` | the exact SO(3) average, `TensISO{4}` or `TensISO{2}` |
| `transverse_isotropify(t, n)` | the exact azimuthal average about `n`, `TensTI{4,T,8}` or `TensTI{2,T,3}` — `ℓ₃ ≠ ℓ₄` and the antisymmetric couplings `ℓ₇`, `ℓ₈` are preserved |
| `mandel66_minor(arr)` / `array_from_mandel66(M)` | 6×6 Kelvin-Mandel ↔ 3×3×3×3, minor-symmetrizing, no major symmetry assumed |
| `ti8_params_from_KM(M)` / `KM_from_ti8_params(p)` | the eight Walpole coefficients of an axially-invariant tensor about `e₃`, read off exactly — the non-major-symmetric counterpart of `TensND.ti_params_from_KM` |
| `ti_average_mandel66(M, n)` / `iso_average_mandel66(M)` | the same averages, on a 6×6 block rather than on a tensor (the ageing-viscoelastic Volterra path) |

See the TensND documentation for the full docstrings.
