# [API — Interactions](@id api-interactions)

```@docs
MeanFieldHomogenization.Interactions
```

## Two-inclusion interaction tensors

```@docs
interaction_tensor
self_interaction_tensor
```

## Periodic images

```@docs
lattice_interaction_tensor
periodic_images
```

## Solid harmonics

The solid, vector and ellipsoidal harmonics, the solutions of the Lamé
equation, their translation and their sums over a periodic array are those of
TensND: [`regular_solid_harmonics`](@extref TensND TensND.regular_solid_harmonics),
[`irregular_solid_harmonics`](@extref TensND TensND.irregular_solid_harmonics),
[`lattice_solid_harmonics`](@extref TensND TensND.lattice_solid_harmonics),
[`regular_lame_solution`](@extref TensND TensND.regular_lame_solution),
[`lame_solution_translation`](@extref TensND TensND.lame_solution_translation),
[`lame_functions`](@extref TensND TensND.lame_functions) and the rest of its
solid and ellipsoidal harmonics.

## Real-space Green operator

```@docs
green_operator
green_operator_iso
```

### Anisotropic reference media

```@docs
green_function_aniso
green_operator_aniso
gauss_legendre_nodes
```
