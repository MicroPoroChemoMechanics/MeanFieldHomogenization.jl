# [API — LayeredSpheroid](@id api-layered-spheroid)

`layer_count`, `layer_modulus`, `layer_interface` and
`layer_volume_fraction` are shared generics extended from
`LayeredSpheres` — see [API — LayeredSphere](layered_sphere.md)
for their docstrings; they apply unchanged to `LayeredSpheroid`.

So are the pointwise-field generics `get_layer`, `local_temperature`,
`local_gradient`, `local_flux` and the four `local_*_*_loc` couplings: one
binding carries both inclusion families, and **all** their docstrings — the
sphere's methods and the spheroid's — are rendered together on that same
page.

```@docs
MeanFieldHomogenization.LayeredSpheroids
MeanFieldHomogenization.LayeredSpheroids.LayeredSpheroid
MeanFieldHomogenization.LayeredSpheroids.layered_spheroid_from_fractions
MeanFieldHomogenization.LayeredSpheroids.layer_q
MeanFieldHomogenization.LayeredSpheroids.layer_semiaxes
MeanFieldHomogenization.LayeredSpheroids.outer_semiaxes
MeanFieldHomogenization.LayeredSpheroids.spheroid_state_sequence
MeanFieldHomogenization.LayeredSpheroids.spheroid_ba_ratios
MeanFieldHomogenization.LayeredSpheroids.LayeredSpheroidTransportFields
MeanFieldHomogenization.LayeredSpheroids.coupling_matrices
MeanFieldHomogenization.LayeredSpheroids.legendre_odd
```
