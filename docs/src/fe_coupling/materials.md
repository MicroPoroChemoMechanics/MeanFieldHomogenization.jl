# [Materials](@id fe-materials)

A **material** bundles a microstructure, a scheme and its internal state. The FE
driver builds it once, allocates one state per quadrature point, and calls
[`material_response`](@ref) in its element loop.

## The contract

Two methods make a material:

```julia
initial_state(m)                              # the state a fresh point starts from
material_response(m, ε, state_old, Δt; cache) # -> MaterialResponse
```

and a response carries three things:

| accessor | |
|:--|:--|
| [`stress`](@ref) | the flux ``\boldsymbol\sigma`` |
| [`tangent`](@ref) | the consistent tangent ``\partial\boldsymbol\sigma/\partial\boldsymbol\varepsilon`` |
| [`state`](@ref) | the **new** internal state |

`state_old` is never mutated. That is what lets a rejected Newton iteration be
retried by simply keeping the old state, and what makes a threaded element loop
safe.

## A linear material, end to end

```@example mat
using MeanFieldHomogenization, TensND

C₀ = TensISO{3}(3 * 30.0, 2 * 18.0)          # matrix: k = 30, μ = 18
rve = RVE(:M)
add_matrix!(rve, Ellipsoid(1.0), Dict(:C => C₀))
add_phase!(rve, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(3 * 90.0, 2 * 60.0));
           fraction = 0.2)

mat = HomogenizedElastic(rve, MoriTanaka())
st  = initial_state(mat)

ε = from_tensors(TensND.Tensors.SymmetricTensor{2, 3}((i, j) -> i == j ? 1.0e-3 : 0.0))
r = material_response(mat, ε, st, 0.0)

σ = to_tensors(stress(r))      # Tensors.SymmetricTensor{2,3}, global frame
ℂ = to_tensors(tangent(r))     # Tensors.SymmetricTensor{4,3}, global frame
σ[1, 1]
```

`to_tensors` is the only sanctioned way out — see
[the frame rule](@ref fe-scale-transition).

## Checking a material

[`check_material_interface`](@ref) exercises the contract and, crucially,
compares the declared tangent against a central finite difference:

```@example mat
check_material_interface(mat)
```

## Making the scheme solve affordable

A self-consistent solve on a cracked, anisotropic RVE costs milliseconds — far
too much per quadrature point, per Newton iteration, per step. But for flat
cracks the compliance contribution ``\mathbb{H}`` is the ``\omega \to 0`` limit
and does **not** depend on the aperture, so the homogenized quantities depend on
the state only through the *discrete* open/closed configuration. Pass a
[`MaterialCache`](@ref) and every point sharing a configuration pays once:

```@example mat
cache = MaterialCache()
for _ in 1:1000
    material_response(mat, ε, st, 0.0; cache = cache)
end
cache_stats(cache)
```

!!! warning "One cache per thread"
    A [`MaterialCache`](@ref) is an unlocked `Dict`. Sharing one across threads
    is a data race whose symptom would be a wrong stiffness, not an error.

## Shipped materials

| | |
|:--|:--|
| [`HomogenizedElastic`](@ref) | linear, from any cell and scheme — the control case for a new coupling |
| [`MicrocrackedMaterial`](@ref) | crack families that open and close; piecewise linear, so the tangent is exact |

[`MicrocrackedMaterial`](@ref) carries the aperture ``\omega_i`` and the
open/closed flag of each family as internal state, updated by

```math
\Delta\omega_i = \hat{\mathbf n}_i \cdot (\mathbb S_i : \Delta\boldsymbol\Sigma)
                  \cdot \hat{\mathbf n}_i ,
```

with ``\mathbb S_i`` the family's own contribution to the macroscopic
compliance ([`crack_family_compliances`](@ref MeanFieldHomogenization.Schemes.crack_family_compliances)).
Steps are split at every closure and reopening, so the result does not depend on
how the loading was subdivided. See it at work on the
[thick-walled cylinder](@ref fe-thick-cylinder).
