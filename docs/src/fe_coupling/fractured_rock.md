# [Building a fractured-rock material](@id fe-fractured-rock)

[`FracturedPoroelasticRock`](@ref) is the material of
[barthelemyARMA2011](@cite): two gradients ``(\boldsymbol{E}, p)`` in, two
fluxes ``(\boldsymbol{\Sigma}, \varphi)`` out, plus a permeability that follows
the fracture apertures. The equations it implements are on
[the coupled poroelastic problem](@ref fe-poro-coupling); this page is how to
build one and what it returns.

## Ingredients

| | |
|:--|:--|
| an `RVE` whose crack families are [`ConductiveCrack`](@ref) | mechanics *and* hydraulics from one microstructure |
| `ω₀` | the initial aspect ratios — what the cubic law is normalized on |
| `k_matrix` | the matrix conductivity: small, but [not zero](@ref fe-permeability) |
| a scheme | `SelfConsistent()` for a connected network, `MoriTanaka()` for a dilute one |

```@example rock
using MeanFieldHomogenization, TensND

C₀, kₛ, C_f = TensISO{3}(3 * 30.0, 2 * 18.0), 1.0e-18, 4.0e-18
props = Dict(:C => C₀, :K => TensISO{3}(kₛ))

rve = RVE(:M)
add_matrix!(rve, Ellipsoid(1.0), props)
add_phase!(rve, :F, ConductiveCrack(1.0; conductivity = C_f), props; density = 0.1)

mat   = FracturedPoroelasticRock(rve, MoriTanaka(); ω₀ = (1.0e-3,), k_matrix = kₛ)
cache = MaterialCache()
st    = initial_state(mat)
nothing # hide
```

## One call, everything the FE code needs

```@example rock
ε = from_tensors(TensND.Tensors.SymmetricTensor{2,3}((i,j) -> i == j == 3 ? -5.0e-4 : 0.0))
r = material_response(mat, (; ε = ε, p = 0.0), st, 0.0; cache = cache)

(σ₃₃ = r.fluxes.σ[3,3], φ = r.fluxes.φ,
 B₃₃ = -r.tangents.σp[3,3], invM = r.tangents.φp,
 ω = apertures(state(r))[1], k = transport_property(mat, state(r))[1,1])
```

The gradients go in as a `NamedTuple`, the fluxes and the four tangent blocks
come out of one response, and the permeability is read off the **new** state —
[`transport_property`](@ref) rather than a flux, because it feeds a different
balance equation.

A transient flow problem also needs ``\varphi`` at the *start* of the step:
[`fluid_content`](@ref) recomputes it from the stored state, so a driver never
has to carry it alongside.

```@example rock
Δφ = r.fluxes.φ - fluid_content(mat, st)
```

## Compression closes the fracture, pressure reopens it

Holding that compressive strain and raising the pore pressure reopens the
fracture, and the permeability recovers with it — the coupling the model exists
to capture:

| ``p`` (GPa) | 0 | 0.005 | 0.010 |
|:--|:--|:--|:--|
| ``\omega`` | 5.52·10⁻⁴ | 6.35·10⁻⁴ | 7.18·10⁻⁴ |
| ``k_{11}`` (m²) | 1.11·10⁻¹⁸ | 1.15·10⁻¹⁸ | 1.20·10⁻¹⁸ |

Once a family closes it leaves the intact matrix behind: ``\boldsymbol{B} = 0``,
``1/M = 0`` and [`transport_property`](@ref) returns `nothing` — the fracture
carries no flow at all.

!!! warning "Returning nothing is an answer, not a failure"
    A driver that forwards it straight into a mobility gets a `MethodError` at
    the worst possible moment. Test for it — the
    [well test](@ref fe-arma2011) does.
