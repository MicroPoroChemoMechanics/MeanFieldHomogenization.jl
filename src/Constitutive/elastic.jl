# =============================================================================
#  elastic.jl — the linear homogenized material.
#
#  The simplest possible law built on a homogenization model, and the one that
#  validates the plumbing end to end: the stiffness is whatever a scheme returns
#  for the RVE, evaluated once, and the response is Hooke's law. There is no
#  internal state and the consistent tangent is exact by construction.
#
#  Its real job is to be the control case of an FE coupling: run it against a
#  problem with a closed-form solution (a thick-walled cylinder against Lamé)
#  and any discrepancy is in the coupling, not in the material.
# =============================================================================

"""
    HomogenizedElastic(cell, scheme; property = :C, kw...)
    HomogenizedElastic(C_hom)

Linear elastic Gauss-point material whose stiffness comes from a homogenization
scheme.

The scheme is run **once**, at construction, and the resulting `C_hom` is stored
in the **canonical frame** — an RVE with oriented inclusions returns its estimate
in a rotated basis, and a material handed to an FE code must speak the global
frame (see [`to_tensors`](@ref)).

```julia
rve = RVE(:M)
add_matrix!(rve, Ellipsoid(1.0), Dict(:C => TensISO{3}(3 * 30.0, 2 * 18.0)))
add_phase!(rve, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(3 * 90.0, 2 * 60.0)); fraction = 0.2)

mat = HomogenizedElastic(rve, MoriTanaka())
r   = material_response(mat, ε, initial_state(mat), 0.0)
σ, ℂ = stress(r), tangent(r)
```

The second form takes a stiffness directly, which is useful for a control run
or when the upscaling was done elsewhere.

Extra keyword arguments are forwarded to
[`homogenize`](@ref MeanFieldHomogenization.Core.homogenize), so solver
tolerances travel with the material:
`HomogenizedElastic(rve, SelfConsistent(); abstol = 1e-12)`.
"""
struct HomogenizedElastic{T, C <: TensND.AbstractTens{4, 3}} <: AbstractMFHMaterial
    C_hom::C
end

function HomogenizedElastic(C_hom::TensND.AbstractTens{4, 3})
    C = _to_canonical(C_hom)
    return HomogenizedElastic{eltype(C), typeof(C)}(C)
end

HomogenizedElastic(
    cell::MFH_Core.AbstractHomogenizationCell, scheme; property::Symbol = :C, kw...
) = HomogenizedElastic(Schemes.homogenize(cell, scheme, property; kw...))

"""
    stiffness(m::HomogenizedElastic) -> Tens{4,3}

The homogenized stiffness the material was built with, in the canonical frame.
"""
stiffness(m::HomogenizedElastic) = m.C_hom

initial_state(::HomogenizedElastic) = NoState()

function material_response(
        m::HomogenizedElastic, gradients::NamedTuple,
        ::AbstractMaterialState, ::Real; cache = nothing
    )
    ε = gradients.ε
    σ = m.C_hom ⊡ ε
    return MaterialResponse((σ = σ,), (σε = m.C_hom,), NoState())
end

"""
    _to_canonical(t) -> AbstractTens

Express a property in the canonical (global) frame — the frame an FE code reads
components in — whenever it is not already there.

The guard is not an optimization. `change_tens` rebuilds a structured tensor as
a generic `Tens`, and a structured type carries information its consumers rely
on: `k_mu` is only defined for `TensISO{4}`, and the algorithm dispatch keys on
the symmetry class. Since `TensISO`, `TensTI` and `TensOrtho` all store their
components in the canonical frame already (a `TensTI` keeps its symmetry axis
separately), converting them is both unnecessary and destructive.

Only a genuinely rotated tensor — what an RVE with tilted inclusions returns —
is converted, and there the conversion is what makes the material correct.
"""
_to_canonical(t::TensND.AbstractTens) =
    TensND.get_basis(t) isa TensND.CanonicalBasis ? t :
    TensND.change_tens(t, TensND.CanonicalBasis{3, eltype(t)}())
