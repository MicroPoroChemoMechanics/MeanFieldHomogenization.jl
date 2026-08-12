# =============================================================================
#  poroelastic.jl — the fractured-rock material of Barthelemy & Daniel (2011).
#
#  A solid matrix holding fracture families, saturated by an incompressible
#  fluid. Two gradients go in, two fluxes come out:
#
#      Σ̇ = ℂ_hom : Ė − ṗ 𝐁 ,        φ̇ = 𝐁 : Ė + ṗ / M ,
#
#  and the permeability follows the apertures, so the flow problem the material
#  is embedded in sees a coefficient that changes with the loading.
#
#  WHAT DRIVES THE MICROSTRUCTURE.  Not Σ, and not the Biot effective stress,
#  but the **Terzaghi** one, Σ' = Σ + p 𝟏. The loading (Σ, p) splits into a dry
#  problem under Σ' — during which fractures open and close — plus a uniform
#  field carrying no strain singularity, which cannot move a flat crack. That is
#  the argument of §1.1 of the paper, and it is why a single mechanical state
#  suffices for the whole coupled problem.
#
#  In terms of the gradients the finite-element code hands over,
#
#      Σ' = Σ + p 𝟏 = ℂ_hom : E + p (𝟏 − 𝐁) ,
#
#  so the aperture update sees the strain increment through ℂ_hom *and* the
#  pressure increment through (𝟏 − 𝐁): with an incompressible solid 𝐁 = 𝟏 and
#  the pressure drops out, as it must.
# =============================================================================

"""
    PoroFracturedState(mech, C)

Internal state of a [`FracturedPoroelasticRock`](@ref): the mechanical state of
the fracture families ([`CrackedState`](@ref)) and their current conductivities
`C[i]`.

`p` is the pore pressure the state was reached at, kept for the same reason the
mechanical state keeps its strain: the driver is an *increment*.

The conductivities are carried separately because they follow the apertures by
the cubic law ``C_i = C_i^0 (\\omega_i/\\omega_i^0)^3`` and feed a *different*
balance equation — Darcy's, through
[`transport_property`](@ref) — rather than the mechanical one.
"""
struct PoroFracturedState{N, S <: CrackedState{N}, T} <: AbstractMaterialState
    mech::S
    C::NTuple{N, T}
    p::T
end

open_set(st::PoroFracturedState) = open_set(st.mech)
apertures(st::PoroFracturedState) = apertures(st.mech)

"Current fracture conductivities ``C_i`` of a [`PoroFracturedState`](@ref)."
conductivities(st::PoroFracturedState) = st.C

"""
    FracturedPoroelasticRock(rve, scheme; ω₀, C₀, k_matrix, porosity_ref, kw...)

The saturated fractured rock of [barthelemyARMA2011](@cite): a Gauss-point
material with **two gradients** ``(\\boldsymbol{E}, p)`` and **two fluxes**
``(\\boldsymbol{\\Sigma}, \\varphi)``, whose fractures open and close and whose
permeability follows their apertures.

- `rve` — matrix plus crack families. Families given as
  [`ConductiveCrack`](@ref MeanFieldHomogenization.Cracks.ConductiveCrack) also carry
  the hydraulic side; ordinary cracks make the material purely poroelastic and
  [`transport_property`](@ref) returns `nothing`.
- `ω₀` — initial aspect ratios, as for [`MicrocrackedMaterial`](@ref).
- `k_matrix` — matrix conductivity, small but **non-zero** (see
  [`fracture_permeability`](@ref MeanFieldHomogenization.Schemes.fracture_permeability)).

The tangent blocks are `:σε` ``= \\mathbb{C}^{\\rm hom}``, `:σp` ``= -\\boldsymbol{B}``,
`:φε` ``= \\boldsymbol{B}`` and `:φp` ``= 1/M``, all recomputed whenever the
open/closed set changes and cached on it.

```julia
mat = FracturedPoroelasticRock(rve, SelfConsistent(); ω₀ = (1.0e-4,), k_matrix = 1.0e-18)
r = material_response(mat, (; ε = ε, p = p), st, Δt; cache = cache)
r.fluxes.σ, r.fluxes.φ, r.tangents.σp, transport_property(mat, r.state)
```

!!! note "Incompressible fluid"
    ``1/M`` comes from [`inverse_biot_modulus`](@ref MeanFieldHomogenization.Poromechanics.inverse_biot_modulus), which assumes
    ``k_f = \\infty`` — the setting of the paper. See that docstring for the
    compressible generalization.
"""
struct FracturedPoroelasticRock{N, M <: MicrocrackedMaterial{N}, T} <: AbstractMFHMaterial
    mech::M
    C₀::NTuple{N, T}
    k_matrix::T
    porosity_ref::T
end

function FracturedPoroelasticRock(
        rve, scheme; ω₀ = nothing, k_matrix::Real = 1.0e-18,
        porosity_ref::Real = 0.0, kw...
    )
    mech = MicrocrackedMaterial(rve, scheme; ω₀ = ω₀, kw...)
    C₀ = map(mech.families) do name
        geom = rve.phases[name].geometry
        geom isa Cracks.ConductiveCrack ? Cracks.fracture_conductivity(geom) : 0.0
    end
    T = float(promote_type(eltype(mech.ω₀), eltype(C₀), typeof(k_matrix)))
    return FracturedPoroelasticRock{length(C₀), typeof(mech), T}(
        mech, map(T, C₀), T(k_matrix), T(porosity_ref)
    )
end

gradient_names(::FracturedPoroelasticRock) = (:ε, :p)
flux_names(::FracturedPoroelasticRock) = (:σ, :φ)
tangent_blocks(::FracturedPoroelasticRock) = (:σε, :σp, :φε, :φp)

initial_state(m::FracturedPoroelasticRock{N}) where {N} =
    PoroFracturedState(initial_state(m.mech), m.C₀, zero(eltype(m.C₀)))

function material_response(
        m::FracturedPoroelasticRock{N}, gradients::NamedTuple,
        st::PoroFracturedState{N}, Δt::Real; cache = nothing
    ) where {N}
    ε = gradients.ε
    p = get(gradients, :p, zero(eltype(ε)))

    # Poroelastic parameters of the CURRENT configuration: the aperture update
    # needs 𝐁 to form the Terzaghi increment, and 𝐁 depends on the open set.
    C_hom, par = _poro_parameters(m, open_set(st), cache)

    # Σ' = ℂ_hom : E + p (𝟏 − 𝐁). The pressure part of the driver vanishes for an
    # incompressible solid (𝐁 = 𝟏), which is the paper's own consistency check.
    𝟏 = TensND.tens_Id2(Val(3), Val(Float64))
    Δp = p - st.p
    Δσ_extra = Δp * _to_canonical(𝟏 - par.B)

    mech = _advance_cracks(m.mech, ε, Δσ_extra, st.mech, cache)

    # Conductivities follow the apertures by the cubic law.
    ω, ω₀ = apertures(mech), m.mech.ω₀
    Cnew = ntuple(i -> m.C₀[i] * (ω[i] / ω₀[i])^3, N)

    C_hom, par = _poro_parameters(m, open_set(mech), cache)
    σ = mech.σ - p * _to_canonical(par.B)
    φ = (par.B ⊡ ε) + p * par.inverse_modulus

    return MaterialResponse(
        (σ = σ, φ = φ),
        (σε = C_hom, σp = -par.B, φε = par.B, φp = par.inverse_modulus),
        PoroFracturedState(mech, Cnew, p),
    )
end

"""
    fluid_content(m::FracturedPoroelasticRock, st; cache = nothing) -> Real

Fluid-content variation ``\\varphi = \\boldsymbol{B} : \\boldsymbol{E} + p/M``
**at** the state `st`, from the natural initial state.

A transient flow problem discretizes ``\\dot\\varphi``, so its residual needs
``\\varphi`` at both ends of the step. [`material_response`](@ref) returns the
new one; this returns the old one, recomputed from the stored strain and
pressure with the poroelastic parameters of that state's own open set — which
is why a driver never has to carry ``\\varphi`` alongside the state.

```julia
Δφ = material_response(m, (; ε, p), st_old, Δt).fluxes.φ - fluid_content(m, st_old)
```
"""
function fluid_content(
        m::FracturedPoroelasticRock{N}, st::PoroFracturedState{N}; cache = nothing
    ) where {N}
    _, par = _poro_parameters(m, open_set(st), cache)
    return (par.B ⊡ st.mech.ε) + st.p * par.inverse_modulus
end

"""
    transport_property(m::FracturedPoroelasticRock, st) -> Union{Nothing,Tens{2,3}}

Effective permeability implied by the current apertures, through
[`fracture_permeability`](@ref MeanFieldHomogenization.Schemes.fracture_permeability).

`nothing` when no family is a
[`ConductiveCrack`](@ref MeanFieldHomogenization.Cracks.ConductiveCrack): the material
is then purely poroelastic and the flow problem is not its business.
"""
function transport_property(m::FracturedPoroelasticRock{N}, st::PoroFracturedState{N}) where {N}
    fams, dens = Any[], Any[]
    for (i, name) in enumerate(m.mech.families)
        geom = m.mech.rve.phases[name].geometry
        geom isa Cracks.ConductiveCrack || continue
        open_set(st)[i] || continue                    # a closed fracture carries no flow
        push!(fams, Cracks.with_conductivity(geom, st.C[i]))
        push!(dens, Schemes.crack_density(m.mech.rve, name))
    end
    isempty(fams) && return nothing
    return Schemes.fracture_permeability(m.k_matrix, fams, dens)
end

# ── Internals ────────────────────────────────────────────────────────────────

# `C_hom` and the poroelastic parameters of one open/closed configuration,
# memoized on it like everything else that depends only on that set.
function _poro_parameters(m::FracturedPoroelasticRock, open, cache)
    return cached!(cache, (objectid(m), :poro, open)) do
        C_hom, _ = _cracked_state_tensors(m.mech, open, cache)
        C_s = _to_canonical(Schemes.matrix_property(m.mech.rve, m.mech.property))
        (C_hom, Poromechanics.poroelastic_parameters(C_hom, C_s, m.porosity_ref))
    end
end
