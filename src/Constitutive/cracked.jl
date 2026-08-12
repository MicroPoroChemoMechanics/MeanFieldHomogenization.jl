# =============================================================================
#  cracked.jl — a microcracked material whose fractures open and close.
#
#  This is the mechanical half of the fractured-reservoir model of
#  Barthelemy & Daniel (ARMA 2011): a solid matrix holding `n` families of flat
#  cracks, each with its own normal, radius, density and aperture. The
#  nonlinearity is entirely geometric — a family is either open (it softens the
#  medium) or closed (it does not, being able to transmit normal and tangential
#  tractions across its faces).
#
#  WHY THE TANGENT IS EXACT.  Between two closure or reopening events, the set of
#  open families does not change, so `C_hom` is constant and the law is strictly
#  linear. The consistent tangent is therefore `C_hom` itself, not an
#  approximation of it, and nothing has to be differentiated. A step that does
#  cross an event is split at the event, which is what `_substep` below does.
#
#  THE APERTURE UPDATE.  Writing `S_hom = S_solid + Σ_i (4π/3) d_i 𝕊_i` (see
#  `Schemes.crack_family_compliances`), family `i` carries the average strain
#  `f_i ⟨ε⟩_i = (4π/3) d_i 𝕊_i : Σ`. With `f_i = (4π/3) d_i ω_i` at constant
#  radius, `⟨ε⟩_{i,nn} = Δω_i / ω_i`, hence
#
#      Δω_i = n̂_i · (𝕊_i : ΔΣ) · n̂_i .
#
#  That is eq. (10) of the paper in its short form: the same `𝕊_i` assembles
#  `S_hom` and drives the aperture, so one identity validates both.
# =============================================================================

"""
    CrackedState(ω, open, ε, σ)

Internal state of a [`MicrocrackedMaterial`](@ref): the current aspect ratio
`ω[i]` of each crack family, whether it is `open[i]`, and the strain `ε` and
stress `σ` the state was reached at.

The law is **incremental**, so both `ε` and `σ` are carried: `ε` to form the
strain increment that drives the apertures, and `σ` because the stress must be
integrated along the path rather than recomputed as
``\\mathbb{C}^{\\rm hom} : \\boldsymbol\\varepsilon``. Those two differ as soon
as a family closes — the medium stiffens from that point on, and only from that
point on, so a total-strain formula would make the stress jump discontinuously
at closure.
"""
struct CrackedState{N, T, E, S} <: AbstractMaterialState
    ω::NTuple{N, T}
    open::NTuple{N, Bool}
    ε::E
    σ::S
end

"""
    open_set(st::CrackedState) -> NTuple{N,Bool}

Which crack families are currently open. This tuple — and nothing else about the
state — is what the homogenized stiffness depends on, which is what makes
[`MaterialCache`](@ref) effective: at most `2^N` distinct stiffnesses exist,
however many quadrature points there are.
"""
open_set(st::CrackedState) = st.open

"""
    apertures(st::CrackedState) -> NTuple{N,T}

Current aspect ratios ``\\omega_i = c_i / a_i`` of the crack families.
"""
apertures(st::CrackedState) = st.ω

"""
    MicrocrackedMaterial(rve, scheme; families, ω₀, kw...)

Gauss-point material for a solid holding crack families that **open and close**
with the loading.

- `rve` — the microstructure, whose crack phases are the families. Their
  densities, normals and radii are fixed data; only the apertures evolve.
- `scheme` — any scheme supporting the per-family decomposition
  ([`SelfConsistent`](@ref MeanFieldHomogenization.SelfConsistent),
  [`MoriTanaka`](@ref MeanFieldHomogenization.MoriTanaka)).
- `families` — the crack phase names, in order. Defaults to every
  [`CrackDensity`](@ref MeanFieldHomogenization.Schemes.CrackDensity) phase of
  `rve`.
- `ω₀` — initial aspect ratio of each family. A family closes when its aspect
  ratio reaches zero, so `ω₀` sets how much compression it tolerates.

The law is piecewise linear: within a branch the tangent is exactly
``\\mathbb{C}^{\\rm hom}`` of the current open set, and a step crossing a closure
is split at the crossing.

```julia
rve = RVE(:M)
add_matrix!(rve, Ellipsoid(1.0), Dict(:C => C₀))
add_phase!(rve, :F1, PennyCrack(1.0), Dict(:C => C₀); density = 0.1)

mat = MicrocrackedMaterial(rve, MoriTanaka(); ω₀ = (1.0e-3,))
r   = material_response(mat, ε, initial_state(mat), 0.0; cache = MaterialCache())
```

!!! note "Closed means invisible, not welded"
    A closed family is dropped from the homogenization, i.e. it transmits both
    normal and tangential tractions. Frictional sliding on closed faces is a
    different model and is not what this material does.

See also [`HomogenizedElastic`](@ref) for the linear control case, and
[`crack_family_compliances`](@ref MeanFieldHomogenization.Schemes.crack_family_compliances)
for the decomposition the aperture update rests on.
"""
struct MicrocrackedMaterial{N, S, R, T, K} <: AbstractMFHMaterial
    rve::R
    scheme::S
    families::NTuple{N, Symbol}
    ω₀::NTuple{N, T}
    property::Symbol
    kw::K
end

function MicrocrackedMaterial(
        rve, scheme;
        families = nothing, ω₀ = nothing, property::Symbol = :C, kw...
    )
    fams = families === nothing ? _crack_phase_names(rve) : Tuple(families)
    isempty(fams) && throw(
        ArgumentError(
            "MicrocrackedMaterial needs at least one crack family; the RVE has " *
                "no phase carrying a CrackDensity."
        )
    )
    N = length(fams)
    w = ω₀ === nothing ? ntuple(_ -> 1.0e-3, N) : Tuple(ω₀)
    length(w) == N || throw(
        ArgumentError("ω₀ has $(length(w)) entries but there are $N crack families")
    )
    all(>(0), w) || throw(ArgumentError("every initial aspect ratio ω₀ must be > 0"))
    return MicrocrackedMaterial{N, typeof(scheme), typeof(rve), eltype(w), typeof(kw)}(
        rve, scheme, fams, w, property, kw
    )
end

_crack_phase_names(rve) = Tuple(
    n for n in Schemes.inclusion_phase_names(rve)
        if rve.amounts[n] isa Schemes.CrackDensity
)

function initial_state(m::MicrocrackedMaterial{N}) where {N}
    z = _zero_tens2(eltype(m.ω₀))
    return CrackedState(m.ω₀, ntuple(_ -> true, N), z, z)
end

function material_response(
        m::MicrocrackedMaterial{N}, gradients::NamedTuple,
        st::CrackedState{N}, ::Real; cache = nothing
    ) where {N}
    ε = gradients.ε
    Δε_total = ε - st.ε
    ω = st.ω
    open = st.open
    σ = st.σ
    remaining = one(eltype(m.ω₀))
    local C_hom

    # Walk the step, splitting it at every event so each piece is traversed with
    # a constant open set — hence a constant stiffness — and INTEGRATING the
    # stress piece by piece. Recomputing `C_hom : ε` at the end instead would
    # make the stress jump at closure, since the medium only stiffens from the
    # closure point onwards.
    #
    # Closure and reopening are searched together. Detecting reopening only at
    # the end of the step would make the answer depend on how the loading was
    # subdivided, which is exactly what sub-stepping exists to avoid.
    for _ in 0:(2N)                   # each family can close and reopen once
        C_hom, 𝕊 = _cracked_state_tensors(m, open, cache)
        Δε_left = remaining * Δε_total
        Δσ = C_hom ⊡ Δε_left
        dω = _aperture_increments(m, 𝕊, Δσ, open)
        α, event = _next_event(m, ω, dω, σ, Δσ, open)

        σ = σ + C_hom ⊡ (α * Δε_left)
        ω = ntuple(i -> open[i] ? ω[i] + α * dω[i] : ω[i], N)
        remaining -= α * remaining
        event === nothing && break

        i, closing = event
        open = ntuple(k -> k == i ? !closing : open[k], N)
        ω = ntuple(
            k -> k == i ? (closing ? zero(eltype(ω)) : eps(eltype(ω))) : ω[k], N
        )
        remaining <= 0 && break
    end
    C_hom, _ = _cracked_state_tensors(m, open, cache)

    return MaterialResponse(
        (σ = σ,), (σε = C_hom,), CrackedState(ω, open, ε, σ)
    )
end

# A genuine zero `Tens{2,3}`: `zero(::TensISO)` narrows the class, and a
# structured zero would not promote the way an accumulator needs to.
_zero_tens2(::Type{T}) where {T} =
    TensND.Tens(Tensors.SymmetricTensor{2, 3}((i, j) -> zero(T)))

# ── Internals ────────────────────────────────────────────────────────────────

# `C_hom` and the per-family compliance contributions for one open/closed
# configuration, memoized on that configuration alone — the reason a scheme
# solve per Gauss point is affordable (see `MaterialCache`).
function _cracked_state_tensors(m::MicrocrackedMaterial, open, cache)
    return cached!(cache, (objectid(m), open)) do
        rve = _rve_with_open(m, open)
        C = Schemes.homogenize(rve, m.scheme, m.property; m.kw...)
        C_can = _to_canonical(C)   # preserves a structured type; see its docstring
        𝕊 = if any(open)
            dec = Schemes.crack_family_compliances(rve, m.scheme, C; property = m.property, m.kw...)
            Dict(k => v for (k, v) in dec.families)
        else
            Dict{Symbol, Any}()
        end
        (C_can, 𝕊)
    end
end

# A copy of the RVE holding only the still-open families. Closed cracks are
# invisible to the homogenization, which is exactly the paper's statement that
# they are reincorporated into the matrix.
function _rve_with_open(m::MicrocrackedMaterial, open)
    rve = m.rve
    kept = Schemes.RVE(
        rve.matrix_name; T = eltype(m.ω₀),
        distribution_shape = rve.distribution_shape
    )
    mp = rve.phases[rve.matrix_name]
    Schemes.add_matrix!(kept, mp.geometry, mp.properties)
    for name in Schemes.inclusion_phase_names(rve)
        idx = findfirst(==(name), m.families)
        idx !== nothing && !open[idx] && continue
        p = rve.phases[name]
        a = rve.amounts[name]
        if a isa Schemes.CrackDensity
            Schemes.add_phase!(
                kept, name, p.geometry, p.properties;
                density = Schemes.amount_value(a),
                symmetrize = Schemes.phase_symmetrize(rve, name)
            )
        else
            Schemes.add_phase!(
                kept, name, p.geometry, p.properties;
                fraction = Schemes.amount_value(a),
                symmetrize = Schemes.phase_symmetrize(rve, name)
            )
        end
    end
    return kept
end

# Δω_i = n̂_i · (𝕊_i : ΔΣ) · n̂_i, with n̂_i the crack normal in GLOBAL components
# (`_frame_columns`, not `crack_normal`, whose components are local).
function _aperture_increments(m::MicrocrackedMaterial{N}, 𝕊, Δσ, open) where {N}
    canon = TensND.CanonicalBasis{3, Float64}()
    return ntuple(N) do i
        open[i] || return zero(eltype(m.ω₀))
        S_i = get(𝕊, m.families[i], nothing)
        S_i === nothing && return zero(eltype(m.ω₀))
        e = TensND.change_tens(S_i ⊡ Δσ, canon)
        A = get_array(e)
        n̂ = MFH_Core._frame_columns(
            Cracks.crack_basis(m.rve.phases[m.families[i]].geometry)
        )[3]
        return sum(n̂[a] * A[a, b] * n̂[b] for a in 1:3, b in 1:3)
    end
end

"""
    _next_event(m, ω, dω, σ, Δσ, open) -> (α, event)

Largest fraction `α ∈ [0,1]` of the remaining increment that can be applied
before the open/closed configuration changes, and the event itself as
`(family, closing)` — or `nothing` when the whole increment fits on the current
branch.

Two kinds of event are searched together:

- **closure** of an open family, when its aspect ratio reaches zero,
  ``\\omega_i + \\alpha\\,\\Delta\\omega_i = 0``;
- **reopening** of a closed family, when the normal traction on its plane turns
  tensile, ``\\hat n_i \\cdot (\\sigma + \\alpha\\,\\Delta\\sigma) \\cdot
  \\hat n_i = 0`` with a positive rate.

The earliest of the two wins.
"""
function _next_event(m::MicrocrackedMaterial{N}, ω, dω, σ, Δσ, open) where {N}
    T = eltype(ω)
    α = one(T)
    event = nothing
    σn = _normal_tractions(m, σ)
    Δσn = _normal_tractions(m, Δσ)
    for i in 1:N
        αᵢ = if open[i]
            dω[i] < 0 ? -ω[i] / dω[i] : nothing         # only a closing family
        elseif Δσn[i] > 0                                # only an opening one
            σn[i] >= 0 ? zero(T) : -σn[i] / Δσn[i]
        else
            nothing
        end
        if αᵢ !== nothing && αᵢ < α
            α = αᵢ
            event = (i, open[i])
        end
    end
    return max(α, zero(T)), event
end

# `n̂_i · σ · n̂_i` for every family, with `n̂_i` in GLOBAL components.
function _normal_tractions(m::MicrocrackedMaterial{N}, σ) where {N}
    A = get_array(TensND.change_tens(σ, TensND.CanonicalBasis{3, eltype(σ)}()))
    return ntuple(N) do i
        n̂ = _family_normal(m, i)
        sum(n̂[a] * A[a, b] * n̂[b] for a in 1:3, b in 1:3)
    end
end

# The crack normal in GLOBAL components. `crack_normal` returns it in the
# crack's own basis, where it is trivially `(0,0,1)` — useless here.
_family_normal(m::MicrocrackedMaterial, i) = MFH_Core._frame_columns(
    Cracks.crack_basis(m.rve.phases[m.families[i]].geometry)
)[3]
