# =============================================================================
#  conductive.jl — the FLOWING crack: a flat inclusion that conducts.
#
#  MeanFieldHomogenization's ordinary crack is *insulating* in transport: it
#  blocks the flux, and its contribution `−K₀·R·K₀` lowers the effective
#  conductivity. A fracture in a rock does the opposite — it is the preferential
#  flow path through an almost impermeable matrix — and that is the object the
#  hydraulic half of a fractured-reservoir model needs.
#
#  THE LIMIT.  Take an oblate spheroid of semi-axes (a, a, c), conductivity k_f,
#  and let it flatten while its conductivity diverges so that the **fracture
#  conductivity**
#
#      C = 2 c k_f          (conductivity × aperture)
#
#  stays finite. Writing γ = C / 2a, the dilute contribution per unit
#  (4π/3)·d in an ISOTROPIC matrix k₀ is
#
#      𝕂 = γ / (1 + π γ / (4 k₀)) · (δ − n̂⊗n̂) ,
#
#  purely in-plane: the normal component vanishes as O(ω). The ideal-fracture
#  limit C → ∞ gives 𝕂 = (4k₀/π)(δ − n̂⊗n̂), i.e. ΔK = (16/3) d k₀ in plane,
#  which is the classical result and the check the tests pin.
#
#  For an anisotropic reference medium there is no such closed form, and rather
#  than re-derive one the same limit is taken NUMERICALLY on the package's own
#  (untouched, validated) spheroid machinery, Richardson-extrapolated in ω. Two
#  Hill-tensor solves buy the general case with no new theory to get wrong.
# =============================================================================

"""
    ConductiveCrack(crack, conductivity)
    ConductiveCrack(a; conductivity, euler_angles = ())

A flat crack that **conducts** rather than blocks: the preferential flow path of
a fractured rock.

`conductivity` is the **fracture conductivity** ``C = 2c\\,k_f`` — the intrinsic
conductivity of the fluid-filled gap times its aperture, with units of
conductivity × length. It is the quantity that stays finite as the crack
flattens, and the one field data reports.

Mechanically a flowing crack is an ordinary open crack, so the whole elastic
branch (`cod_tensor`, ℍ, ℕ, the `delta_*` seam) is inherited unchanged from the
wrapped [`EllipticCrack`](@ref); only the transport response differs.

```julia
cr = ConductiveCrack(1.0; conductivity = 2.0e-13, euler_angles = (π/4, 0.0))
conductivity_contribution(cr, TensISO{3}(1.0e-18))     # 𝕂, POSITIVE, in-plane
```

!!! note "Penny shape only"
    The closed form above is derived for a circular crack. An elliptical one has
    a different in-plane structure and is rejected rather than approximated.
"""
struct ConductiveCrack{T <: Number, C <: EllipticCrack{T}} <: MFH_Core.AbstractCrack{T}
    crack::C
    conductivity::T

    # Inner constructor only: it validates, and it keeps the default
    # field-typed one from being ambiguous with the convenience method below.
    function ConductiveCrack(crack::EllipticCrack{T}, conductivity::Real) where {T}
        crack.a ≈ crack.b || throw(
            ArgumentError(
                "ConductiveCrack is derived for a circular (penny) crack; got " *
                    "a = $(crack.a), b = $(crack.b)."
            )
        )
        conductivity > 0 ||
            throw(ArgumentError("fracture conductivity must be > 0"))
        return new{T, typeof(crack)}(crack, T(conductivity))
    end
end

ConductiveCrack(a::Real; conductivity::Real, euler_angles = ()) =
    ConductiveCrack(PennyCrack(a; euler_angles = euler_angles), conductivity)

# ── Everything geometric is the wrapped crack ────────────────────────────────

MFH_Core.shape_trait(c::ConductiveCrack) = MFH_Core.shape_trait(c.crack)
MFH_Core.dimension(c::ConductiveCrack) = MFH_Core.dimension(c.crack)
MFH_Core.inclusion_basis(c::ConductiveCrack) = MFH_Core.inclusion_basis(c.crack)
crack_basis(c::ConductiveCrack) = crack_basis(c.crack)
crack_normal(c::ConductiveCrack) = crack_normal(c.crack)
aspect_ratio(c::ConductiveCrack) = aspect_ratio(c.crack)
semi_major(c::ConductiveCrack) = semi_major(c.crack)
semi_minor(c::ConductiveCrack) = semi_minor(c.crack)
crack_chi(c::ConductiveCrack) = crack_chi(c.crack)

"The fracture conductivity ``C = 2c\\,k_f`` of a [`ConductiveCrack`](@ref)."
fracture_conductivity(c::ConductiveCrack) = c.conductivity

"""
    with_conductivity(c::ConductiveCrack, C) -> ConductiveCrack

The same fracture with a new conductivity — how the cubic aperture law
``C_i = C_i^0 (\\omega_i/\\omega_i^0)^3`` is applied during a simulation.
"""
with_conductivity(c::ConductiveCrack, C::Real) = ConductiveCrack(c.crack, C)

# Elasticity: identical to the wrapped crack.
cod_tensor(c::ConductiveCrack, C₀::TensND.AbstractTens{4, 3}; kw...) =
    cod_tensor(c.crack, C₀; kw...)

# ── Transport: the conductive limit ──────────────────────────────────────────

"""
    conductivity_contribution(c::ConductiveCrack, K₀; kw...) -> Tens{2,3}

Size-independent **conductivity contribution** ``\\mathbb K`` of a flowing
crack, such that `delta_conductivity(c, 𝕂, d)` ``= (4\\pi/3)\\,d\\,\\mathbb K``
is the dilute correction to the effective conductivity.

Positive and purely in-plane, in contrast with the insulating crack whose
contribution is negative and normal to the crack plane.
"""
MFH_Core.conductivity_contribution(
    c::ConductiveCrack, K₀::TensND.AbstractTens{2, 3}; kw...
) = _conductive_crack_K(c, K₀)

# Isotropic reference: the closed form.
function _conductive_crack_K(c::ConductiveCrack, K₀::TensND.TensISO{2, 3})
    k₀ = MFH_Core.extract_iso_conductivity(K₀)
    γ = c.conductivity / (2 * semi_major(c))
    T = float(promote_type(typeof(γ), typeof(k₀)))
    κ = T(γ) / (one(T) + T(π) * T(γ) / (4 * T(k₀)))
    # (δ − n̂⊗n̂) in the crack basis, where n̂ is e₃ by construction.
    P = Tensors.SymmetricTensor{2, 3}((i, j) -> (i == j && i < 3) ? one(T) : zero(T))
    return TensND.Tens(κ * P, crack_basis(c))
end

# Anisotropic reference: the ω → 0 limit of the package's own spheroid
# contribution, Richardson-extrapolated. `𝕂(ω) = 𝕂 + 𝒪(ω)`, so combining ω and
# 10ω as `(10𝕂(ω) − 𝕂(10ω))/9` cancels the leading error — checked against the
# closed form above to seven digits.
function _conductive_crack_K(c::ConductiveCrack, K₀::TensND.AbstractTens{2, 3})
    ω = 1.0e-4
    return (10 * _flat_spheroid_limit(c, K₀, ω) - _flat_spheroid_limit(c, K₀, 10ω)) / 9
end

function _flat_spheroid_limit(c::ConductiveCrack, K₀, ω)
    a = semi_major(c)
    k_f = c.conductivity / (2 * a * ω)          # C = 2·c·k_f with c = aω
    ell = Elasticity.Ellipsoid(a, a, a * ω, crack_basis(c))
    N = MFH_Core.conductivity_contribution(ell, TensND.TensISO{3}(k_f), K₀)
    return ω * N
end
