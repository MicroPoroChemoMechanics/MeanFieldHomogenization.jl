# =============================================================================
#  arc.jl — the circular arc crack: a crack along an arc of circle of radius R
#  and half-angle α, unbounded along the axis of the circle. It is not flat,
#  so it has no COD tensor; its contributions are integrated from its intensity
#  factors (Rice), in closed form.
# =============================================================================

"""
    ArcShape

Shape tag of a [`CircularArcCrack`](@ref): a cylindrical crack, unbounded along
its generators, normalized as a ribbon crack by its half-length ``R\\alpha``:
density ``\\varepsilon = N (R\\alpha)^2`` per unit area of the cross section and
``\\Delta\\mathbb S = \\pi\\,\\varepsilon\\,\\mathbb H``.
"""
struct ArcShape end

function _check_arc(R, α)
    is_hard_numeric(typeof(R)) || return nothing
    R > 0 && 0 < α ≤ π || throw(
        ArgumentError(
            "a circular arc crack needs a radius R > 0 and a half-angle 0 < α ≤ π, " *
                "got R = $R, α = $α. For α → 0 use `RibbonCrack(R * α)`."
        )
    )
    return nothing
end

"""
    CircularArcCrack{T, B<:AbstractBasis}

Crack along the arc of circle of radius ``R`` and half-angle ``\\alpha``
(``0 < \\alpha \\le \\pi``), unbounded along the axis of the circle. In its frame
``(\\underline{\\ell}, \\underline{m}, \\underline{n})``, ``\\underline{\\ell}`` is
the axis, ``\\underline{m}`` the direction of the chord and ``\\underline{n}`` the
normal at the middle of the arc, pointing away from the center: the arc is
``\\underline{x} = R(\\sin\\theta\\,\\underline{m} + \\cos\\theta\\,\\underline{n})``,
``|\\theta| \\le \\alpha``.

``\\alpha \\to 0`` recovers the [`RibbonCrack`](@ref) of half-width ``R\\alpha``
with the same frame; ``\\alpha = \\pi`` closes the arc into a cylindrical cut.
"""
struct CircularArcCrack{T <: Number, B <: TensND.AbstractBasis} <: MFH_Core.AbstractCrack{T}
    R::T
    α::T
    basis::B
    function CircularArcCrack{T, B}(R, α, basis) where {T <: Number, B <: TensND.AbstractBasis}
        _check_arc(R, α)
        return new{T, B}(R, α, basis)
    end
end


"""
    CircularArcCrack(R, α; euler_angles=())
    CircularArcCrack(R, α, R::AbstractMatrix)
    CircularArcCrack(R, α, basis::AbstractBasis)

Circular arc crack of radius `R` and half-angle `α`, its frame
``(\\underline{\\ell}, \\underline{m}, \\underline{n})`` given by ZYZ Euler angles,
the columns of a rotation matrix, or a TensND basis.
"""
function CircularArcCrack(R::TR, α::Tα; euler_angles::Tuple{Vararg{Real}} = ()) where {TR, Tα}
    T = TensND.float_like(promote_type(TR, Tα))
    return CircularArcCrack(T(R), T(α), TensND.euler_basis(euler_angles; T = T))
end
CircularArcCrack(R, α, Q::AbstractMatrix) =
    CircularArcCrack(R, α, TensND.RotatedBasis(Matrix{float(eltype(Q))}(Q)))
function CircularArcCrack(R::TR, α::Tα, basis::TensND.AbstractBasis) where {TR, Tα}
    T = TensND.float_like(promote_type(TR, Tα))
    return CircularArcCrack{T, typeof(basis)}(T(R), T(α), basis)
end

MFH_Core.shape_trait(::CircularArcCrack) = ArcShape
_crack_density_factor(::Type{ArcShape}, ::Type{T}) where {T} = T(π)
_rice_size_power(::Type{ArcShape}) = 2

Base.:(==)(x::T, y::T) where {T <: CircularArcCrack} = x.R == y.R && x.α == y.α && x.basis == y.basis
Base.hash(x::CircularArcCrack, h::UInt) = hash(x.basis, hash(x.α, hash(x.R, hash(typeof(x), h))))
Base.show(io::IO, c::CircularArcCrack{T}) where {T} =
    print(io, "CircularArcCrack{", T, "} (radius R=", c.R, ", half-angle α=", c.α, ")")

# ── Intensity factors ───────────────────────────────────────────────────────

# Influence arrays at the tip θ = sα (s = ±1), in the frame (ℓ, m, n) = (1, 2, 3).
# In-plane modes after Cotterell & Rice (1980) as written by Kachanov & Sevostianov
# (2018, eq. 1.9.27), with σ₁₁ → Σ_mm, σ₂₂ → Σ_nn and σ₁₂ → Σ_mn for the tip
# θ = +α, the other tip by the mirror m → −m:
#   K_{I,II} = √(πR sin α) [A (Σmm + Σnn) + D (Σnn − Σmm) + s C Σmn];
# anti-plane mode by the analogy with conduction (eq. 1.10.9, k₀ → μ):
#   K_III = √(πR sin α) [cos(α/2) Σℓn + s sin(α/2) Σℓm].
# Each mode is G:Σ with G symmetric; 𝒦_qij = n_q G^I_ij + ν_q G^II_ij + τ_q G^III_ij.
function _arc_influence(R, α, s)
    T = promote_type(typeof(R), typeof(α))
    z = zero(T)
    sq = sqrt(T(π) * R * sin(α))
    c1, s1 = cos(α / 2), sin(α / 2)
    c3, s3 = cos(3α / 2), sin(3α / 2)
    w = (s1 * c1)^2 / 2
    den = 1 + s1^2
    GI = _arc_mode(sq * c1 / (2den), sq * (c3 / 2 - w * c1 / den), -s * sq * (s3 + s1^3))
    GII = _arc_mode(sq * s1 / (2den), sq * (s3 / 2 - w * s1 / den), s * sq * (c3 + c1 * s1^2))
    GIII = SMatrix{3, 3, T}(z, s * sq * s1 / 2, sq * c1 / 2, s * sq * s1 / 2, z, z, sq * c1 / 2, z, z)
    θ = s * α
    n_t = SVector(z, sin(θ), cos(θ))
    ν_t = s * SVector(z, cos(θ), -sin(θ))
    τ_t = n_t × ν_t
    𝒦 = SArray{Tuple{3, 3, 3}, T}(
        ntuple(Val(27)) do idx
            q, i, j = Tuple(CartesianIndices((3, 3, 3))[idx])
            n_t[q] * GI[i, j] + ν_t[q] * GII[i, j] + τ_t[q] * GIII[i, j]
        end
    )
    return (; 𝒦, n = n_t, ν = ν_t, τ = τ_t)
end

# The symmetric G with G:Σ = A (Σmm + Σnn) + D (Σnn − Σmm) + C Σmn.
function _arc_mode(A, D, C)
    z = zero(promote_type(typeof(A), typeof(D), typeof(C)))
    return SMatrix{3, 3}(z, z, z, z, A - D, C / 2, z, C / 2, A + D)
end

# The flux intensity factor at the tip `s`: K_T = 𝐤⋅σ∞ (eq. 1.10.9 with q → −σ∞).
function _arc_flux_influence(R, α, s)
    T = promote_type(typeof(R), typeof(α))
    sq = sqrt(T(π) * R * sin(α))
    θ = s * α
    k = sq * SVector(zero(T), s * sin(α / 2), cos(α / 2))
    return (; k, n = SVector(zero(T), sin(θ), cos(θ)), ν = s * SVector(zero(T), cos(θ), -sin(θ)))
end

"""
    sif(crack::CircularArcCrack, C₀, Σ; tip=1) -> (𝐊, (Kᴵ, Kᴵᴵ, Kᴵᴵᴵ))
    sif(crack::CircularArcCrack, K₀, σ∞; tip=1) -> K_T

Intensity factors of a circular arc crack at its tip ``\\theta = +\\alpha``
(`tip = 1`) or ``\\theta = -\\alpha`` (`tip = -1`), in the conventions of
[`sif`](@ref): ``\\underline{K}`` on (``\\underline{n}``, ``\\underline{\\nu}``,
``\\underline{\\tau}``) at the tip. In-plane modes of [sih1962](@citet), corrected
by [cotterell1980](@citet), as given by [kachanov2018; eq. (1.9.27)](@citet);
anti-plane mode by the analogy with
conduction [kachanov2018; eq. (1.10.9)](@cite). They do not depend on the
moduli of the isotropic matrix.
"""
function sif(crack::CircularArcCrack, C₀::TensND.AbstractTens{4, 3}, Σ; tip::Integer = 1, kw...)
    _iso_matrix(C₀, crack)
    g = _arc_influence(crack.R, crack.α, _tip_sign(tip))
    S = SMatrix{3, 3}(TensND.components(Σ, crack.basis))
    K = SVector(ntuple(q -> sum(g.𝒦[q, i, j] * S[i, j] for i in 1:3, j in 1:3), 3))
    𝐊 = TensND.Tens(K, crack.basis)
    return 𝐊, (K ⋅ g.n, K ⋅ g.ν, K ⋅ g.τ)
end

function sif(crack::CircularArcCrack, K₀::TensND.AbstractTens{2, 3}, σ∞; tip::Integer = 1, kw...)
    _iso_matrix(K₀, crack)
    g = _arc_flux_influence(crack.R, crack.α, _tip_sign(tip))
    return g.k ⋅ SVector{3}(TensND.components(σ∞, crack.basis))
end

_tip_sign(tip::Integer) = tip == 1 ? 1 : tip == -1 ? -1 :
    throw(ArgumentError("`tip` is 1 (θ = +α) or -1 (θ = -α), got $tip"))

# ── Contributions ───────────────────────────────────────────────────────────

"""
    inverse_property_contribution(crack::CircularArcCrack, C₀; method=:analytical, reltol=1e-8)
    inverse_property_contribution(crack::CircularArcCrack, K₀; method=:analytical, reltol=1e-8)

Compliance contribution ``\\mathbb H`` (or resistivity contribution
``\\boldsymbol{R}``) of a circular arc crack in an isotropic matrix, normalized as a
ribbon by its half-length: ``\\Delta\\mathbb S = \\pi N (R\\alpha)^2\\,\\mathbb H``.

`method = :analytical` evaluates the closed forms obtained by integrating the
intensity factors along the growth of both tips from ``\\alpha = 0``
[kachanov2018; eqs. (1.9.28) and (1.10.12)](@cite); `method = :rice` performs the
same integration numerically ([`rice_cod_tensor`](@ref) with the influence of the
whole stress, the crack not being flat). The closed forms accept symbolic radius,
angle and moduli.
"""
function inverse_property_contribution(
        crack::CircularArcCrack, C₀::TensND.AbstractTens{4, 3};
        method::Symbol = :analytical, reltol = 1.0e-8, kw...
    )
    _iso_matrix(C₀, crack)
    # a static branch per value of the run-time Symbol keeps the result inferred
    return method === :analytical ? _arc_compliance(Val(:analytical), crack, C₀, reltol) :
        method === :rice ? _arc_compliance(Val(:rice), crack, C₀, reltol) : _arc_method_error(method)
end

function inverse_property_contribution(
        crack::CircularArcCrack, K₀::TensND.AbstractTens{2, 3};
        method::Symbol = :analytical, reltol = 1.0e-8, kw...
    )
    _iso_matrix(K₀, crack)
    return method === :analytical ? _arc_resistivity(Val(:analytical), crack, K₀, reltol) :
        method === :rice ? _arc_resistivity(Val(:rice), crack, K₀, reltol) : _arc_method_error(method)
end

function _arc_compliance(::Val{:analytical}, crack, C₀, _)
    E, ν = TensND.E_nu(C₀)
    R, α = crack.R, crack.α
    T = promote_type(typeof(E), typeof(ν), typeof(R), typeof(α))
    Es = E / (1 - ν^2)
    μ = E / (2 * (1 + ν))
    c = cos(α)
    norm = T(π) * (R * α)^2
    # V Δ𝕊 of one crack (Kachanov & Sevostianov 2018, eqs. 1.9.28 and 1.10.12 with k₀ → μ)
    p = T(π) * R^2 * (1 - c) / Es
    Smmmm = p * (one(T) / 4 - c - c^2 / 4 + 2 / (3 - c))
    Snnnn = p * (-T(15) / 4 - c - c^2 / 4 + 18 / (3 - c))
    Smmnn = p * (T(7) / 4 + c + c^2 / 4 - 6 / (3 - c))
    Smnmn = p * (5 - c^2) / 4
    Sℓmℓm = T(π) * R^2 * (1 - c)^2 / (8μ)
    Sℓnℓn = T(π) * R^2 * (1 - c) * (3 + c) / (8μ)
    H = zeros(T, 3, 3, 3, 3)
    H[2, 2, 2, 2] = Smmmm / norm
    H[3, 3, 3, 3] = Snnnn / norm
    H[2, 2, 3, 3] = H[3, 3, 2, 2] = Smmnn / norm
    for (i, j, k, l) in ((2, 3, 2, 3), (3, 2, 2, 3), (2, 3, 3, 2), (3, 2, 3, 2))
        H[i, j, k, l] = Smnmn / norm
    end
    for (i, j, k, l) in ((1, 2, 1, 2), (2, 1, 1, 2), (1, 2, 2, 1), (2, 1, 2, 1))
        H[i, j, k, l] = Sℓmℓm / norm
    end
    for (i, j, k, l) in ((1, 3, 1, 3), (3, 1, 1, 3), (1, 3, 3, 1), (3, 1, 3, 1))
        H[i, j, k, l] = Sℓnℓn / norm
    end
    return TensND.Tens(TensND.Tensors.SymmetricTensor{4, 3, T}((i, j, k, l) -> H[i, j, k, l]), crack.basis)
end

function _arc_resistivity(::Val{:analytical}, crack, K₀, _)
    k₀ = first(TensND.get_data(K₀))
    R, α = crack.R, crack.α
    T = promote_type(typeof(k₀), typeof(R), typeof(α))
    c = cos(α)
    norm = T(π) * (R * α)^2
    rmm = T(π) * R^2 * (1 - c)^2 / (2k₀)
    rnn = T(π) * R^2 * (1 - c) * (3 + c) / (2k₀)
    z = zero(T)
    return TensND.Tens(TensND.Tensors.SymmetricTensor{2, 3, T}((z, z, z, rmm / norm, z, rnn / norm)), crack.basis)
end

# Both tips advance by R dα′ as the half-angle grows from 0 to α.
_arc_front(R) = α′ -> map(s -> (R, _arc_influence(R, α′, s).ν, _arc_influence(R, α′, s).n, _arc_influence(R, α′, s).𝒦), (1, -1))
_arc_front_thermal(R) = α′ -> map(s -> (g = _arc_flux_influence(R, α′, s); (R, g.ν, g.n, g.k)), (1, -1))

_arc_compliance(::Val{:rice}, crack, C₀, reltol) = _rice_compliance(
    _arc_front(crack.R), zero(crack.α), crack.α, C₀, ArcShape;
    size = crack.R * crack.α, basis = crack.basis, reltol
)
_arc_resistivity(::Val{:rice}, crack, K₀, reltol) = _rice_compliance(
    _arc_front_thermal(crack.R), zero(crack.α), crack.α, K₀, ArcShape;
    size = crack.R * crack.α, basis = crack.basis, reltol
)
_arc_method_error(M) = throw(
    ArgumentError("unknown method `:$M` for a circular arc crack: use `:analytical` or `:rice`")
)

# The arc has no COD tensor: the bundle is assembled from ℍ directly.
function inverse_and_property_contribution(crack::CircularArcCrack, C₀::TensND.AbstractTens{4, 3}; kw...)
    H = inverse_property_contribution(crack, C₀; kw...)
    return (H, -(C₀ ⊡ H ⊡ C₀))
end
function inverse_and_property_contribution(crack::CircularArcCrack, K₀::TensND.AbstractTens{2, 3}; kw...)
    R = inverse_property_contribution(crack, K₀; kw...)
    return (R, -(K₀ ⋅ R ⋅ K₀))
end
