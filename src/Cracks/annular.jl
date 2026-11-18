# =============================================================================
#  annular.jl — flat cracks whose fronts are circles: the annular crack
#  (a penny-shaped crack bonded on a central disc) and the user-defined
#  `RiceCrack`, both computed by Rice's formula (rice.jl).
# =============================================================================

"""
    CircularFronts

Shape tag of a flat crack whose fronts are concentric circles, normalized by its
outer radius ``a`` exactly as a penny-shaped crack: density ``\\varepsilon =
N a^3``, ``\\mathbb H = \\tfrac34\\,\\underline{n}\\stackrel{s}{\\otimes}\\boldsymbol{B}
\\stackrel{s}{\\otimes}\\underline{n}`` and ``\\Delta\\mathbb S = \\tfrac{4\\pi}{3}
\\varepsilon\\,\\mathbb H``. Carried by [`AnnularCrack`](@ref) and
[`RiceCrack`](@ref).
"""
struct CircularFronts <: CrackShape end

function _check_annulus(a, c)
    is_hard_numeric(typeof(a)) || return nothing
    0 < c < a || throw(
        ArgumentError(
            "an annular crack needs 0 < c < a (inner and outer radii), got a = $a, c = $c. " *
                "For c = 0 use `PennyCrack(a)`."
        )
    )
    return nothing
end

"""
    AnnularCrack{T, B<:AbstractBasis}

Flat crack occupying the annulus ``c < r < a`` of its plane, of normal
``\\underline{n}`` (third vector of `basis`): a penny-shaped crack of radius ``a``
whose faces are bonded on the central disc of radius ``c``.

Its intensity factors at both fronts, under normal and tangential tractions, are
computed in an isotropic matrix by a Galerkin method on the integral equations of
the crack (see [`sif`](@ref)); its contributions follow by Rice's formula
([`rice_cod_tensor`](@ref)) along the growth of the inner front from ``r = a`` to
``r = c``. ``c \\to 0`` recovers the penny-shaped crack of radius ``a``.
"""
struct AnnularCrack{T <: Number, B <: TensND.AbstractBasis} <: MFH_Core.AbstractCrack{T}
    a::T
    c::T
    basis::B
    function AnnularCrack{T, B}(a, c, basis) where {T <: Number, B <: TensND.AbstractBasis}
        _check_annulus(a, c)
        return new{T, B}(a, c, basis)
    end
end


"""
    AnnularCrack(a, c; euler_angles=())
    AnnularCrack(a, c, R::AbstractMatrix)
    AnnularCrack(a, c, basis::AbstractBasis)

Annular crack of outer radius `a` and inner radius `c` (``0 < c < a``), oriented by
ZYZ Euler angles (its normal is the third vector of the rotated basis), by the
columns of a rotation matrix, or by a TensND basis.
"""
function AnnularCrack(a::Ta, c::Tc; euler_angles::Tuple{Vararg{Real}} = ()) where {Ta, Tc}
    T = TensND.float_like(promote_type(Ta, Tc))
    return AnnularCrack(T(a), T(c), TensND.euler_basis(euler_angles; T = T))
end
AnnularCrack(a, c, R::AbstractMatrix) =
    AnnularCrack(a, c, TensND.RotatedBasis(Matrix{float(eltype(R))}(R)))
function AnnularCrack(a::Ta, c::Tc, basis::TensND.AbstractBasis) where {Ta, Tc}
    T = TensND.float_like(promote_type(Ta, Tc))
    return AnnularCrack{T, typeof(basis)}(T(a), T(c), basis)
end

"""
    RiceCrack{T, B, FI, FII, FIII, FT}

Flat crack of outer radius ``a`` whose growth from a nucleus is described by the
user: its fronts are circles of radius ``\\rho`` swept from ``\\rho_0`` to
``\\rho_1``, and its intensity factors per unit traction at a front of radius
``\\rho`` are given as functions of ``(\\rho, \\nu)``, ``\\nu`` the Poisson ratio of
the isotropic matrix:

- `K_I(ρ, ν)`: opening mode under a unit normal traction;
- `K_II(ρ, ν)`, `K_III(ρ, ν)`: in-plane and anti-plane sliding modes at the point
  of the front where the unit tangential traction is respectively normal and
  tangent to the front;
- `K_T(ρ)`: flux intensity factor under a unit normal flux, `ρ -> K_I(ρ, 0)` by
  default (the two problems coincide for a flat crack, see
  [Contributions from intensity factors](@ref th-rice-contributions)).

[`cod_tensor`](@ref) then follows by Rice's formula, and the crack enters every
scheme as any other crack (density ``\\varepsilon = N a^3``).
"""
struct RiceCrack{T <: Number, B <: TensND.AbstractBasis, FI, FII, FIII, FT} <: MFH_Core.AbstractCrack{T}
    a::T
    ρ₀::T
    ρ₁::T
    K_I::FI
    K_II::FII
    K_III::FIII
    K_T::FT
    basis::B
end

"""
    RiceCrack(a; K_I, K_II, K_III, K_T, from=0, to=a, euler_angles=())

User-defined crack of outer radius `a` grown along circular fronts of radius ``\\rho``
from `from` to `to`, with intensity factors per unit traction `K_I(ρ, ν)`,
`K_II(ρ, ν)`, `K_III(ρ, ν)` and flux intensity factor `K_T(ρ)` (see
[`RiceCrack`](@ref)). A penny-shaped crack:

```julia
RiceCrack(1.0;
    K_I = (ρ, ν) -> 2sqrt(ρ / π),
    K_II = (ρ, ν) -> 4sqrt(ρ / π) / (2 - ν),
    K_III = (ρ, ν) -> 4(1 - ν) * sqrt(ρ / π) / (2 - ν))
```
"""
function RiceCrack(
        a::Ta; K_I, K_II, K_III, K_T = ρ -> K_I(ρ, zero(ρ)), from = zero(a), to = a,
        euler_angles::Tuple{Vararg{Real}} = ()
    ) where {Ta}
    T = TensND.float_like(promote_type(Ta, typeof(from), typeof(to)))
    basis = TensND.euler_basis(euler_angles; T = T)
    return RiceCrack{T, typeof(basis), typeof(K_I), typeof(K_II), typeof(K_III), typeof(K_T)}(
        T(a), T(from), T(to), K_I, K_II, K_III, K_T, basis
    )
end

const _CircularFrontCrack = Union{AnnularCrack, RiceCrack}

MFH_Core.shape_trait(::_CircularFrontCrack) = CircularFronts
semi_major(c::_CircularFrontCrack) = c.a
semi_minor(c::_CircularFrontCrack) = c.a

function MFH_Core.shape_tensor(c::Union{AnnularCrack{T}, RiceCrack{T}}) where {T}
    z = zero(T)
    return TensND.Tens(TensND.Tensors.SymmetricTensor{2, 3, T}((c.a, z, z, c.a, z, z)), c.basis)
end

Base.:(==)(x::T, y::T) where {T <: AnnularCrack} = x.a == y.a && x.c == y.c && x.basis == y.basis
Base.hash(x::AnnularCrack, h::UInt) = hash(x.basis, hash(x.c, hash(x.a, hash(typeof(x), h))))
Base.show(io::IO, c::AnnularCrack{T}) where {T} =
    print(io, "AnnularCrack{", T, "} (outer radius a=", c.a, ", inner radius c=", c.c, ")")
Base.show(io::IO, c::RiceCrack{T}) where {T} =
    print(io, "RiceCrack{", T, "} (radius a=", c.a, ", fronts from ρ=", c.ρ₀, " to ρ=", c.ρ₁, ")")

# Isotropic matrix required: the intensity factors are those of an isotropic body.
_iso_matrix(C₀::TensND.TensISO{4, 3}, _) = C₀
_iso_matrix(K₀::TensND.TensISO{2, 3}, _) = K₀
_iso_matrix(P₀, who) = throw(
    ArgumentError(
        "the intensity factors of $(_who(who)) are those of an isotropic matrix, " *
            "got a property of type $(nameof(typeof(P₀))). Register the phase with " *
            "`symmetrize = IsoSymmetrize()`, or give an isotropic reference."
    )
)

_who(name::AbstractString) = name
_who(x) = string(nameof(typeof(x)))

# Four points of a circular front of radius ρ, weight `w` each (a quarter of the
# area swept per unit growth parameter), with the matrix 𝐌 = kI n⊗n + kII e_r⊗e_r
# + kIII e_θ⊗e_θ in the crack frame. Four points integrate exactly the
# trigonometric polynomials of degree 2 that 𝐌ᵀ⋅𝐁ᴿ⋅𝐌 is along the front.
function _circle_front(w, outward::Bool, kI, kII, kIII)
    T = promote_type(typeof(w), typeof(kI), typeof(kII), typeof(kIII))
    z, o = zero(T), one(T)
    sgn = outward ? o : -o
    return map(((o, z), (z, o), (-o, z), (z, -o))) do (cθ, sθ)
        er = SVector(cθ, sθ, z)
        eθ = SVector(-sθ, cθ, z)
        n = SVector(z, z, o)
        M = kI * n * transpose(n) + kII * er * transpose(er) + kIII * eθ * transpose(eθ)
        (w, sgn * er, M)
    end
end

_circle_front_thermal(w, outward::Bool, kT) =
    map(p -> (p[1], p[2], kT), _circle_front(w, outward, kT, kT, kT))

# ── AnnularCrack: the inner front grows from r = a to r = c ─────────────────

# Intensity factors per unit traction at the inner front of the annulus
# (ρ < r < a), in the matrix of Poisson ratio ν: `_annulus_solve` works on the
# unit annulus, and an intensity factor scales as √a.
function _annulus_inner_sifs(ρ, a, ν; shear::Bool = true)
    S = _annulus_solve(ρ / a, ν; shear)
    sa = sqrt(a)
    shear || return (sa * S.kI.in,)
    return (sa * S.kI.in, sa * S.kII.in, sa * S.kIII.in)
end

_annulus_front(a, ν) = ρ -> begin
    kI, kII, kIII = _annulus_inner_sifs(ρ, a, ν)
    _circle_front(2 * oftype(kI, π) * ρ / 4, false, kI, kII, kIII)
end
_annulus_front_thermal(a) = ρ -> begin
    (kI,) = _annulus_inner_sifs(ρ, a, zero(ρ); shear = false)
    _circle_front_thermal(2 * oftype(kI, π) * ρ / 4, false, kI)
end

"""
    cod_tensor(crack::AnnularCrack, C₀; method=:rice, reltol=1e-8, K_interface=nothing)
    cod_tensor(crack::AnnularCrack, K₀; method=:rice, reltol=1e-8, α_interface=nothing)

COD tensor ``\\boldsymbol{B}`` (or thermal COD scalar ``b``) of an annular crack in
an isotropic matrix, normalized by its outer radius ``a``.

- `method = :rice` (default): Rice's formula, the intensity factors of the
  inner front integrated as it grows from ``r = a`` to ``r = c``
  ([`rice_cod_tensor`](@ref));
- `method = :opening`: the mean opening and sliding of the crack at its final
  shape, from the same Galerkin solution — one solve instead of one per
  quadrature node of the growth, and the independent check of the former.

In conduction, ``b`` follows from the opening mode alone: for a flat crack in an
isotropic body, the flux intensity factor per unit normal flux equals the stress
intensity factor per unit normal traction.
"""
function cod_tensor(
        crack::AnnularCrack, C₀::TensND.AbstractTens{4, 3};
        K_interface::Union{Nothing, TensND.AbstractTens{2, 3}} = nothing,
        method::Symbol = :rice, reltol = 1.0e-8, kw...
    )
    MFH_Core._bump!(MFH_Core.COD_CALLS)
    E, ν = TensND.E_nu(_iso_matrix(C₀, crack))
    # `method` is a run-time Symbol: a static branch per value keeps the result inferred.
    B = method === :rice ? _annular_cod(Val(:rice), crack, C₀, E, ν, reltol) :
        method === :opening ? _annular_cod(Val(:opening), crack, C₀, E, ν, reltol) :
        _annular_method_error(method)
    K_interface === nothing && return B
    return _apply_interface_stiffness(B, K_interface, semi_minor(crack))
end

function cod_tensor(
        crack::AnnularCrack, K₀::TensND.AbstractTens{2, 3};
        α_interface::Union{Nothing, Real} = nothing,
        method::Symbol = :rice, reltol = 1.0e-8, kw...
    )
    k₀ = first(TensND.get_data(_iso_matrix(K₀, crack)))
    b = method === :rice ? _annular_cod_thermal(Val(:rice), crack, K₀, k₀, reltol) :
        method === :opening ? _annular_cod_thermal(Val(:opening), crack, K₀, k₀, reltol) :
        _annular_method_error(method)
    α_interface === nothing && return b
    return b / (1 + semi_minor(crack) * α_interface * b)
end

_annular_cod(::Val{:rice}, crack, C₀, E, ν, reltol) = rice_cod_tensor(
    _annulus_front(crack.a, ν), crack.c, crack.a, C₀, CircularFronts;
    size = crack.a, basis = crack.basis, reltol
)

function _annular_cod(::Val{:opening}, crack, C₀, E, ν, _)
    S = _annulus_solve(crack.c / crack.a, ν)
    Es = E / (1 - ν^2)
    μ = E / (2 * (1 + ν))
    T = typeof(S.opening / Es)
    # π a³ B_nn = ∫ w dS / σ and π a³ B_tt = ∫ [[u_x]] dS / τ, on the unit annulus.
    Bt = S.slide / (T(π) * μ)
    return _diagonal_cod(Bt, Bt, S.opening / (T(π) * Es), crack.basis)
end


_annular_cod_thermal(::Val{:rice}, crack, K₀, k₀, reltol) = rice_cod_tensor(
    _annulus_front_thermal(crack.a), crack.c, crack.a, K₀, CircularFronts;
    size = crack.a, basis = crack.basis, reltol
)

function _annular_cod_thermal(::Val{:opening}, crack, K₀, k₀, _)
    S = _annulus_solve(crack.c / crack.a, zero(k₀); shear = false)
    # The temperature jump is (E*/2k₀) times the opening of the mechanical problem.
    return S.opening / (2 * oftype(S.opening, π) * k₀)
end


_annular_method_error(M) = throw(
    ArgumentError("unknown method `:$M` for an annular crack: use `:rice` or `:opening`")
)

"""
    sif(crack::AnnularCrack, C₀, Σ; front=:outer, y₀=ℓ) -> (𝐊, (Kᴵ, Kᴵᴵ, Kᴵᴵᴵ))
    sif(crack::AnnularCrack, K₀, σ∞; front=:outer) -> K_T

Intensity factors of an annular crack in an isotropic matrix under the remote stress
`Σ`, at the point of the outer (`front = :outer`) or inner (`front = :inner`) front
in the direction `y₀` of the crack plane, with the conventions of
[`sif`](@ref) [barthelemyIJSS2026](@cite): ``\\underline{K}`` is decomposed on
``(\\underline{n}, \\underline{\\nu}, \\underline{\\tau})``, ``\\underline{\\nu}``
the outer normal to the front in the crack plane and ``\\underline{\\tau} =
\\underline{n}\\wedge\\underline{\\nu}``.
"""
function sif(
        crack::AnnularCrack, C₀::TensND.AbstractTens{4, 3}, Σ;
        front::Symbol = :outer, y₀ = TensND.tens_basis(crack.basis, 1), kw...
    )
    _, ν = TensND.E_nu(_iso_matrix(C₀, crack))
    S = _annulus_solve(crack.c / crack.a, ν)
    k = _annulus_front_values(S, front)
    n̂ = crack_normal(crack)
    êr = _in_plane_unit(y₀, n̂)
    êθ = n̂ × êr
    t = Σ ⋅ n̂
    sa = sqrt(crack.a)
    𝐊 = sa * (k.I * (t ⋅ n̂) * n̂ + k.II * (t ⋅ êr) * êr + k.III * (t ⋅ êθ) * êθ)
    ν̂ = front === :outer ? êr : -êr
    τ̂ = n̂ × ν̂
    return 𝐊, (𝐊 ⋅ n̂, 𝐊 ⋅ ν̂, 𝐊 ⋅ τ̂)
end

function sif(
        crack::AnnularCrack, K₀::TensND.AbstractTens{2, 3}, σ∞;
        front::Symbol = :outer, kw...
    )
    _iso_matrix(K₀, crack)
    S = _annulus_solve(crack.c / crack.a, zero(crack.a); shear = false)
    return sqrt(crack.a) * _annulus_front_values(S, front).I * (crack_normal(crack) ⋅ σ∞)
end

function _annulus_front_values(S, front::Symbol)
    front === :outer && return (I = S.kI.out, II = _maybe(S.kII, :out), III = _maybe(S.kIII, :out))
    front === :inner && return (I = S.kI.in, II = _maybe(S.kII, :in), III = _maybe(S.kIII, :in))
    throw(ArgumentError("`front` is `:outer` or `:inner`, got `:$front`"))
end
_maybe(::Nothing, ::Symbol) = nothing
_maybe(k, f::Symbol) = getfield(k, f)

# The unit vector of the crack plane in the direction of `y`.
function _in_plane_unit(y, n̂)
    v = y - (y ⋅ n̂) * n̂
    return v / norm(v)
end

# ── RiceCrack: the user's intensity factors ─────────────────────────────────

_rice_crack_front(c::RiceCrack, ν) = ρ -> _circle_front(
    2 * oftype(float(ρ), π) * ρ / 4, c.ρ₁ ≥ c.ρ₀,
    c.K_I(ρ, ν), c.K_II(ρ, ν), c.K_III(ρ, ν)
)
_rice_crack_front_thermal(c::RiceCrack) =
    ρ -> _circle_front_thermal(2 * oftype(float(ρ), π) * ρ / 4, c.ρ₁ ≥ c.ρ₀, c.K_T(ρ))

# The growth runs from ρ₀ to ρ₁ in either direction; the swept area is counted
# positively by integrating over the ordered interval.
_ordered(x, y) = TensND.primal(x) ≤ TensND.primal(y) ? (x, y) : (y, x)

function cod_tensor(
        crack::RiceCrack, C₀::TensND.AbstractTens{4, 3};
        K_interface::Union{Nothing, TensND.AbstractTens{2, 3}} = nothing,
        reltol = 1.0e-8, kw...
    )
    MFH_Core._bump!(MFH_Core.COD_CALLS)
    _, ν = TensND.E_nu(_iso_matrix(C₀, crack))
    B = rice_cod_tensor(
        _rice_crack_front(crack, ν), _ordered(crack.ρ₀, crack.ρ₁)..., C₀, CircularFronts;
        size = crack.a, basis = crack.basis, reltol
    )
    K_interface === nothing && return B
    return _apply_interface_stiffness(B, K_interface, semi_minor(crack))
end

function cod_tensor(
        crack::RiceCrack, K₀::TensND.AbstractTens{2, 3};
        α_interface::Union{Nothing, Real} = nothing, reltol = 1.0e-8, kw...
    )
    b = rice_cod_tensor(
        _rice_crack_front_thermal(crack), _ordered(crack.ρ₀, crack.ρ₁)..., _iso_matrix(K₀, crack), CircularFronts;
        size = crack.a, basis = crack.basis, reltol
    )
    α_interface === nothing && return b
    return b / (1 + semi_minor(crack) * α_interface * b)
end
