# =============================================================================
#  elasticity.jl — the elastic n-layer confocal spheroid, case I (axisymmetric).
#
#  The conduction solver next door propagates a 2𝒩-vector through per-interface
#  transfer matrices. Elasticity cannot reuse that shape: the confocal surfaces
#  are not homothetic, so the harmonic degrees COUPLE, and — unlike conduction,
#  where the coupling appears only at an imperfect interface — here they couple
#  across a perfect one too. What is assembled instead is one global system,
#  four scalar conditions per interface projected on the Legendre degrees.
#
#  THE DERIVATION IS NOT IN THE LITERATURE and lives on the theory page
#  `theory/layered_spheroid_elasticity.md`, checked there and in
#  `test/LayeredSpheroids/test_pn_symbolic.jl`. In outline, with Papkovich–Neuber
#  potentials `φ₀, φ₃` (case I fixes the gauge `φ₁ = φ₂ = 0`) and
#  `Φ = φ₀ + z φ₃`, `z = c p q`:
#
#      2μ u = ∇Φ - 4(1-ν) φ₃ ê₃
#      σ    = -2ν (∂_z φ₃) 𝟙 + ∇∇Φ - 4(1-ν) sym(ê₃ ⊗ ∇φ₃)
#
#  A confocal interface shares its geometry, so every geometric prefactor
#  cancels from a matching condition and what must be continuous at `q = q_ℓ`,
#  for every `p`, is
#
#      U_q/μ,      (1-p²) U_p/μ,      T_q,      (1-p²) T_p
#
#  the tractions carrying no `1/μ` because Papkovich–Neuber's stress carries no
#  `μ` at all. The `(1-p²)` on the tangential lines is what keeps the coupling
#  BANDED: a bare `Pₙ′` reaches every lower degree, whereas `(1-p²)Pₙ′` and
#  `p Pₙ` reach one, and `p²Pₙ` two. Truncating the series is legitimate for
#  that reason.
#
#  Parity splits the unknowns: `φ₀` lives on the even degrees and `φ₃` on the
#  odd ones, which is why `legendre_degrees` takes a degree list rather than
#  assuming one.
# =============================================================================

"""
    AxisymmetricCase()

Case I of Duan et al. (2005): a remote strain `diag(ε_t, ε_t, ε_a)` about the
spheroid's axis. The Papkovich–Neuber gauge is `φ₁ = φ₂ = 0`, both surviving
potentials are of order `m = 0`, and the fields are axisymmetric — `u_φ` and
`σ_φq` vanish structurally rather than by assumption.

A singleton, so the two remaining elementary problems (transverse and
longitudinal shear, which need orders `m = 2` and `m = 1`) can be added by
dispatch without disturbing this one.
"""
struct AxisymmetricCase end

"""
    _leg_pqq(kind, x, degrees) -> (v, d, dd)

Legendre values, first and second derivatives at `x`, for the given `degrees`.

The second derivative is not tabulated anywhere: it comes from Legendre's own
equation, `(1-x²)y'' - 2x y' + n(n+1) y = 0`, which both `Pₙ` and `Qₙ` satisfy
on either branch.
"""
function _leg_pqq(kind::Symbol, x, degrees)
    v, d = legendre_degrees(kind, x, degrees)
    ix = inv(1 - x^2)
    dd = [(2 * x * d[i] - degrees[i] * (degrees[i] + 1) * v[i]) * ix for i in eachindex(degrees)]
    return v, d, dd
end

"Even degrees `0, 2, …` and odd degrees `1, 3, …`, `𝒩` of each."
@inline _case1_degrees(𝒩::Int) = (collect(0:2:(2𝒩 - 2)), collect(1:2:(2𝒩 - 1)))

"""
    _case1_modes(𝒩, role) -> Vector{Tuple{Symbol,Int}}

The unknown amplitudes of one region. `:A`/`:B` are the `Pₙ(q)`/`Qₙ(q)` parts of
`φ₀` (even degrees), `:C`/`:D` the same for `φ₃` (odd degrees).

- `:core` — `Qₙ(q)` is singular on the focal segment, so only `:A`, `:C` survive.
- `:shell` — all four.
- `:matrix` — only the decaying `:B`, `:D`; the growing part is the prescribed
  remote field.

Degree `0` is absent from `:A` in every region: `φ₀ = P₀(p)P₀(q) = 1` is a
constant, contributes nothing to `∇Φ`, and would put an exactly null column in
the system.
"""
function _case1_modes(𝒩::Int, role::Symbol)
    even, odd = _case1_degrees(𝒩)
    A = [(:A, n) for n in even if n > 0]
    B = [(:B, n) for n in even]
    C = [(:C, m) for m in odd]
    D = [(:D, m) for m in odd]
    role === :core && return vcat(A, C)
    role === :shell && return vcat(A, B, C, D)
    role === :matrix && return vcat(B, D)
    throw(ArgumentError("_case1_modes: unknown role $role"))
end

"""
    _quad_eltype(::Type{T}) -> Type

Element type of the Gauss–Legendre nodes. They are *constants* of the
quadrature, so a `ForwardDiff.Dual` element type wants ordinary `Float64`
nodes — computing them in dual arithmetic would only carry zero partials
around. `BigFloat` is the one case worth honoring, because there the nodes
themselves must be more accurate.
"""
@inline _quad_eltype(::Type{T}) where {T} = real(T) === BigFloat ? BigFloat : Float64

"""
    _case1_block(modes, q, c, μ, ν, 𝒩, xg, wg, even, odd) -> Matrix

The `4𝒩 × length(modes)` contribution of one region to the conditions at
`q`. Rows are, in order: `U_q/μ` on the even degrees, `(1-p²)U_p/μ` on the odd
ones, `T_q` on the even, `(1-p²)T_p` on the odd.

The projections are computed by Gauss–Legendre quadrature in `p`. That is exact,
not approximate: each condition is a polynomial in `p` once the shared radicals
are removed, so a rule with enough nodes integrates it to the last bit.
"""
function _case1_block(modes, q, c, μ, ν, 𝒩::Int, xg, wg, even, odd)
    Tel = promote_type(typeof(q), typeof(c), typeof(μ), typeof(ν), eltype(xg))
    M = zeros(Tel, 4𝒩, length(modes))
    maxdeg = max(maximum(even), maximum(odd))
    degs = 0:maxdeg
    Pp = [legendre_degrees(:P0, x, degs) for x in xg]
    Pv, Pd, Pdd = _leg_pqq(:P0, q, degs)
    Qv, Qd, Qdd = _leg_pqq(:Q0, q, degs)

    for (j, (kind, n)) in enumerate(modes)
        F, dF, ddF = (kind === :A || kind === :C) ?
            (Pv[n + 1], Pd[n + 1], Pdd[n + 1]) :
            (Qv[n + 1], Qd[n + 1], Qdd[n + 1])
        for (g, x) in enumerate(xg)
            Pn, dPn = Pp[g][1][n + 1], Pp[g][2][n + 1]
            Uq, Up, Tq, Tp = (kind === :A || kind === :B) ?
                _case1_phi0_terms(Pn, dPn, F, dF, ddF, x, q) :
                _case1_phi3_terms(Pn, dPn, F, dF, ddF, x, q, c, ν)
            pb2 = 1 - x^2
            for (r, k) in enumerate(even)
                cc = wg[g] * Pp[g][1][k + 1] * (2k + 1) / 2
                M[r, j] += cc * Uq / μ
                M[2𝒩 + r, j] += cc * Tq
            end
            for (r, k) in enumerate(odd)
                cc = wg[g] * Pp[g][1][k + 1] * (2k + 1) / 2
                M[𝒩 + r, j] += cc * pb2 * Up / μ
                M[3𝒩 + r, j] += cc * pb2 * Tp
            end
        end
    end
    return M
end

"The four condition values of a `φ₀` mode `Pₙ(p) F(q)`, where `Φ = φ₀`."
@inline function _case1_phi0_terms(Pn, dPn, F, dF, ddF, p, q)
    pb2, qb2, w2 = 1 - p^2, q^2 - 1, q^2 - p^2
    return (
        Pn * dF,
        dPn * F,
        qb2 * w2 * Pn * ddF + pb2 * (q * Pn * dF - p * dPn * F),
        w2 * dPn * dF + p * Pn * dF - q * dPn * F,
    )
end

"The four condition values of a `φ₃` mode `P_m(p) G(q)`, where `Φ = c p q φ₃`."
@inline function _case1_phi3_terms(Pm, dPm, G, dG, ddG, p, q, c, ν)
    pb2, qb2, w2 = 1 - p^2, q^2 - 1, q^2 - p^2
    Φp = c * q * (Pm + p * dPm) * G
    Φq = c * p * Pm * (G + q * dG)
    Φpq = c * (Pm + p * dPm) * (G + q * dG)
    Φqq = c * p * Pm * (2 * dG + q * ddG)
    φ3, φ3p, φ3q = Pm * G, dPm * G, Pm * dG
    return (
        Φq - 4 * c * p * (1 - ν) * φ3,
        Φp - 4 * c * q * (1 - ν) * φ3,
        qb2 * w2 * Φqq + pb2 * (q * Φq - p * Φp) -
            2 * ν * c * w2 * (q * pb2 * φ3p + p * qb2 * φ3q) -
            4 * (1 - ν) * c * p * qb2 * w2 * φ3q,
        w2 * Φpq + p * Φq - q * Φp - 2 * (1 - ν) * c * w2 * (q * φ3q + p * φ3p),
    )
end

"Poisson's ratio from the isotropic pair."
@inline _poisson(κ, μ) = (3 * κ - 2 * μ) / (2 * (3 * κ + μ))

"""
    _case1_remote(c, κ₀, μ₀, εa, εt) -> (A₂, C₁)

Amplitudes of the remote uniform strain `diag(ε_t, ε_t, ε_a)`, as coefficients
on `P₂(p)P₂(q)` and `P₁(p)P₁(q)`.

A uniform strain excites **two** coefficients and no more, which is the
"collapse" the single-inclusion case is checked against:

    φ₀ = -(4/3) c² μ₀ ε_t P₂(p)P₂(q)   (plus a constant, which is inert)
    φ₃ = γ c P₁(p)P₁(q),   γ = -μ₀(ε_a + 2ε_t)/(1 - 2ν₀)
"""
@inline function _case1_remote(c, κ₀, μ₀, εa, εt)
    ν₀ = _poisson(κ₀, μ₀)
    γ = -μ₀ * (εa + 2 * εt) / (1 - 2 * ν₀)
    return (-4 * c^2 * μ₀ * εt / 3, γ * c)
end

"""
    spheroid_elastic_coefficients(s, C₀, εa, εt; case = AxisymmetricCase(),
                                  ngauss = 0) -> NamedTuple

Solve the elastic `n`-layer confocal spheroid under an axisymmetric remote
strain `diag(εt, εt, εa)` about the spheroid's axis.

Returns `(; modes, amplitudes, residual)`: `modes[ℓ]` lists the
`(kind, degree)` pairs of region `ℓ` (`1:N` the layers, `N+1` the matrix) and
`amplitudes[ℓ]` their values, with the matrix's prescribed remote part left
out. `residual` is the relative least-squares residual, which is a *diagnostic*
and not a fitting error — see the note below.

!!! note "Why the system is solved in least squares"
    Each interface contributes `4𝒩` conditions and each region `4𝒩-1`
    amplitudes, degree `0` of `φ₀` being a constant potential that moves
    nothing. The assembled system is therefore over-determined by one row per
    interface, and those rows are **redundant, not conflicting**: the residual
    comes out at machine precision. Solving in least squares is the honest way
    to use that, and a residual that stops being negligible is a signal that
    something upstream is wrong.

!!! note "How far `Nseries` is worth pushing"
    A single homogeneous spheroid is exact at any truncation — its series
    collapses to two coefficients. A genuinely layered one converges
    geometrically, the residual falling by roughly a factor 3 per unit of
    `Nseries`. In `Float64` that stops paying at about `Nseries = 12`
    (relative accuracy ~`1e-7`): past there the least-squares conditioning
    dominates, the residual stops falling, and the answer parts company with a
    256-bit one. Raise the element type rather than the truncation. The
    conduction solver records the same limit after Barthélémy & Bignonnet's
    appendix C; the elastic blocks are wider, so it arrives sooner.

Only [`PerfectInterface`](@ref) is supported so far; imperfect interfaces need
their jump terms added to the four conditions. Oblate spheroids are refused:
their confocal parameter is complex and nothing here has been checked against a
reference for it.
"""
function spheroid_elastic_coefficients(
        s::LayeredSpheroid{T, N, Q}, C₀, εa, εt;
        case::AxisymmetricCase = AxisymmetricCase(), ngauss::Int = 0,
    ) where {T, N, Q}
    s.prolate || throw(
        ArgumentError(
            "spheroid_elastic_coefficients: oblate spheroids are not supported " *
                "yet — the confocal parameter is complex there, and nothing in " *
                "this solver has been checked against a reference for it"
        )
    )
    for ℓ in 1:N
        layer_interface(s, ℓ) isa PerfectInterface || throw(
            ArgumentError(
                "spheroid_elastic_coefficients: only PerfectInterface is supported; " *
                    "got $(typeof(layer_interface(s, ℓ))) at layer $ℓ. A uniform " *
                    "spring or membrane law does not fit this formulation — see " *
                    "the theory page — because it puts an ODD power of the metric " *
                    "factor w = √(q²-p²) into the matching condition, which stops " *
                    "it being a polynomial identity in p. Model the imperfection " *
                    "as a thin confocal interphase instead: that is an extra " *
                    "layer, which this solver already handles. Note that a " *
                    "confocal shell is ω times thicker at the equator than at " *
                    "the pole, ω being the aspect ratio."
            )
        )
    end

    κμ = ntuple(k -> _iso_bulk_shear(layer_modulus(s, k)), Val(N))
    κ₀, μ₀ = _iso_bulk_shear(C₀)
    Tp = promote_type(
        T, real(Q), typeof(κ₀), typeof(μ₀), typeof(εa), typeof(εt),
        ntuple(k -> typeof(κμ[k][1]), N)..., ntuple(k -> typeof(κμ[k][2]), N)...,
        interfaces_eltype(s.interfaces)
    )

    𝒩 = s.Nseries
    even, odd = _case1_degrees(𝒩)
    # A rule exact for the highest polynomial degree the conditions reach: the
    # multipliers add at most 4 to a degree of `2𝒩-1`, and the projection adds
    # the test degree again.
    n_gauss = ngauss > 0 ? ngauss : 4𝒩 + 6
    xg, wg = QuadGK.gauss(_quad_eltype(Tp), n_gauss)

    c = real(s.c)
    qs = ntuple(k -> real(s.q[k]), Val(N))
    roles = ntuple(ℓ -> ℓ == 1 ? :core : :shell, Val(N))
    modes = Vector{Vector{Tuple{Symbol, Int}}}(undef, N + 1)
    for ℓ in 1:N
        modes[ℓ] = _case1_modes(𝒩, roles[ℓ])
    end
    modes[N + 1] = _case1_modes(𝒩, :matrix)

    offs = cumsum(vcat(0, [length(m) for m in modes]))
    ncol = offs[end]
    A = zeros(Tp, 4𝒩 * N, ncol)
    rhs = zeros(Tp, 4𝒩 * N)

    for ℓ in 1:N
        rows = (4𝒩 * (ℓ - 1) + 1):(4𝒩 * ℓ)
        κi, μi = κμ[ℓ]
        κo, μo = ℓ < N ? κμ[ℓ + 1] : (κ₀, μ₀)
        νi, νo = _poisson(κi, μi), _poisson(κo, μo)
        A[rows, (offs[ℓ] + 1):offs[ℓ + 1]] .=
            _case1_block(modes[ℓ], qs[ℓ], c, μi, νi, 𝒩, xg, wg, even, odd)
        A[rows, (offs[ℓ + 1] + 1):offs[ℓ + 2]] .=
            -_case1_block(modes[ℓ + 1], qs[ℓ], c, μo, νo, 𝒩, xg, wg, even, odd)
        if ℓ == N
            A₂, C₁ = _case1_remote(c, κ₀, μ₀, εa, εt)
            Mr = _case1_block(
                [(:A, 2), (:C, 1)], qs[ℓ], c, μo, νo, 𝒩, xg, wg, even, odd
            )
            rhs[rows] .= Mr * [A₂, C₁]
        end
    end

    sol = A \ rhs
    nr = sqrt(sum(abs2, rhs))
    residual = sqrt(sum(abs2, A * sol - rhs)) / (nr > 0 ? nr : one(nr))
    amplitudes = [sol[(offs[ℓ] + 1):offs[ℓ + 1]] for ℓ in 1:(N + 1)]
    return (; modes, amplitudes, residual)
end

"""
    spheroid_core_strain(s, C₀, εa, εt; kw...) -> (εa_in, εt_in)

Uniform strain in the **core** of an elastic confocal spheroid under the
axisymmetric remote strain `diag(εt, εt, εa)`.

The core carries only the regular modes, and the two that survive — degree `2`
of `φ₀` and degree `1` of `φ₃` — are exactly a uniform strain, read back
through the same relations that build the remote field:

    ε_t = -3A₂ / (4 c² μ₁),   ε_a = -γ(1 - 2ν₁)/μ₁ - 2ε_t,   γ = C₁/c.

For a single layer this is Eshelby's result, which is what the tests check it
against.
"""
function spheroid_core_strain(s::LayeredSpheroid, C₀, εa, εt; kw...)
    r = spheroid_elastic_coefficients(s, C₀, εa, εt; kw...)
    κ₁, μ₁ = _iso_bulk_shear(layer_modulus(s, 1))
    ν₁ = _poisson(κ₁, μ₁)
    c = real(s.c)
    A₂ = zero(eltype(r.amplitudes[1]))
    C₁ = zero(eltype(r.amplitudes[1]))
    for (m, v) in zip(r.modes[1], r.amplitudes[1])
        m === (:A, 2) && (A₂ = v)
        m === (:C, 1) && (C₁ = v)
    end
    εt_in = -3 * A₂ / (4 * c^2 * μ₁)
    γ_in = C₁ / c
    εa_in = -γ_in * (1 - 2 * ν₁) / μ₁ - 2 * εt_in
    return εa_in, εt_in
end
