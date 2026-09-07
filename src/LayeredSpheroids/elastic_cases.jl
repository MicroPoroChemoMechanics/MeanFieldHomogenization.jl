# =============================================================================
#  elastic_cases.jl — the three elementary problems, and the strain
#  concentration tensor they add up to.
#
#  `pn_modes.jl` turns one harmonic mode into `u` and the traction; this file
#  says WHICH modes each elementary problem uses, what its remote field is, and
#  how the six independent remote strains give the concentration tensor a
#  mean-field scheme consumes.
#
#  THE DECOMPOSITION. For a spheroid the concentration tensor is transversely
#  isotropic about the axis, and the six-dimensional space of symmetric
#  second-order tensors splits into three subspaces it does not mix:
#
#      axisymmetric      (2)  ê₃⊗ê₃ and 𝟙 - ê₃⊗ê₃        case I
#      transverse shear  (2)  ε₁₁-ε₂₂ and 2ε₁₂           case II
#      longitudinal      (2)  2ε₁₃ and 2ε₂₃              case III
#
#  Each case is run twice (the two members of its subspace, related by a
#  rotation about the axis and distinguished only by `cos` against `sin`), so
#  six solves fill the tensor — and the two members of a pair must agree, which
#  is a free check on the whole construction.
#
#  NO RIGID-BODY ROTATIONS ARE NEEDED, and that is worth stating because Duan
#  et al. (2005) add two to case III and attribute to their omission the error
#  in Riccardi & Montheillet (1999). With the FULL four-potential set a rigid
#  rotation about ê₂ is already spanned: `φ₁ = A z`, `φ₃ = -A x` gives
#  `u ∝ (z, 0, -x)`, the antisymmetric combination of the very two potentials
#  whose symmetric combination is the remote shear. Duan's extra unknowns are
#  an artifact of a gauge that drops one of them, not a physical requirement.
#  The Eshelby oracle confirms it: case III lands on the analytic answer to
#  `1e-15` with no rotation in the unknown list.
# =============================================================================

"""
    ModeGroup(parts, degrees)

Modes that share a single amplitude, and the degrees they run over.

`parts` is a list of `(potential, m, trig, weight)`. Most groups have one part.
Case II needs more: its remote field ties `φ₁` and `φ₂` together — Duan's
"``(\\varphi_1,\\varphi_2)`` from a single ``\\Psi`` at ``m=1``" — and without
the tie the interface system is underdetermined, eight families answering six
conditions.
"""
struct ModeGroup
    parts::Vector{NTuple{4, Any}}
    degrees::Vector{Int}
end

ModeGroup(pot::Int, m::Int, trig::Symbol, degrees) =
    ModeGroup([(pot, m, trig, 1.0)], collect(degrees))

"""
    TransverseShearCase(trig)

Case II: a remote transverse shear. `trig = :sin` loads `2\\varepsilon_{12}`,
`trig = :cos` loads `\\varepsilon_{11}-\\varepsilon_{22}`; the two are the same
problem rotated by ``45°`` about the axis.

Active potentials are `φ₀, φ₃` at order `m = 2` and the tied pair `(φ₁, φ₂)` at
`m = 1`.
"""
struct TransverseShearCase
    trig::Symbol
end

"""
    LongitudinalShearCase(trig)

Case III: a remote shear between the axis and a transverse direction.
`trig = :cos` loads `2\\varepsilon_{13}`, `trig = :sin` loads
`2\\varepsilon_{23}`.

Active potentials are `φ₀, φ₃` at order `m = 1` and `φ₁` (resp. `φ₂`) at
`m = 0`. No rigid-body rotation is added — see this file's header.
"""
struct LongitudinalShearCase
    trig::Symbol
end

const ElasticCase = Union{AxisymmetricCase, TransverseShearCase, LongitudinalShearCase}

"""
    _case_groups(case, D) -> Vector{ModeGroup}

The unknown families of one region, `D` degrees each.

**Parity is part of the answer, not a detail.** Under `p → -p` (that is,
`z → -z`) a mode picks up `(-1)^{n+m}`, and each case's remote field fixes which
parity of `Φ` is admissible; the potentials then inherit one parity each. Case I
puts `φ₀` on the even degrees and `φ₃` on the odd ones, and so on. Admitting
both parities does **not** merely waste columns: it splices in the *other*
problem of the same order — the one whose remote field is odd where this one's
is even — and the truncated system then leaks between them. That mistake put
case I out by 10% while leaving cases II and III exact, whose higher orders
happen to be less forgiving of it.
"""
_case_groups(::AxisymmetricCase, D::Int) = [
    # `φ₀` on the even degrees, degree 0 INCLUDED. Only its *regular* part is a
    # constant and therefore inert; the irregular one is `P₀(p)Q₀(q) =
    # arccoth q`, an essential mode — dropping it makes the interface system
    # genuinely inconsistent rather than merely rank-deficient, with a residual
    # of `1e-1` instead of `1e-16`. The regular degree-0 column is exactly
    # zero, which the least-squares solve absorbs.
    ModeGroup(0, 0, :cos, 0:2:(2D)),
    ModeGroup(3, 0, :cos, 1:2:(2D - 1)),     # φ₃ odd
]

_case_groups(cs::TransverseShearCase, D::Int) = [
    ModeGroup(0, 2, cs.trig, 2:2:(2D)),
    ModeGroup(3, 2, cs.trig, 3:2:(2D + 1)),
    # The tie. For `:sin` the remote is `φ₁ = A y`, `φ₂ = A x`; for `:cos` it is
    # `φ₁ = A x`, `φ₂ = -A y`. Same `A` either way.
    ModeGroup(
        cs.trig === :sin ?
            NTuple{4, Any}[(1, 1, :sin, 1.0), (2, 1, :cos, 1.0)] :
            NTuple{4, Any}[(1, 1, :cos, 1.0), (2, 1, :sin, -1.0)],
        collect(1:2:(2D - 1))
    ),
]

_case_groups(cs::LongitudinalShearCase, D::Int) = [
    ModeGroup(0, 1, cs.trig, 2:2:(2D)),
    ModeGroup(cs.trig === :cos ? 1 : 2, 0, :cos, 1:2:(2D - 1)),
    ModeGroup(3, 1, cs.trig, 1:2:(2D - 1)),
]

"""
    _case_remote(case, groups, c, κ₀, μ₀, ε) -> Vector{Pair}

Amplitudes of the remote uniform strain, as `(group index, degree) => value`.

Each case's remote field is two coefficients at most, which is why a single
homogeneous inclusion is an *exact* oracle rather than a converged one. The
derivations are on the theory page; in each, `Φ` comes out with no `φ₀` part
except in case I, and

    case I    φ₀ = -(4/3) c² μ₀ ε_t P₂(p)P₂(q),  φ₃ = γ c P₁(p)P₁(q),
              γ = -μ₀(ε_a + 2ε_t)/(1 - 2ν₀)
    case II   the tied pair at degree 1, amplitude -A c
    case III  φ₁ (or φ₂) = α z at degree 1, φ₃ = α x (or α y) at degree 1,
              amplitudes α c and -α c

with `A = α = μ₀ ε/(2ν₀ - 1)`. The signs follow from
`ρ = -c P₁¹(p)P₁¹(q)` and `z = c P₁(p)P₁(q)`.
"""
function _case_remote(::AxisymmetricCase, c, κ₀, μ₀, εa, εt)
    ν₀ = _poisson(κ₀, μ₀)
    γ = -μ₀ * (εa + 2 * εt) / (1 - 2 * ν₀)
    return [(1, 2) => -4 * c^2 * μ₀ * εt / 3, (2, 1) => γ * c]
end

function _case_remote(::TransverseShearCase, c, κ₀, μ₀, ε)
    A = μ₀ * ε / (2 * _poisson(κ₀, μ₀) - 1)
    return [(3, 1) => -A * c]
end

function _case_remote(::LongitudinalShearCase, c, κ₀, μ₀, ε)
    α = μ₀ * ε / (2 * _poisson(κ₀, μ₀) - 1)
    return [(2, 1) => α * c, (3, 1) => -α * c]
end

"""
    _group_fields(group, regularity, n, ϕ, p, q, c, μ, ν, ::Type{T})

Unit-amplitude displacement and traction of one group, summing its parts. A
part whose order exceeds the degree contributes nothing, `Pₙᵐ` vanishing there.
"""
function _group_fields(g::ModeGroup, reg::Symbol, n::Int, ϕ, p, q, c, μ, ν, ::Type{T}) where {T}
    u = ntuple(_ -> zero(T), 3)
    t = ntuple(_ -> zero(T), 3)
    for (pot, m, trig, w) in g.parts
        n < m && continue
        uu, tt = mode_fields(PNMode(pot, reg, n, m, trig), ϕ, p, q, c, μ, ν, T)
        u = ntuple(k -> u[k] + w * uu[k], 3)
        t = ntuple(k -> t[k] + w * tt[k], 3)
    end
    return u, t
end

"""
    _elastic_block(groups, regularities, q, c, μ, ν, ϕ₀, xg, wg, tdegs, ::Type{T})

One region's contribution to the six conditions at `q`, projected on `tdegs`.

The conditions are the continuity of `u` and of the traction
`σ·e_q`, three components each. Displacements carry `1/μ`; tractions do not,
Papkovich–Neuber's stress carrying no `μ` at all.

The azimuth is evaluated at a single generic `ϕ₀`, and that is exact rather
than a sampling: every field component of a given case carries ONE azimuthal
harmonic, and it is the same on both sides of the interface, so it divides out
of a matching condition. `ϕ₀` only has to avoid the zeros of `sin mϕ` and
`cos mϕ`.

The **tangential** rows carry a factor `1 - p²`, for the reason
[the banding argument](@ref th-spheroid-banding) gives: a bare `Pₙ′` reaches
every lower degree, so a truncated projection of it loses information, while
`(1-p²)Pₙ′` reaches one. The weight is legitimate — `1-p²` is shared geometry
across a confocal interface, and it is the weight for which the `Pₙ′` are
orthogonal. Leaving it out is not a small error: it puts case I out by 10%,
while cases II and III survive because their `Pₙᵐ` with `m ≥ 1` already carry
`(1-p²)^{m/2}`.
"""
function _elastic_block(
        groups, regs, q, c, μ, ν, ϕ₀, xg, wg, tdegs, ::Type{T}
    ) where {T}
    cols = [(gi, r, n) for gi in eachindex(groups) for r in regs for n in groups[gi].degrees]
    nt = length(tdegs)
    M = zeros(T, 6 * nt, length(cols))
    maxdeg = maximum(tdegs)
    Pp = [legendre_degrees(:P0, x, 0:maxdeg) for x in xg]
    for (j, (gi, r, n)) in enumerate(cols)
        for (g, x) in enumerate(xg)
            u, t = _group_fields(groups[gi], r, n, ϕ₀, x, q, c, μ, ν, T)
            # (u_φ, u_p, u_q, σ_φq, σ_pq, σ_qq); the two tangential pairs get
            # the `1 - p²` weight, the normal ones do not.
            pb2 = one(T) - x^2
            vals = (
                pb2 * u[1], pb2 * u[2], u[3],
                pb2 * t[1], pb2 * t[2], t[3],
            )
            for (rr, k) in enumerate(tdegs)
                cc = wg[g] * Pp[g][1][k + 1] * (2k + 1) / 2
                for s in 1:6
                    M[(s - 1) * nt + rr, j] += cc * vals[s]
                end
            end
        end
    end
    return M, cols
end

"""
    _solve_elastic(s, C₀, case, remote; D, ngauss, ϕ₀) -> NamedTuple

Assemble and solve one elementary problem on the whole stack.

Returns `(; amplitudes, groups, residual)`, where `amplitudes[ℓ]` are the
unknowns of region `ℓ` (`1:N` the layers, `N+1` the matrix) in the order
`_elastic_block` lists its columns.

Like the case-I solver, the system is over-determined and solved in least
squares: the extra rows are redundant rather than conflicting, and `residual`
says so — it comes out at machine precision for a single inclusion, where the
series collapses to two coefficients and the answer is exact.
"""
function _solve_elastic(
        s::LayeredSpheroid{T, N, Q}, C₀, case::ElasticCase, remote;
        D::Int = 6, ngauss::Int = 0, ϕ₀ = 0.37,
    ) where {T, N, Q}
    s.prolate || throw(
        ArgumentError(
            "elastic confocal spheroid: oblate is not supported — the confocal " *
                "parameter is complex there and nothing has been checked against " *
                "a reference for it"
        )
    )
    for ℓ in 1:N
        layer_interface(s, ℓ) isa PerfectInterface || throw(
            ArgumentError(
                "elastic confocal spheroid: only PerfectInterface is supported. " *
                    "A uniform spring or membrane law puts an odd power of the " *
                    "metric factor w = √(q²-p²) into the matching condition, so it " *
                    "stops being a polynomial identity in p — see the theory page. " *
                    "Model the imperfection as a thin confocal interphase instead."
            )
        )
    end

    κμ = ntuple(k -> _iso_bulk_shear(layer_modulus(s, k)), Val(N))
    κ₀, μ₀ = _iso_bulk_shear(C₀)
    Tp = promote_type(
        T, real(Q), typeof(κ₀), typeof(μ₀), typeof(ϕ₀),
        ntuple(k -> typeof(κμ[k][1]), N)..., ntuple(k -> typeof(κμ[k][2]), N)...,
        interfaces_eltype(s.interfaces), typeof(last(first(remote)))
    )

    groups = _case_groups(case, D)
    # Both parities have to be tested: the normal conditions project onto the
    # even degrees and the tangential ones onto the odd, so a set covering only
    # `D` degrees supplies half the equations the unknowns need.
    tdegs = collect(0:(2D + 1))
    n_gauss = ngauss > 0 ? ngauss : 6D + 12
    xg, wg = QuadGK.gauss(_quad_eltype(Tp), n_gauss)
    c = real(s.c)
    qs = ntuple(k -> real(s.q[k]), Val(N))

    regs = Vector{Any}(undef, N + 1)
    regs[1] = (:regular,)
    for ℓ in 2:N
        regs[ℓ] = (:regular, :irregular)
    end
    regs[N + 1] = (:irregular,)

    colcount = [
        length(
            [
                (gi, r, n) for gi in eachindex(groups) for r in regs[ℓ]
                    for n in groups[gi].degrees
            ]
        ) for ℓ in 1:(N + 1)
    ]
    offs = cumsum(vcat(0, colcount))
    nrow = 6 * length(tdegs) * N
    A = zeros(Tp, nrow, offs[end])
    rhs = zeros(Tp, nrow)

    for ℓ in 1:N
        nt = 6 * length(tdegs)
        rows = (nt * (ℓ - 1) + 1):(nt * ℓ)
        κi, μi = κμ[ℓ]
        κo, μo = ℓ < N ? κμ[ℓ + 1] : (κ₀, μ₀)
        νi, νo = _poisson(κi, μi), _poisson(κo, μo)
        Mi, _ = _elastic_block(groups, regs[ℓ], qs[ℓ], c, μi, νi, ϕ₀, xg, wg, tdegs, Tp)
        Mo, _ = _elastic_block(groups, regs[ℓ + 1], qs[ℓ], c, μo, νo, ϕ₀, xg, wg, tdegs, Tp)
        A[rows, (offs[ℓ] + 1):offs[ℓ + 1]] .= Mi
        A[rows, (offs[ℓ + 1] + 1):offs[ℓ + 2]] .= -Mo
        if ℓ == N
            for ((gi, n), amp) in remote
                Mr, _ = _elastic_block(
                    [groups[gi]], (:regular,), qs[ℓ], c, μo, νo, ϕ₀, xg, wg, tdegs, Tp
                )
                rhs[rows] .+= amp * Mr[:, findfirst(==(n), groups[gi].degrees)]
            end
        end
    end

    # Drop the identically zero columns before solving. There is at least one:
    # the REGULAR degree-0 mode of `φ₀` is a constant potential, so it moves
    # nothing and its column is exactly zero. `Float64` QR tolerated that; with
    # a `ForwardDiff.Dual` element type it raised `SingularException`. Filtering
    # is both the fix and better conditioning, and it catches any other null
    # direction a future case might introduce.
    keep = [j for j in axes(A, 2) if any(!iszero, @view A[:, j])]
    solk = A[:, keep] \ rhs
    sol = zeros(Tp, size(A, 2))
    sol[keep] .= solk
    nr = sqrt(sum(abs2, rhs))
    residual = sqrt(sum(abs2, A * sol - rhs)) / (nr > 0 ? nr : one(nr))
    amplitudes = [
        [
            (col, sol[offs[ℓ] + i]) for (i, col) in enumerate(
                    [
                        (gi, r, n) for gi in eachindex(groups) for r in regs[ℓ]
                        for n in groups[gi].degrees
                    ]
                )
        ] for ℓ in 1:(N + 1)
    ]
    return (; amplitudes, groups, residual)
end

"""
    _surface_moment(groups, amps, q, c, μ, ν, ::Type{T}; np, nϕ) -> 3×3

The unnormalized surface moment on one confocal surface,

```math
\\oint_{q}\\operatorname{sym}(\\underline u\\otimes\\underline e_q)\\,\\mathrm dS,
\\qquad \\mathrm dS = c^2\\,\\bar q\\,w\\,\\mathrm dp\\,\\mathrm d\\varphi .
```

Averages are built from these by [`_avg_strain`](@ref); nothing here is divided
by a volume, so the same routine serves a shell and the whole inclusion.
"""
function _surface_moment(
        groups, amps, q, c, μ, ν, ::Type{T}; np::Int = 40, nϕ::Int = 48
    ) where {T}
    xg, wg = QuadGK.gauss(_quad_eltype(T), np)
    E = zeros(T, 3, 3)
    qb = sqrt(q^2 - one(q))
    dϕ = 2 * T(π) / nϕ
    for (ip, x) in enumerate(xg)
        w = sqrt(q^2 - x^2)
        wp = c^2 * qb * w * wg[ip] * dϕ
        for k in 0:(nϕ - 1)
            ϕ = 2 * T(π) * k / nϕ
            e = _cartesian_in_chart(ϕ, x, q, T)
            U = zeros(T, 3)
            for ((gi, r, n), amp) in amps
                u, _ = _group_fields(groups[gi], r, n, ϕ, x, q, c, μ, ν, T)
                for i in 1:3, a in 1:3
                    U[i] += amp * u[a] * e[i][a]
                end
            end
            n3 = ntuple(i -> e[i][3], 3)
            for i in 1:3, j in 1:3
                E[i, j] += wp * (U[i] * n3[j] + U[j] * n3[i]) / 2
            end
        end
    end
    return E
end

"Confocal volume up to `q`: `(4π/3) c³ q(q²-1)`, and `0` at `q = 1`."
@inline _confocal_volume(q, c) = 4 * oftype(float(q), π) / 3 * c^3 * q * (q^2 - one(q))

"""
    _avg_strain(groups, amps, q_out, q_in, c, μ, ν, ::Type{T}; kw...) -> 3×3

Volume-averaged strain over one confocal shell `q_in < q ≤ q_out`, as the
difference of two surface moments,

```math
\\langle\\boldsymbol\\varepsilon\\rangle_k \\, V_k
   = \\oint_{q=q_{out}} \\operatorname{sym}(\\underline u\\otimes\\underline e_q)\\,\\mathrm dS
   - \\oint_{q=q_{in}}  \\operatorname{sym}(\\underline u\\otimes\\underline e_q)\\,\\mathrm dS .
```

Pass `q_in = 1` for the core: the confocal surface degenerates to the focal
segment there and `dS = c²\\bar q w\\,\\mathrm dp\\,\\mathrm d\\varphi` vanishes with
`\\bar q`, so the term drops out on its own.

For the **whole** inclusion take `q_in = 1` and the outermost layer's
amplitudes: with perfect interfaces `u` is continuous, so the inner boundaries
cancel in pairs and only the outer surface survives, whatever `N` is.
"""
function _avg_strain(
        groups, amps, q_out, q_in, c, μ, ν, ::Type{T}; kw...
    ) where {T}
    E = _surface_moment(groups, amps, q_out, c, μ, ν, T; kw...)
    if q_in > one(q_in) + eps(float(real(T)))
        E = E .- _surface_moment(groups, amps, q_in, c, μ, ν, T; kw...)
    end
    return E ./ (_confocal_volume(q_out, c) - _confocal_volume(q_in, c))
end

"""
    _basis_loadings(::Type{T}) -> Vector{Tuple{case, args, E}}

The six independent remote strains that fill the concentration tensor, each
with the elementary problem that solves it and the strain tensor it is:

| loading | case | remote strain |
|:--|:--|:--|
| 1 | `AxisymmetricCase`, `(1,0)` | `ê₃⊗ê₃` |
| 2 | `AxisymmetricCase`, `(0,1)` | `𝟙 - ê₃⊗ê₃` |
| 3 | `TransverseShearCase(:sin)` | `ê₁⊗ê₂ + ê₂⊗ê₁` |
| 4 | `TransverseShearCase(:cos)` | `ê₁⊗ê₁ - ê₂⊗ê₂` |
| 5 | `LongitudinalShearCase(:cos)` | `ê₁⊗ê₃ + ê₃⊗ê₁` |
| 6 | `LongitudinalShearCase(:sin)` | `ê₂⊗ê₃ + ê₃⊗ê₂` |

Loadings 3–4 and 5–6 are the same problem rotated about the axis, so their
responses must match up to that rotation — which is a check the assembly gets
for free.
"""
function _basis_loadings(::Type{T}) where {T}
    sym(f) = TensND.Tens(TensND.SymmetricTensor{2, 3}((i, j) -> T(f(i, j))))
    return [
        (AxisymmetricCase(), (one(T), zero(T)), sym((i, j) -> i == j == 3 ? 1 : 0)),
        (AxisymmetricCase(), (zero(T), one(T)), sym((i, j) -> i == j && i != 3 ? 1 : 0)),
        (TransverseShearCase(:sin), (one(T),), sym((i, j) -> (i, j) in ((1, 2), (2, 1)) ? 1 : 0)),
        (TransverseShearCase(:cos), (one(T),), sym((i, j) -> i == j == 1 ? 1 : (i == j == 2 ? -1 : 0))),
        (LongitudinalShearCase(:cos), (one(T),), sym((i, j) -> (i, j) in ((1, 3), (3, 1)) ? 1 : 0)),
        (LongitudinalShearCase(:sin), (one(T),), sym((i, j) -> (i, j) in ((2, 3), (3, 2)) ? 1 : 0)),
    ]
end

"""
    spheroid_strain_concentration(s::LayeredSpheroid, C₀; D = 6, kw...) -> Tens{4,3}

Volume-averaged strain concentration tensor `𝔸` of an elastic `n`-layer
confocal spheroid in an infinite isotropic matrix `C₀`, defined by
`⟨ε⟩ = 𝔸 : E` over the whole composite inclusion.

This is what a mean-field scheme consumes. It is assembled from **six** solves,
one per independent remote strain — [`_basis_loadings`](@ref) lists them — each
averaged by the surface integral of [`_avg_strain`](@ref). The result is
returned as a general fourth-order tensor rather than a `TensTI{4}`: it *is*
transversely isotropic about the spheroid's axis, but the tests check that
rather than the type asserting it, and `𝔸` has no major symmetry to exploit.

Returns `(; A, residuals)`, `residuals` being the six least-squares residuals —
diagnostics, machine-precision for a single inclusion.

!!! note "The axis"
    `𝔸` comes out in the global frame, so a spheroid whose `axis` is not `ê₃`
    is not yet handled here: the elementary problems are written about `ê₃`.
    Build the spheroid with the default axis and rotate the result.
"""
function spheroid_strain_concentration(
        s::LayeredSpheroid{T, N, Q}, C₀; D::Int = 6, kw...
    ) where {T, N, Q}
    isapprox(s.axis[3], one(T); atol = sqrt(eps(float(real(T))))) || throw(
        ArgumentError(
            "spheroid_strain_concentration: the elementary problems are written " *
                "about ê₃; got axis = $(s.axis). Build with the default axis and " *
                "rotate the result."
        )
    )
    κ₀, μ₀ = _iso_bulk_shear(C₀)
    κμ = ntuple(k -> _iso_bulk_shear(layer_modulus(s, k)), Val(N))
    # The LAYER moduli belong in this promotion, and leaving them out is the
    # same defect v0.10.1 fixed on the conduction side: a single `Dual` layer
    # among `Float64` ones then meets a `Float64` buffer and throws.
    Tp = promote_type(
        T, real(Q), typeof(κ₀), typeof(μ₀),
        ntuple(k -> typeof(κμ[k][1]), N)..., ntuple(k -> typeof(κμ[k][2]), N)...,
        interfaces_eltype(s.interfaces)
    )
    c = real(s.c)
    q_N = real(s.q[N])
    κN, μN = κμ[N]
    νN = _poisson(κN, μN)

    loads = _basis_loadings(Tp)
    Xc = Matrix{Tp}(undef, 6, 6)
    Yc = Matrix{Tp}(undef, 6, 6)
    resids = Vector{Tp}(undef, 6)
    for (k, (case, args, E)) in enumerate(loads)
        remote = _case_remote(case, c, κ₀, μ₀, args...)
        r = _solve_elastic(s, C₀, case, remote; D, kw...)
        # `u` is continuous, so the outer boundary is read from layer N.
        εav = _avg_strain(r.groups, r.amplitudes[N], q_N, one(q_N), c, μN, νN, Tp)
        Xc[:, k] .= KM(E)
        Yc[:, k] .= KM(TensND.Tens(TensND.SymmetricTensor{2, 3}((i, j) -> εav[i, j])))
        resids[k] = r.residual
    end
    return (; A = inv_KM(Yc * inv(Xc)), residuals = resids)
end

"""
    spheroid_layer_strain_concentration(s, C₀; D = 6, kw...) -> (; A, layers, residuals)

Per-layer strain concentration tensors of an elastic confocal spheroid, plus
their volume-weighted sum.

`layers[k]` is `𝔸_k` with `⟨ε⟩_k = 𝔸_k : E` averaged over layer `k` alone, and
`A` is `Σ_k f_k 𝔸_k`. The two are computed independently — the layers from
differences of surface moments on their own boundaries, `A` from a single
moment on the outer one — so their agreement is a check rather than a
restatement.

Per-layer averages are what a **stiffness contribution tensor** needs:
`N_C = Σ_k f_k (ℂ_k - ℂ_0) : 𝔸_k` cannot be recovered from the total alone,
the layers having different moduli.
"""
function spheroid_layer_strain_concentration(
        s::LayeredSpheroid{T, N, Q}, C₀; D::Int = 6, kw...
    ) where {T, N, Q}
    isapprox(s.axis[3], one(T); atol = sqrt(eps(float(real(T))))) || throw(
        ArgumentError(
            "spheroid_layer_strain_concentration: the elementary problems are " *
                "written about ê₃; got axis = $(s.axis)."
        )
    )
    κ₀, μ₀ = _iso_bulk_shear(C₀)
    κμ = ntuple(k -> _iso_bulk_shear(layer_modulus(s, k)), Val(N))
    Tp = promote_type(
        T, real(Q), typeof(κ₀), typeof(μ₀),
        ntuple(k -> typeof(κμ[k][1]), N)..., ntuple(k -> typeof(κμ[k][2]), N)...,
        interfaces_eltype(s.interfaces)
    )
    c = real(s.c)
    qs = ntuple(k -> real(s.q[k]), Val(N))

    loads = _basis_loadings(Tp)
    Xc = Matrix{Tp}(undef, 6, 6)
    Yl = [Matrix{Tp}(undef, 6, 6) for _ in 1:N]
    Yt = Matrix{Tp}(undef, 6, 6)
    resids = Vector{Tp}(undef, 6)

    for (kk, (case, args, E)) in enumerate(loads)
        remote = _case_remote(case, c, κ₀, μ₀, args...)
        r = _solve_elastic(s, C₀, case, remote; D, kw...)
        Xc[:, kk] .= KM(E)
        for ℓ in 1:N
            κℓ, μℓ = κμ[ℓ]
            νℓ = _poisson(κℓ, μℓ)
            q_in = ℓ == 1 ? one(qs[1]) : qs[ℓ - 1]
            εk = _avg_strain(r.groups, r.amplitudes[ℓ], qs[ℓ], q_in, c, μℓ, νℓ, Tp)
            Yl[ℓ][:, kk] .= KM(TensND.Tens(TensND.SymmetricTensor{2, 3}((i, j) -> εk[i, j])))
        end
        κN, μN = κμ[N]
        εt = _avg_strain(r.groups, r.amplitudes[N], qs[N], one(qs[N]), c, μN, _poisson(κN, μN), Tp)
        Yt[:, kk] .= KM(TensND.Tens(TensND.SymmetricTensor{2, 3}((i, j) -> εt[i, j])))
        resids[kk] = r.residual
    end

    iX = inv(Xc)
    return (;
        A = inv_KM(Yt * iX),
        layers = ntuple(ℓ -> inv_KM(Yl[ℓ] * iX), Val(N)),
        residuals = resids,
    )
end
