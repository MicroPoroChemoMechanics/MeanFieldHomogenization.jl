# =============================================================================
#  annular_solver.jl — intensity factors of a flat annular crack in an
#  isotropic matrix.
#
#  The crack occupies c < r < a in the plane of normal n. Its faces carry a
#  uniform traction; the displacement jump is sought by a Galerkin method on the
#  regularized form of the crack equations, whose kernel 1/ρ is only
#  logarithmically singular:
#
#    mode 0 (normal traction, and the conduction problem):
#       (E*/4) ∬ w′(r) w′(s) r s Q₁(r, s) dr ds = 2π σ ∫ w r dr
#    mode 1 (tangential traction along e_x, jump p(r) cos θ e_r − q(r) sin θ e_θ):
#       (μ/4) ∬ [D(r) D(s)/(1 − ν) + C(r) C(s)] r s Q₁(r, s) dr ds
#          = π τ ∫ (p + q) r dr,   D = (r p)′/r − q/r,   C = −(r q)′/r + p/r
#
#  with Q₁(r, s) = ∫₀^{2π} cos φ / ρ dφ. The derivation is §4 of
#  docs/src/theory/rice_crack_contributions.md.
#
#  Unknowns: w, p, q = h √(1 − t²) Σₙ cₙ Uₙ(t) on r = m + h t, so that the
#  square-root behavior at both fronts is built in, and the intensity factors
#  are read from Σ cₙ Uₙ(±1).
# =============================================================================

"""
    _ring_kernel(r, s, d) -> T

``s\\,Q_1(r, s)``, with ``Q_1(r, s) = \\int_0^{2\\pi}\\cos\\varphi\\,
\\rho^{-1}\\,\\mathrm{d}\\varphi`` and ``\\rho^2 = r^2 + s^2 - 2 r s \\cos\\varphi``.

With ``k^2 = 4 r s/(r + s)^2`` and ``k' = |r - s|/(r + s)``,
``Q_1 = 4[(2 - k^2)K(k) - 2E(k)]/(k^2 (r + s))``. Along the arithmetic–geometric
mean started at ``(a_0, b_0, c_0) = (1, k', k)``, ``E = K(1 - \\sum_{n \\ge 0}
2^{n-1} c_n^2)``, so the ``c_0^2`` terms cancel exactly and

```math
(2 - k^2) K - 2E = 2K \\sum_{n \\ge 1} 2^{n-1} c_n^2 ,
\\qquad c_1 = \\frac{\\min(r, s)}{r + s},
```

a sum of positive terms evaluated without cancellation, with no square root of zero
(safe under ForwardDiff). The separation `d = r - s` is passed by the caller, which
computes it without cancellation near the diagonal; at `d = 0` exactly (a single
point of a quadrature, where the kernel has an integrable logarithmic singularity)
the function returns zero.
"""
function _ring_kernel(r, s, d)
    T = promote_type(typeof(r), typeof(s), typeof(d))
    iszero(d) && return zero(T)
    σ = r + s
    kp = abs(d) / σ
    c = min(r, s) / σ
    a = (1 + kp) / 2
    b = sqrt(kp)
    S = c^2
    w = one(T)
    tol = Elliptic._agm_tol(T)
    for _ in 1:60
        Elliptic._agm_converged(T, a, b, tol) && break
        c = (a - b) / 2
        a, b = (a + b) / 2, sqrt(a * b)
        w *= 2
        S += w * c^2
    end
    K = T(π) / (2a)
    k2 = 4 * r * s / σ^2
    return s * 8 * K * S / (k2 * σ)
end

# Tₙ₊₁(t) and h (1 − t²) Uₙ(t)/s for n = 0 … N − 1, written into `Tv` and `Gv`, by
# the three-term recurrence of Uₙ and Tₙ₊₁ = t Uₙ − Uₙ₋₁.
function _annulus_basis!(Tv, Gv, t, h, s)
    u_prev = zero(t)
    u = one(t)
    g = h * (1 - t^2) / s
    for n in eachindex(Tv, Gv)
        Tv[n] = t * u - u_prev
        Gv[n] = g * u
        u_prev, u = u, 2t * u - u_prev
    end
    return Tv, Gv
end

"""
    _annulus_order(λ) -> Int

Number ``N`` of Chebyshev modes used for an annulus of radius ratio
``\\lambda = c/a``. The jump varies over a length ``\\sim c`` near the inner
front, and the error on the intensity factors decreases as
``\\exp(-0.8\\,N\\sqrt{\\lambda})`` (measured); the rule below keeps it near
``10^{-9}`` down to ``\\lambda = 0.01`` and caps the cost below. The integrals of
the jump converge much faster (``10^{-12}`` from ``N = 32`` at ``\\lambda = 0.01``).
Test `test/Cracks/test_annular.jl`, "Chebyshev order".
"""
_annulus_order(λ) = clamp(ceil(Int, 12 + 10 / sqrt(TensND.primal(λ))), 24, 160)


"""
    _annulus_solve(λ, ν; N=_annulus_order(λ), reltol=1e-11, shear=true) -> NamedTuple

Galerkin solution for the annulus ``\\lambda < r < 1`` under a unit traction, with
unit moduli (``E^* = E/(1-\\nu^2) = 1`` in mode 0, ``\\mu = 1`` in mode 1).

Returns the intensity factors per unit traction at the inner (`in`) and outer (`out`)
fronts — `kI`, `kII`, `kIII` — and the integrals of the jump over the crack, `opening`
(``\\int w\\,\\mathrm{d}S`` per unit ``\\sigma/E^*``) and `slide`
(``\\int [\\![u_x]\\!]\\,\\mathrm{d}S`` per unit ``\\tau/\\mu``). With `shear = false`
only mode 0 is solved (the conduction problem needs no more), and the mode-1 fields
are `nothing`.
"""
function _annulus_solve(
        λ, ν; N::Union{Nothing, Int} = nothing, reltol = 1.0e-11, shear::Bool = true
    )
    T₀ = promote_type(typeof(λ), typeof(ν))
    # An adaptive quadrature and a pivoted solve: no symbolic counterpart.
    is_hard_numeric(T₀) || throw(
        ArgumentError(
            "the annular crack is solved numerically (adaptive quadrature, linear solve): " *
                "its radius ratio and the Poisson ratio must be numbers, got $T₀"
        )
    )
    T = float(T₀)
    N = something(N, _annulus_order(λ))
    N ≥ 2 || throw(ArgumentError("the annulus solver needs N ≥ 2 Chebyshev modes, got N = $N"))
    # Angular nodes and quadrature limits carry no derivative: QuadGK builds its
    # rule in the type of the limits, which must be a plain float.
    R = TensND.primal_type(T)
    h = (1 - λ) / 2
    m = (1 + λ) / 2
    M = 2N + 8
    A0 = zeros(T, N, N)
    A1 = shear ? zeros(T, 2N, 2N) : zeros(T, 0, 0)
    Tx = Vector{T}(undef, N)
    Gx = Vector{T}(undef, N)
    Tt = similar(Tx)
    Gt = similar(Gx)
    n1 = T.(1:N)
    I = Vector{T}(undef, 2N)
    for j in 1:M
        θx = R(2j - 1) * R(π) / (2M)
        x = cos(θx)
        r = m + h * x
        _annulus_basis!(Tx, Gx, x, h, r)
        # The separation r − s = h (x − t) is written as a product, exact near
        # the diagonal where m + h x − (m + h t) loses every digit.
        integrand!(v, θt) = begin
            t = cos(θt)
            s = m + h * t
            d = 2h * sin((θx + θt) / 2) * sin((θt - θx) / 2)
            q = _ring_kernel(r, s, d)
            _annulus_basis!(Tt, Gt, t, h, s)
            @views v[1:N] .= q .* Tt
            @views v[(N + 1):end] .= q .* Gt
            v
        end
        quadgk!(integrand!, I, zero(R), θx, R(π); rtol = reltol)
        IT = I[1:N]
        IG = I[(N + 1):end]
        κ = T(π) / M * r * h^2 / 4
        A0 .+= κ .* ((n1 .* Tx) * transpose(n1 .* IT))
        if shear
            dD = vcat(-(n1 .* Tx) .+ Gx, -Gx)
            dC = vcat(Gx, (n1 .* Tx) .- Gx)
            ID = vcat(-(n1 .* IT) .+ IG, -IG)
            IC = vcat(IG, (n1 .* IT) .- IG)
            A1 .+= κ .* (dD * transpose(ID) ./ (1 - ν) .+ dC * transpose(IC))
        end
    end
    π² = T(π)^2
    L0 = zeros(T, N)
    L0[1] = π² * h^2 * m
    L0[2] = π² * h^3 / 2
    w = lu((A0 .+ transpose(A0)) ./ 2) \ L0
    # Σₙ cₙ Uₙ(±1), with Uₙ(1) = n + 1 and Uₙ(−1) = (−1)ⁿ (n + 1)
    edge(cn, sgn) = sum(cn[k] * sgn^(k - 1) * k for k in eachindex(cn))
    sq = sqrt(T(π) * h) / 4
    kI = (; in = sq * edge(w, -1), out = sq * edge(w, 1))
    opening = π² * h^2 * (m * w[1] + h * w[2] / 2)
    shear || return (; kI, kII = nothing, kIII = nothing, opening, slide = nothing)
    L1 = zeros(T, 2N)
    L1[1] = L1[N + 1] = π² * h^2 * m / 2
    L1[2] = L1[N + 2] = π² * h^3 / 4
    cαβ = lu((A1 .+ transpose(A1)) ./ 2) \ L1
    α = cαβ[1:N]
    β = cαβ[(N + 1):end]
    kII = (; in = 2 / (1 - ν) * sq * edge(α, -1), out = 2 / (1 - ν) * sq * edge(α, 1))
    kIII = (; in = 2 * sq * edge(β, -1), out = 2 * sq * edge(β, 1))
    slide = π² * h^2 / 2 * (m * (α[1] + β[1]) + h * (α[2] + β[2]) / 2)
    return (; kI, kII, kIII, opening, slide)
end
