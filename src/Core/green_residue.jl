# =============================================================================
#  green_residue.jl
#
#  Factored Masson / Cauchy residue algorithm used by both `Elasticity`
#  (Hill polarization tensor — Masson 2008) and `Cracks` (line-integrated
#  Green kernel ``\\hat{\\mathbf Q}^{\\star}_{nn}`` — Cauchy residue).
#
#  Strategy
#  ========
#  Both algorithms reduce their integrand to the same rational-function
#  pattern :
#
#      f(z) = N(z) / Q(z)
#
#  where
#
#    * `Q(z) = det K(z)` — degree-6 polynomial in `z`
#    * `K(z)` — acoustic tensor evaluated on a caller-supplied parametric
#               wave-vector `ζ(z) = α₀ + α₁ z` (linear polynomial)
#
#  The common building blocks are factored in [`_build_poly_system`](@ref):
#  the coefficient matrices `α₀, α₁, α₂` such that
#  `K(z) = α₀ + α₁ z + α₂ z²`, the polynomial adjugate `adj(K)(z)` and the
#  determinant `Q(z)`, plus its derivative `Q'(z)` and the roots in the
#  upper half plane.
#
#  The algorithm-specific numerator and post-processing (Masson log
#  factor vs Cauchy residue without log) are delegated to a caller-supplied
#  closure in the downstream wrappers — see
#  `Elasticity/hill_3d_aniso_residue.jl` and
#  `Cracks/green_residue.jl`.
# =============================================================================

"""
    _build_acoustic_coeffs(C, α₀ζ, α₁ζ) -> (A₀, A₁, A₂)

Build the three 3×3 coefficient matrices of
``\\boldsymbol{K}(z) = \\boldsymbol{A}_0 + \\boldsymbol{A}_1\\,z + \\boldsymbol{A}_2\\,z^2``
given the stiffness `C` (Float64 `3×3×3×3` array) and the two linear
coefficients `α₀ζ`, `α₁ζ` of the wave-vector parametrization
``\\underline{\\zeta}(z) = \\underline{\\zeta}_0 + \\underline{\\zeta}_1\\,z``.

Each `Aₖ` is a 3×3 matrix of real scalars ; they are assembled below
into `ComplexF64` polynomial entries before root finding.
"""
function _build_acoustic_coeffs(
        C::AbstractArray{<:Real, 4},
        α₀ζ::AbstractVector{<:Real},
        α₁ζ::AbstractVector{<:Real}
    )
    A₀ = zeros(Float64, 3, 3)
    A₁ = zeros(Float64, 3, 3)
    A₂ = zeros(Float64, 3, 3)
    @inbounds for j in 1:3, i in 1:3
        s0 = 0.0; s1 = 0.0; s2 = 0.0
        for k in 1:3, l in 1:3
            cc = Float64(C[i, k, j, l])
            s0 += cc * α₀ζ[k] * α₀ζ[l]
            s1 += cc * (α₀ζ[k] * α₁ζ[l] + α₁ζ[k] * α₀ζ[l])
            s2 += cc * α₁ζ[k] * α₁ζ[l]
        end
        A₀[i, j] = s0; A₁[i, j] = s1; A₂[i, j] = s2
    end
    return A₀, A₁, A₂
end

"""
    _build_poly_system(C, α₀ζ, α₁ζ) -> (K_poly, adj_poly, Q, dQ, roots_uhp)

Build the polynomial acoustic tensor
``\\boldsymbol{K}(z) = \\boldsymbol{A}_0 + \\boldsymbol{A}_1\\,z + \\boldsymbol{A}_2\\,z^2``, its
polynomial adjugate ``\\mathrm{adj}(\\boldsymbol{K})(z)``, the determinant
``Q(z) = \\det\\boldsymbol{K}(z)`` and its derivative ``Q'(z)``, then compute
the roots of ``Q`` lying in the strict upper half plane.

All polynomial entries are `Polynomial{ComplexF64,:z}` (so that complex
roots can be handled uniformly). The roots filter keeps only points
with `imag(zr) > 1e-8`.
"""
function _build_poly_system(
        C::AbstractArray{<:Real, 4},
        α₀ζ::AbstractVector{<:Real},
        α₁ζ::AbstractVector{<:Real}
    )
    _bump!(RESIDUE_SOLVES)
    A₀, A₁, A₂ = _build_acoustic_coeffs(C, α₀ζ, α₁ζ)

    poly(coefs) = Polynomial(ComplexF64.(coefs), :z)
    K_poly = Matrix{Polynomial{ComplexF64, :z}}(undef, 3, 3)
    @inbounds for j in 1:3, i in 1:3
        K_poly[i, j] = poly([A₀[i, j], A₁[i, j], A₂[i, j]])
    end

    # Complementary row/column indices for a fixed 3×3 pattern — a static
    # tuple lookup instead of `setdiff(1:3, [j])`, which allocates two
    # `Vector`s (plus the single-element literal) per one of the 9 entries.
    _other2(m::Int) = m == 1 ? (2, 3) : m == 2 ? (1, 3) : (1, 2)
    adj_poly = Matrix{Polynomial{ComplexF64, :z}}(undef, 3, 3)
    @inbounds for j in 1:3, i in 1:3
        rows = _other2(j)
        cols = _other2(i)
        minor = K_poly[rows[1], cols[1]] * K_poly[rows[2], cols[2]] -
            K_poly[rows[1], cols[2]] * K_poly[rows[2], cols[1]]
        adj_poly[i, j] = isodd(i + j) ? -minor : minor
    end

    Q = Polynomial(ComplexF64[0.0], :z)
    @inbounds for i in 1:3
        Q = Q + K_poly[1, i] * adj_poly[i, 1]
    end
    dQ = derivative(Q)

    Q_coeffs = coeffs(Q)
    while length(Q_coeffs) > 1 && abs(Q_coeffs[end]) < 1.0e-14 * abs(Q_coeffs[1])
        pop!(Q_coeffs)
    end
    roots_all = PolynomialRoots.roots(Q_coeffs)

    # Newton polish: `PolynomialRoots.roots` (Durand-Kerner) typically leaves
    # |Q(zᵣ)| ~ 1e-10…1e-12; 2 Newton iterations push this to O(eps).  Skip
    # roots where |Q'(zᵣ)| is too small (multiple roots).
    Q_scale = maximum(abs, Q_coeffs)
    dQ_thresh = 1.0e-14 * Q_scale
    @inbounds for _ in 1:2
        for i in eachindex(roots_all)
            zr = roots_all[i]
            Qv = Q(zr)
            dQv = dQ(zr)
            abs(dQv) < dQ_thresh && continue
            roots_all[i] = zr - Qv / dQv
        end
    end

    # After polish, genuine UHP roots have imag > O(eps); tighten the filter
    # to catch only real-axis roots.
    roots_uhp = ComplexF64[zr for zr in roots_all if imag(zr) > 1.0e-12]

    return (K_poly = K_poly, adj_poly = adj_poly, Q = Q, dQ = dQ, roots_uhp = roots_uhp)
end

# ─── Multiplicity detection
#
# The Bairstow / Durand-Kerner root finder splits a true multiplicity-k root
# into k clustered numerical roots, each at a relative distance ε^(1/k) from
# the true location. Distance-based clustering would need a per-mult
# tolerance schedule; instead we verify
# the multiplicity directly via the polynomial-derivative criterion:
#
#       Q^(k-1)(z) ≈ 0 in a SCALE-RELATIVE sense  ⇔  mult(z) ≥ k.
#
# Threshold 1e-3 (relative to max|coeff|) is calibrated for degree-≤6 acoustic
# polynomials with Bairstow-precision roots — it catches multiplicities up to
# 6 without false positives on generically-distinct roots whose separation is
# typically O(0.1) in normalized problems.

"""
    _polynomial_scale(Q) -> Float64

Largest |coefficient| of the polynomial; used as the denominator of the
scale-relative multiplicity test. Falls back to 1 if Q is identically zero.
"""
@inline function _polynomial_scale(Q::Polynomial)
    s = 0.0
    @inbounds for c in coeffs(Q)
        a = abs(c)
        a > s && (s = a)
    end
    return s == 0.0 ? 1.0 : s
end

"""
    _gather_almost_multiple_roots(Q, roots; ε_poly=1e-3, ε_match=1e-4,
                                 ε_cluster=1e-2, ref=nothing)
        -> (roots′, mults, refidx)

Promote near-multiple Bairstow / Durand-Kerner roots to exact multiple ones
by checking whether ``Q`` vanishes at the trial root for successively higher
derivative levels (semantically: ``Q^{(k-1)}(z) \\approx 0 \\Rightarrow`` multiplicity ``\\ge k``) **and**
whether a candidate "absorber" Bairstow root lies within a true-cluster
distance. Each element of the returned vectors corresponds to one *cluster
representative*; roots that have been absorbed into a higher-multiplicity
cluster get multiplicity 0 and are skipped by downstream callers.

Three tolerances are used:

* `ε_poly` — relative threshold on ``|Q^{(k-1)}(z)| / \\mathrm{scale}(Q^{(k-1)})``
  for the polynomial-vanishing test. 1e-3 catches multiplicities up to 6 in
  a normalized ``Q`` of degree ``\\le 6``.
* `ε_match` — absolute distance in the complex plane below which an
  externally-supplied reference point is considered to coincide with a
  Bairstow root. Defaults to the Bairstow / Durand-Kerner precision floor,
  ~1e-4; **NOT** scale-dependent.
* `ε_cluster` — max in-cluster distance. A true multiplicity-``k`` root cluster
  splits under finite precision into ``k`` Bairstow roots within ``\\varepsilon^{1/k}``,
  giving ~1e-8 for ``k=2``, ~5e-6 for ``k=3``, ~1e-4 for ``k=4``. The default 1e-3
  catches multiplicities up to 4 reliably while rejecting clustered-but-
  distinct simple roots (which generically sit ``\\ge O(10^{-2})`` apart even when
  they are close to each other in absolute terms). Multiplicities 5 and 6
  fall back to DECUHR. **This guard is essential**: the polynomial-derivative
  test alone gives false positives e.g. for the rotationally-symmetric
  TI-aligned spheroid (``Q`` has three close-but-simple root pairs and ``|Q'|/
  \\mathrm{scale}(Q') < \\varepsilon_{\\mathrm{poly}}`` at each one).

If `ref` is provided (typically `complex(0,1)` for the Hill log_I term), it
is treated as an extra reference point that must appear in the returned
list with its true multiplicity, even if no Bairstow root landed nearby.
"""
function _gather_almost_multiple_roots(
        Q::Polynomial,
        roots::AbstractVector{ComplexF64};
        ε_poly::Float64 = 1.0e-3,
        ε_match::Float64 = 1.0e-4,
        ε_cluster::Float64 = 1.0e-3,
        ref::Union{Nothing, ComplexF64} = nothing
    )
    z_list = ComplexF64[r for r in roots]
    mults = Int[1 for _ in roots]

    # Insert the reference point at the beginning if it does not already
    # appear in the cluster list (needed so the log_I term sees its
    # multiplicity even when none of the Bairstow roots collapsed onto i).
    refidx = -1
    if ref !== nothing
        matched = false
        @inbounds for j in eachindex(z_list)
            if abs(z_list[j] - ref) < ε_match
                refidx = j
                matched = true
                break
            end
        end
        if !matched
            push!(z_list, ref)
            push!(mults, 0)            # ref is not (yet) known to be a root
            refidx = length(z_list)
        end
    end

    # Iterate from the END so that earlier (likely-distinct) roots stay put
    # and later (likely-clustered) roots feed their multiplicity into them.
    for i in length(z_list):-1:1
        if mults[i] > 0 || i == refidx
            z0 = z_list[i]
            QQ = (i == refidx) ? Q : derivative(Q)
            scale = _polynomial_scale(QQ)
            keep_going = true
            while abs(QQ(z0)) < ε_poly * scale && keep_going
                # Find the nearest active root with index < i to absorb.
                nearest = -1
                d_min = Inf
                @inbounds for j in 1:(i - 1)
                    if mults[j] > 0
                        d = abs(z_list[j] - z0)
                        if d < d_min
                            d_min = d
                            nearest = j
                        end
                    end
                end
                # Reject the absorption if the nearest candidate is farther
                # away than ε_cluster: the Q^(k-1) test alone would group
                # clustered-but-distinct simple roots, which then break the
                # downstream multiplicity-k residue formula.
                if nearest > 0 && d_min < ε_cluster
                    mults[nearest] -= 1
                    mults[i] += 1
                    QQ = derivative(QQ)
                    scale = _polynomial_scale(QQ)
                else
                    keep_going = false
                end
            end
        end
    end

    return z_list, mults, refidx
end

# ─── Multiplicity-aware residue formulas
#
# Two families of formulas are needed, both derived analytically (Maple) for
# the Hill residue algorithm:
#
#   * `_residue_logI_mult` — residue at z=i, where the Masson log factor
#     itself has a branch to handle. Multiplicities 0, 1, 2, and the special
#     deg(Q)=6 closed form for mult=3.
#
#   * `_residue_logz_mult` — residue at any other UHP root, with the Masson
#     log factor applied multiplicatively. Multiplicities 1 and 2; mult=3
#     would require a 100-line formula not ported here — the upstream
#     fallback to DECUHR handles it.
#
# Returning NaN signals an unsupported configuration (mult ≥ 3 in log_z,
# or any mult ≥ 4 in log_I) and triggers the DECUHR fallback in the caller.

"""
    _residue_logI(P, Q, mult) -> ComplexF64

Residue at ``z = i`` (with Masson log factor folded in) for a numerator ``P`` and
denominator ``Q`` of multiplicity `mult` at that point. Returns NaN+NaN·im if
the multiplicity is beyond what is implemented.
"""
function _residue_logI(P::Polynomial, Q::Polynomial, mult::Int)
    z = ComplexF64(0.0, 1.0)
    if mult == 0
        return P(z) / Q(z)
    elseif mult == 1
        Q1 = derivative(Q); q1 = Q1(z)
        Q2 = derivative(Q1); q2 = Q2(z)
        p0 = P(z); P1 = derivative(P); p1 = P1(z)
        return (5.0 * im * p0 * q1 + 6.0 * p1 * q1 - 3.0 * p0 * q2) / (6.0 * q1 * q1)
    elseif mult == 2
        p0 = P(z); P1 = derivative(P); p1 = P1(z); P2 = derivative(P1); p2 = P2(z)
        Q1 = derivative(Q); Q2 = derivative(Q1); q2 = Q2(z)
        Q3 = derivative(Q2); q3 = Q3(z); Q4 = derivative(Q3); q4 = Q4(z)
        t1 = q2 * q2
        t7 = q3 * q3
        return (
            -99.0 * t1 * p0 - 15.0 * p0 * q2 * q4 + 20.0 * t7 * p0
                + 150.0 * t1 * p1 * z + 90.0 * t1 * p2
                - 50.0 * z * p0 * q2 * q3 - 60.0 * p1 * q2 * q3
        ) / (90.0 * t1 * q2)
    elseif mult == 3 && degree(Q) == 6
        # Closed form for the special degenerate case Q = Q[6]·(z²+1)³·…
        # (only the leading coefficient of Q matters; the rest cancels).
        p0 = P(z); P1 = derivative(P); p1 = P1(z)
        P2 = derivative(P1); p2 = P2(z); P3 = derivative(P2); p3 = P3(z)
        return (z * p3 / 48.0 + 16.0 / 35.0 * p0 - 33.0 / 80.0 * z * p1 - 7.0 / 48.0 * p2) / Q[6]
    else
        return ComplexF64(NaN, NaN)
    end
end

"""
    _residue_logz(P, Q, z, mult) -> ComplexF64

Residue at a general UHP root ``z`` (with Masson log factor folded in) for a
numerator ``P`` and denominator ``Q`` with multiplicity `mult` at ``z``. Returns
NaN+NaN·im if `mult > 2` (the 3-formula was deemed too long to port; the
caller falls back to DECUHR in that case).
"""
function _residue_logz(P::Polynomial, Q::Polynomial, z::ComplexF64, mult::Int)
    if mult == 0
        return ComplexF64(0.0, 0.0)
    elseif mult == 1
        Q1 = derivative(Q); q1 = Q1(z)
        t2 = 1.0 + z * z
        t3 = sqrt(t2)
        L = -(2.0 * log(z + t3) - im * π)
        return L * P(z) / (q1 * t3 * t2)
    elseif mult == 2
        p0 = P(z); P1 = derivative(P); p1 = P1(z)
        Q1 = derivative(Q); Q2 = derivative(Q1); q2 = Q2(z)
        Q3 = derivative(Q2); q3 = Q3(z)
        t2 = z * z
        t3 = 1.0 + t2
        t4 = sqrt(t3)
        t6 = log(z + t4)
        t10 = im * q2
        t29 = q3 * p0
        t32 = im * p0
        t44 = -18.0 * t6 * p0 * q2 * z + 9.0 * z * p0 * π * t10 +
            6.0 * t6 * p1 * t2 * q2 - 3.0 * π * p1 * t2 * t10 +
            6.0 * t6 * p1 * q2 - 3.0 * p1 * π * t10 -
            2.0 * t6 * t29 + q3 * π * t32 -
            2.0 * t6 * t2 * t29 + π * t2 * q3 * t32 +
            6.0 * q2 * t4 * p0
        t45 = t3 * t3
        t49 = q2 * q2
        return -2.0 / 3.0 / t49 / t4 / t45 * t44
    else
        return ComplexF64(NaN, NaN)
    end
end
