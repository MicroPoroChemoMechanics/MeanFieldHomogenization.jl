# =============================================================================
#  lattice_sums.jl — solid harmonics of a periodic array of sources.
#
#  The potential of a unit source repeated on the lattice of periods
#  (L₁, L₂, L₃), made periodic by a uniform background of opposite sign, is the
#  function S with
#
#       ΔS = −4π [ Σ_i δ(x − x_i) − 1/V ] ,   ⟨S⟩ = 0 ,   V = L₁ L₂ L₃ .
#
#  A decaying harmonic is a derivative of 1/r: Y_n^m = 𝒟_n^m (1/r), with 𝒟_n^m
#  the harmonic polynomial (−1)^n (n−m)! (n+m)! y_n^m / (2n−1)!! of the three
#  partial derivatives. Its periodic counterpart is the same derivative of S,
#
#       Y*_n^m(x) = 𝒟_n^m S(x) ,
#
#  which is what `lattice_harmonics` returns; at a lattice point the source of
#  that point is left out, S − 1/r being regular there.
#
#  S is evaluated by Ewald's splitting, with a parameter α,
#
#       S(x) = Σ_i erfc(α r_i)/r_i + (4π/V) Σ_{k≠0} e^{−k²/4α²} e^{ik·x} / k² − π/(α² V) ,
#
#  and 𝒟_n^m goes through each sum term by term. On a radial function a
#  harmonic polynomial of the three partial derivatives acts as (Hobson's theorem)
#  𝒴(∇) f(r) = [(1/r) d/dr]^n f · 𝒴(x), so that
#
#       𝒟_n^m [erfc(αr)/r] = w_n(r) Y_n^m(x) ,
#       w_0 = erfc(αr) ,  w_n = w_{n−1} + (2α²r²)^n e^{−α²r²} / [(2n−1)!! √π α r] ,
#
#  and on a plane wave it acts as 𝒟_n^m e^{ik·x} = 𝒟_n^m(ik) e^{ik·x}. Both sums
#  converge as a Gaussian, whatever the degree: no order of summation has to be
#  prescribed, unlike for the direct sum of the degree-2 harmonics.
#
#  The background is not a detail. Because of it S − 1/r is not harmonic
#  (its Laplacian is 4π/V), and the regular expansion of a periodic dipole
#  about a lattice point holds one term more than the harmonics of degree 2
#  give: a uniform field, the same in the three directions (see `mem.jl`).
# =============================================================================

# Smallest reduced cutoff ξ = α r_c for which the neglected terms, of the order
# of (2ξ²)^n e^{−ξ²}, are below the rounding of a double at every degree.
function _ewald_reduced_cutoff(nmax::Integer)
    ξ = 6.0
    while ξ^2 - nmax * log(2 * ξ^2) < 37
        ξ += 0.25
    end
    return ξ
end

"""
    lattice_harmonics(x, periods, nmax) -> HarmonicTable

Decaying solid harmonics of a source repeated on the orthogonal lattice of
`periods` ``(L_1, L_2, L_3)``, at the point `x` and up to the degree `nmax`:
``Y_n^{*m} = \\mathcal{D}_n^m S``, with ``S`` the periodic potential of the unit
source with its uniform background and ``\\mathcal{D}_n^m`` the derivative such
that ``Y_n^m = \\mathcal{D}_n^m(1/r)``. The degree 0 is ``S`` itself.

At a lattice point (`x` a combination of the periods with integer
coefficients) the source of that point is left out: the table then holds the
field its images send to it.

The sums are Ewald's, which converge as a Gaussian at every degree, the
conditionally convergent degree 2 included.
"""
lattice_harmonics(x::AbstractVector, periods, nmax::Integer) =
    first(_lattice_sums(x, periods, nmax, Val(false)))

"""
    lattice_biharmonics(x, periods, nmax) -> (HarmonicTable, HarmonicTable)

The tables of [`lattice_harmonics`](@ref) and of their biharmonic
counterparts, at the point `x` and up to the degree `nmax`:
``Y_n^{*m} = \\mathcal{D}_n^m S`` and ``B_n^{*m} = \\mathcal{D}_n^m \\rho``, where
``\\rho`` is the periodic function of zero mean with ``\\Delta\\rho = 2S``. It is
the periodic counterpart of ``r``, for which ``\\Delta r = 2/r`` and
``\\mathcal{D}_n^m r = -r^2\\,Y_n^m/(2n-1)``: the table ``B^*`` replaces
``-r^2 Y_n^m/(2n-1)`` in the sums over the images of the solutions of the
Lamé equation. At a lattice point the source of that point is left out of
both.
"""
lattice_biharmonics(x::AbstractVector, periods, nmax::Integer) =
    _lattice_sums(x, periods, nmax, Val(true))

# Both tables by Ewald's splitting with the parameter α; the result does not
# depend on α, which the tests check. With `Val(false)` the second table is
# not computed (`nothing`). The images whose integer coordinates are listed in
# `exclude` are left out of the sums: their short-range terms are skipped and
# the long-range part they carry into the reciprocal sum, (1 − w_n) Y_n, is
# taken off. That part is small, hence free of cancellation, only for the
# images within 1/α or so, which is what `α` is for.
function _lattice_sums(x::AbstractVector, periods, nmax::Integer, bih::Val, α = nothing; exclude = ())
    length(x) == 3 == length(periods) ||
        throw(ArgumentError("lattice_harmonics: a point and three periods in 3D are expected"))
    nmax ≥ 0 || throw(ArgumentError("lattice_harmonics: the degree must be nonnegative, got $nmax"))
    T = float(promote_type(eltype(x), eltype(periods)))
    L = ntuple(i -> T(periods[i]), 3)
    V = prod(L)
    a = _ewald_parameter(α, T, V)
    ξ = _ewald_reduced_cutoff(nmax)
    S, ρ = _ewald_buffers(Complex{T}, _packed(nmax, nmax), bih)
    on_site = _ewald_real!(S, ρ, x, L, a, ξ, Int(nmax), exclude)
    _ewald_reciprocal!(S, ρ, x, L, a, ξ, Int(nmax))
    _ewald_constants!(S, ρ, a, V, on_site)
    return HarmonicTable(Int(nmax), S), _table(Int(nmax), ρ)
end

# α = √π / V^{1/3} balances the two sums.
_ewald_parameter(::Nothing, ::Type{T}, V) where {T} = sqrt(T(π)) / cbrt(V)
_ewald_parameter(α, ::Type{T}, V) where {T} = T(α)

_ewald_buffers(::Type{C}, n::Int, ::Val{false}) where {C} = (zeros(C, n), nothing)
_ewald_buffers(::Type{C}, n::Int, ::Val{true}) where {C} = (zeros(C, n), zeros(C, n))
_table(nmax::Int, ::Nothing) = nothing
_table(nmax::Int, ρ::Vector) = HarmonicTable(nmax, ρ)

# The degree 0 only: the term k = 0 the reciprocal sum leaves out, and the
# value at a lattice point of the source left out (erf(αr)/r → 2α/√π for S;
# r − ψ(r) → 1/(α√π) for ρ). The constant of ρ makes its mean zero.
function _ewald_constants!(S, ρ, α, V, on_site::Bool)
    S[1] -= π / (α^2 * V)
    on_site && (S[1] -= 2 * α / sqrt(oftype(α, π)))
    ρ === nothing && return nothing
    ρ[1] += π / (4 * α^4 * V)
    on_site && (ρ[1] -= 1 / (α * sqrt(oftype(α, π))))
    return nothing
end

# Real-space sums over the images within the cutoff. For S, Σ_i w_n(r_i) Y_n^m(x_i);
# for ρ, whose short-range part is ψ(r) = r erfc(αr) − e^{−α²r²}/(α√π) (so that
# (1/r) dψ/dr = erfc(αr)/r), Hobson's theorem gives
#     𝒟_n^m ψ = w_{n−1}(r) 𝒟_n^m r = −w_{n−1}(r) r² Y_n^m/(2n−1)    (n ≥ 1).
# Returns whether `x` is a lattice point, whose own source is then skipped.
function _ewald_real!(S::Vector{Complex{T}}, ρ, x, L::NTuple{3}, α, ξ, nmax::Int, exclude = ()) where {T}
    r_c = ξ / α
    reach = ntuple(i -> ceil(Int, primal(r_c + abs(x[i])) / primal(L[i])), 3)
    on_site = false
    for n₁ in (-reach[1]):reach[1], n₂ in (-reach[2]):reach[2], n₃ in (-reach[3]):reach[3]
        xn = (T(x[1]) + n₁ * L[1], T(x[2]) + n₂ * L[2], T(x[3]) + n₃ * L[3])
        r² = xn[1]^2 + xn[2]^2 + xn[3]^2
        if iszero(r²)
            on_site = true
            continue
        end
        # an excluded image counts with the sign −1 for its long-range part alone
        excluded = (n₁, n₂, n₃) in exclude
        excluded || r² ≤ r_c^2 || continue
        _ewald_real_image!(S, ρ, xn, r², α, nmax, excluded)
    end
    return on_site
end

# The short-range terms of one image at xn: w_n Y_n for S and 𝒟_n ψ for ρ, or,
# for an excluded image, minus the long-range part, −(1 − w_n) Y_n and
# −(𝒟_n r − 𝒟_n ψ).
function _ewald_real_image!(S::Vector{Complex{T}}, ρ, xn, r², α, nmax::Int, excluded::Bool) where {T}
    r = sqrt(r²)
    Y = irregular_harmonics(collect(xn), nmax)
    e = exp(-α^2 * r²)
    g = e / (sqrt(T(π)) * α * r)
    w = erfc(α * r)
    σ = excluded ? -one(T) : one(T)
    _ewald_real_biharmonic!(ρ, Y, r, _long_range(w, excluded), e, α, 0, excluded)
    dfact = one(T)                       # (2n − 1)!!
    for n in 0:nmax
        if n > 0
            _ewald_real_biharmonic!(ρ, Y, r, _long_range(w, excluded), e, α, n, excluded)    # w = w_{n−1} here
            dfact *= 2n - 1
            w += (2 * α^2 * r²)^n * g / dfact
        end
        f = σ * _long_range(w, excluded)
        for m in 0:n
            S[_packed(n, m)] += f * Y[n, m]
        end
    end
    return nothing
end

# The weight w_n of the short-range part, or 1 − w_n, that of the long-range one.
_long_range(w, excluded::Bool) = excluded ? one(w) - w : w

# The degree n of the short-range part of ρ, from w = w_{n−1}(r) (n ≥ 1), and its
# value at n = 0; for an excluded image, minus the long-range part, w being then
# 1 − w_{n−1} and ψ being replaced by r − ψ.
_ewald_real_biharmonic!(::Nothing, Y, r, w, e, α, n::Int, excluded::Bool) = nothing
function _ewald_real_biharmonic!(ρ::Vector, Y, r, w, e, α, n::Int, excluded::Bool)
    if n == 0
        ψ = r * erfc(α * r) - e / (α * sqrt(oftype(α, π)))
        ρ[1] += excluded ? -(r - ψ) : ψ
        return nothing
    end
    f = (excluded ? w : -w) * r^2 / (2n - 1)
    for m in 0:n
        ρ[_packed(n, m)] += f * Y[n, m]
    end
    return nothing
end

# Reciprocal sums. For S, (4π/V) Σ_{k≠0} e^{−k²/4α²} 𝒟_n^m(ik) e^{ik·x} / k², with
# 𝒟_n^m(ik) = (−i)^n (n−m)! (n+m)! y_n^m(k) / (2n−1)!!. For ρ, the long-range part
# of r, whose Fourier transform −8π/k⁴ is split as −8π [1 − (1 + k²/4α²) e^{−k²/4α²}]/k⁴
# (short range, regular at k = 0) plus −8π (1 + k²/4α²) e^{−k²/4α²}/k⁴.
function _ewald_reciprocal!(S::Vector{Complex{T}}, ρ, x, L::NTuple{3}, α, ξ, nmax::Int) where {T}
    k_c = 2 * α * ξ
    V = prod(L)
    b = ntuple(i -> 2 * T(π) / L[i], 3)
    reach = ntuple(i -> ceil(Int, primal(k_c) / primal(b[i])), 3)
    coef = [exp(logfactorial(n - m) + logfactorial(n + m) - _logdoublefactorial(n)) for n in 0:nmax for m in 0:n]
    for m₁ in (-reach[1]):reach[1], m₂ in (-reach[2]):reach[2], m₃ in (-reach[3]):reach[3]
        m₁ == m₂ == m₃ == 0 && continue
        k = (m₁ * b[1], m₂ * b[2], m₃ * b[3])
        k² = k[1]^2 + k[2]^2 + k[3]^2
        k² ≤ k_c^2 || continue
        y = regular_harmonics(collect(k), nmax)
        wave = exp(-k² / (4 * α^2)) * cis(k[1] * x[1] + k[2] * x[2] + k[3] * x[3]) / V
        _ewald_reciprocal_add!(S, 4 * T(π) / k² * wave, y, coef, nmax)
        _ewald_reciprocal_add!(ρ, -8 * T(π) * (1 + k² / (4 * α^2)) / k²^2 * wave, y, coef, nmax)
    end
    return nothing
end

_ewald_reciprocal_add!(::Nothing, weight, y, coef, nmax::Int) = nothing
function _ewald_reciprocal_add!(acc::Vector, weight, y, coef, nmax::Int)
    phase = one(weight)                      # (−i)^n
    for n in 0:nmax
        for m in 0:n
            acc[_packed(n, m)] += weight * phase * coef[_packed(n, m)] * y[n, m]
        end
        phase *= -im
    end
    return nothing
end

# log (2n − 1)!!, with (−1)!! = 1.
_logdoublefactorial(n::Integer) = n == 0 ? 0.0 : logfactorial(2n) - n * log(2.0) - logfactorial(n)
