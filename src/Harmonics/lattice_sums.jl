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
function lattice_harmonics(x::AbstractVector, periods, nmax::Integer)
    length(x) == 3 == length(periods) ||
        throw(ArgumentError("lattice_harmonics: a point and three periods in 3D are expected"))
    T = float(promote_type(eltype(x), eltype(periods)))
    L = ntuple(i -> T(periods[i]), 3)
    V = prod(L)
    α = sqrt(T(π)) / cbrt(V)
    ξ = _ewald_reduced_cutoff(nmax)
    acc = zeros(Complex{T}, _packed(nmax, nmax))
    on_site = _ewald_real!(acc, x, L, α, ξ, Int(nmax))
    _ewald_reciprocal!(acc, x, L, α, ξ, Int(nmax))
    acc[1] -= T(π) / (α^2 * V)
    # the source left out at a lattice point takes erf(αr)/r → 2α/√π with it
    on_site && (acc[1] -= 2 * α / sqrt(T(π)))
    return HarmonicTable(Int(nmax), acc)
end

# Σ_i w_n(r_i) Y_n^m(x_i) over the images x_i within the cutoff. Returns whether
# `x` is a lattice point, whose own source is then skipped.
function _ewald_real!(acc::Vector{Complex{T}}, x, L::NTuple{3}, α, ξ, nmax::Int) where {T}
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
        r² ≤ r_c^2 || continue
        r = sqrt(r²)
        Y = irregular_harmonics(collect(xn), nmax)
        g = exp(-α^2 * r²) / (sqrt(T(π)) * α * r)
        w = erfc(α * r)
        dfact = one(T)                       # (2n − 1)!!
        for n in 0:nmax
            if n > 0
                dfact *= 2n - 1
                w += (2 * α^2 * r²)^n * g / dfact
            end
            for m in 0:n
                acc[_packed(n, m)] += w * Y[n, m]
            end
        end
    end
    return on_site
end

# (4π/V) Σ_{k≠0} e^{−k²/4α²} 𝒟_n^m(ik) e^{ik·x} / k², with
# 𝒟_n^m(ik) = (−i)^n (n−m)! (n+m)! y_n^m(k) / (2n−1)!!.
function _ewald_reciprocal!(acc::Vector{Complex{T}}, x, L::NTuple{3}, α, ξ, nmax::Int) where {T}
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
        weight = (4 * T(π) / V) * exp(-k² / (4 * α^2)) / k² * cis(k[1] * x[1] + k[2] * x[2] + k[3] * x[3])
        phase = one(Complex{T})              # (−i)^n
        for n in 0:nmax
            for m in 0:n
                acc[_packed(n, m)] += weight * phase * coef[_packed(n, m)] * y[n, m]
            end
            phase *= -im
        end
    end
    return acc
end

# log (2n − 1)!!, with (−1)!! = 1.
_logdoublefactorial(n::Integer) = n == 0 ? 0.0 : logfactorial(2n) - n * log(2.0) - logfactorial(n)
