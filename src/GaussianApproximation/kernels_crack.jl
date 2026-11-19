# =============================================================================
#  kernels_crack.jl — action of the crack operator on one planar Gaussian.
#
#  In the plane of a crack of normal n, the traction t = Σ·n needed to open the
#  crack by [[u]] is t = T * [[u]], an operator whose symbol is homogeneous of
#  degree one in the in-plane wave vector ξ = |ξ| e(θ):
#
#      T̂(ξ) = |ξ| T*(θ) ,   T*(θ + π) = T*(θ) .
#
#  T* is the reduced kernel Q̂*ₙₙ of `Cracks` in elasticity (in an isotropic
#  medium (μ/2) [𝟏 + ν/(1-ν) (n⊗n + e⊗e)]), and half the square root of a
#  quadratic form in conduction. The Gaussian φ(x) = exp(-|x|²/(H h²)) / (πH)
#  has the transform h² exp(-|ξ|² H h²/4), so the traction at x = ρ h (cos ϑ,
#  sin ϑ) due to one Gaussian is
#
#      (T * φ)(x) = (1/h) Σₖ [Tᶜₖ cos 2kϑ + Tˢₖ sin 2kϑ] Zₖ(ρ) ,
#
#  with Tᶜₖ, Tˢₖ the Fourier coefficients of T* in 2θ and Zₖ a radial profile
#  closed in modified Bessel functions.
# =============================================================================

"""
    harmonic_profile(k, ρ², H)

Radial profile of the harmonic ``2k`` of the traction due to a planar Gaussian of
shape parameter `H`, at the squared distance `ρ²` (in grid steps) from its node:

```math
Z_k(\\rho) = \\frac{1}{2\\pi^{3/2}H^{3/2}}\\int_0^{2\\pi}\\cos 2k\\varphi\\,
    \\Bigl(1 - \\frac{2\\rho^2\\cos^2\\varphi}{H}\\Bigr)\\,
    \\mathrm{e}^{-\\rho^2\\cos^2\\varphi/H}\\,\\mathrm{d}\\varphi
  = \\frac{(-1)^k\\,\\mathrm{e}^{-z/2}}{\\sqrt{\\pi}\\,H^{3/2}}
    \\Bigl[(1-z)\\,I_k + \\tfrac{z}{2}\\,(I_{k-1} + I_{k+1})\\Bigr],
```

with ``z = \\rho^2/H`` and the modified Bessel functions taken at ``z/2``.
``Z_0 \\simeq -1/(2\\pi\\rho^3)`` far from the node: the field of a point
dislocation loop.
"""
function harmonic_profile(k::Integer, ρ², H)
    z = ρ² / H
    # scaled Bessel functions exp(-x) Iₙ(x): finite at any distance
    i₋, i₀, i₊ = besselix(abs(k - 1), z / 2), besselix(k, z / 2), besselix(k + 1, z / 2)
    return (isodd(k) ? -1 : 1) * ((1 - z) * i₀ + z * (i₋ + i₊) / 2) / (sqrt(oftype(z, π)) * H * sqrt(H))
end

"""
    SymbolHarmonics{m, T}

Fourier coefficients in ``2\\theta`` of the symbol ``\\boldsymbol{T}^\\star(\\theta)``
of a crack operator, an `m × m` symmetric matrix: `cosines[k + 1]` and
`sines[k + 1]` multiply ``\\cos 2k\\theta`` and ``\\sin 2k\\theta``.
"""
struct SymbolHarmonics{m, T <: Number, L}
    cosines::Vector{SMatrix{m, m, T, L}}
    sines::Vector{SMatrix{m, m, T, L}}
end

"""
    symbol_harmonics(symbol, Val(m); angles=32, rtol=1e-12) -> SymbolHarmonics

Harmonics of `symbol(θ)`, an `m × m` static matrix of period ``\\pi``, from its
values at `angles` equally spaced angles. Harmonics beyond the last one larger
than `rtol` times the mean are dropped: an isotropic medium keeps two.
"""
function symbol_harmonics(symbol::F, ::Val{m}; angles::Integer = 32, rtol::Real = 1.0e-12) where {F, m}
    θs = (0:(angles - 1)) .* (π / angles)
    values = [symbol(θ) for θ in θs]
    cosines = [(k == 0 ? 1 : 2) * sum(v * cos(2k * θ) for (v, θ) in zip(values, θs)) / angles for k in 0:(angles ÷ 2 - 1)]
    sines = [2 * sum(v * sin(2k * θ) for (v, θ) in zip(values, θs)) / angles for k in 0:(angles ÷ 2 - 1)]
    size_of(M) = maximum(abs ∘ TensND.primal, M)
    # the truncation is decided on the values, whatever derivatives they carry
    last = findlast(k -> size_of(cosines[k]) + size_of(sines[k]) > rtol * size_of(cosines[1]), eachindex(cosines))
    last == length(cosines) && throw(
        ArgumentError(
            "the symbol of the crack operator is not resolved by $angles angles " *
                "(strong anisotropy): raise `angles`."
        )
    )
    return SymbolHarmonics(cosines[1:last], sines[1:last])
end

# Kernel of the crack operator at the differences of nodes of the grid.
function _harmonic_samples(grid::GaussianGrid{2}, hs::SymbolHarmonics{m}, vm::Val{m}) where {m}
    return sample_kernel(grid.dims, vm) do Δ
        ρ² = sum(abs2, Δ)
        ϑ = atan(Δ[2], Δ[1])
        sum(
            (c * cos(2k * ϑ) + s * sin(2k * ϑ)) * (harmonic_profile(k, ρ², grid.H) / grid.h)
                for (k, c, s) in zip(0:(length(hs.cosines) - 1), hs.cosines, hs.sines)
        )
    end
end

# ─── Symbols, in the frame of the crack (n = e₃) ─────────────────────────────

# Isotropic elasticity: T*(θ) = (μ/2) [𝟏 + ν/(1-ν) (n⊗n + e⊗e)].
function _crack_symbol(C₀::TensND.TensISO{4, 3}, ::TensND.AbstractBasis)
    E, ν = TensND.E_nu(C₀)
    μ, η = E / (2 * (1 + ν)), ν / (1 - ν)
    n = SVector(0, 0, 1)
    return θ -> (e = SVector(cos(θ), sin(θ), 0); (μ / 2) * (I + η * (n * n' + e * e')))
end

# Any anisotropy: the reduced kernel Q̂*ₙₙ of `Cracks`, with the stiffness read
# in the frame of the crack.
function _crack_symbol(C₀::TensND.AbstractTens{4, 3}, basis::TensND.AbstractBasis)
    C = Array(TensND.components(C₀, basis))
    n = SVector(0.0, 0.0, 1.0)
    return θ -> SMatrix{3, 3}(Cracks._Qnn_star_nestedquadgk(C, SVector(cos(θ), sin(θ), zero(θ)), n))
end

# Conduction: T*(θ) = ½ √(Kₙₙ (e·K·e) - (e·K·n)²), (k₀/2) in an isotropic medium.
function _crack_symbol(K₀::TensND.AbstractTens{2, 3}, basis::TensND.AbstractBasis)
    K = SMatrix{3, 3}(TensND.components(K₀, basis))
    return θ -> (e = SVector(cos(θ), sin(θ), 0); SMatrix{1, 1}(sqrt(K[3, 3] * (transpose(e) * K * e) - (transpose(e) * K[:, 3])^2) / 2))
end
