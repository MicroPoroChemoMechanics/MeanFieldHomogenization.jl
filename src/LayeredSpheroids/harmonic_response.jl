# =============================================================================
#  harmonic_response.jl — response of a layered spheroid to the spheroidal
#  harmonics of any degree and order, for the multipole expansion method.
#
#  In the confocal coordinates (ξ, η, φ) of focal distance c (`s.c`, real for
#  a prolate spheroid, −i·focal for an oblate one), the regular and decaying
#  spheroidal solid harmonics are (kushch2013, Eqs. (6.11), (6.12))
#
#      f_n^m = P̄_n^m(ξ) χ_n^m(η, φ) ,   F_n^m = Q̄_n^m(ξ) χ_n^m(η, φ) ,
#      P̄_n^m = (n−m)!/(n+m)! P_n^m ,    Q̄_n^m = (n−m)!/(n+m)! Q_n^m ,
#
#  with χ_n^m the surface harmonic of the solid harmonics of a sphere and, at
#  a negative order, the radial factor of the order |m|. A spheroid placed in
#  the regular field Σ a_n^m f_n^m sends Σ b_n^m F_n^m, and the transfer matrices
#  of `conductivity.jl`, written here at the order m and the degrees
#  max(m, 1), …, nmax, give b = 𝐙 a: the core holds no decaying part, so the
#  product S of the transfer matrices maps the regular amplitudes of the core
#  to the regular (S₁₁) and decaying (S₂₁) amplitudes in the matrix, and
#  𝐙 = S₂₁ S₁₁⁻¹ up to the normalization P̄, Q̄. A perfect interface gives a
#  diagonal 𝐙; a Kapitza or surface-conductive interface couples the degrees
#  of the same parity.
# =============================================================================

"""
    harmonic_response(spheroid::LayeredSpheroid, K₀, m, nmax) -> Matrix

Response of a layered spheroid in a matrix of conductivity `K₀` to the regular
spheroidal harmonics of the order `m` and of the degrees
``n = \\max(|m|, 1), \\dots,`` `nmax`: the matrix ``\\mathbf{Z}`` such that the
regular field ``\\sum_n a_n^m f_n^m`` makes the spheroid send
``\\sum_n b_n^m F_n^m`` with ``b = \\mathbf{Z}\\,a``, the harmonics being those of
[kushch2013](@cite) (Eqs. (6.11), (6.12)) in the confocal coordinates of the
spheroid. The degrees couple at a Kapitza or a surface-conductive interface,
the parities never.
"""
function harmonic_response(s::LayeredSpheroid, K₀, m::Integer, nmax::Integer)
    μ = abs(Int(m))
    degrees = max(μ, 1):Int(nmax)
    isempty(degrees) && throw(
        ArgumentError("harmonic_response: the order $m has no degree between 1 and nmax = $nmax")
    )
    k₀ = _as_scalar_k(K₀)
    tables = _spheroid_tables(s, μ, degrees)
    S = _spheroid_transfer(s, k₀, tables)[end]
    𝒩 = length(degrees)
    Z = S[(𝒩 + 1):(2𝒩), 1:𝒩] / S[1:𝒩, 1:𝒩]
    # from the amplitudes of P_n^m, Q_n^m to those of P̄_n^m, Q̄_n^m
    w = [_factorial_ratio(real(eltype(Z)), n, μ) for n in degrees]
    return Z .* transpose(w) ./ w
end
