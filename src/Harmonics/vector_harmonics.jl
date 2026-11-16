# =============================================================================
#  vector_harmonics.jl — vector solid harmonics and the solutions of the Lamé
#  equation built on them (kushch2013, chapter 4).
#
#  In the complex Cartesian basis e₁ = (i₁ + i i₂)/2, e₂ = conj(e₁), e₃ = i₃,
#  the three families of harmonic vectors are (kushch2013, Eqs. (4.14), (4.15))
#
#    Y⁽¹⁾_nm = e₁ Y_{n+1}^{m−1} − e₂ Y_{n+1}^{m+1} − e₃ Y_{n+1}^m            ( = grad Y_n^m )
#    Y⁽²⁾_nm = (i/n) [ e₁ (n+m) Y_n^{m−1} + e₂ (n−m) Y_n^{m+1} − e₃ m Y_n^m ]
#    Y⁽³⁾_nm = [ e₁ (n+m−1)(n+m) Y_{n−1}^{m−1} − e₂ (n−m−1)(n−m) Y_{n−1}^{m+1}
#                + e₃ (n−m)(n+m) Y_{n−1}^m ] / [n(2n−1)]
#
#    y⁽¹⁾_nm = e₁ y_{n−1}^{m−1} − e₂ y_{n−1}^{m+1} + e₃ y_{n−1}^m            ( = grad y_n^m )
#    y⁽²⁾_nm = (i/(n+1)) [ −e₁ (n−m+1) y_n^{m−1} − e₂ (n+m+1) y_n^{m+1} + e₃ m y_n^m ]
#    y⁽³⁾_nm = [ e₁ (n−m+1)(n−m+2) y_{n+1}^{m−1} − e₂ (n+m+1)(n+m+2) y_{n+1}^{m+1}
#                − e₃ (n−m+1)(n+m+1) y_{n+1}^m ] / [(n+1)(2n+3)]
#
#  The first is a gradient, the second a rotation field (Y⁽²⁾_nm = grad Y_n^m × x / n),
#  both divergence-free; the third has the divergence −Y_n^m (resp. −y_n^m).
#
#  The solutions of the Lamé equation of an isotropic solid of Poisson ratio ν
#  are the first two families as they are, and a third one that is the
#  Papkovich-Neuber field of the third harmonic vector:
#
#    U⁽¹⁾ = Y⁽¹⁾ ,  U⁽²⁾ = Y⁽²⁾ ,  U⁽³⁾ = 4(1−ν) Y⁽³⁾ − grad(x·Y⁽³⁾) ,
#
#  with x·Y⁽³⁾_nm = r² Y_n^m/(2n−1) and x·y⁽³⁾_nm = −r² y_n^m/(2n+3). This is the
#  third solution of kushch2013, Eq. (4.27), written without a biharmonic
#  function; the Poisson ratio enters it through the factor 4(1−ν) alone.
# =============================================================================

# e₁ a + e₂ b + e₃ c in Cartesian components.
@inline _complex_cartesian(a, b, c) = SVector((a + b) / 2, im * (a - b) / 2, c)

# A harmonic of a table, zero outside the degrees it holds.
@inline function _entry(h::HarmonicTable{C}, n::Integer, m::Integer) where {C}
    return 0 ≤ n ≤ h.nmax ? h[n, m] : zero(C)
end

"""
    vector_irregular_harmonic(i, n, m, Y::HarmonicTable) -> SVector{3}

Cartesian components of the decaying vector solid harmonic
``\\underline{Y}^{(i)}_{nm}`` (``i = 1, 2, 3``, ``n \\ge 1`` for ``i > 1``) from the
table `Y` of the decaying scalar harmonics at the point, which must reach the
degree ``n + 1``.
"""
function vector_irregular_harmonic(i::Integer, n::Integer, m::Integer, Y::HarmonicTable)
    i == 1 && return _complex_cartesian(_entry(Y, n + 1, m - 1), -_entry(Y, n + 1, m + 1), -_entry(Y, n + 1, m))
    i == 2 && return (im / n) * _complex_cartesian(
        (n + m) * _entry(Y, n, m - 1), (n - m) * _entry(Y, n, m + 1), -m * _entry(Y, n, m)
    )
    i == 3 || throw(ArgumentError("vector solid harmonics: the family is 1, 2 or 3, got $i"))
    return _complex_cartesian(
        (n + m - 1) * (n + m) * _entry(Y, n - 1, m - 1), -(n - m - 1) * (n - m) * _entry(Y, n - 1, m + 1),
        (n - m) * (n + m) * _entry(Y, n - 1, m)
    ) / (n * (2n - 1))
end

"""
    vector_regular_harmonic(i, n, m, y::HarmonicTable) -> SVector{3}

Cartesian components of the regular vector solid harmonic
``\\underline{y}^{(i)}_{nm}`` (``i = 1, 2, 3``) from the table `y` of the regular
scalar harmonics at the point, which must reach the degree ``n + 1``.
"""
function vector_regular_harmonic(i::Integer, n::Integer, m::Integer, y::HarmonicTable)
    i == 1 && return _complex_cartesian(_entry(y, n - 1, m - 1), -_entry(y, n - 1, m + 1), _entry(y, n - 1, m))
    i == 2 && return (im / (n + 1)) * _complex_cartesian(
        -(n - m + 1) * _entry(y, n, m - 1), -(n + m + 1) * _entry(y, n, m + 1), m * _entry(y, n, m)
    )
    i == 3 || throw(ArgumentError("vector solid harmonics: the family is 1, 2 or 3, got $i"))
    return _complex_cartesian(
        (n - m + 1) * (n - m + 2) * _entry(y, n + 1, m - 1), -(n + m + 1) * (n + m + 2) * _entry(y, n + 1, m + 1),
        -(n - m + 1) * (n + m + 1) * _entry(y, n + 1, m)
    ) / ((n + 1) * (2n + 3))
end

"""
    lame_irregular(i, n, m, x, ν) -> SVector{3}

Decaying solution ``\\underline{U}^{(i)}_{nm}`` of the Lamé equation at the
point `x`, for the Poisson ratio `ν`: the vector harmonics
``\\underline{Y}^{(1)}_{nm}`` and ``\\underline{Y}^{(2)}_{nm}``, and
``\\underline{U}^{(3)}_{nm} = 4(1-\\nu)\\,\\underline{Y}^{(3)}_{nm}
- \\underline{\\operatorname{grad}}\\,(\\underline{x}\\cdot\\underline{Y}^{(3)}_{nm})``
[kushch2013](@cite).
"""
function lame_irregular(i::Integer, n::Integer, m::Integer, x::AbstractVector, ν)
    Y = irregular_harmonics(x, n + 1)
    i < 3 && return vector_irregular_harmonic(i, n, m, Y)
    r² = sum(abs2, x)
    # grad(x·Y⁽³⁾) = grad(r² Y_n^m)/(2n−1) = (2 x Y_n^m + r² Y⁽¹⁾_nm)/(2n−1)
    grad = (2 * Y[n, m] * SVector{3}(x) + r² * vector_irregular_harmonic(1, n, m, Y)) / (2n - 1)
    return 4 * (1 - ν) * vector_irregular_harmonic(3, n, m, Y) - grad
end

"""
    lame_regular(i, n, m, x, ν) -> SVector{3}

Regular solution ``\\underline{u}^{(i)}_{nm}`` of the Lamé equation at the point
`x`, for the Poisson ratio `ν`: the counterpart of [`lame_irregular`](@ref) on
the regular vector harmonics. The solutions of the lowest degrees are the
dilatation ``\\underline{u}^{(3)}_{00}``, the translations
``\\underline{u}^{(1)}_{1m}``, the rotations ``\\underline{u}^{(2)}_{1m}`` and
the uniform deviatoric strains ``\\underline{u}^{(1)}_{2m}``.
"""
function lame_regular(i::Integer, n::Integer, m::Integer, x::AbstractVector, ν)
    y = regular_harmonics(x, n + 1)
    i < 3 && return vector_regular_harmonic(i, n, m, y)
    r² = sum(abs2, x)
    # grad(x·y⁽³⁾) = −grad(r² y_n^m)/(2n+3) = −(2 x y_n^m + r² y⁽¹⁾_nm)/(2n+3)
    grad = -(2 * y[n, m] * SVector{3}(x) + r² * vector_regular_harmonic(1, n, m, y)) / (2n + 3)
    return 4 * (1 - ν) * vector_regular_harmonic(3, n, m, y) - grad
end

# ─── Translation ─────────────────────────────────────────────────────────────
#
#  About a center at R from the first one, a decaying vector harmonic expands
#  on the regular ones (kushch2013, Eq. (4.85)):
#
#    Y⁽ⁱ⁾_n′m′(r + R) = Σ_{j ≤ i} Σ_{n,m} (−1)^{n+m} α⁽ⁱ⁾⁽ʲ⁾_n′nm′m(R) y⁽ʲ⁾_nm(r) ,
#
#    α⁽¹⁾⁽¹⁾ = α⁽²⁾⁽²⁾ = α⁽³⁾⁽³⁾ = Y_{n′+n}^{m′−m} ,
#    α⁽²⁾⁽¹⁾ = i (m/n + m′/n′) Y_{n′+n−1}^{m′−m} ,          α⁽³⁾⁽²⁾ = −α⁽²⁾⁽¹⁾ ,
#    α⁽³⁾⁽¹⁾ = [1 + m m′/(n n′) − (n′²−m′²)/(n′(2n′−1)) − (n²−m²)/(n(2n−1))] Y_{n′+n−2}^{m′−m} .
#
#  The translation of the Lamé solutions follows from the Papkovich-Neuber form
#  of the third one. With x = r + R the position from the first center,
#
#    U⁽³⁾ = [4(1−ν) B − grad(r·B)] − grad(R·B) ,     B = Y⁽³⁾_n′m′(r + R) ,
#
#  and, B being expanded as above, the bracket is term by term a regular Lamé
#  solution: u⁽³⁾_nm for y⁽³⁾_nm, 4(1−ν) u⁽²⁾_nm for y⁽²⁾_nm (which is tangent to
#  the spheres, r·y⁽²⁾ = 0) and (4(1−ν) − n) u⁽¹⁾_nm for y⁽¹⁾_nm = grad y_n^m
#  (r·grad y_n^m = n y_n^m). The last term is the gradient of the harmonic function
#  R·B, whose components are scalar harmonics of degree n′ − 1 that translate as
#  such:
#
#    R·Y⁽³⁾_n′m′(r + R) = Σ_{n,m} (−1)^{n+m} H_n′nm′m(R) y_n^m(r) ,
#    H_n′nm′m = [ (R·e₁)(n′+m′−1)(n′+m′) Y_{n′−1+n}^{m′−1−m} − (R·e₂)(n′−m′−1)(n′−m′) Y_{n′−1+n}^{m′+1−m}
#               + R₃ (n′−m′)(n′+m′) Y_{n′−1+n}^{m′−m} ] / [n′(2n′−1)] .
#
#  Hence U⁽ⁱ⁾_n′m′(r + R) = Σ_{j ≤ i} Σ_{n,m} (−1)^{n+m} η⁽ⁱ⁾⁽ʲ⁾_n′nm′m u⁽ʲ⁾_nm(r), with
#  η = α for i < 3 and
#
#    η⁽³⁾⁽³⁾ = α⁽³⁾⁽³⁾ ,   η⁽³⁾⁽²⁾ = 4(1−ν) α⁽³⁾⁽²⁾ ,   η⁽³⁾⁽¹⁾ = (4(1−ν) − n) α⁽³⁾⁽¹⁾ − H .
#
#  Each coefficient is affine in 4(1−ν): `lame_translation` returns the two
#  parts, (a, b) with η = a + 4(1−ν) b. At n = 0 only j = 3 exists (u⁽¹⁾_00 and
#  u⁽²⁾_00 vanish).

"""
    lame_translation(i, j, n′, n, m′, m, Y::HarmonicTable, R) -> (a, b)

Coefficient ``\\eta^{(i)(j)}_{n'n\\,m'm} = a + 4(1-\\nu)\\,b`` of the regular Lamé
solution ``\\underline{u}^{(j)}_{nm}`` in the expansion of the decaying one
``\\underline{U}^{(i)}_{n'm'}`` about a center at `R` from its own,

```math
\\underline{U}^{(i)}_{n'm'}(\\underline{r} + \\underline{R}) = \\sum_{j \\le i}\\sum_{n,m}
  (-1)^{n+m}\\,\\eta^{(i)(j)}_{n'n\\,m'm}(\\underline{R})\\;\\underline{u}^{(j)}_{nm}(\\underline{r}),
\\qquad \\|\\underline{r}\\| < \\|\\underline{R}\\| ,
```

from the table `Y` of the decaying scalar harmonics at `R`, which must reach
the degree ``n' + n``. The two parts are separated because the Poisson ratio of
the matrix is all that distinguishes one isotropic matrix from another here.
"""
function lame_translation(
        i::Integer, j::Integer, n′::Integer, n::Integer, m′::Integer, m::Integer,
        Y::HarmonicTable{C}, R::AbstractVector
    ) where {C}
    z = zero(C)
    (j > i || (n == 0 && j < 3)) && return (z, z)
    i == j && return (_entry(Y, n′ + n, m′ - m), z)
    if i - j == 1
        α = im * (m / n + m′ / n′) * _entry(Y, n′ + n - 1, m′ - m)      # α⁽²⁾⁽¹⁾ = −α⁽³⁾⁽²⁾
        return i == 2 ? (α, z) : (z, -α)
    end
    α = (1 + m * m′ / (n * n′) - (n′^2 - m′^2) / (n′ * (2n′ - 1)) - (n^2 - m^2) / (n * (2n - 1))) *
        _entry(Y, n′ + n - 2, m′ - m)
    H = (
        (R[1] + im * R[2]) / 2 * (n′ + m′ - 1) * (n′ + m′) * _entry(Y, n′ - 1 + n, m′ - 1 - m) -
            (R[1] - im * R[2]) / 2 * (n′ - m′ - 1) * (n′ - m′) * _entry(Y, n′ - 1 + n, m′ + 1 - m) +
            R[3] * (n′ - m′) * (n′ + m′) * _entry(Y, n′ - 1 + n, m′ - m)
    ) / (n′ * (2n′ - 1))
    return (-n * α - H, α)
end
