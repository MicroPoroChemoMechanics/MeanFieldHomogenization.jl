# =============================================================================
#  vector_harmonics.jl — vector solid harmonics and the solutions of the Lamé
#  equation built on them (kushch2013, chapter 4).
#
#  In the complex Cartesian basis e₁ = (i₁ + i i₂)/2, e₂ = conj(e₁), e₃ = i₃,
#  the three families of harmonic vectors are (kushch2013, Eqs. (4.14), (4.15))
#
#    Y⁽¹⁾_ts = e₁ Y_{t+1}^{s−1} − e₂ Y_{t+1}^{s+1} − e₃ Y_{t+1}^s            ( = ∇Y_t^s )
#    Y⁽²⁾_ts = (i/t) [ e₁ (t+s) Y_t^{s−1} + e₂ (t−s) Y_t^{s+1} − e₃ s Y_t^s ]
#    Y⁽³⁾_ts = [ e₁ (t+s−1)(t+s) Y_{t−1}^{s−1} − e₂ (t−s−1)(t−s) Y_{t−1}^{s+1}
#                + e₃ (t−s)(t+s) Y_{t−1}^s ] / [t(2t−1)]
#
#    y⁽¹⁾_ts = e₁ y_{t−1}^{s−1} − e₂ y_{t−1}^{s+1} + e₃ y_{t−1}^s            ( = ∇y_t^s )
#    y⁽²⁾_ts = (i/(t+1)) [ −e₁ (t−s+1) y_t^{s−1} − e₂ (t+s+1) y_t^{s+1} + e₃ s y_t^s ]
#    y⁽³⁾_ts = [ e₁ (t−s+1)(t−s+2) y_{t+1}^{s−1} − e₂ (t+s+1)(t+s+2) y_{t+1}^{s+1}
#                − e₃ (t−s+1)(t+s+1) y_{t+1}^s ] / [(t+1)(2t+3)]
#
#  The first is a gradient, the second a rotation field (Y⁽²⁾_ts = ∇Y_t^s × x / t),
#  both divergence-free; the third has the divergence −Y_t^s (resp. −y_t^s).
#
#  The solutions of the Lamé equation of an isotropic solid of Poisson ratio ν
#  are the first two families as they are, and a third one that is the
#  Papkovich-Neuber field of the third harmonic vector:
#
#    U⁽¹⁾ = Y⁽¹⁾ ,  U⁽²⁾ = Y⁽²⁾ ,  U⁽³⁾ = 4(1−ν) Y⁽³⁾ − ∇(x·Y⁽³⁾) ,
#
#  with x·Y⁽³⁾_ts = r² Y_t^s/(2t−1) and x·y⁽³⁾_ts = −r² y_t^s/(2t+3). This is the
#  third solution of kushch2013, Eq. (4.27), written without a biharmonic
#  function; the Poisson ratio enters it through the factor 4(1−ν) alone.
# =============================================================================

# e₁ a + e₂ b + e₃ c in Cartesian components.
@inline _complex_cartesian(a, b, c) = SVector((a + b) / 2, im * (a - b) / 2, c)

# A harmonic of a table, zero outside the degrees it holds.
@inline function _entry(h::HarmonicTable{C}, t::Integer, s::Integer) where {C}
    return 0 ≤ t ≤ h.tmax ? h[t, s] : zero(C)
end

"""
    vector_irregular_harmonic(i, t, s, Y::HarmonicTable) -> SVector{3}

Cartesian components of the decaying vector solid harmonic
``\\underline{Y}^{(i)}_{ts}`` (``i = 1, 2, 3``, ``t \\ge 1`` for ``i > 1``) from the
table `Y` of the decaying scalar harmonics at the point, which must reach the
degree ``t + 1``.
"""
function vector_irregular_harmonic(i::Integer, t::Integer, s::Integer, Y::HarmonicTable)
    i == 1 && return _complex_cartesian(_entry(Y, t + 1, s - 1), -_entry(Y, t + 1, s + 1), -_entry(Y, t + 1, s))
    i == 2 && return (im / t) * _complex_cartesian(
        (t + s) * _entry(Y, t, s - 1), (t - s) * _entry(Y, t, s + 1), -s * _entry(Y, t, s)
    )
    i == 3 || throw(ArgumentError("vector solid harmonics: the family is 1, 2 or 3, got $i"))
    return _complex_cartesian(
        (t + s - 1) * (t + s) * _entry(Y, t - 1, s - 1), -(t - s - 1) * (t - s) * _entry(Y, t - 1, s + 1),
        (t - s) * (t + s) * _entry(Y, t - 1, s)
    ) / (t * (2t - 1))
end

"""
    vector_regular_harmonic(i, t, s, y::HarmonicTable) -> SVector{3}

Cartesian components of the regular vector solid harmonic
``\\underline{y}^{(i)}_{ts}`` (``i = 1, 2, 3``) from the table `y` of the regular
scalar harmonics at the point, which must reach the degree ``t + 1``.
"""
function vector_regular_harmonic(i::Integer, t::Integer, s::Integer, y::HarmonicTable)
    i == 1 && return _complex_cartesian(_entry(y, t - 1, s - 1), -_entry(y, t - 1, s + 1), _entry(y, t - 1, s))
    i == 2 && return (im / (t + 1)) * _complex_cartesian(
        -(t - s + 1) * _entry(y, t, s - 1), -(t + s + 1) * _entry(y, t, s + 1), s * _entry(y, t, s)
    )
    i == 3 || throw(ArgumentError("vector solid harmonics: the family is 1, 2 or 3, got $i"))
    return _complex_cartesian(
        (t - s + 1) * (t - s + 2) * _entry(y, t + 1, s - 1), -(t + s + 1) * (t + s + 2) * _entry(y, t + 1, s + 1),
        -(t - s + 1) * (t + s + 1) * _entry(y, t + 1, s)
    ) / ((t + 1) * (2t + 3))
end

"""
    lame_irregular(i, t, s, x, ν) -> SVector{3}

Decaying solution ``\\underline{U}^{(i)}_{ts}`` of the Lamé equation at the
point `x`, for the Poisson ratio `ν`: the vector harmonics
``\\underline{Y}^{(1)}_{ts}`` and ``\\underline{Y}^{(2)}_{ts}``, and
``\\underline{U}^{(3)}_{ts} = 4(1-\\nu)\\,\\underline{Y}^{(3)}_{ts}
- \\underline{\\nabla}(\\underline{x}\\cdot\\underline{Y}^{(3)}_{ts})``
[kushch2013](@cite).
"""
function lame_irregular(i::Integer, t::Integer, s::Integer, x::AbstractVector, ν)
    Y = irregular_harmonics(x, t + 1)
    i < 3 && return vector_irregular_harmonic(i, t, s, Y)
    r² = sum(abs2, x)
    # ∇(x·Y⁽³⁾) = ∇(r² Y_t^s)/(2t−1) = (2 x Y_t^s + r² Y⁽¹⁾_ts)/(2t−1)
    grad = (2 * Y[t, s] * SVector{3}(x) + r² * vector_irregular_harmonic(1, t, s, Y)) / (2t - 1)
    return 4 * (1 - ν) * vector_irregular_harmonic(3, t, s, Y) - grad
end

"""
    lame_regular(i, t, s, x, ν) -> SVector{3}

Regular solution ``\\underline{u}^{(i)}_{ts}`` of the Lamé equation at the point
`x`, for the Poisson ratio `ν`: the counterpart of [`lame_irregular`](@ref) on
the regular vector harmonics. The solutions of the lowest degrees are the
dilatation ``\\underline{u}^{(3)}_{00}``, the translations
``\\underline{u}^{(1)}_{1s}``, the rotations ``\\underline{u}^{(2)}_{1s}`` and
the uniform deviatoric strains ``\\underline{u}^{(1)}_{2s}``.
"""
function lame_regular(i::Integer, t::Integer, s::Integer, x::AbstractVector, ν)
    y = regular_harmonics(x, t + 1)
    i < 3 && return vector_regular_harmonic(i, t, s, y)
    r² = sum(abs2, x)
    # ∇(x·y⁽³⁾) = −∇(r² y_t^s)/(2t+3) = −(2 x y_t^s + r² y⁽¹⁾_ts)/(2t+3)
    grad = -(2 * y[t, s] * SVector{3}(x) + r² * vector_regular_harmonic(1, t, s, y)) / (2t + 3)
    return 4 * (1 - ν) * vector_regular_harmonic(3, t, s, y) - grad
end

# ─── Translation ─────────────────────────────────────────────────────────────
#
#  About a center at R from the first one, a decaying vector harmonic expands
#  on the regular ones (kushch2013, Eq. (4.85)):
#
#    Y⁽ⁱ⁾_ts(r + R) = Σ_{j ≤ i} Σ_{k,l} (−1)^{k+l} α⁽ⁱ⁾⁽ʲ⁾_tksl(R) y⁽ʲ⁾_kl(r) ,
#
#    α⁽¹⁾⁽¹⁾ = α⁽²⁾⁽²⁾ = α⁽³⁾⁽³⁾ = Y_{t+k}^{s−l} ,
#    α⁽²⁾⁽¹⁾ = i (l/k + s/t) Y_{t+k−1}^{s−l} ,          α⁽³⁾⁽²⁾ = −α⁽²⁾⁽¹⁾ ,
#    α⁽³⁾⁽¹⁾ = [1 + ls/(kt) − (t²−s²)/(t(2t−1)) − (k²−l²)/(k(2k−1))] Y_{t+k−2}^{s−l} .
#
#  The translation of the Lamé solutions follows from the Papkovich-Neuber form
#  of the third one. With x = r + R the position from the first center,
#
#    U⁽³⁾ = [4(1−ν) B − ∇(r·B)] − ∇(R·B) ,     B = Y⁽³⁾_ts(r + R) ,
#
#  and, B being expanded as above, the bracket is term by term a regular Lamé
#  solution: u⁽³⁾_kl for y⁽³⁾_kl, 4(1−ν) u⁽²⁾_kl for y⁽²⁾_kl (which is tangent to
#  the spheres, r·y⁽²⁾ = 0) and (4(1−ν) − k) u⁽¹⁾_kl for y⁽¹⁾_kl = ∇y_k^l
#  (r·∇y_k^l = k y_k^l). The last term is the gradient of the harmonic function
#  R·B, whose components are scalar harmonics of degree t − 1 that translate as
#  such:
#
#    R·Y⁽³⁾_ts(r + R) = Σ_{k,l} (−1)^{k+l} H_tksl(R) y_k^l(r) ,
#    H_tksl = [ (R·e₁)(t+s−1)(t+s) Y_{t−1+k}^{s−1−l} − (R·e₂)(t−s−1)(t−s) Y_{t−1+k}^{s+1−l}
#               + R₃ (t−s)(t+s) Y_{t−1+k}^{s−l} ] / [t(2t−1)] .
#
#  Hence U⁽ⁱ⁾_ts(r + R) = Σ_{j ≤ i} Σ_{k,l} (−1)^{k+l} η⁽ⁱ⁾⁽ʲ⁾_tksl u⁽ʲ⁾_kl(r), with
#  η = α for i < 3 and
#
#    η⁽³⁾⁽³⁾ = α⁽³⁾⁽³⁾ ,   η⁽³⁾⁽²⁾ = 4(1−ν) α⁽³⁾⁽²⁾ ,   η⁽³⁾⁽¹⁾ = (4(1−ν) − k) α⁽³⁾⁽¹⁾ − H .
#
#  Each coefficient is affine in 4(1−ν): `lame_translation` returns the two
#  parts, (a, b) with η = a + 4(1−ν) b. At k = 0 only j = 3 exists (u⁽¹⁾_00 and
#  u⁽²⁾_00 vanish).

"""
    lame_translation(i, j, t, k, s, l, Y::HarmonicTable, R) -> (a, b)

Coefficient ``\\eta^{(i)(j)}_{tksl} = a + 4(1-\\nu)\\,b`` of the regular Lamé
solution ``\\underline{u}^{(j)}_{kl}`` in the expansion of the decaying one
``\\underline{U}^{(i)}_{ts}`` about a center at `R` from its own,

```math
\\underline{U}^{(i)}_{ts}(\\underline{r} + \\underline{R}) = \\sum_{j \\le i}\\sum_{k,l}
  (-1)^{k+l}\\,\\eta^{(i)(j)}_{tksl}(\\underline{R})\\;\\underline{u}^{(j)}_{kl}(\\underline{r}),
\\qquad \\|\\underline{r}\\| < \\|\\underline{R}\\| ,
```

from the table `Y` of the decaying scalar harmonics at `R`, which must reach
the degree ``t + k``. The two parts are separated because the Poisson ratio of
the matrix is all that distinguishes one isotropic matrix from another here.
"""
function lame_translation(
        i::Integer, j::Integer, t::Integer, k::Integer, s::Integer, l::Integer,
        Y::HarmonicTable{C}, R::AbstractVector
    ) where {C}
    z = zero(C)
    (j > i || (k == 0 && j < 3)) && return (z, z)
    i == j && return (_entry(Y, t + k, s - l), z)
    if i - j == 1
        α = im * (l / k + s / t) * _entry(Y, t + k - 1, s - l)      # α⁽²⁾⁽¹⁾ = −α⁽³⁾⁽²⁾
        return i == 2 ? (α, z) : (z, -α)
    end
    α = (1 + l * s / (k * t) - (t^2 - s^2) / (t * (2t - 1)) - (k^2 - l^2) / (k * (2k - 1))) *
        _entry(Y, t + k - 2, s - l)
    H = (
        (R[1] + im * R[2]) / 2 * (t + s - 1) * (t + s) * _entry(Y, t - 1 + k, s - 1 - l) -
            (R[1] - im * R[2]) / 2 * (t - s - 1) * (t - s) * _entry(Y, t - 1 + k, s + 1 - l) +
            R[3] * (t - s) * (t + s) * _entry(Y, t - 1 + k, s - l)
    ) / (t * (2t - 1))
    return (-k * α - H, α)
end
