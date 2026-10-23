# =============================================================================
#  alv_scheme_algebra.jl — the closed-form ALV schemes, once for every
#  representation of a Volterra operator.
#
#  Voigt, Reuss, dilute, dilute dual, Mori-Tanaka and Maxwell, and the dilute
#  concentration and contribution, are the same few Volterra sums, products and
#  inverses whatever the operator is written as: a dense block matrix (6n×6n at
#  order 4, 3n×3n at order 2), or the n×n parameter matrices of an isotropic
#  (α, β), transversely isotropic (ℓ₁…ℓ₆) or orthotropic (o₁…o₁₂) operator.
#  Each representation is an algebra type that supplies its identity, product
#  and inverse; the schemes are written once against it, and the exported
#  `*_alv`, `*_alv_order2`, `*_alv_iso`, `*_alv_ti` and `*_alv_ortho` functions
#  check their arguments and call them.
# =============================================================================

abstract type _ALVAlgebra end
"Dense block matrices, `B×B` blocks per pair of times: 6 at order 4, 3 at order 2."
struct _DenseALV{B} <: _ALVAlgebra end
"Isotropic parameter pairs ``(\\alpha, \\beta)``."
struct _IsoALV <: _ALVAlgebra end
"Transversely isotropic Walpole parameters ``(\\ell_1, \\dots, \\ell_6)``."
struct _TIALV <: _ALVAlgebra end
"Orthotropic parameters ``(o_1, \\dots, o_{12})``."
struct _OrthoALV <: _ALVAlgebra end

# The number of times of the grid, the element type of one operator, and the
# map of a function over its components (a dense matrix is its own component).
_alg_ntimes(::_DenseALV{B}, C) where {B} = size(C, 1) ÷ B
_alg_ntimes(::_ALVAlgebra, C) = size(first(C), 1)
_alg_eltype(::_DenseALV, C) = eltype(C)
_alg_eltype(::_ALVAlgebra, C) = mapreduce(eltype, promote_type, C)
_alg_map(::_DenseALV, f, X...) = f(X...)
_alg_map(::_ALVAlgebra, f, X...) = map(f, X...)

_alg_one(::_DenseALV{B}, n, T) where {B} = Matrix{T}(LinearAlgebra.I, B * n, B * n)
_alg_one(::_IsoALV, n, T) = _iso_identity(n, T)
_alg_one(::_TIALV, n, T) = _ti_identity(n, T)
_alg_one(::_OrthoALV, n, T) = _ortho_identity(n, T)

_alg_prod(::_DenseALV, X, Y) = X * Y
_alg_prod(::_IsoALV, X, Y) = _iso_prod(X, Y)
_alg_prod(::_TIALV, X, Y) = _ti_prod(X, Y)
_alg_prod(::_OrthoALV, X, Y) = _ortho_prod(X, Y)

_alg_inv(::_DenseALV{B}, X) where {B} = volterra_inverse(X; block_size = B)
_alg_inv(::_IsoALV, X) = _iso_inv(X)
_alg_inv(::_TIALV, X) = _ti_inv(X)
_alg_inv(::_OrthoALV, X) = _ortho_inv(X)

_alg_add(a, X, Y) = _alg_map(a, +, X, Y)
_alg_sub(a, X, Y) = _alg_map(a, -, X, Y)
_alg_scale(a, c, X) = _alg_map(a, x -> c .* x, X)
_alg_convert(a, ::Type{T}, X) where {T} = _alg_map(a, x -> Matrix{T}(x), X)

# The element type of a scheme: every operator of the lists, every extra one,
# the fractions, and the extra scalars, so that a dual number anywhere carries.
function _alg_promote(a, lists, fractions, operators = (), scalars = ())
    T = mapreduce(typeof, promote_type, fractions; init = Union{})
    for list in lists, X in list
        T = promote_type(T, _alg_eltype(a, X))
    end
    for X in operators
        T = promote_type(T, _alg_eltype(a, X))
    end
    for s in scalars
        T = promote_type(T, typeof(s))
    end
    return T
end

# `X₀ + Σᵣ fᵣ Xᵣ`, accumulated in the order of the phases.
function _alg_accumulate(a, X₀, Xs, fractions)
    return foldl(zip(Xs, fractions); init = X₀) do acc, (X, f)
        _alg_map(a, (s, x) -> s .+ f .* x, acc, X)
    end
end

function _alg_voigt(a, Cs, fractions)
    n = _alg_ntimes(a, first(Cs))
    T = _alg_promote(a, (Cs,), fractions)
    return _alg_accumulate(a, _alg_map(a, zero, _alg_one(a, n, T)), Cs, fractions)
end

_alg_reuss(a, Cs, fractions) = _alg_inv(a, _alg_voigt(a, [_alg_inv(a, C) for C in Cs], fractions))

function _alg_dilute(a, C₀, Ns, fractions)
    T = _alg_promote(a, (Ns,), fractions, (C₀,))
    return _alg_accumulate(a, _alg_convert(a, T, C₀), Ns, fractions)
end

_alg_dilute_dual(a, C₀, Hs, fractions) = _alg_inv(a, _alg_dilute(a, _alg_inv(a, C₀), Hs, fractions))

# 𝔸 = (𝕀 + ℙ ∘ (ℂᴱ - ℂ₀))⁻¹ and ℕ = (ℂᴱ - ℂ₀) ∘ 𝔸.
function _alg_dilute_concentration(a, Cᴱ, C₀, P)
    T = _alg_promote(a, (), (), (Cᴱ, C₀, P))
    Id = _alg_one(a, _alg_ntimes(a, C₀), T)
    return _alg_inv(a, _alg_add(a, Id, _alg_prod(a, P, _alg_sub(a, Cᴱ, C₀))))
end

_alg_dilute_contribution(a, Cᴱ, C₀, P) =
    _alg_prod(a, _alg_sub(a, Cᴱ, C₀), _alg_dilute_concentration(a, Cᴱ, C₀, P))

# ℂ₀ + (Σᵣ fᵣ ℕᵣ) ∘ (f₀ 𝕀 + Σₛ fₛ 𝔸ₛ)⁻¹: the inverse on the RIGHT, the normal
# block of a structured operator and the time axis not commuting.
function _alg_mori_tanaka(a, C₀, As, Ns, fractions, f₀)
    n = _alg_ntimes(a, C₀)
    T = _alg_promote(a, (As, Ns), fractions, (C₀,), (f₀,))
    Id = _alg_one(a, n, T)
    num = _alg_accumulate(a, _alg_map(a, zero, Id), Ns, fractions)
    den = _alg_accumulate(a, _alg_scale(a, T(f₀), Id), As, fractions)
    return _alg_add(a, _alg_convert(a, T, C₀), _alg_prod(a, num, _alg_inv(a, den)))
end

# ℂ₀ + ℕ ∘ (𝕀 - ℙ_Ω ∘ ℕ)⁻¹, ℕ = Σᵣ fᵣ ℕᵣ.
function _alg_maxwell(a, C₀, Ns, fractions, H₀)
    n = _alg_ntimes(a, C₀)
    T = _alg_promote(a, (Ns,), fractions, (C₀, H₀))
    Id = _alg_one(a, n, T)
    Σ = _alg_accumulate(a, _alg_map(a, zero, Id), Ns, fractions)
    factor = _alg_prod(a, Σ, _alg_inv(a, _alg_sub(a, Id, _alg_prod(a, H₀, Σ))))
    return _alg_add(a, _alg_convert(a, T, C₀), factor)
end

# The phase lists of a scheme come in step, one entry per phase.
_alg_check(name, lists...) = allequal(map(length, lists)) ||
    throw(ArgumentError("$name: phase counts mismatch ($(join(map(length, lists), ", ")))"))

# Dense operators are square block matrices of one size, a multiple of the block.
function _alg_check_dense(name, ::_DenseALV{B}, Xs...) where {B}
    sz = size(first(Xs))
    all(X -> size(X) == sz, Xs) && sz[1] == sz[2] && sz[1] % B == 0 || throw(
        DimensionMismatch("$name: expected square $(B)n×$(B)n block matrices of one size, got $(map(size, Xs))")
    )
    return nothing
end
