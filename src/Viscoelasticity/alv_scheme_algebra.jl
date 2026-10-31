# =============================================================================
#  alv_scheme_algebra.jl — the closed-form ALV schemes, once for every
#  representation of a Volterra operator.
#
#  Voigt, Reuss, dilute, dilute dual, Mori-Tanaka and Maxwell, and the dilute
#  concentration and contribution, are the same few Volterra sums, products and
#  inverses whatever the operator is written as: a dense block matrix (6n×6n at
#  order 4, 3n×3n at order 2), or a `VolterraOperator` stored by the blocks of
#  its symmetry class. Each representation is an algebra type that supplies its
#  identity, product and inverse; the schemes are written once against it, and
#  the exported `*_alv` and `*_alv_order2` functions check their arguments and
#  call them.
# =============================================================================

abstract type _ALVAlgebra end
"Dense block matrices, `B×B` blocks per pair of times: 6 at order 4, 3 at order 2."
struct _DenseALV{B} <: _ALVAlgebra end

# The number of times of the grid, the element type of one operator, and the
# map of a function over it.
_alg_ntimes(::_DenseALV{B}, C) where {B} = size(C, 1) ÷ B
_alg_eltype(::_DenseALV, C) = eltype(C)
_alg_map(::_DenseALV, f, X...) = f(X...)

_alg_one(::_DenseALV{B}, n, T) where {B} = Matrix{T}(LinearAlgebra.I, B * n, B * n)

_alg_prod(::_DenseALV, X, Y) = X * Y

_alg_inv(::_DenseALV{B}, X) where {B} = volterra_inverse(X; block_size = B)

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

# Dense blocks: the identity goes onto the diagonal of the product rather than
# into a 6n×6n matrix of its own, built at every call of a fixed-point loop.
function _alg_dilute_concentration(::_DenseALV{B}, Cᴱ, C₀, P) where {B}
    M = P * (Cᴱ - C₀)
    view(M, LinearAlgebra.diagind(M)) .+= one(eltype(M))
    return volterra_inverse(M; block_size = B)
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

# ── Operators stored by the blocks of their class ──────────────────────────

"""
Volterra operators stored by the blocks of their symmetry class
([`VolterraOperator`](@ref)), on one grid: the scheme formulas run on them with
their own `*`, `inv` and `+`, each operation in the cheapest class that holds its
operands (an isotropic operator joins the class of the other).
"""
struct _OpALV{O, G <: TimeGrid} <: _ALVAlgebra
    grid::G
end
_OpALV{O}(grid::G) where {O, G <: TimeGrid} = _OpALV{O, G}(grid)
_alg_ntimes(a::_OpALV, C) = length(a.grid)
_alg_eltype(::_OpALV, C) = eltype(C)
# The identity of the order: 𝕀 = 𝕁 + 𝕂 at order 4, 𝟏 at order 2.
_alg_one(a::_OpALV{4}, n, ::Type{T}) where {T} = VolterraOperator(TensISO{3}(one(T), one(T)), a.grid)
_alg_one(a::_OpALV{2}, n, ::Type{T}) where {T} = VolterraOperator(TensISO{3}(one(T)), a.grid)
_alg_prod(::_OpALV, X, Y) = X * Y
_alg_inv(::_OpALV, X) = inv(X)
_alg_add(::_OpALV, X, Y) = X + Y
_alg_sub(::_OpALV, X, Y) = X - Y
_alg_scale(::_OpALV, c, X) = c * X
_alg_convert(::_OpALV, ::Type{T}, X) where {T} = _with_eltype(T, X)
_alg_map(::_OpALV, ::typeof(zero), X) = zero(X)
_alg_accumulate(::_OpALV, X₀, Xs, fractions) =
    foldl((acc, (X, f)) -> acc + f * X, zip(Xs, fractions); init = X₀)

"""
    _in_class_algebra(f, singles::Tuple, lists::Tuple) -> VolterraOperator

`f(algebra, singles..., lists...)` evaluated on [`VolterraOperator`](@ref)s:
`singles` are operators, `lists` vectors of them, one per phase. An operator
given as a plain ``6n\\times 6n`` matrix enters in the cheapest class it belongs
to — isotropic, the class of an operator among the others, transversely isotropic
about ``\\underline{e}_3``, orthotropic in the canonical frame, dense otherwise;
one given as a `VolterraOperator` enters as it is.
"""
function _in_class_algebra(f::F, singles::Tuple, lists::Tuple) where {F}
    ops = collect(Iterators.flatten((singles, lists...)))
    grid = _operator_grid(ops)
    candidates = _algebra_candidates(ops)
    as_op(M) = _as_class_operator(M, grid, candidates)
    return f(_OpALV{_operands_order(ops, grid)}(grid), map(as_op, singles)..., map(l -> [as_op(M) for M in l], lists)...)
end

"""
    _in_cheapest_algebra(f, singles::Tuple, lists::Tuple) -> Matrix

[`_in_class_algebra`](@ref), the result written out as a dense block matrix, for
the callers that compute on dense matrices.
"""
_in_cheapest_algebra(f::F, singles::Tuple, lists::Tuple) where {F} = Matrix(_in_class_algebra(f, singles, lists))

# The order of the operands: that of an operator among them, or, for plain matrices
# alone (order 4: the dense callers), 4.
function _operands_order(ops, grid)
    for M in ops
        M isa VolterraOperator && return _order(M.class)
    end
    return 4
end

# The grid of the operators, or, for plain matrices alone, a grid of as many
# instants (its values are not read: only the equality of grids is).
function _operator_grid(ops)
    for M in ops
        M isa VolterraOperator && return M.grid
    end
    return TimeGrid(collect(1.0:(size(first(ops), 1) ÷ 6)))
end

const _ISO4 = TensISO{3}(1.0, 1.0)
const _TI4_E3 = TensTI{4}(1.0, 1.0, 0.0, 1.0, 1.0, (0.0, 0.0, 1.0))
const _ORTHO4_CANON = TensOrtho{4, Float64}(ntuple(_ -> 1.0, 9), TensND.CanonicalBasis{3, Float64}())
const _ISO2 = TensISO{3}(1.0)
const _TI2_E3 = TensTI{2}(1.0, 1.0, (0.0, 0.0, 1.0))
const _ORTHO2_CANON = TensOrtho{2, Float64}((1.0, 1.0, 1.0), TensND.CanonicalBasis{3, Float64}())

function _algebra_candidates(ops)
    own = Any[c for c in map(_own_class, ops) if c !== nothing]
    return Any[_ISO4, _ISO2, own..., _TI4_E3, _TI2_E3, _ORTHO4_CANON, _ORTHO2_CANON]
end
# The class an operand brings, if it is stored in one.
_own_class(R::VolterraOperator{<:_StructuredClass}) = R.class
_own_class(M) = nothing

_as_class_operator(R::VolterraOperator, grid, candidates) = (_same_grid(R.grid, grid, "ALV scheme"); R)
# The isotropy test of the dense paths, on an operator stored by blocks.
_is_iso_block(R::VolterraOperator{<:TensISO{4, 3}}; kw...) = true
_is_iso_block(R::VolterraOperator; kw...) = _is_iso_block(Matrix(R); kw...)

# A full matrix: the isotropic test reads the entries in place, the commonest
# case (an iterate of a self-consistent scheme); the others go through `_compact`.
# The order of a full matrix is read on its size against the grid.
_as_class_operator(M::AbstractMatrix, grid, candidates) =
    size(M, 1) == 6 * length(grid) && _is_iso_block(M) ? VolterraOperator(_ISO4, iso_params_from_blocks(M), grid) :
    _compact(VolterraOperator(M, grid), candidates)

# ── Dense operators, computed in the cheapest algebra they share ───────────

# The product, inverse and right division of dense block matrices (6n×6n), each
# taken in the cheapest algebra its operands belong to: an isotropic product
# costs two n×n ones where the dense one costs a (6n)³ gemm.
_alv_prod(X, Y) = _in_cheapest_algebra(_alg_prod, (X, Y), ())
_alv_inv(X) = _in_cheapest_algebra(_alg_inv, (X,), ())
_alv_right_divide(X, Y) = _in_cheapest_algebra((a, X, Y) -> _alg_prod(a, X, _alg_inv(a, Y)), (X, Y), ())

# The dilute concentration (𝕀 + ℙ ∘ (ℂᴱ − ℂ₀))⁻¹ of a phase against a reference, as a
# dense block matrix.
_alv_dilute_concentration(Cᴱ, C₀, P) = _in_cheapest_algebra(_alg_dilute_concentration, (Cᴱ, C₀, P), ())
