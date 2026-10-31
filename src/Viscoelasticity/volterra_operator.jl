# =============================================================================
#  volterra_operator.jl — aging viscoelastic operators and histories on one
#  shared time grid, written as their elastic counterparts.
#
#  An operator of aging linear viscoelasticity (a relaxation R(t, t′), a creep
#  J(t, t′), an aging conductivity, …) has two faces: a function of two times,
#  and, on a grid t₁ < … < tₙ, its trapezoidal matrix (trapezoidal.jl), which
#  maps the values ε(tⱼ) of a history to the values σ(tᵢ) of the response.
#  A history has two faces too: a function of one time, and its values on the
#  grid.
#
#  When the values of the operator belong to one symmetry class (isotropic,
#  cubic, transversely isotropic, orthotropic, in one frame), each tensor
#  value is a few small blocks (TensND `symmetry_blocks`), and the operator
#  is one (m·n)×(m·n) matrix per block of size m instead of one 6n×6n matrix:
#  composition, sum and Volterra inverse act block by block. Two isotropic
#  blocks of n×n replace a 6n×6n matrix: 18 times less memory, 108 times
#  fewer flops in a product.
# =============================================================================

# ── The grid ────────────────────────────────────────────────────────────────

"""
    TimeGrid(times)

The instants ``t_1 < \\dots < t_n`` on which every operator and every history of
one aging viscoelastic computation is discretized. It is shared: an operator and a
history carry the grid they were built on, [`time_grid`](@ref) returns it after
any computation, and two operands on different grids are refused rather than
interpolated.
"""
struct TimeGrid{T <: Real, V <: AbstractVector{T}}
    times::V
    function TimeGrid(times::V) where {T <: Real, V <: AbstractVector{T}}
        isempty(times) && throw(ArgumentError("TimeGrid: no instant; give at least one"))
        # Instants must compare, to be ordered: symbols have no place here.
        is_hard_numeric(T) || throw(ArgumentError("TimeGrid: the instants are numbers that compare; got a $T"))
        all(i -> times[i] < times[i + 1], firstindex(times):(lastindex(times) - 1)) || throw(
            ArgumentError("TimeGrid: the instants must increase strictly; got $(times)")
        )
        return new{T, V}(times)
    end
end
TimeGrid(g::TimeGrid) = g

Base.length(g::TimeGrid) = length(g.times)
Base.getindex(g::TimeGrid, i) = g.times[i]
Base.firstindex(g::TimeGrid) = firstindex(g.times)
Base.lastindex(g::TimeGrid) = lastindex(g.times)
Base.iterate(g::TimeGrid, s...) = iterate(g.times, s...)
Base.eltype(::Type{<:TimeGrid{T}}) where {T} = T
Base.collect(g::TimeGrid) = collect(g.times)
Base.:(==)(a::TimeGrid, b::TimeGrid) = a === b || a.times == b.times
Base.show(io::IO, g::TimeGrid) = print(io, "TimeGrid(", length(g), " instants, ", first(g.times), " … ", last(g.times), ")")

# Two operands of one computation share their grid.
_same_grid(a::TimeGrid, b::TimeGrid, what) = a == b || throw(
    ArgumentError("$what: the operands live on different time grids ($a and $b); build them on one TimeGrid")
)

# ── Symmetry classes of the values ─────────────────────────────────────────

# Values that no symmetry class describes: a full block (6×6 Kelvin-Mandel at
# order 4, 3×3 at order 2), or a scalar.
"The class of an operator whose values are full blocks: Kelvin-Mandel ``6\\times6`` at order 4, ``3\\times3`` at order 2."
struct DenseClass{O} end
"The class of a scalar operator."
struct ScalarClass end

# The structured classes are TensND tensors of the class, standing for their
# class and frame (their coefficients are never read).
const _StructuredClass = Union{
    TensISO{4, 3}, TensISO{2, 3}, TensCubic,
    TensTI{4, <:Any, 5}, TensTI{4, <:Any, 6}, TensTI{2, <:Any, 2},
    TensOrtho{4}, TensOrtho{2},
}
const _OperatorClass = Union{_StructuredClass, DenseClass, ScalarClass}

# The class of a kernel value.
_class_of(x::_StructuredClass) = x
_class_of(::TensND.AbstractTens{4, 3}) = DenseClass{4}()
_class_of(::TensND.AbstractTens{2, 3}) = DenseClass{2}()
_class_of(::Number) = ScalarClass()
function _class_of(x::AbstractMatrix)
    size(x) == (6, 6) && return DenseClass{4}()
    size(x) == (3, 3) && return DenseClass{2}()
    throw(ArgumentError("a kernel value given as a matrix is 6×6 (Kelvin-Mandel, order 4) or 3×3 (order 2); got $(size(x))"))
end

_order(::TensND.AbstractTens{O}) where {O} = O
_order(::DenseClass{O}) where {O} = O
_order(::ScalarClass) = 0

# The dimension of the values of a history the operator acts on: Kelvin-Mandel
# vectors of symmetric second-order tensors at order 4, vectors at order 2.
_value_dim(c) = _order(c) == 4 ? 6 : 3
_value_dim(::ScalarClass) = 1

_block_sizes(c::_StructuredClass) = map(copies -> Val(length(first(copies))), TensND.symmetry_layout(c))
_block_sizes(::DenseClass{4}) = (Val(6),)
_block_sizes(::DenseClass{2}) = (Val(3),)
_block_sizes(::ScalarClass) = (Val(1),)
_val(::Val{m}) where {m} = m

# The orthonormal basis of the values in which the blocks sit, and their places.
_class_basis(c::_StructuredClass) = TensND.symmetry_basis(c)
_class_layout(c::_StructuredClass) = TensND.symmetry_layout(c)
_class_basis(::DenseClass{O}) where {O} = SMatrix{_value_dim(DenseClass{O}()), _value_dim(DenseClass{O}()), Float64}(I)
_class_layout(::DenseClass{4}) = ((SVector(1, 2, 3, 4, 5, 6),),)
_class_layout(::DenseClass{2}) = ((SVector(1, 2, 3),),)
_class_basis(::ScalarClass) = SMatrix{1, 1, Float64}(I)
_class_layout(::ScalarClass) = ((SVector(1),),)

# The full matrix of a value: Kelvin-Mandel at order 4, components at order 2.
_value_matrix(x::AbstractMatrix) = SMatrix{size(x)...}(x)
_value_matrix(x::TensND.AbstractTens{4, 3}) = SMatrix{6, 6}(_to_mandel(x))
_value_matrix(x::TensND.AbstractTens{2, 3}) = SMatrix{3, 3}(TensND.canonical_components(x))

# The blocks of a kernel value in a class. A value of the class itself is read
# directly when it shares the frame; a value of another class or frame means
# the law changes class over time, which no compact operator can hold.
_value_blocks(c::_StructuredClass, x::_StructuredClass) =
    _same_class(c, x) ? TensND.symmetry_blocks(x) : _value_blocks_by_matrix(c, x)
_value_blocks(c::_StructuredClass, x) = _value_blocks_by_matrix(c, x)

# A value of another type that belongs to the class all the same — an isotropic
# value in a transversely isotropic law, or the zero of one, which TensND
# returns isotropic — is read through its matrix; one off the class is refused.
function _value_blocks_by_matrix(c, x)
    M = _value_matrix(x)
    b = TensND.symmetry_blocks(c, M)
    is_hard_numeric(eltype(M)) && !(maximum(abs, TensND.symmetry_matrix(c, b) - M) ≤ _CLASS_RTOL * maximum(abs, M)) &&
        _class_change_error(c, x)
    return b
end
_value_blocks(::DenseClass, x) = (_value_matrix(x),)
_value_blocks(::ScalarClass, x::Number) = (SMatrix{1, 1}(x),)

_class_change_error(c, x) = throw(
    ArgumentError(
        "the law changes its symmetry class or frame over time (a $(nameof(typeof(c))) first, then a " *
            "$(nameof(typeof(x)))); return its values as Kelvin-Mandel matrices to build a dense operator"
    )
)

# The value of the class whose blocks are `b`: a tensor of the class, the full
# matrix, or the scalar.
_class_value(c::_StructuredClass, b) = TensND.from_symmetry_blocks(c, b)
_class_value(::DenseClass, b) = b[1]
_class_value(::ScalarClass, b) = b[1][1]

# The full matrix (Kelvin-Mandel at order 4) of the value whose blocks are `b`.
_class_matrix(c::_StructuredClass, b) = TensND.symmetry_matrix(c, b)
_class_matrix(::Union{DenseClass, ScalarClass}, b) = b[1]

# The blocks, in class `c`, of a full matrix, and whether it belongs to `c`.
_matrix_blocks(c::_StructuredClass, M) = TensND.symmetry_blocks(c, M)
_matrix_blocks(::Union{DenseClass, ScalarClass}, M) = (M,)

# One class and frame: the same blocks in the same adapted basis. Classes of
# one family (a symmetric and a general TI tensor, say) share their layout.
_same_class(a, b) = false
_same_class(::DenseClass{O}, ::DenseClass{O}) where {O} = true
_same_class(::ScalarClass, ::ScalarClass) = true
_same_class(a::_StructuredClass, b::_StructuredClass) =
    typeof(TensND.symmetry_layout(a)) == typeof(TensND.symmetry_layout(b)) &&
    _same_basis(TensND.symmetry_basis(a), TensND.symmetry_basis(b))

# Numbers compare by value; symbols only structurally (`==` on them is an
# expression, not a Bool).
_same_basis(P::AbstractMatrix{T}, Q::AbstractMatrix{S}) where {T, S} =
    _same_basis(Val(is_hard_numeric(promote_type(T, S))), P, Q)
_same_basis(::Val{true}, P, Q) = P == Q
_same_basis(::Val{false}, P, Q) = isequal(P, Q)

_class_name(::TensISO) = "isotropic"
_class_name(::TensCubic) = "cubic"
_class_name(::TensTI) = "transversely isotropic"
_class_name(::TensOrtho) = "orthotropic"
_class_name(::DenseClass) = "dense"
_class_name(::ScalarClass) = "scalar"

# ── The operator ────────────────────────────────────────────────────────────

"""
    VolterraOperator(law, times)
    VolterraOperator(f, times)
    VolterraOperator(C, times)
    VolterraOperator(M, times; like = nothing)

A linear operator of aging viscoelasticity on the time grid `times` (a vector of
instants or a [`TimeGrid`](@ref)): a relaxation ``R(t,t')``, a creep compliance
``J(t,t')``, an aging conductivity, … It maps the values of a history on the grid
to the values of the response, ``\\sigma(t_i)=\\int_0^{t_i}R(t_i,\\tau)\\,
\\mathrm{d}\\varepsilon(\\tau)``, by the trapezoidal rule of
[sanahuja2013](@citet).

It is built from

- a law (`ViscoLaw`, `maxwell_iso`, … or any function `f(t, t′)`) returning
  tensors, ``6\\times 6`` Kelvin-Mandel or ``3\\times 3`` matrices, or scalars;
- a constant tensor `C` (an elastic stiffness: ``R(t,t')=C``);
- the matrix `M` of an operator already discretized on the grid
  (``6n\\times 6n`` at order 4, ``3n\\times 3n`` at order 2, ``n\\times n`` for a
  scalar), as one scale hands it to the next; `like = C` stores it in the
  symmetry class and frame of the tensor `C`, and checks that it belongs to it.

When the values belong to one symmetry class, only its blocks are stored
(TensND `symmetry_blocks`): two ``n\\times n`` matrices for an isotropic operator
instead of one ``6n\\times 6n``. It computes as its elastic counterpart does:

| elastic | aging viscoelastic |
|:--|:--|
| `C₁ ⊡ C₂` | composition ``R_1\\circ R_2`` (in this order: it does not commute) |
| `inv(C)` | Volterra inverse: the creep operator of a relaxation one |
| `C₁ + C₂`, `C₁ - C₂`, `c * C` | sum, difference, product by a number |
| `I`, `one(R)` | the identity operator |
| `C ⊡ ε` | the response to the history `ε` (a [`TimeHistory`](@ref)) |

At order 2 `⋅` replaces `⊡`.

It **is** the discretized matrix, stored compactly: an `AbstractMatrix` of size
``6n\\times 6n`` (``3n\\times 3n`` at order 2), whose matrix product is the
composition, matrix inverse the Volterra inverse and sum the sum of operators, so
that it goes wherever such a matrix goes. `Matrix(R)` materializes it,
[`operator_value`](@ref)`(R, i, j)` is its value at the pair of instants
``(t_i, t_j)`` as a tensor of its class, [`operator_law`](@ref)`(R)` the law of
``(t, t')`` back, [`time_grid`](@ref)`(R)` its grid.
"""
struct VolterraOperator{C <: _OperatorClass, T, B <: Tuple{Matrix{T}, Vararg{Matrix{T}}}, G <: TimeGrid} <: AbstractMatrix{T}
    class::C
    blocks::B
    grid::G
end

"""
    time_grid(x) -> TimeGrid

The time grid of an operator or a history, or of the result of a computation on
them.
"""
time_grid(R::VolterraOperator) = R.grid

Base.eltype(::Type{<:VolterraOperator{C, T}}) where {C, T} = T

function VolterraOperator(law::AbstractViscoLaw, times)
    grid = TimeGrid(times)
    t, n = grid.times, length(grid)
    c = _law_class(law, t)
    sizes = _block_sizes(c)
    T = mapreduce(eltype, promote_type, _value_blocks(c, visco_eval(law, t[1], t[1])))
    Ms = map(s -> zeros(T, _val(s) * n, _val(s) * n), sizes)
    _fill_trapezoidal_classes!(Ms, law, t, sizes, x -> _value_blocks(c, x))
    return VolterraOperator(c, Ms, grid)
end

VolterraOperator(f, times) = VolterraOperator(ViscoLaw(f), times)

# The class of a law: that of its first value on the diagonal of the grid that is not
# zero. A zero says nothing (TensND makes the zero of any tensor isotropic), and a
# phase that precipitates later is zero first.
function _law_class(law, t)
    for tᵢ in t
        v = visco_eval(law, tᵢ, tᵢ)
        iszero(v) || return _class_of(v)
    end
    return _class_of(visco_eval(law, t[1], t[1]))
end

# A constant tensor: its trapezoidal matrix is block diagonal, the tensor on
# the diagonal (the weights ½(f + f) on it, ½(f − f) = 0 below).
function VolterraOperator(C::Union{TensND.AbstractTens, Number}, times)
    grid = TimeGrid(times)
    c = _class_of(C)
    return VolterraOperator(c, map(b -> _block_diagonal(b, length(grid)), _value_blocks(c, C)), grid)
end

_block_diagonal(b::SMatrix{m, m, T}, n) where {m, T} = kron(Matrix{T}(I, n, n), Matrix(b))

function VolterraOperator(M::AbstractMatrix, times; like = nothing)
    grid = TimeGrid(times)
    n = length(grid)
    size(M, 1) == size(M, 2) && size(M, 1) in (n, 3n, 6n) || throw(
        DimensionMismatch(
            "VolterraOperator: a $(size(M)) matrix on $n instants; expected $(n)×$(n) (scalar), " *
                "$(3n)×$(3n) (order 2) or $(6n)×$(6n) (order 4)"
        )
    )
    d = size(M, 1) ÷ n
    c = d == 6 ? DenseClass{4}() : d == 3 ? DenseClass{2}() : ScalarClass()
    return _in_class(VolterraOperator(c, (Matrix(M),), grid), like)
end

_in_class(R, ::Nothing) = R
_in_class(R, like) = convert_class(R, _class_of(like); check = true)

# ── Entries and conversions ────────────────────────────────────────────────

# The blocks at the pair of instants (i, j), as static matrices.
@inline _entry_blocks(R::VolterraOperator, i, j) = _entry_blocks(R.class, R.blocks, i, j)
@inline function _entry_blocks(c, blocks, i, j)
    return map(blocks, _block_sizes(c)) do B, s
        m = _val(s)
        SMatrix{m, m}(view(B, (m * (i - 1) + 1):(m * i), (m * (j - 1) + 1):(m * j)))
    end
end

"""
    operator_value(R::VolterraOperator, i, j)

The value of the operator `R` at the pair of instants ``(t_i, t_j)`` of its grid
(its trapezoidal weight, not its kernel: see [`operator_law`](@ref)), as a tensor
of its class, a full matrix for a dense operator, a number for a scalar one.
"""
operator_value(R::VolterraOperator, i::Integer, j::Integer) = _class_value(R.class, _entry_blocks(R, i, j))

# The matrix interface: entry (I, J) of the full discretized matrix.
Base.size(R::VolterraOperator) = (d = _value_dim(R.class) * length(R.grid); (d, d))
function Base.getindex(R::VolterraOperator{<:Any, T}, I::Integer, J::Integer) where {T}
    @boundscheck checkbounds(R, I, J)
    d = _value_dim(R.class)
    i, r = divrem(I - 1, d)
    j, c = divrem(J - 1, d)
    j > i && return zero(T)
    return T(_class_matrix(R.class, _entry_blocks(R, i + 1, j + 1))[r + 1, c + 1])
end

"""
    Matrix(R::VolterraOperator) -> Matrix

The full discretized matrix of `R`: ``6n\\times 6n`` (Kelvin-Mandel blocks) at
order 4, ``3n\\times 3n`` at order 2, ``n\\times n`` for a scalar operator.
"""
Base.Matrix(R::VolterraOperator) = _dense_matrix(R.class, R.blocks, length(R.grid))

# The full matrix of the operator of class `c` whose block matrices are `blocks`, on n instants.
function _dense_matrix(c, blocks, n)
    T = mapreduce(eltype, promote_type, blocks)
    d = _value_dim(c)
    M = zeros(T, d * n, d * n)
    for j in 1:n, i in j:n
        M[(d * (i - 1) + 1):(d * i), (d * (j - 1) + 1):(d * j)] .= _class_matrix(c, _entry_blocks(c, blocks, i, j))
    end
    return M
end
Base.Matrix(R::VolterraOperator{<:DenseClass}) = copy(R.blocks[1])
# Isotropic: the single pass that writes each entry once (conversions.jl).
Base.Matrix(R::VolterraOperator{<:TensISO{4, 3}}) = iso_blocks_from_params(R.blocks...)
Base.Matrix(R::VolterraOperator{ScalarClass}) = copy(R.blocks[1])

"""
    convert_class(R::VolterraOperator, class; check = true) -> VolterraOperator

`R` with its values stored in the symmetry class and frame of the tensor `class`
(or dense, `DenseClass{4}()`). With `check`, the values must belong to the class:
a value off it would be projected silently, so it is refused.
"""
function convert_class(R::VolterraOperator, c; check::Bool = true)
    typeof(R.class) == typeof(c) && _same_class(R.class, c) && return R
    Rc, failure = _converted(R, c, check ? _CLASS_RTOL : nothing)
    failure === nothing || throw(
        ArgumentError(
            "the operator does not belong to the $(_class_name(c)) class given: its value at instants " *
                "$(failure[1:2]) is off it by $(failure[3]); give a tensor of its class as `like`, or keep it dense"
        )
    )
    return Rc
end

# A value belongs to a class when the class rebuilds it to this relative
# accuracy, the scale being the largest entry of the operator.
const _CLASS_RTOL = 1.0e-12

# `R` in class `c`, and the first pair of instants whose value is off the class
# by more than `rtol` (nothing when all belong, or when `rtol` is nothing; the
# operator is then left unfinished). Its type depends on `c` only.
function _converted(R::VolterraOperator, c, rtol)
    n, sizes = length(R.grid), _block_sizes(c)
    T = promote_type(eltype(R), eltype(_class_basis(c)))
    Ms = map(s -> zeros(T, _val(s) * n, _val(s) * n), sizes)
    tol = _class_tolerance(R, rtol)
    for j in 1:n, i in j:n
        A = _class_matrix(R.class, _entry_blocks(R, i, j))
        b = _matrix_blocks(c, A)
        r = _class_residual(tol, c, A, b)
        r === nothing || return VolterraOperator(c, Ms, R.grid), (i, j, r)
        map(Ms, b, sizes) do M, x, s
            m = _val(s)
            M[(m * (i - 1) + 1):(m * i), (m * (j - 1) + 1):(m * j)] .= x
        end
    end
    return VolterraOperator(c, Ms, R.grid), nothing
end

# An isotropic operator belongs to every class of its order, linearly: the blocks of
# α 𝕁 + β 𝕂 in class `c` are kron(α, b(𝕁)) + kron(β, b(𝕂)), b(𝕁) and b(𝕂) the small
# constant blocks of the two projectors. No check, no conversion entry by entry.
function _converted(R::VolterraOperator{<:TensISO{4, 3}}, c, rtol)
    α, β = R.blocks
    bJ = _matrix_blocks(c, _value_matrix(TensISO{3}(1.0, 0.0)))
    bK = _matrix_blocks(c, _value_matrix(TensISO{3}(0.0, 1.0)))
    return VolterraOperator(c, map((j, k) -> kron(α, Matrix(j)) + kron(β, Matrix(k)), bJ, bK), R.grid), nothing
end
function _converted(R::VolterraOperator{<:TensISO{2, 3}}, c, rtol)
    λ = R.blocks[1]
    b𝟏 = _matrix_blocks(c, _value_matrix(TensISO{3}(1.0)))
    return VolterraOperator(c, map(b -> kron(λ, Matrix(b)), b𝟏), R.grid), nothing
end

"""
    _iso_part(R::VolterraOperator) -> VolterraOperator

The isotropic part of an order-4 operator, read on its blocks: the average over the
rotations of each of its values, ``\\alpha = \\underline{s}\\cdot M\\cdot\\underline{s}`` (``\\underline{s}``
the spherical Kelvin-Mandel direction) and ``\\beta = (\\operatorname{tr} M - \\alpha)/5``,
both linear in the blocks.
"""
function _iso_part(R::VolterraOperator)
    n = length(R.grid)
    Q = _class_basis(R.class)
    s = transpose(Q) * SVector(1, 1, 1, 0, 0, 0) / sqrt(3 * one(eltype(Q)))
    T = promote_type(eltype(R), eltype(Q))
    α, trace = zeros(T, n, n), zeros(T, n, n)
    map(R.blocks, _class_layout(R.class)) do B, copies
        m = length(first(copies))
        W = reshape(B, m, n, m, n)
        for a in 1:m
            trace .+= length(copies) .* view(W, a, :, a, :)
        end
        for c in copies, a in 1:m, b in 1:m
            α .+= (s[c[a]] * s[c[b]]) .* view(W, a, :, b, :)
        end
    end
    return VolterraOperator(_ISO4, (α, (trace .- α) ./ 5), R.grid)
end

"""
    _iso_part2(R::VolterraOperator) -> VolterraOperator

The isotropic part of an order-2 operator, read on its blocks: a third of the trace of
each of its values, the trace being that of the blocks, each counted once per copy (the
adapted basis is orthonormal).
"""
function _iso_part2(R::VolterraOperator)
    n = length(R.grid)
    trace = zeros(eltype(R), n, n)
    map(R.blocks, _class_layout(R.class)) do B, copies
        m = length(first(copies))
        W = reshape(B, m, n, m, n)
        for a in 1:m
            trace .+= length(copies) .* view(W, a, :, a, :)
        end
    end
    return VolterraOperator(_ISO2, (trace ./ 3,), R.grid)
end

# The absolute tolerance of a membership test. Values that do not compare
# (symbols) are taken as given: no residual can be judged on them.
_class_tolerance(R, ::Nothing) = nothing
_class_tolerance(R, rtol) = is_hard_numeric(eltype(R)) ? rtol * max(maximum(B -> maximum(abs, B; init = zero(eltype(R))), R.blocks), floatmin(Float64)) : nothing

# The residual of a value off the class when it exceeds the tolerance.
_class_residual(::Nothing, c, A, b) = nothing
function _class_residual(tol, c, A, b)
    r = maximum(abs, _class_matrix(c, b) - A)
    return r ≤ tol ? nothing : r
end

"""
    _compact(R::VolterraOperator, candidates) -> VolterraOperator

`R` in the first of the classes `candidates` (tensors standing for a class and a
frame) its values all belong to, as it is otherwise.
"""
_compact(R::VolterraOperator{<:Union{TensISO{4, 3}, TensISO{2, 3}}}, candidates) = R
function _compact(R::VolterraOperator, candidates)
    # Values that do not compare keep their storage: no residual can say they belong.
    is_hard_numeric(eltype(R)) || return R
    for c in candidates
        _order(c) == _order(R.class) || continue
        # Already in this class (or in the cheapest one): nothing to convert.
        _same_class(R.class, c) && return R
        Rc, failure = _converted(R, c, _CLASS_RTOL)
        failure === nothing && return Rc
    end
    return R
end

# ── Algebra ─────────────────────────────────────────────────────────────────

# Two operands brought to one class: kept when they share it, an isotropic one
# taken into the class of the other (it belongs to every class), dense otherwise.
function _common_class(A::VolterraOperator, B::VolterraOperator, what)
    _same_grid(A.grid, B.grid, what)
    _order(A.class) == _order(B.class) || throw(
        ArgumentError("$what: operators of orders $(_order(A.class)) and $(_order(B.class)) do not combine")
    )
    return _combine(_combine_rule(_family(A.class), _family(B.class)), A, B, what)
end

# The family of a class, its layout: what decides, from the types alone, how two
# operators combine (a symmetric and a general TI tensor are one family).
_family(c::_StructuredClass) = typeof(TensND.symmetry_layout(c))
_family(::DenseClass{O}) where {O} = DenseClass{O}
_family(::ScalarClass) = ScalarClass
const _IsoFamily = Union{typeof(TensND.symmetry_layout(TensISO{3}(1.0, 1.0))), typeof(TensND.symmetry_layout(TensISO{3}(1.0)))}

# The rule, from two constant types: folded at compile time, so that the result
# type of a composition is inferred.
_combine_rule(F, G) = F === G ? Val(:same) : F <: _IsoFamily ? Val(:left_iso) : G <: _IsoFamily ? Val(:right_iso) : Val(:dense)

# One family in one frame stays in it; in two frames (two families of fibers of
# different axes, say), the full blocks. The frames are values: a run-time branch,
# whose result types are those of the class and of the dense class.
function _combine(::Val{:same}, A, B, what)
    _same_class(A.class, B.class) && return A, B
    D = _dense_class(A.class)
    return first(_converted(A, D, nothing)), first(_converted(B, D, nothing))
end
# An isotropic operator belongs to every class of its order: it joins the other's.
_combine(::Val{:left_iso}, A, B, what) = first(_converted(A, B.class, nothing)), B
_combine(::Val{:right_iso}, A, B, what) = A, first(_converted(B, A.class, nothing))
# Two families: the full blocks.
function _combine(::Val{:dense}, A, B, what)
    D = _dense_class(A.class)
    return first(_converted(A, D, nothing)), first(_converted(B, D, nothing))
end
_dense_class(c) = DenseClass{_order(c)}()
_dense_class(c::ScalarClass) = c

_blockwise(f::F, A::VolterraOperator, B::VolterraOperator) where {F} = VolterraOperator(A.class, map(f, A.blocks, B.blocks), A.grid)
_blockwise(f::F, A::VolterraOperator) where {F} = VolterraOperator(A.class, map(f, A.blocks), A.grid)

for op in (:+, :-)
    @eval function Base.$op(A::VolterraOperator, B::VolterraOperator)
        A′, B′ = _common_class(A, B, $(string(op)))
        return _blockwise($op, A′, B′)
    end
end
Base.:-(A::VolterraOperator) = _blockwise(-, A)
# Element by element: `Matrix{Num} / Num` would take Symbolics' array method,
# which unwraps the coefficients.
Base.:*(c::Number, A::VolterraOperator) = _blockwise(X -> c .* X, A)
Base.:*(A::VolterraOperator, c::Number) = c * A
# Its ambiguity with Symbolics' `/(::AbstractArray{<:Real}, ::Num)` is resolved in the Symbolics extension.
Base.:/(A::VolterraOperator, c::Number) = _blockwise(X -> X ./ c, A)

# The composition: per block, the product of the matrices of the block.
function _compose(A::VolterraOperator, B::VolterraOperator)
    A′, B′ = _common_class(A, B, "composition")
    return _blockwise(*, A′, B′)
end
TensND.:⊡(A::VolterraOperator, B::VolterraOperator) = _order_check(4, A, "⊡", _compose(A, B))
TensND.:⋅(A::VolterraOperator, B::VolterraOperator) = _order_check(2, A, "⋅", _compose(A, B))
Base.:*(A::VolterraOperator, B::VolterraOperator) = _compose(A, B)

_order_check(O, A, op, result) = _order(A.class) == O ? result : throw(
        ArgumentError("`$op` composes operators of order $O; this one is of order $(_order(A.class)) (use $(_order(A.class) == 2 ? "⋅" : "⊡") or *)")
    )

# A constant tensor, a number or `I` combined with an operator is the constant
# operator on its grid.
_as_operator(x::VolterraOperator, ::VolterraOperator) = x
_as_operator(x::Union{TensND.AbstractTens, Number}, R::VolterraOperator) = VolterraOperator(x, R.grid)
# A dense matrix is the discretized matrix of an operator on the grid, combined
# as a matrix; a tensor (an AbstractMatrix too at order 2) is a constant operator.
# Other matrix types take the generic AbstractMatrix methods: claiming them all
# here would be ambiguous with StaticArrays, Symbolics, …
_with_matrix(op::F, A::VolterraOperator, M::StridedMatrix) where {F} = op(Matrix(A), M)
_with_matrix(op::F, M::StridedMatrix, A::VolterraOperator) where {F} = op(M, Matrix(A))
_with_matrix(op::F, A::VolterraOperator, x::TensND.AbstractTens) where {F} = op(A, _as_operator(x, A))
_with_matrix(op::F, x::TensND.AbstractTens, A::VolterraOperator) where {F} = op(_as_operator(x, A), A)
# A second-order tensor is both: a tensor.
_with_matrix(op::F, A::VolterraOperator, x::TensND.AbstractTens{2}) where {F} = op(A, _as_operator(x, A))
_with_matrix(op::F, x::TensND.AbstractTens{2}, A::VolterraOperator) where {F} = op(_as_operator(x, A), A)
for op in (:+, :-)
    @eval Base.$op(A::VolterraOperator, x::Union{StridedMatrix, TensND.AbstractTens}) = _with_matrix($op, A, x)
    @eval Base.$op(x::Union{StridedMatrix, TensND.AbstractTens}, A::VolterraOperator) = _with_matrix($op, x, A)
    @eval Base.$op(A::VolterraOperator, J::UniformScaling) = $op(A, J.λ * one(A))
    @eval Base.$op(J::UniformScaling, A::VolterraOperator) = $op(J.λ * one(A), A)
end
Base.:*(A::VolterraOperator, M::StridedMatrix) = Matrix(A) * M
Base.:*(M::StridedMatrix, A::VolterraOperator) = M * Matrix(A)
TensND.:⊡(A::VolterraOperator, x::TensND.AbstractTens{4}) = A ⊡ _as_operator(x, A)
TensND.:⊡(x::TensND.AbstractTens{4}, A::VolterraOperator) = _as_operator(x, A) ⊡ A

"""
    inv(R::VolterraOperator) -> VolterraOperator

The Volterra inverse of `R`: the operator ``R^{-1}`` such that ``R\\circ R^{-1}``
is the identity on the grid. The inverse of a relaxation operator is the creep
operator of the same material. Computed block by block (forward substitution,
[`volterra_inverse`](@ref)).
"""
Base.inv(R::VolterraOperator) =
    VolterraOperator(R.class, map((B, s) -> volterra_inverse(B; block_size = _val(s)), R.blocks, _block_sizes(R.class)), R.grid)

# The Frobenius norm of the full matrix: each block counts once per copy, the
# adapted basis being orthonormal.
LinearAlgebra.norm(R::VolterraOperator) =
    sqrt(sum(map((B, copies) -> length(copies) * sum(abs2, B), R.blocks, _class_layout(R.class))))

Base.one(R::VolterraOperator{<:Any, T}) where {T} =
    VolterraOperator(R.class, map(B -> Matrix{T}(I, size(B)...), R.blocks), R.grid)
Base.zero(R::VolterraOperator) = _blockwise(zero, R)

# `R` with element type `T`, as the scheme formulas promote their operands.
_with_eltype(::Type{T}, R::VolterraOperator{<:Any, T}) where {T} = R
_with_eltype(::Type{T}, R::VolterraOperator) where {T} = _blockwise(B -> Matrix{T}(B), R)

Base.:(==)(A::VolterraOperator, B::VolterraOperator) =
    _same_class(A.class, B.class) ? A.grid == B.grid && A.blocks == B.blocks : Matrix(A) == Matrix(B)
function Base.isapprox(A::VolterraOperator, B::VolterraOperator; kw...)
    A′, B′ = _common_class(A, B, "isapprox")
    return all(map((X, Y) -> isapprox(X, Y; kw...), A′.blocks, B′.blocks))
end
# Against a plain matrix, as matrices.
Base.isapprox(A::VolterraOperator, M::AbstractMatrix; kw...) = isapprox(Matrix(A), M; kw...)
Base.isapprox(M::AbstractMatrix, A::VolterraOperator; kw...) = isapprox(M, Matrix(A); kw...)

Base.show(io::IO, ::MIME"text/plain", R::VolterraOperator) = show(io, R)
function Base.show(io::IO, R::VolterraOperator)
    sizes = join(map(s -> "$(_val(s))", _block_sizes(R.class)), ", ")
    return print(
        io, "VolterraOperator: ", _class_name(R.class), ", order ", _order(R.class), ", ",
        length(R.grid), " instants, blocks of sizes (", sizes, "), ", eltype(R)
    )
end

# ── Histories ───────────────────────────────────────────────────────────────

"""
    TimeHistory(f, times)
    TimeHistory(values, times)

A history on the time grid `times`: a strain ``\\varepsilon(t)`` or a stress
``\\sigma(t)`` (symmetric second-order tensors), a temperature gradient or a
field (vectors), or a scalar. It is given as a function `f(t)` evaluated at the
instants, or by its values: a vector of them, one per instant, or a matrix whose
column ``i`` is the value at ``t_i`` (Kelvin-Mandel components for a symmetric
tensor).

`R ⊡ ε` (order 4), `K ⋅ g` (order 2) or `R * h` is the response of an operator.
`h[i]` is the value at ``t_i`` as Kelvin-Mandel components (or the vector),
`Matrix(h)` all of them, [`time_grid`](@ref)`(h)` the grid.
"""
struct TimeHistory{T, G <: TimeGrid}
    values::Matrix{T}
    grid::G
    function TimeHistory(values::AbstractMatrix{T}, times) where {T}
        grid = TimeGrid(times)
        size(values, 2) == length(grid) && size(values, 1) in (1, 3, 6) || throw(
            DimensionMismatch(
                "TimeHistory: $(size(values)) values on $(length(grid)) instants; expected one column per " *
                    "instant of 6 (Kelvin-Mandel), 3 (vector) or 1 (scalar) components"
            )
        )
        return new{T, typeof(grid)}(Matrix(values), grid)
    end
end

TimeHistory(values::AbstractVector, times) = TimeHistory(reduce(hcat, map(_history_components, values)), times)
TimeHistory(f, times) = (g = TimeGrid(times); TimeHistory([f(t) for t in g.times], g))

_history_components(x::Number) = SVector(x)
_history_components(x::AbstractVector) = SVector{length(x)}(x)
_history_components(x::TensND.AbstractTens{1, 3}) = SVector{3}(TensND.canonical_components(x))
_history_components(x::TensND.AbstractTens{2, 3}) = SVector{6}(TensND.mandel(TensND.canonical_components(x); symmetrize = true))

time_grid(h::TimeHistory) = h.grid
Base.getindex(h::TimeHistory, i::Integer) = h.values[:, i]
Base.firstindex(h::TimeHistory) = 1
Base.lastindex(h::TimeHistory) = length(h.grid)
Base.Matrix(h::TimeHistory) = copy(h.values)
Base.length(h::TimeHistory) = length(h.grid)
Base.:(==)(a::TimeHistory, b::TimeHistory) = a.grid == b.grid && a.values == b.values
Base.isapprox(a::TimeHistory, b::TimeHistory; kw...) = a.grid == b.grid && isapprox(a.values, b.values; kw...)
Base.show(io::IO, h::TimeHistory) =
    print(io, "TimeHistory: ", size(h.values, 1), " components on ", length(h.grid), " instants, ", eltype(h.values))

"""
    R * h, R ⊡ h, R ⋅ h -> TimeHistory

The response of the operator `R` to the history `h` on their common grid:
``y(t_i) = \\int_0^{t_i}R(t_i,\\tau)\\,\\mathrm{d}h(\\tau)``. In the adapted basis of
the class each block acts on its own components.
"""
function Base.:*(R::VolterraOperator, h::TimeHistory)
    _same_grid(R.grid, h.grid, "operator applied to a history")
    d, n = _value_dim(R.class), length(h.grid)
    size(h.values, 1) == d || throw(
        DimensionMismatch("an operator of order $(_order(R.class)) acts on histories of $d components; this one has $(size(h.values, 1))")
    )
    Q = _class_basis(R.class)
    V = transpose(Q) * h.values
    W = zeros(promote_type(eltype(R), eltype(V)), d, n)
    map(R.blocks, _class_layout(R.class)) do B, copies
        for c in copies
            W[c, :] .= reshape(B * vec(V[c, :]), length(c), n)
        end
    end
    return TimeHistory(Q * W, h.grid)
end
# A vector stacked instant by instant, as the full matrix would multiply it.
function Base.:*(R::VolterraOperator, x::StridedVector)
    d, n = _value_dim(R.class), length(R.grid)
    length(x) == d * n || throw(DimensionMismatch("a $(size(R)) operator times a vector of length $(length(x))"))
    return vec((R * TimeHistory(reshape(x, d, n), R.grid)).values)
end
TensND.:⊡(R::VolterraOperator, h::TimeHistory) = _order_check(4, R, "⊡", R * h)
TensND.:⋅(R::VolterraOperator, h::TimeHistory) = _order_check(2, R, "⋅", R * h)

# ── Back to a function of (t, t′) ─────────────────────────────────────────

"""
    operator_law(R::VolterraOperator; mode = :relaxation) -> SampledViscoLaw

The law of ``(t, t')`` whose operator on the grid of `R` is `R`: its values at the
pairs of instants are tensors of the class of `R` (full matrices for a dense
operator, numbers for a scalar one). The kernel is recovered block by block, as
for a matrix.
"""
function operator_law(R::VolterraOperator; mode::Symbol = :relaxation)
    n, sizes = length(R.grid), _block_sizes(R.class)
    F = map((B, s) -> _kernel_samples(B, n, s), R.blocks, sizes)
    samples = [_class_value(R.class, map((Fₖ, s) -> SMatrix{_val(s), _val(s)}(Fₖ[i, j]), F, sizes)) for i in 1:n, j in 1:n]
    return SampledViscoLaw(R.grid.times, samples; mode)
end

# On an operator: its kernel read back on the grid, switched, discretized again, in the
# class of its values.
"""
    precipitated(R::VolterraOperator, t_s) -> VolterraOperator
    dissolved(R::VolterraOperator, t_d) -> VolterraOperator

The same transforms on a relaxation operator, on its grid: its kernel
([`operator_law`](@ref)) switched on at ``t_s`` or off at ``t_d``, and discretized again.
"""
precipitated(R::VolterraOperator, t_s::Real) = VolterraOperator(precipitated(operator_law(R), t_s), R.grid)
dissolved(R::VolterraOperator, t_d::Real) = VolterraOperator(dissolved(operator_law(R), t_d), R.grid)

# The Volterra inverse of the matrix interface, block by block.
function volterra_inverse(R::VolterraOperator; block_size::Int = _value_dim(R.class))
    block_size == _value_dim(R.class) || throw(
        ArgumentError("volterra_inverse: this operator has blocks of size $(_value_dim(R.class)), not $block_size")
    )
    return inv(R)
end
