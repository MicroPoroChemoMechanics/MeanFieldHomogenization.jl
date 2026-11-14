# =============================================================================
#  layered_cylinder_alv.jl — n-layer cylinder in aging linear viscoelasticity,
#  orders 4 (elasticity) and 2 (conduction).
#
#  The radial problems of `LayeredCylinders/kernel.jl` are written for
#  operators that do not commute: every product there is in the order the
#  equilibrium equations give it. They are solved here with the n × n Volterra
#  matrices of the moduli in place of numbers, and the results reassembled in
#  6n × 6n (resp. 3n × 3n) block matrices. No formula is restated.
#
#  The axis n of the cylinder is any direction. Its layers and the reference,
#  isotropic or transversely isotropic about n, are written in the frame of the
#  cylinder, the radial problems are solved there, and the result is brought
#  back to the canonical frame at the end (see "The frame of the cylinder").
# =============================================================================

const _LC = LayeredCylinders

# ── Transversely isotropic blocks about e₃ ───────────────────────────────────

"""
    ti_params_from_blocks(M) -> (; zz, ze, mz, me, T, L)

The six ``n\\times n`` matrices of a ``6n\\times 6n`` block matrix whose every
``6\\times 6`` Mandel block is transversely isotropic about ``\\underline e_3``:
per block, ``zz = M_{33}``, ``ze = 2M_{31}``, ``mz = M_{13}``,
``me = M_{11} + M_{12}``, ``T = M_{11} - M_{12}``, ``L = M_{44}`` — for a
stiffness of Hill moduli ``(k, l, m, n, p)``, ``(n, 2l, l, 2k, 2m, 2p)``. In
the Walpole basis ``\\ell_1 = zz``, ``\\ell_2 = me``, ``\\ell_3 = ze/\\sqrt2``,
``\\ell_4 = \\sqrt2\\,mz``, ``\\ell_5 = T``, ``\\ell_6 = L``. Inverse:
[`ti_blocks_from_params`](@ref).
"""
function ti_params_from_blocks(M::AbstractMatrix)
    sz = size(M, 1)
    sz == size(M, 2) && sz % 6 == 0 || throw(
        ArgumentError("ti_params_from_blocks: M must be square of size 6n; got $(size(M))")
    )
    return (;
        zz = copy(_slots(M, 3, 3)), ze = 2 .* _slots(M, 3, 1), mz = copy(_slots(M, 1, 3)),
        me = _slots(M, 1, 1) .+ _slots(M, 1, 2), T = _slots(M, 1, 1) .- _slots(M, 1, 2),
        L = copy(_slots(M, 4, 4)),
    )
end

"""
    ti_blocks_from_params(c) -> Matrix

The ``6n\\times 6n`` block matrix of the six ``n\\times n`` matrices
`(; zz, ze, mz, me, T, L)` of [`ti_params_from_blocks`](@ref).
"""
function ti_blocks_from_params(c::NamedTuple{(:zz, :ze, :mz, :me, :T, :L)})
    n = size(c.zz, 1)
    E = typeof(one(promote_type(map(eltype, values(c))...)) / 2)
    M = zeros(E, 6n, 6n)
    _slots(M, 3, 3) .= c.zz
    for a in 1:2
        _slots(M, 3, a) .= c.ze ./ 2
        _slots(M, a, 3) .= c.mz
        _slots(M, a, a) .= (c.me .+ c.T) ./ 2
        _slots(M, a, 3 - a) .= (c.me .- c.T) ./ 2
        _slots(M, 3 + a, 3 + a) .= c.L
    end
    _slots(M, 6, 6) .= c.T
    return M
end

# The entries (a, b) of every 3×3 block of a 3n×3n block matrix.
_slots3(M::AbstractMatrix, a::Int, b::Int) = view(M, a:3:size(M, 1), b:3:size(M, 2))

"""
    ti_order2_params_from_blocks(M) -> (; t, a)
    ti_order2_blocks_from_params(c) -> Matrix

The transverse and axial ``n\\times n`` matrices of a ``3n\\times 3n`` block matrix
whose every block is ``\\mathrm{diag}(t, t, a)``, and the inverse map.
"""
function ti_order2_params_from_blocks(M::AbstractMatrix)
    sz = size(M, 1)
    sz == size(M, 2) && sz % 3 == 0 || throw(
        ArgumentError("ti_order2_params_from_blocks: M must be square of size 3n; got $(size(M))")
    )
    return (; t = copy(_slots3(M, 1, 1)), a = copy(_slots3(M, 3, 3)))
end

function ti_order2_blocks_from_params(c::NamedTuple{(:t, :a)})
    n = size(c.t, 1)
    M = zeros(promote_type(eltype(c.t), eltype(c.a)), 3n, 3n)
    _slots3(M, 1, 1) .= c.t
    _slots3(M, 2, 2) .= c.t
    _slots3(M, 3, 3) .= c.a
    return M
end

# A block matrix that its transversely isotropic parameters do not rebuild is
# not of that class: a layer or a reference of another symmetry, or of another
# axis. Compared where the entries compare.
function _check_ti_blocks(M::AbstractMatrix, rebuilt::AbstractMatrix, who::AbstractString)
    is_hard_numeric(eltype(M)) || return nothing
    norm(M - rebuilt) ≤ _LC._TI_RTOL * norm(M) || _not_ti_about_axis(who)
    return nothing
end

# ── The frame of the cylinder ────────────────────────────────────────────────
#
# The radial problems are solved in a frame of the cylinder, its axis n being
# the third vector. An operator transversely isotropic about n is stored by
# TensND in the basis adapted to its class, whose blocks do not depend on the
# direction of the axis: read in the class about e₃, the same blocks are the
# operator written in the frame of the cylinder; read back in the class about
# n, they are the operator in the canonical frame. So a law or a reference is
# written in the frame of the cylinder once, everything is computed there, and
# the result alone is brought back. Neither move touches the order of the
# Volterra products: a frame does not depend on time.

const _E₃ = (0.0, 0.0, 1.0)
_canonical_axis(cyl::LayeredCylinder) = TensND.isparallel(cyl.axis, _E₃)

# The transversely isotropic class about n: general at order 4 (a concentration
# has no major symmetry).
_ti_class(::Val{4}, n) = TensTI{4}(1.0, 1.0, 0.0, 0.0, 1.0, 1.0, n)
_ti_class(::Val{2}, n) = TensTI{2}(1.0, 1.0, n)

# A property or a reference as a Volterra operator on the grid.
_alv_operator(R::VolterraOperator, times) = R
_alv_operator(P::Union{TensND.AbstractTens, AbstractMatrix, TensND.AbstractTens{2}}, times) = VolterraOperator(P, times)
_alv_operator(P, times) = VolterraOperator(_as_visco_law(P), times)

# The block matrix of `P` in the frame of `cyl`, `b` rows per instant; refused
# when `P` is not transversely isotropic about the axis.
function _cylinder_frame_blocks(o::Val, cyl::LayeredCylinder, P, times, who::AbstractString)
    _canonical_axis(cyl) && return _alv_blocks(P, times, _block_rows(o))
    R, failure = _converted(_alv_operator(P, times), _ti_class(o, cyl.axis), _CLASS_RTOL)
    failure === nothing || _not_ti_about_axis(who)
    return Matrix(VolterraOperator(_ti_class(o, _E₃), R.blocks, R.grid))
end

# The block matrix `M` of an operator computed in the frame of `cyl`, in the
# canonical frame.
function _canonical_frame_blocks(o::Val, cyl::LayeredCylinder, M::AbstractMatrix, times)
    _canonical_axis(cyl) && return M
    R = VolterraOperator(M, times; like = _ti_class(o, _E₃))
    return Matrix(VolterraOperator(_ti_class(o, cyl.axis), R.blocks, R.grid))
end

_not_ti_about_axis(who::AbstractString) = throw(
    ArgumentError(
        "$who of the LayeredCylinder is not transversely isotropic about its axis: in " *
            "aging viscoelasticity its layers and the reference must be isotropic or " *
            "transversely isotropic about the axis of the cylinder"
    )
)

_finite_blocks(c::NamedTuple, who::AbstractString) =
    any(M -> any(_nonfinite, M), values(c)) && throw(
    ArgumentError("$who of the LayeredCylinder has an infinite modulus, which has no Volterra matrix")
)

# A Volterra matrix the recurrences divide by: its diagonal (the instantaneous
# modulus at each instant) must not vanish. Checked where the numbers compare.
function _check_invertible(M::AbstractMatrix, what::AbstractString, who::AbstractString)
    is_hard_numeric(eltype(M)) || return nothing
    any(iszero, diag(M)) && throw(
        ArgumentError(
            "$who of the LayeredCylinder has a vanishing $what at some instant: the ALV " *
                "recurrences divide by it (a void layer is not supported in elasticity)"
        )
    )
    return nothing
end

# Hill moduli (n × n matrices) of a 6n × 6n transversely isotropic operator.
function _hill_volterra(M::AbstractMatrix, who::AbstractString)
    c = ti_params_from_blocks(M)
    _finite_blocks(c, who)
    _check_ti_blocks(M, ti_blocks_from_params(c), who)
    L = (; k = c.me ./ 2, l = c.mz, m = c.T ./ 2, n = c.zz, p = c.L ./ 2, lt = c.ze ./ 2)
    _check_invertible(L.k + L.m, "plane-strain modulus k + m", who)
    _check_invertible(L.m, "transverse shear modulus m", who)
    _check_invertible(L.p, "axial shear modulus p", who)
    return L
end

# A layer may be impermeable, its conductivity zero at every instant; otherwise
# the recurrence divides by it. The reference carries the remote gradient.
function _conductivity_volterra(M::AbstractMatrix, who::AbstractString; reference::Bool = false)
    c = ti_order2_params_from_blocks(M)
    _finite_blocks(c, who)
    _check_ti_blocks(M, ti_order2_blocks_from_params(c), who)
    (reference || !_LC._is_impermeable(c.t)) && _check_invertible(c.t, "transverse conductivity", who)
    return c
end

# The block matrix of a layer property (a tensor, a law) or of a reference (a
# law, or the matrix of a running estimate), `b` rows per instant.
_alv_blocks(P::AbstractMatrix, times, b::Int) = P
# An order-2 tensor of TensND is an `AbstractMatrix` too, but an elastic property.
_alv_blocks(P::TensND.AbstractTens{2}, times, b::Int) = _trapezoidal_relaxation(heaviside_law(P), times, b)
_alv_blocks(P, times, b::Int) = _trapezoidal_relaxation(_as_visco_law(P), times, b)

"""
    _cylinder_alv_solution(Val(order), cyl, ref, times) -> NamedTuple

The radial problems of `cyl` against the aging reference `ref` on the instants
`times`, with ``n\\times n`` Volterra matrices as moduli: the solution of
`LayeredCylinders._elastic_solution` (order 4) or `_conduction_solution`
(order 2), whose entries are the matrices mapping the history of the remote
loading to the history of each state.
"""
function _cylinder_alv_solution(o::Val, cyl::LayeredCylinder, ref::_ALVReference, times::AbstractVector{<:Real})
    R₀, Rs = _cylinder_alv_blocks(o, cyl, ref, times)
    # The type of a discretized law is known only once it is evaluated: the
    # solution is computed behind a function barrier, on the block matrices.
    return _cylinder_alv_from_blocks(o, cyl, R₀, Rs)
end

# The block matrices of the reference and of the layers in the frame of the
# cylinder, `b` rows per instant.
_cylinder_alv_blocks(o::Val, cyl::LayeredCylinder{T, N}, ref::_ALVReference, times::AbstractVector{<:Real}) where {T, N} = (
    _cylinder_frame_blocks(o, cyl, ref, times, "the reference medium"),
    ntuple(k -> _cylinder_frame_blocks(o, cyl, layer_modulus(cyl, k), times, "layer $k"), Val(N)),
)

_block_rows(::Val{4}) = 6
_block_rows(::Val{2}) = 3

_volterra_entries(::Val{4}, M, who; kw...) = _hill_volterra(M, who)
_volterra_entries(::Val{2}, M, who; kw...) = _conductivity_volterra(M, who; kw...)

function _cylinder_alv_from_blocks(
        o::Val, cyl::LayeredCylinder{T, N}, R₀::AbstractMatrix, Rs::Tuple
    ) where {T, N}
    L₀ = _volterra_entries(o, R₀, "the reference medium"; reference = true)
    Ls = ntuple(k -> _volterra_entries(o, Rs[k], "layer $k"), Val(N))
    E = promote_type(
        T, map(eltype, values(L₀))..., map(L -> eltype(first(L)), Ls)...,
        interfaces_eltype(cyl.interfaces)
    )
    # The elastic solution, its entries being Volterra matrices.
    return _cylinder_alv_solve(o, E, cyl, L₀, Ls)
end

_cylinder_alv_solve(::Val{4}, ::Type{E}, cyl, L₀, Ls) where {E} = _LC._elastic_solution(E, cyl, L₀, Ls)
_cylinder_alv_solve(::Val{2}, ::Type{E}, cyl, K₀, Ks) where {E} = _LC._conduction_solution(E, cyl, K₀, Ks)

# ── Operators of the cylinder and of its layers ──────────────────────────────
#
#  `layer = nothing` is the whole cylinder (its outer interface included when
#  `external`, the default); `layer = k` is layer k, its material by default,
#  plus the jump across its outer boundary when `external` and across its inner
#  one when `internal`, as in elasticity.

_strain_entries_alv(::Val{4}, sol, cyl, ::Nothing, external, internal) = _LC._whole_strain(sol, external)
_strain_entries_alv(::Val{2}, sol, cyl, ::Nothing, external, internal) = _LC._whole_gradient(sol, external)
function _strain_entries_alv(::Val{4}, sol, cyl, layer::Int, external, internal)
    _LC._check_layer(cyl, layer)
    return _LC._layer_strain(sol, layer, external, internal)
end
function _strain_entries_alv(::Val{2}, sol, cyl, layer::Int, external, internal)
    _LC._check_layer(cyl, layer)
    return _LC._layer_gradient(sol, layer, external, internal)
end

_blocks_alv(c::NamedTuple{(:zz, :ze, :mz, :me, :T, :L)}) = ti_blocks_from_params(c)
_blocks_alv(c::NamedTuple{(:t, :a)}) = ti_order2_blocks_from_params(c)

# The block matrix of an operator of the radial problems, in the canonical frame.
_canonical_blocks(o::Val, cyl::LayeredCylinder, c::NamedTuple, times) = _canonical_frame_blocks(o, cyl, _blocks_alv(c), times)

function _gradient_gradient_concentration_alv(
        o::Val, cyl::LayeredCylinder, ref::_ALVReference, times::AbstractVector{<:Real};
        layer::Union{Nothing, Int} = nothing, external::Bool = layer === nothing,
        internal::Bool = false,
    )
    sol = _cylinder_alv_solution(o, cyl, ref, times)
    return _canonical_blocks(o, cyl, _strain_entries_alv(o, sol, cyl, layer, external, internal), times)
end

_contribution_entries_alv(::Val{4}, sol, external) = _LC._contribution(sol, external)
_contribution_entries_alv(::Val{2}, sol, external) = _LC._conduction_contribution(sol, external)

_property_contribution_alv(
    o::Val, cyl::LayeredCylinder, ref::_ALVReference, times::AbstractVector{<:Real};
    external::Bool = true,
) = _canonical_blocks(o, cyl, _contribution_entries_alv(o, _cylinder_alv_solution(o, cyl, ref, times), external), times)

_flux_entries_alv(::Val{4}, sol, cyl, ::Nothing, external) = _LC._whole_stress(sol, external)
_flux_entries_alv(::Val{2}, sol, cyl, ::Nothing, external) = _LC._whole_flux(sol, external)
_flux_entries_alv(::Val{4}, sol, cyl, layer::Int, external) = (_LC._check_layer(cyl, layer); _LC._layer_stress(sol, layer))
_flux_entries_alv(::Val{2}, sol, cyl, layer::Int, external) = (_LC._check_layer(cyl, layer); _LC._layer_flux(sol, layer))

"""
    flux_gradient_concentration_alv(cyl::LayeredCylinder, ref, times; layer = nothing, external = true) -> Matrix

Volterra block matrix of the average flux of a [`LayeredCylinder`](@ref) per
unit remote gradient history, against the aging reference `ref`: in elasticity
the average stress per unit remote strain, in conduction the average of
``\\boldsymbol k\\cdot\\nabla T`` per unit remote gradient. Without `layer`, the whole
cylinder, with the surface stress (flux) of its membranes (conducting
surfaces), that of the outer one when `external`. With `layer = k`, the
material of layer ``k``: its relaxation operator applied on the left of its
average gradient, ``\\widetilde{\\mathbb C}_k\\circ\\widetilde{\\mathbb A}_k``; `external`
then plays no part, a layer's material carrying no interface.
"""
function flux_gradient_concentration_alv(
        cyl::LayeredCylinder, ref::_ALVReference, times::AbstractVector{<:Real};
        layer::Union{Nothing, Int} = nothing, external::Bool = true,
    )
    o = _alv_order(ref, times)
    return _canonical_blocks(o, cyl, _flux_entries_alv(o, _cylinder_alv_solution(o, cyl, ref, times), cyl, layer, external), times)
end

# The concentration A and the average flux B of the cylinder from one solution.
function _concentration_and_flux_alv(
        o::Val, cyl::LayeredCylinder, ref::_ALVReference, times::AbstractVector{<:Real};
        external::Bool = true,
    )
    sol = _cylinder_alv_solution(o, cyl, ref, times)
    return _canonical_blocks(o, cyl, _strain_entries_alv(o, sol, cyl, nothing, external, false), times),
        _canonical_blocks(o, cyl, _flux_entries_alv(o, sol, cyl, nothing, external), times)
end

# ── Pointwise operators ──────────────────────────────────────────────────────

"""
    LayeredCylinderALVFields(cyl, ref, times)

Pointwise solution of a [`LayeredCylinder`](@ref) against the aging reference
`ref` on the instants `times`: the amplitudes of the radial problems in every
layer and in the matrix, as Volterra matrices. Build it once for many points
and pass it to [`local_gradient_gradient_concentration_alv`](@ref) and
[`local_flux_gradient_concentration_alv`](@ref).
"""
struct LayeredCylinderALVFields{O, C <: LayeredCylinder, S, M, R}
    order::Val{O}
    cylinder::C
    sol::S
    matrix::M
    relaxations::R      # block matrices of the layers, then of the reference
end

function LayeredCylinderALVFields(
        cyl::LayeredCylinder, ref::_ALVReference, times::AbstractVector{<:Real}
    )
    o = _alv_order(ref, times)
    R₀, Rs = _cylinder_alv_blocks(o, cyl, ref, times)
    sol = _cylinder_alv_from_blocks(o, cyl, R₀, Rs)
    # the laws, discretized once, of the layers then of the reference, in the
    # canonical frame where the pointwise operators are written
    relaxations = map(M -> _canonical_frame_blocks(o, cyl, M, times), (Rs..., R₀))
    return LayeredCylinderALVFields(o, cyl, sol, _matrix_amplitudes_alv(o, sol), relaxations)
end

_matrix_amplitudes_alv(::Val{4}, sol) = _LC._matrix_amplitudes(sol)
_matrix_amplitudes_alv(::Val{2}, sol) = _LC._matrix_transport_amplitudes(sol)

# The b × b array of n × n matrices `M` rotated by the b × b matrix `R` of
# numbers, R ∘ M ∘ Rᵀ, written out as a bn × bn time-major block matrix.
function _rotated_blocks(M::AbstractMatrix{<:AbstractMatrix}, R::AbstractMatrix)
    b = size(M, 1)
    n = size(M[1, 1], 1)
    out = zeros(promote_type(eltype(M[1, 1]), eltype(R)), b * n, b * n)
    for J in 1:b, I in 1:b
        blk = view(out, I:b:(b * n), J:b:(b * n))
        for q in 1:b, p in 1:b
            iszero(R[I, p] * R[J, q]) || (blk .+= (R[I, p] * R[J, q]) .* M[p, q])
        end
    end
    return out
end

# 𝔸(x) in the cylindrical frame of the point, as an array of n × n matrices.
_local_blocks_alv(f::LayeredCylinderALVFields{4}, r, k::Int) =
    _LC._local_mandel(_LC._local_strain_entries(f.sol, f.matrix, k, r))

function _local_blocks_alv(f::LayeredCylinderALVFields{2}, r, k::Int)
    amps = k ≤ layer_count(f.cylinder) ? f.sol.transverse.amps[k] : f.matrix
    return _LC._local_conduction_matrix(_LC._two_mode_gradients(r, amps...)...)
end

_frame_rotation(::LayeredCylinderALVFields{4}, Q) = TensND.mandel_rotation(Q)
_frame_rotation(::LayeredCylinderALVFields{2}, Q) = Q

function _local_alv(f::LayeredCylinderALVFields, point, side::Symbol, layer)
    r, _, Q = _LC._cylindrical_frame(f.cylinder, point...)
    k = _LC._resolve(f, r, side, layer)
    return _rotated_blocks(_local_blocks_alv(f, r, k), _frame_rotation(f, Q)), k
end

"""
    local_gradient_gradient_concentration_alv(f::LayeredCylinderALVFields, point...; side = :outer, layer = nothing) -> Matrix
    local_gradient_gradient_concentration_alv(cyl, ref, times, point...; side, layer) -> Matrix

Volterra block matrix (``6n\\times 6n`` in elasticity, ``3n\\times 3n`` in
conduction, canonical frame) of the pointwise concentration of a
[`LayeredCylinder`](@ref): it maps the history of the remote strain (gradient)
to the history of the strain (gradient) at the point, given as a vector or by
its cylindrical coordinates ``(r, \\theta, z)``. `side` and `layer` have the
meaning they have for the elastic
[`local_gradient_gradient_concentration`](@ref MeanFieldHomogenization.LayeredSpheres.local_gradient_gradient_concentration).
"""
local_gradient_gradient_concentration_alv(
    f::LayeredCylinderALVFields, point...; side::Symbol = :outer, layer = nothing
) = first(_local_alv(f, point, side, layer))

"""
    local_flux_gradient_concentration_alv(f::LayeredCylinderALVFields, point...; side = :outer, layer = nothing) -> Matrix
    local_flux_gradient_concentration_alv(cyl, ref, times, point...; side, layer) -> Matrix

Volterra block matrix of the pointwise flux of a [`LayeredCylinder`](@ref) per
unit remote gradient history: the relaxation operator of the region applied on
the left of [`local_gradient_gradient_concentration_alv`](@ref). In conduction
it is the flux ``\\underline q = -\\boldsymbol k\\cdot\\nabla T``, as in the elastic
[`local_flux_gradient_concentration`](@ref MeanFieldHomogenization.LayeredSpheres.local_flux_gradient_concentration).
"""
function local_flux_gradient_concentration_alv(
        f::LayeredCylinderALVFields, point...; side::Symbol = :outer, layer = nothing
    )
    A, k = _local_alv(f, point, side, layer)
    return _flux_sign(f.order) * (f.relaxations[k] * A)
end

# The pointwise flux: the stress in elasticity, q = −k·∇T in conduction.
_flux_sign(::Val{4}) = 1
_flux_sign(::Val{2}) = -1

for name in (:local_gradient_gradient_concentration_alv, :local_flux_gradient_concentration_alv)
    @eval $name(
        cyl::LayeredCylinder, ref::_ALVReference, times::AbstractVector{<:Real}, point...; kw...
    ) = $name(LayeredCylinderALVFields(cyl, ref, times), point...; kw...)
end
