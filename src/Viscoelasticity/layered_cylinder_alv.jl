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
#  The block matrices of this pipeline are written in the canonical frame, so
#  the axis of the cylinder must be e₃, and each layer and the reference
#  isotropic or transversely isotropic about e₃.
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
    norm(M - rebuilt) ≤ _LC._TI_RTOL * norm(M) || throw(
        ArgumentError(
            "$who of the LayeredCylinder is not transversely isotropic about e₃: in aging " *
                "viscoelasticity the cylinder, its layers and the reference must be isotropic " *
                "or transversely isotropic about e₃"
        )
    )
    return nothing
end

function _check_canonical_axis(cyl::LayeredCylinder)
    TensND.isparallel(cyl.axis, (0, 0, 1)) || throw(
        ArgumentError(
            "the ALV kernels of a LayeredCylinder are written in the canonical frame: its " *
                "axis must be e₃; got $(cyl.axis)"
        )
    )
    return nothing
end

_finite_blocks(c::NamedTuple, who::AbstractString) =
    any(M -> any(_nonfinite, M), values(c)) && throw(
    ArgumentError("$who of the LayeredCylinder has an infinite modulus, which has no Volterra matrix")
)

# Hill moduli (n × n matrices) of a 6n × 6n transversely isotropic operator.
function _hill_volterra(M::AbstractMatrix, who::AbstractString)
    c = ti_params_from_blocks(M)
    _check_ti_blocks(M, ti_blocks_from_params(c), who)
    _finite_blocks(c, who)
    return (; k = c.me ./ 2, l = c.mz, m = c.T ./ 2, n = c.zz, p = c.L ./ 2, lt = c.ze ./ 2)
end

function _conductivity_volterra(M::AbstractMatrix, who::AbstractString)
    c = ti_order2_params_from_blocks(M)
    _check_ti_blocks(M, ti_order2_blocks_from_params(c), who)
    _finite_blocks(c, who)
    return c
end

# The block matrix of a layer property (a tensor, a law) or of a reference (a
# law, or the matrix of a running estimate), `b` rows per instant.
_alv_blocks(P::AbstractMatrix, times, b::Int) = P
# An order-2 tensor of TensND is an `AbstractMatrix` too, but an elastic property.
_alv_blocks(P::TensND.AbstractTens{2}, times, b::Int) = _trapezoidal_relaxation(heaviside_law(P), times, b)
_alv_blocks(P, times, b::Int) = _trapezoidal_relaxation(_as_visco_law(P), times, b)

# All the matrices of a solution in one element type (a dual in one layer).
_common_matrices(E::Type, c::NamedTuple) = map(M -> convert(Matrix{E}, M), c)

"""
    _cylinder_alv_solution(Val(order), cyl, ref, times) -> NamedTuple

The radial problems of `cyl` against the aging reference `ref` on the instants
`times`, with ``n\\times n`` Volterra matrices as moduli: the solution of
`LayeredCylinders._elastic_solution` (order 4) or `_conduction_solution`
(order 2), whose entries are the matrices mapping the history of the remote
loading to the history of each state.
"""
function _cylinder_alv_solution(
        ::Val{4}, cyl::LayeredCylinder{T, N}, ref::_ALVReference, times::AbstractVector{<:Real}
    ) where {T, N}
    _check_canonical_axis(cyl)
    L₀ = _hill_volterra(_alv_blocks(ref, times, 6), "the reference medium")
    Ls = ntuple(k -> _hill_volterra(_alv_blocks(layer_modulus(cyl, k), times, 6), "layer $k"), Val(N))
    E = promote_type(
        T, eltype(L₀.k), map(L -> eltype(L.k), Ls)..., interfaces_eltype(cyl.interfaces)
    )
    radii = map(E, cyl.radii)
    L₀ = _common_matrices(E, L₀)
    Ls = map(L -> _common_matrices(E, L), Ls)
    itfs = cyl.interfaces
    dilatation, axial = _LC._solve_axisymmetric(radii, Ls, itfs, L₀)
    shear = _LC._solve_shear(radii, Ls, itfs, L₀)
    antiplane = _LC._solve_potential(_LC.Antiplane(), radii, map(L -> L.p, Ls), itfs, L₀.p)
    fractions = ntuple(k -> E(layer_volume_fraction(cyl, k)), Val(N))
    return (; radii, Ls, L₀, fractions, interfaces = itfs, dilatation, axial, shear, antiplane)
end

function _cylinder_alv_solution(
        ::Val{2}, cyl::LayeredCylinder{T, N}, ref::_ALVReference, times::AbstractVector{<:Real}
    ) where {T, N}
    _check_canonical_axis(cyl)
    K₀ = _conductivity_volterra(_alv_blocks(ref, times, 3), "the reference medium")
    Ks = ntuple(k -> _conductivity_volterra(_alv_blocks(layer_modulus(cyl, k), times, 3), "layer $k"), Val(N))
    E = promote_type(
        T, eltype(K₀.t), map(K -> eltype(K.t), Ks)..., interfaces_eltype(cyl.interfaces)
    )
    radii = map(E, cyl.radii)
    K₀ = _common_matrices(E, K₀)
    Ks = map(K -> _common_matrices(E, K), Ks)
    transverse = _LC._solve_potential(
        _LC.TransverseConduction(), radii, map(K -> K.t, Ks), cyl.interfaces, K₀.t
    )
    fractions = ntuple(k -> E(layer_volume_fraction(cyl, k)), Val(N))
    return (; radii, Ks, K₀, fractions, interfaces = cyl.interfaces, transverse)
end

# ── Whole-cylinder operators ─────────────────────────────────────────────────

_gradient_gradient_concentration_alv(
    o::Val{4}, cyl::LayeredCylinder, ref::_ALVReference, times::AbstractVector{<:Real};
    external::Bool = true,
) = ti_blocks_from_params(_LC._whole_strain(_cylinder_alv_solution(o, cyl, ref, times), external))

_property_contribution_alv(
    o::Val{4}, cyl::LayeredCylinder, ref::_ALVReference, times::AbstractVector{<:Real};
    external::Bool = true,
) = ti_blocks_from_params(_LC._contribution(_cylinder_alv_solution(o, cyl, ref, times), external))

_gradient_gradient_concentration_alv(
    o::Val{2}, cyl::LayeredCylinder, ref::_ALVReference, times::AbstractVector{<:Real};
    external::Bool = true,
) = ti_order2_blocks_from_params(
    _LC._whole_gradient(_cylinder_alv_solution(o, cyl, ref, times), external)
)

_property_contribution_alv(
    o::Val{2}, cyl::LayeredCylinder, ref::_ALVReference, times::AbstractVector{<:Real};
    external::Bool = true,
) = ti_order2_blocks_from_params(
    _LC._conduction_contribution(_cylinder_alv_solution(o, cyl, ref, times), external)
)

# The concentration A and the average flux B of the cylinder from one solution.
function _concentration_and_flux_alv(
        o::Val{4}, cyl::LayeredCylinder, ref::_ALVReference, times::AbstractVector{<:Real};
        external::Bool = true,
    )
    sol = _cylinder_alv_solution(o, cyl, ref, times)
    return ti_blocks_from_params(_LC._whole_strain(sol, external)),
        ti_blocks_from_params(_LC._whole_stress(sol, external))
end

function _concentration_and_flux_alv(
        o::Val{2}, cyl::LayeredCylinder, ref::_ALVReference, times::AbstractVector{<:Real};
        external::Bool = true,
    )
    sol = _cylinder_alv_solution(o, cyl, ref, times)
    return ti_order2_blocks_from_params(_LC._whole_gradient(sol, external)),
        ti_order2_blocks_from_params(_LC._whole_flux(sol, external))
end
