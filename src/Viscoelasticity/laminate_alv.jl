# =============================================================================
#  laminate_alv.jl — the periodic multilayer in ageing linear viscoelasticity.
#
#  The laminate solution is pure algebra: products of Kelvin-Mandel matrices
#  and ONE inversion restricted to the out-of-plane subspace. Replacing each
#  scalar by a discretized Volterra operator therefore turns the elastic
#  kernel into the ALV one WITHOUT rewriting the physics — the very same
#  `Core.laminate_stiffness` runs, with its `opinv` / `opinv_avg` callables
#  swapped for the block-Volterra inversion. Julia specializes on the function
#  type, so there is no abstraction cost and only one place where a physics
#  bug could live.
#
#  ── Layout ────────────────────────────────────────────────────────────────
#  `trapezoidal_matrix` returns a `(6n × 6n)` matrix laid out as `n × n` time
#  blocks of `6 × 6` Kelvin-Mandel (order 4), or `(3n × 3n)` with `3 × 3`
#  blocks (order 2). The out-of-plane restriction therefore picks Mandel slots
#  `(3, 4, 5)` **inside each time block**, giving a `(3n × 3n)` matrix in the
#  same time-major layout, which `volterra_inverse(…; block_size = 3)`
#  inverts. In transport the out-of-plane subspace is one-dimensional and the
#  inversion is `block_size = 1`.
# =============================================================================

"""
    _alv_op_indices(n_times, nc, slots) -> Vector{Int}

Row/column indices selecting `slots` (Kelvin-Mandel components) inside every
time block of an ALV matrix with `nc` components per block. Produces the
out-of-plane restriction in the same time-major order as the input, so the
result is again a valid Volterra matrix.
"""
function _alv_op_indices(n_times::Int, nc::Int, slots)
    idx = Vector{Int}(undef, n_times * length(slots))
    k = 0
    @inbounds for t in 0:(n_times - 1), s in slots
        k += 1
        idx[k] = nc * t + s
    end
    return idx
end

"""
    _plane_pinv_alv(M, n_times, nc, slots, bs) -> Matrix

Volterra counterpart of [`Core.plane_pinv`](@ref): restrict `M` to the
out-of-plane subspace of every time block, invert the restriction as a
block-lower-triangular Volterra operator, and embed the result back with
zeros elsewhere.

The elastic version inverts a 3×3 (resp. 1×1) block in closed form; here the
same restriction is a `(3n × 3n)` (resp. `(n × n)`) Volterra operator, and
`volterra_inverse` plays the role of `_inv3`.
"""
function _plane_pinv_alv(M::AbstractMatrix, n_times::Int, nc::Int, slots, bs::Int)
    idx = _alv_op_indices(n_times, nc, slots)
    Mop = M[idx, idx]
    iMop = volterra_inverse(Mop; block_size = bs)
    out = zeros(eltype(iMop), size(M))
    @inbounds out[idx, idx] = iMop
    return out
end

"""
    _alv_rotate_blocks(M, Q, n_times, nc) -> Matrix

Apply a Kelvin-Mandel (Bond) rotation `Q` to every time block of an ALV
matrix, i.e. express the whole Volterra operator in the layer frame. A no-op
when the laminate normal is the third canonical axis, which is why the
canonical laminate pays nothing.
"""
function _alv_rotate_blocks(M::AbstractMatrix, Q::AbstractMatrix, n_times::Int, nc::Int)
    out = similar(M)
    @inbounds for i in 0:(n_times - 1), j in 0:(n_times - 1)
        rows = (nc * i + 1):(nc * i + nc)
        cols = (nc * j + 1):(nc * j + nc)
        out[rows, cols] = Q' * M[rows, cols] * Q
    end
    return out
end

# Scalar Heaviside weights of the trapezoidal discretization: the `n × n`
# Volterra matrix of a UNIT constant (elastic) kernel. A constant property `A`
# discretizes to the time blocks `W[i,j] · A`, which is how an elastic
# interface compliance enters an otherwise viscoelastic laminate.
_alv_heaviside_weights(times) = trapezoidal_matrix(heaviside_law(1.0), times)

"""
    _alv_kron_time(W, B) -> Matrix

Assemble the ALV matrix of a **constant** (elastic) block `B` from the scalar
Heaviside weights `W`: time block `(i, j)` is `W[i,j] · B`.
"""
function _alv_kron_time(W::AbstractMatrix, B::AbstractMatrix)
    n = size(W, 1)
    nc = size(B, 1)
    T = promote_type(eltype(W), eltype(B))
    out = zeros(T, n * nc, n * nc)
    @inbounds for i in 1:n, j in 1:i
        out[(nc * (i - 1) + 1):(nc * i), (nc * (j - 1) + 1):(nc * j)] = W[i, j] * B
    end
    return out
end

# The ALV kernel is numerical by construction: a trapezoidal Volterra
# discretization on a grid of times, assembled into dense `zeros(Float64, …)`
# blocks. A symbolic frame therefore cannot be honored — say so, rather than
# letting `_basis_matrix` fail on `Float64(::Sym)` several frames deeper.
function _alv_check_numeric_frame(basis)
    return is_hard_numeric(eltype(basis)) || throw(
        ArgumentError(
            "laminate_alv: the ageing-viscoelastic kernel is a numerical " *
                "Volterra discretization and needs a numeric frame; got a " *
                "$(eltype(basis)) basis. Evaluate the frame first, or use the " *
                "elastic `Laminated` scheme, which is symbolic end to end."
        )
    )
end

"""
    laminate_alv(lam, ::Val{order}; times, property) -> Matrix

Effective ageing-viscoelastic operator of a periodic multilayer cell, as a
`(6n × 6n)` relaxation matrix (`order = 4`) or a `(3n × 3n)` one
(`order = 2`), with `n = length(times)`.

Each layer carries a [`ViscoLaw`](@ref) under `property`; the interfaces stay
elastic (their compliances are numbers), which covers the usual case of an
ageing bulk with a time-independent interface. Reached through
`homogenize_alv(lam, Laminated(), :C; times = …)`.
"""
function laminate_alv(lam::Laminates.Laminate, ::Val{4}; times, property::Symbol = :C)
    names = Laminates.layer_names(lam)
    n = length(times)
    basis = Laminates.laminate_basis(lam)
    _alv_check_numeric_frame(basis)
    rotate = !(basis isa TensND.CanonicalBasis)
    Q = rotate ? MFH_Core._bond6(MFH_Core._basis_matrix(basis)) : nothing

    Cs = map(names) do nm
        law = Laminates.layer_property(lam, nm, property)
        law isa ViscoLaw || throw(
            ArgumentError(
                "laminate_alv: layer :$(nm) property :$(property) is not a ViscoLaw"
            )
        )
        M = _trapezoidal_relaxation(law, times, 6)
        return rotate ? _alv_rotate_blocks(M, Q, n, 6) : M
    end
    fs = [Laminates.layer_volume_fraction(lam, nm) for nm in names]

    opinv = M -> _plane_pinv_alv(M, n, 6, (3, 4, 5), 3)
    P_int, C_surf = _alv_interface_terms(lam, times, Val(4))
    return MFH_Core.laminate_stiffness(
        Cs, fs, P_int, C_surf; opinv = opinv, opinv_avg = opinv
    )
end

function laminate_alv(lam::Laminates.Laminate, ::Val{2}; times, property::Symbol = :K)
    names = Laminates.layer_names(lam)
    n = length(times)
    basis = Laminates.laminate_basis(lam)
    basis isa TensND.CanonicalBasis || throw(
        ArgumentError(
            "laminate_alv at order 2 currently needs a canonical frame " *
                "(normal = e₃); rotate the layer properties instead"
        )
    )

    Ks = map(names) do nm
        law = Laminates.layer_property(lam, nm, property)
        law isa ViscoLaw || throw(
            ArgumentError(
                "laminate_alv: layer :$(nm) property :$(property) is not a ViscoLaw"
            )
        )
        return _trapezoidal_relaxation(law, times, 3)
    end
    fs = [Laminates.layer_volume_fraction(lam, nm) for nm in names]

    opinv = M -> _plane_pinv_alv(M, n, 3, (3,), 1)
    P_int, K_surf = _alv_interface_terms(lam, times, Val(2))
    return MFH_Core.laminate_stiffness(
        Ks, fs, P_int, K_surf; opinv = opinv, opinv_avg = opinv
    )
end

# Interface terms, lifted to Volterra operators. The interfaces are elastic,
# so each is a constant block times the scalar Heaviside weights.
function _alv_interface_terms(lam::Laminates.Laminate, times, ::Val{order}) where {order}
    nc = order == 4 ? 6 : 3
    n = length(times)
    Z = zeros(Float64, n * nc, n * nc)
    all(itf -> itf isa PerfectInterface, lam.interfaces) && return (Z, Z)
    P, S = Laminates._interface_terms(lam, Float64, Val(order))
    W = _alv_heaviside_weights(times)
    return (_alv_kron_time(W, Matrix(P)), _alv_kron_time(W, Matrix(S)))
end

# ── Public dispatch ─────────────────────────────────────────────────────────

"""
    homogenize_alv(lam::Laminate, ::Laminated, prop; times, kw...)

Ageing-viscoelastic effective operator of a periodic multilayer, dispatching
on the order of the layer laws exactly as the elastic `Laminated` scheme
dispatches on the order of the layer tensors.
"""
function homogenize_alv(
        lam::Laminates.Laminate, ::Laminated, prop::Symbol;
        times::AbstractVector{<:Real}, kw...
    )
    law = Laminates.layer_property(lam, first(Laminates.layer_names(lam)), prop)
    law isa ViscoLaw ||
        throw(ArgumentError("homogenize_alv: property :$(prop) is not a ViscoLaw"))
    order = _alv_property_order(law, first(times))
    return laminate_alv(lam, Val(order); times = times, property = prop)
end

# Voigt / Reuss bounds on an ALV laminate — the same free oracles as in the
# elastic case (in-plane Voigt and out-of-plane Reuss are exact).
function homogenize_alv(
        lam::Laminates.Laminate, ::Voigt, prop::Symbol;
        times::AbstractVector{<:Real}, kw...
    )
    names = Laminates.layer_names(lam)
    B = _alv_property_order(Laminates.layer_property(lam, first(names), prop), first(times)) == 4 ? 6 : 3
    Ms = [
        _trapezoidal_relaxation(Laminates.layer_property(lam, nm, prop), times, B)
            for nm in names
    ]
    fs = [Laminates.layer_volume_fraction(lam, nm) for nm in names]
    return sum(fs[i] * Ms[i] for i in eachindex(Ms))
end

function homogenize_alv(
        lam::Laminates.Laminate, ::Reuss, prop::Symbol;
        times::AbstractVector{<:Real}, kw...
    )
    names = Laminates.layer_names(lam)
    B = _alv_property_order(Laminates.layer_property(lam, first(names), prop), first(times)) == 4 ? 6 : 3
    Ss = [
        volterra_inverse(
            _trapezoidal_relaxation(Laminates.layer_property(lam, nm, prop), times, B);
            block_size = B
        ) for nm in names
    ]
    fs = [Laminates.layer_volume_fraction(lam, nm) for nm in names]
    return volterra_inverse(sum(fs[i] * Ss[i] for i in eachindex(Ss)); block_size = B)
end
