# =============================================================================
#  axi_supershape_pore.jl — a superspheroidal cavity, by axisymmetric Fourier
#  elements.
#
#  The same morphology family as `FESupershapePore`, and the same contract, but
#  the axisymmetric member of it — so the solve is two-dimensional. Where the
#  three-dimensional cell divides its cost by eight with an octant, separating
#  the azimuth into Fourier modes divides it by orders of magnitude: three
#  assemblies and eight solves on a triangle mesh of the meridian half-plane.
#
#  ## Notation, because `p` is taken three times in this package
#
#  Here `p` is the **concavity exponent** of the shape, with `m = 2p` the
#  exponent of the level set, as everywhere in `Superspheres`. It is *not* the
#  confocal angular coordinate of `theory/layered_spheroid.md`, nor the mode-1
#  nodal unknown of `axi_fourier.jl`. Coordinates here are cylindrical
#  `(ρ, θ, z)` throughout — the geometry is not confocal and has no use for that
#  chart — and the aspect ratio is written `c/a` in full.
#
#  ## Why it is a *heterogeneous* inclusion
#
#  Same reason as the three-dimensional pore: `is_homogeneous_inclusion` asks
#  whether a single `ℂ₁` describes the interior, and for a cavity `inv(ℂ₁)` is
#  meaningless. The package's exact identities then collapse to `ℕ = -ℂ₀:𝔸` and
#  `ℍ = 𝔸:𝕊₀`, with the stress side identically zero.
# =============================================================================

"""
    AxiSupershapePoreShape

Shape trait of [`FEAxiSupershapePore`](@ref). No kernel dispatches on it — the
inclusion supplies its own localization tensors.
"""
struct AxiSupershapePoreShape end

"""
    FEAxiSupershapePore(shape; opts, basis, euler_angles, backend,
                        elastic = nothing, transport = nothing, guard = :warn)

A **superspheroidal cavity** solved by axisymmetric Fourier finite elements.

`shape` is a [`Superspheroid`](@ref MeanFieldHomogenization.Superspheres.Superspheroid);
its axis of revolution is the third axis of `basis`. The answer is transversely
isotropic — five constants in elasticity, two in transport — which is exactly
what the Fourier modes deliver: mode 0 gives a `2×2` block, modes 1 and 2 a
scalar each.

`elastic` and `transport` take a trained surrogate in place of the solve, one
per physics, exactly as for [`FESupershapePore`](@ref). That is the only route
to a sensitivity with respect to the morphology: the finite-element solve runs
in `Float64` and memoizes on the reference medium alone, so a derivative in `p`
would come back a silent zero.

`radius_ratio` multiplies the shape's **bounding radius**, not `a` — for an
elongated superspheroid the two differ by the aspect ratio, and using `a` lets
the outer boundary come within `1.2c` of the body.
"""
struct FEAxiSupershapePore{T <: Number, S <: Superspheroid{T}, B <: TensND.AbstractBasis} <:
    Core.AbstractCustomInclusion{T}
    shape::S
    basis::B
    mesh::FEAxiMeshOptions
    cache::FECache
    backend::FEBackend
    # Typed `Any` for the same reason as in `supershape_pore.jl`:
    # `NeuralInclusions` is included after this module.
    elastic::Any
    transport::Any
    guard::Symbol
end

function FEAxiSupershapePore(
        shape::Superspheroid{T};
        opts::FEAxiMeshOptions = FEAxiMeshOptions(),
        basis::Union{Nothing, TensND.AbstractBasis} = nothing,
        euler_angles::Tuple{Vararg{Real}} = (),
        backend::FEBackend = AutoBackend(),
        elastic = nothing,
        transport = nothing,
        guard::Symbol = :warn,
    ) where {T}
    guard in (:warn, :error, :none) ||
        throw(ArgumentError("`guard` must be :warn, :error or :none, got :$guard"))
    _check_pore_surrogate(elastic, 4, :elastic, shape)
    _check_pore_surrogate(transport, 2, :transport, shape)
    bas = basis === nothing ? Core._default_basis(Float64, euler_angles) : basis
    return FEAxiSupershapePore{T, typeof(shape), typeof(bas)}(
        shape, bas, opts, FECache(), backend, elastic, transport, guard
    )
end

Base.show(io::IO, s::FEAxiSupershapePore) =
    print(io, "FEAxiSupershapePore(", s.shape, ")")

# ─── Level 0 of the contract ─────────────────────────────────────────────────

Core.dimension(::FEAxiSupershapePore) = 3
Core.inclusion_basis(s::FEAxiSupershapePore) = s.basis
Core.shape_trait(::FEAxiSupershapePore) = AxiSupershapePoreShape()
Core.is_homogeneous_inclusion(::FEAxiSupershapePore) = false
_fe_cache(s::FEAxiSupershapePore) = s.cache

function Core.shape_tensor(s::FEAxiSupershapePore{T}) where {T}
    D = zeros(T, 3, 3)
    D[1, 1] = D[2, 2] = s.shape.a
    D[3, 3] = s.shape.c
    return TensND.Tens(D, s.basis)
end

_axi_pore_outer_radius(s::FEAxiSupershapePore) =
    s.mesh.radius_ratio * Float64(bounding_radius(s.shape))

"Columns of `_AXI_Q` spanning Fourier mode `m`: two for mode 0, one each after."
_axi_mode_cols(m::Int) = m == 0 ? (1:2) : m == 1 ? (3:3) : (5:5)

# ─── The two solves ──────────────────────────────────────────────────────────

"""
    _axi_pore_run_elastic(pore, C₀) -> NamedTuple

The corrected solve in elasticity: three modes, the modal fixed point, and the
transversely isotropic reassembly into the global frame. See the driver's header
for why `𝔽 = -P₀` carries no `V_D`.
"""
function _axi_pore_run_elastic(
        s::FEAxiSupershapePore, C₀::TensND.AbstractTens{4, 3}
    )
    μ, ν = _fe_iso_moduli(C₀; what = "`FEAxiSupershapePore`")
    s.elastic === nothing || return _pore_surrogate_response(s.elastic, s, C₀)
    R = _fe_frame(s)
    C0_66 = Core.mandel66_minor(_to_local4(C₀, R))
    Dmap = [AXI_SET_MATRIX => C0_66]
    backend = _axi_setup(s).backend
    V_D = Float64(shape_volume(s.shape))

    blocks = map((0, 1, 2)) do m
        mode = _axi_mode_setup(s, :elasticity, m)
        nload = m == 0 ? 2 : 1
        L_s, L_u, _Vm = _axi_pore_solve_mode(
            backend, mode, Dmap,
            (N, dNρ, dNz, ρ) -> _axi_B_elast(m, N, dNρ, dNz, ρ),
            v -> _axi_project(m, v), nload,
            j -> (x -> _axi_bc_affine(m, j, x[1], x[2])),
            j -> (x -> _axi_bc_dipole(m, j, x[1], x[2], μ, ν, V_D)),
            _axi_dof_map(m), V_D,
        )
        Qm = _AXI_Q[:, _axi_mode_cols(m)]
        P0 = Qm' * C0_66 * Qm
        Z = zeros(nload, nload)
        # `𝔹ᴱ = 𝔹ᵖ = 0` for a cavity, so this degenerates *identically* into
        # `(𝕀 - 𝕃_u 𝔽)⁻¹ 𝕃_s` with `𝔽 = -P₀`. Reused rather than rewritten.
        A, _B, X = _axi_correct(L_s, Z, L_u, Z, P0)
        (; m, L_s, L_u, A, X)
    end

    A66 = _axi_assemble_66(blocks[1].A, blocks[2].A[1, 1], blocks[3].A[1, 1])
    A66_E = _axi_assemble_66(
        blocks[1].L_s, blocks[2].L_s[1, 1], blocks[3].L_s[1, 1]
    )
    s.cache.assemblies += 1
    return (;
        A = _from_local66(A66, R),
        A_uncorrected = _from_local66(A66_E, R),
        blocks, volume = V_D,
        dipole_norm = maximum(_axi_pore_dipole_norm(b, C0_66) for b in blocks),
    )
end

"""
    _axi_pore_dipole_norm(block, C0_66) -> Float64

`‖𝕃_u 𝔽‖` on one modal block — the diagnostic to watch rather than the answer.
It is `O((a/R)³)`, so a log-log slope of `-3` against `R` says the correction is
wired right, and the *corrected* answer falling faster than that says the sign
is right too.
"""
function _axi_pore_dipole_norm(block, C0_66::AbstractMatrix)
    Qm = _AXI_Q[:, _axi_mode_cols(block.m)]
    return LinearAlgebra.opnorm(block.L_u * (Qm' * C0_66 * Qm))
end

"""
    _axi_pore_run_cond(pore, K₀) -> NamedTuple

The corrected solve in transport: modes 0 and 1, giving `R₃₃` and `R₁₁`.
"""
function _axi_pore_run_cond(s::FEAxiSupershapePore, K₀::TensND.AbstractTens{2, 3})
    k₀ = _fe_iso_scalar(K₀; what = "`FEAxiSupershapePore`")
    s.transport === nothing || return _pore_surrogate_response(s.transport, s, K₀)
    R = _fe_frame(s)
    Dmap = [AXI_SET_MATRIX => _to_local2(K₀, R)]
    backend = _axi_setup(s).backend
    V_D = Float64(shape_volume(s.shape))

    blocks = map((0, 1)) do m
        mode = _axi_mode_setup(s, :conduction, m)
        L_s, L_u, _Vm = _axi_pore_solve_mode(
            backend, mode, Dmap,
            (N, dNρ, dNz, ρ) -> _axi_B_cond(m, N, dNρ, dNz, ρ),
            g -> _axi_project_cond(m, g), 1,
            _j -> (x -> (_axi_bc_affine_cond(m, x[1], x[2]),)),
            _j -> (x -> (_axi_bc_dipole_cond(m, x[1], x[2], k₀, V_D),)),
            # Transport solves one scalar, so the dof map is the identity.
            ones(3, 1), V_D,
        )
        Z = zeros(1, 1)
        A, _B, X = _axi_correct(L_s, Z, L_u, Z, fill(k₀, 1, 1))
        (; m, L_s, L_u, A, X)
    end

    diag3(t, a) = [t 0.0 0.0; 0.0 t 0.0; 0.0 0.0 a]
    A33 = diag3(blocks[2].A[1, 1], blocks[1].A[1, 1])
    A33_E = diag3(blocks[2].L_s[1, 1], blocks[1].L_s[1, 1])
    s.cache.assemblies += 1
    return (;
        A = _from_local33(A33, R),
        A_uncorrected = _from_local33(A33_E, R),
        blocks, volume = V_D,
        dipole_norm = maximum(abs(b.L_u[1, 1] * k₀) for b in blocks),
    )
end

_axi_pore_run(s::FEAxiSupershapePore, P₀::TensND.AbstractTens{4, 3}) =
    _axi_pore_run_elastic(s, P₀)
_axi_pore_run(s::FEAxiSupershapePore, P₀::TensND.AbstractTens{2, 3}) =
    _axi_pore_run_cond(s, P₀)

"""
    fe_axi_pore_localization(pore, P₀) -> Tens

The cavity's localization tensor, memoized on the reference medium expressed in
the pore's own frame — so a whole family of *orientations* of the same shape in
the same matrix shares one solve.
"""
function fe_axi_pore_localization(s::FEAxiSupershapePore, P₀::TensND.AbstractTens{O, 3}) where {O}
    # No `.A` here: with a surrogate attached, `_axi_pore_run_*` returns the
    # tensor itself rather than the solve's named tuple — the same contract as
    # `fe_cell_localization` in the three-dimensional pore. Asking for `.A`
    # made every shipped axisymmetric model unusable.
    has_surrogate(s, O) && return _axi_pore_run(s, P₀)
    r = get!(() -> _axi_pore_run(s, P₀), s.cache.tensors, _fe_cache_key(P₀))
    return r isa TensND.AbstractTens ? r : r.A
end

"Bypass the cache and return every intermediate, including `A_uncorrected`."
fe_axi_pore_breakdown(s::FEAxiSupershapePore, P₀::TensND.AbstractTens) =
    _axi_pore_run(s, P₀)

# ─── Gate B, with an exactly zero stress side ────────────────────────────────

Core.strain_strain_loc(
    s::FEAxiSupershapePore, ::TensND.AbstractTens{4, 3},
    C₀::TensND.AbstractTens{4, 3}; kw...
) = fe_axi_pore_localization(s, C₀)

Core.gradient_gradient_loc(
    s::FEAxiSupershapePore, ::TensND.AbstractTens{2, 3},
    K₀::TensND.AbstractTens{2, 3}; kw...
) = fe_axi_pore_localization(s, K₀)

"""
    Core.stress_strain_loc(pore, C₁, C₀)
    Core.flux_gradient_loc(pore, K₁, K₀)

**Exactly zero**, as for [`FESupershapePore`](@ref): a cavity carries no stress
and no flux, not to the discretization error but by definition.
"""
Core.stress_strain_loc(
    ::FEAxiSupershapePore, ::TensND.AbstractTens{4, 3},
    ::TensND.AbstractTens{4, 3}; kw...
) = TensND.TensCanonical(Tensors.Tensor{4, 3}((i, j, k, l) -> 0.0))

Core.flux_gradient_loc(
    ::FEAxiSupershapePore, ::TensND.AbstractTens{2, 3},
    ::TensND.AbstractTens{2, 3}; kw...
) = TensND.TensCanonical(Tensors.Tensor{2, 3}((i, j) -> 0.0))

# ─── Surrogate seam, shape parameters, sensitivity ───────────────────────────

has_surrogate(s::FEAxiSupershapePore, order::Integer) =
    (order == 4 ? s.elastic : s.transport) !== nothing

pore_shape_params(s::FEAxiSupershapePore) = pore_shape_params(s.shape)

function Schemes._geom_field(g::FEAxiSupershapePore, name::Symbol)
    ps = pore_shape_params(g)
    haskey(ps, name) || throw(
        ArgumentError(
            "`FEAxiSupershapePore` has no shape parameter :$name; it has " *
                "$(join(keys(ps), ", "))"
        )
    )
    return getproperty(ps, name)
end

function Schemes._replace_geom_field(
        g::FEAxiSupershapePore, ::Val{name}, ::Nothing, value
    ) where {name}
    (has_surrogate(g, 4) || has_surrogate(g, 2)) || _no_fe_sensitivity(g, name)
    return FEAxiSupershapePore(
        _rebuild_pore_shape(g.shape, Val(name), value);
        opts = g.mesh, basis = g.basis, backend = g.backend,
        elastic = g.elastic, transport = g.transport, guard = g.guard,
    )
end

Schemes._replace_geom_field(
    g::FEAxiSupershapePore, ::Val{name}, ::Int, _value
) where {name} = throw(
    ArgumentError(
        "`FEAxiSupershapePore` has no indexed geometry field; its parameters " *
            "are $(join(keys(pore_shape_params(g)), ", "))"
    )
)

# ─── Diagnostics ─────────────────────────────────────────────────────────────

"""
    fe_axi_pore_mesh_report(pore) -> NamedTuple

Cell and node counts, and the one comparison that says whether the meridian
spline resolves the profile: the **measured** revolution volume of the matrix
against `V_Ω - V_D` in closed form. The gap is the geometry error, and it is the
number to watch when `p` goes concave.
"""
function fe_axi_pore_mesh_report(s::FEAxiSupershapePore)
    st = _axi_setup(s)
    counts = fe_axi_grid_counts(st.backend, st.grid)
    R = _axi_pore_outer_radius(s)
    V_Ω = 4π * R^3 / 3
    V_D = Float64(shape_volume(s.shape))
    v_matrix = fe_axi_region_volume(st.backend, st.grid, AXI_SET_MATRIX)
    return (;
        backend = st.backend, counts.ncells, counts.nnodes,
        R, volume_cavity_exact = V_D, volume_cell_exact = V_Ω,
        volume_matrix = v_matrix, volume_matrix_exact = V_Ω - V_D,
        volume_error = (v_matrix - (V_Ω - V_D)) / (V_Ω - V_D),
    )
end
