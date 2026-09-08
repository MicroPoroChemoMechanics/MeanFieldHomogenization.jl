# =============================================================================
#  axi_driver.jl — the corrected finite Eshelby cell, axisymmetric declination.
#
#  The general 2 × N scheme of Adessina, Barthélémy, Lavergne & Ben Fraj
#  (*Int. J. Eng. Sci.* 119, 2017), one Fourier mode at a time.  For each mode
#  and each Kelvin basis tensor of that mode, two problems are solved on the
#  truncated cell:
#
#    E-problem   u|∂Ω = E·x                        → ⟨ε⟩ = 𝔸ᴱ:E, ⟨σ⟩ = 𝔹ᴱ:E
#    p-problem   u|∂Ω = +∇G(x):(V_D P)             → ⟨ε⟩ = 𝔸ᵖ:P, ⟨σ⟩ = 𝔹ᵖ:P
#
#  The polarization of the *infinite*-medium problem is its own consequence,
#
#      P = ⟨σ − ℂ₀:ε⟩_D = (𝔹ᴱ − ℂ₀:𝔸ᴱ):E + (𝔹ᵖ − ℂ₀:𝔸ᵖ):P ,
#
#  a linear fixed point solved in closed form by `_axi_correct`.
#
#  Transport is the same algebra with `ε ↦ ∇T`, `σ ↦ k∇T`, `ℂ₀ ↦ k₀` and the
#  scalar dipole field `T = M·x/(4π k₀ r³)`.
#
#  Cost: three assemblies (modes 0, 1, 2) and eight solves in *two* dimensions,
#  where the three-dimensional formulation of the paper needs six solves on a
#  mesh two orders of magnitude larger.
#
#  Nothing here knows which finite-element library is at work: every call into
#  the discretization goes through the nine generics of `backends.jl`.
# =============================================================================

"""
Whole discretization of an axisymmetric inclusion: the resolved backend, its
grid and its modes.

The backend is pinned here at the first solve rather than read afresh each
time — the grid and the mode spaces are backend-native objects, so a single
inclusion must never mix two of them.
"""
struct AxiSetup{B, G}
    backend::B
    grid::G
    modes::Dict{Tuple{Symbol, Int}, Any}
end

function _axi_setup(incl::FEExcenteredSphere)
    if incl.cache.setup === nothing
        b = _resolve_backend(incl.backend)
        incl.cache.setup = AxiSetup(b, fe_axi_grid(b, incl), Dict{Tuple{Symbol, Int}, Any}())
    end
    return incl.cache.setup
end

"Number of scalar unknowns, and the components the axis pins to zero, per mode."
_axi_mode_shape(physics::Symbol, m::Int) = physics === :elasticity ?
    (_axi_ncomp(m), _axi_axis_zeros(m)) : (1, m == 0 ? () : (1,))

function _axi_mode_setup(incl::FEExcenteredSphere, physics::Symbol, m::Int)
    s = _axi_setup(incl)
    return get!(s.modes, (physics, m)) do
        ncomp, axis_zeros = _axi_mode_shape(physics, m)
        fe_axi_mode(s.backend, s.grid, incl.mesh.order, ncomp, axis_zeros)
    end
end

# ─── One mode, both problems ─────────────────────────────────────────────────

"""
    _axi_solve_mode(backend, mode, Dmap, Bop, proj, nload, bc_E, bc_p)
        -> (𝔸ᴱ, 𝔹ᴱ, 𝔸ᵖ, 𝔹ᵖ, V)

Assemble one mode, factorize **once**, and run the `nload` affine problems
followed by the `nload` dipole problems. `bc_E(j)` and `bc_p(j, V)` return the
boundary data of load `j` as a function of the boundary point.

The single factorization is the reason the Dirichlet data is applied by
[`fe_axi_set_dirichlet!`](@ref) into a plain vector rather than baked into the
operator: only the right-hand side changes from one load to the next.
"""
function _axi_solve_mode(backend, mode, Dmap, Bop, proj, nload, bc_E, bc_p)
    K = fe_axi_stiffness(backend, mode, Dmap, Bop)
    ndofs, free, presc = fe_axi_dof_split(backend, mode)
    F = LinearAlgebra.lu(K[free, free])
    Kfp = K[free, presc]
    sets = (AXI_SET_CORE, AXI_SET_SHELL)

    function solve_with(f)
        u = zeros(ndofs)
        fe_axi_set_dirichlet!(backend, mode, u, f)
        u[free] .= F \ Vector(-Kfp * u[presc])
        return u
    end

    A_E = zeros(nload, nload)
    B_E = zeros(nload, nload)
    V = 0.0
    for j in 1:nload
        e, s, V = fe_axi_average(
            backend, mode, Dmap, solve_with(bc_E(j)), Bop, proj, sets
        )
        A_E[:, j] .= e
        B_E[:, j] .= s
    end

    A_p = zeros(nload, nload)
    B_p = zeros(nload, nload)
    for j in 1:nload
        e, s, _ = fe_axi_average(
            backend, mode, Dmap, solve_with(bc_p(j, V)), Bop, proj, sets
        )
        A_p[:, j] .= e
        B_p[:, j] .= s
    end
    return A_E, B_E, A_p, B_p, V
end

# ─── The two physics ─────────────────────────────────────────────────────────

"""
    _axi_run_elastic(incl, C₀) -> NamedTuple

Full corrected solve in elasticity: the three modes, the fixed point, and the
two localization tensors reassembled in the global frame.
"""
function _axi_run_elastic(incl::FEExcenteredSphere, C₀::TensND.AbstractTens{4, 3})
    μ, ν = _fe_iso_moduli(C₀; what = "`FEExcenteredSphere`")
    R = _fe_frame(incl)
    C0_66 = Core.mandel66_minor(_to_local4(C₀, R))
    Dmap = [
        AXI_SET_CORE => _axi_check_ti(
            Core.mandel66_minor(_to_local4(incl.props[1], R)), "the core"
        ),
        AXI_SET_SHELL => _axi_check_ti(
            Core.mandel66_minor(_to_local4(incl.props[2], R)), "the shell"
        ),
        AXI_SET_MATRIX => C0_66,
    ]
    backend = _axi_setup(incl).backend

    blocks = map((0, 1, 2)) do m
        mode = _axi_mode_setup(incl, :elasticity, m)
        nload = m == 0 ? 2 : 1
        A_E, B_E, A_p, B_p, V = _axi_solve_mode(
            backend, mode, Dmap,
            (N, dNρ, dNz, ρ) -> _axi_B_elast(m, N, dNρ, dNz, ρ),
            v -> _axi_project(m, v), nload,
            j -> (x -> _axi_bc_affine(m, j, x[1], x[2])),
            (j, V) -> (x -> _axi_bc_dipole(m, j, x[1], x[2], μ, ν, V)),
        )
        cols = m == 0 ? (1:2) : m == 1 ? (3:3) : (5:5)
        Qm = _AXI_Q[:, cols]
        A, B, X = _axi_correct(A_E, B_E, A_p, B_p, Qm' * C0_66 * Qm)
        (; m, A_E, B_E, A_p, B_p, A, B, X, V)
    end

    A66 = _axi_assemble_66(blocks[1].A, blocks[2].A[1, 1], blocks[3].A[1, 1])
    B66 = _axi_assemble_66(blocks[1].B, blocks[2].B[1, 1], blocks[3].B[1, 1])
    A66_E = _axi_assemble_66(blocks[1].A_E, blocks[2].A_E[1, 1], blocks[3].A_E[1, 1])
    B66_E = _axi_assemble_66(blocks[1].B_E, blocks[2].B_E[1, 1], blocks[3].B_E[1, 1])
    incl.cache.assemblies += 1
    return (;
        A = _from_local66(A66, R), B = _from_local66(B66, R),
        A_uncorrected = _from_local66(A66_E, R),
        B_uncorrected = _from_local66(B66_E, R),
        blocks, volume = blocks[1].V,
    )
end

"""
    _axi_run_cond(incl, K₀) -> NamedTuple

Full corrected solve in transport: modes 0 and 1, the fixed point, and the two
localization tensors reassembled in the global frame.
"""
function _axi_run_cond(incl::FEExcenteredSphere, K₀::TensND.AbstractTens{2, 3})
    k₀ = _fe_iso_scalar(K₀; what = "`FEExcenteredSphere`")
    R = _fe_frame(incl)
    Dmap = [
        AXI_SET_CORE => _axi_check_ti2(_to_local2(incl.props[1], R), "the core"),
        AXI_SET_SHELL => _axi_check_ti2(_to_local2(incl.props[2], R), "the shell"),
        AXI_SET_MATRIX => _to_local2(K₀, R),
    ]
    backend = _axi_setup(incl).backend

    blocks = map((0, 1)) do m
        mode = _axi_mode_setup(incl, :conduction, m)
        A_E, B_E, A_p, B_p, V = _axi_solve_mode(
            backend, mode, Dmap,
            (N, dNρ, dNz, ρ) -> _axi_B_cond(m, N, dNρ, dNz, ρ),
            g -> _axi_project_cond(m, g), 1,
            _j -> (x -> (_axi_bc_affine_cond(m, x[1], x[2]),)),
            (_j, V) -> (x -> (_axi_bc_dipole_cond(m, x[1], x[2], k₀, V),)),
        )
        A, B, X = _axi_correct(A_E, B_E, A_p, B_p, fill(k₀, 1, 1))
        (; m, A_E, B_E, A_p, B_p, A, B, X, V)
    end

    diag3(t, a) = [t 0.0 0.0; 0.0 t 0.0; 0.0 0.0 a]
    A33 = diag3(blocks[2].A[1, 1], blocks[1].A[1, 1])
    B33 = diag3(blocks[2].B[1, 1], blocks[1].B[1, 1])
    A33_E = diag3(blocks[2].A_E[1, 1], blocks[1].A_E[1, 1])
    B33_E = diag3(blocks[2].B_E[1, 1], blocks[1].B_E[1, 1])
    incl.cache.assemblies += 1
    return (;
        A = _from_local33(A33, R), B = _from_local33(B33, R),
        A_uncorrected = _from_local33(A33_E, R),
        B_uncorrected = _from_local33(B33_E, R),
        blocks, volume = blocks[1].V,
    )
end

_axi_run(incl, P₀::TensND.AbstractTens{4, 3}) = _axi_run_elastic(incl, P₀)
_axi_run(incl, P₀::TensND.AbstractTens{2, 3}) = _axi_run_cond(incl, P₀)

# ─── Public entry points ─────────────────────────────────────────────────────

function _fe_axi_localization(
        incl::FEExcenteredSphere, P₀::TensND.AbstractTens; kw...
    )
    res = get!(() -> _axi_run(incl, P₀), incl.cache.tensors, _fe_cache_key(P₀))
    return res.A, res.B
end

"""
    fe_axi_breakdown(incl, P₀) -> NamedTuple

Diagnostic view of the corrected axisymmetric solve. Besides the corrected
localization tensors `A`, `B` it returns their **uncorrected** counterparts —
those of the plain truncated cell, `u|∂Ω = E·x` — and the per-mode blocks
`(A_E, B_E, A_p, B_p, A, B, X)`, plus the measured inclusion volume.

`A_uncorrected` drifts with `radius_ratio` like `(a/R)³` while `A` does not:
that contrast is the practical proof that the correction is wired correctly.
Bypasses the cache.
"""
fe_axi_breakdown(incl::FEExcenteredSphere, P₀::TensND.AbstractTens) = _axi_run(incl, P₀)

"""
    fe_axi_mesh_report(incl) -> NamedTuple

Mesh diagnostics: cell and node counts per region, and the measured volumes of
the core, the shell and the whole cell against their exact values. Builds the
grid if it does not exist yet, and caches it.

The volumes are measured on the geometric interpolation with a quadrature of
its own, so they do not match the solver's `volume` to the last digit; both
converge to the same limit.
"""
function fe_axi_mesh_report(incl::FEExcenteredSphere)
    s = _axi_setup(incl)
    counts = fe_axi_grid_counts(s.backend, s.grid)
    vol(set) = fe_axi_region_volume(s.backend, s.grid, set)
    a = Float64(incl.a)
    ac = Float64(core_radius(incl))
    R = incl.mesh.radius_ratio * a
    v_core, v_shell, v_matrix = vol(AXI_SET_CORE), vol(AXI_SET_SHELL), vol(AXI_SET_MATRIX)
    return (;
        backend = s.backend,
        ncells = counts.ncells,
        nnodes = counts.nnodes,
        ncells_core = counts.ncells_by_set[AXI_SET_CORE],
        ncells_shell = counts.ncells_by_set[AXI_SET_SHELL],
        ncells_matrix = counts.ncells_by_set[AXI_SET_MATRIX],
        volume_core = v_core,
        volume_core_exact = 4π * ac^3 / 3,
        volume_shell = v_shell,
        volume_shell_exact = 4π * (a^3 - ac^3) / 3,
        volume_cell = v_core + v_shell + v_matrix,
        volume_cell_exact = 4π * R^3 / 3,
    )
end
