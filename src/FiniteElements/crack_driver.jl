# =============================================================================
#  crack_driver.jl — the corrected finite Eshelby cell for a flat crack.
#
#  Weak form: pure linear elasticity, `∫ σ(u):ε(v) dΩ = 0`, no body force.  The
#  crack is a **zero-thickness discontinuity** (duplicated nodes) whose lips are
#  traction-free *naturally* — no interface term, no multiplier, no contact.
#  The only Dirichlet condition is on the outer sphere, and its value is where
#  the whole method lives.
#
#  Per evaluation (Adessina et al. 2017, crack declination — 3 + 3 instead of
#  the general 6 + 6):
#
#    1. one assembly, one factorization of the free-free block;
#    2. three "traction" solves,  u|∂Ω = (𝕊₀:Σ⁽ⁱ⁾)·x  with Σ⁽ⁱ⁾·n̂ = eᵢ   → 𝐁ₛ
#    3. three "dipole"   solves,  u|∂Ω = -∇G(x):Πₘ,  Πₘ = b·S_f·ℂ₀:(eₘ⊗ˢn̂) → 𝐁ᵤ
#    4. 𝐁∞ = (𝟏 - 𝐁ᵤ)⁻¹·𝐁ₛ
#
#  The minus in step 3 is *not* the one in `axi_driver.jl`, which imposes
#  `+∇G:M`.  The universal rule is `u = +∇G:M` with `M` the polarization
#  moment; the moment of a displacement discontinuity is
#  `M = -S_f ℂ₀:(⟨[[u]]⟩⊗ˢn̂)`, and `Πₘ` above is its opposite, so the sign is
#  carried here instead.  The two files agree; they put the minus in different
#  places.
#
#  Step 3 is the correction: 𝐁ₛ is the COD of the *finite* cell, 𝐁ᵤ its
#  response to the crack's own far field, and step 4 solves the resulting
#  linear fixed point in closed form.
# =============================================================================

"Whole discretization of a crack: the resolved backend, its mesh and its space."
struct CrackSetup{B, G, S}
    backend::B
    grid::G
    space::S
end

function _crack_setup(crack::FEEllipticCrack)
    if crack.cache.setup === nothing
        b = _resolve_backend(crack.backend)
        grid = fe_crack_grid(b, crack)
        crack.cache.setup = CrackSetup(b, grid, fe_crack_space(b, grid, crack.mesh.order))
    end
    return crack.cache.setup
end

# Σ⁽ⁱ⁾ with Σ⁽ⁱ⁾·e₃ = eᵢ, and eₘ ⊗ˢ e₃ — in the *local* crack frame, where the
# normal is e₃ by construction of the mesh.
const _SIGMA_T = (
    Tensors.SymmetricTensor{2, 3}([0.0 0.0 1.0; 0.0 0.0 0.0; 1.0 0.0 0.0]),
    Tensors.SymmetricTensor{2, 3}([0.0 0.0 0.0; 0.0 0.0 1.0; 0.0 1.0 0.0]),
    Tensors.SymmetricTensor{2, 3}([0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 1.0]),
)
const _EM_N = (
    Tensors.SymmetricTensor{2, 3}([0.0 0.0 0.5; 0.0 0.0 0.0; 0.5 0.0 0.0]),
    Tensors.SymmetricTensor{2, 3}([0.0 0.0 0.0; 0.0 0.0 0.5; 0.0 0.5 0.0]),
    Tensors.SymmetricTensor{2, 3}([0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 1.0]),
)

"""
    _solve_cod_local(crack, C, μ, ν) -> (B_s, B_u, B_inf)

The 3 + 3 corrected scheme, entirely in the crack's local frame. Returns plain
``3\\times 3`` matrices.
"""
function _solve_cod_local(
        crack::FEEllipticCrack,
        C::Tensors.SymmetricTensor{4, 3, Float64}, μ::Float64, ν::Float64
    )
    s = _crack_setup(crack)
    a, b = Float64(crack.a), Float64(crack.b)
    S_f = π * a * b

    K = fe_crack_stiffness(s.backend, s.space, C)
    crack.cache.assemblies += 1
    ndofs, free, presc = fe_crack_dof_split(s.backend, s.space)

    # One factorization of the free-free block serves all six right-hand sides:
    # the prescribed dofs are eliminated by hand rather than `apply!`-ed into
    # the matrix, which would consume it.
    F = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(K[free, free]))
    Kfp = K[free, presc]

    function solve_with(f)
        u = zeros(ndofs)
        fe_crack_set_dirichlet!(s.backend, s.space, u, f)
        u[free] .= F \ Vector(-Kfp * u[presc])
        return u
    end

    S₀ = inv(C)
    jump(u) = fe_crack_mean_jump(s.backend, s.space, u, S_f, b)

    B_s = zeros(3, 3)
    for i in 1:3
        E = S₀ ⊡ _SIGMA_T[i]
        B_s[:, i] .= jump(solve_with(x -> E ⋅ Tensors.Vec{3}(Tuple(x))))
    end

    B_u = zeros(3, 3)
    for m in 1:3
        Π = Matrix((b * S_f) * (C ⊡ _EM_N[m]))
        B_u[:, m] .= jump(
            solve_with(x -> -Core._dipole_displacement_iso(μ, ν, collect(x), Π))
        )
    end

    B_inf = (LinearAlgebra.I - B_u) \ B_s
    return B_s, B_u, B_inf
end

# ─── Frame handling ──────────────────────────────────────────────────────────

# ─── Frame handling ──────────────────────────────────────────────────────────
#
#  The mesh is built once, in the crack's *local* frame (crack in the z = 0
#  plane, semi-axes along x and y, normal along z).  The reference medium is
#  therefore rotated into that frame before the solve and the resulting COD
#  tensor rotated back afterwards.  Besides being necessary, this is what makes
#  the memoization pay off under orientation averaging: a whole family of
#  identically-shaped cracks at different orientations, embedded in the same
#  (isotropic, or symmetrization-projected) matrix, share one single solve.

"Rotation matrix whose columns are the crack's local axes in global coordinates."
function _frame_matrix(crack)
    l̂, m̂, n̂ = Core._frame_columns(Core.inclusion_basis(crack))
    return hcat(l̂, m̂, n̂)
end

"Components of `C₀` in the crack's local frame, as a `SymmetricTensor{4,3}`."
function _local_stiffness(crack, C₀::TensND.AbstractTens{4, 3})
    Cg = Core._C_array(C₀)
    R = _frame_matrix(crack)
    Cl = zeros(Float64, 3, 3, 3, 3)
    @inbounds for p in 1:3, q in 1:3, r in 1:3, s in 1:3
        acc = 0.0
        for i in 1:3, j in 1:3, k in 1:3, l in 1:3
            acc += R[i, p] * R[j, q] * R[k, r] * R[l, s] * Cg[i, j, k, l]
        end
        Cl[p, q, r, s] = acc
    end
    return Tensors.SymmetricTensor{4, 3}((i, j, k, l) -> Cl[i, j, k, l])
end

"Cache key: the local-frame stiffness, rounded so that arithmetic noise between
two otherwise identical scheme iterations does not miss the cache."
_cache_key(C::Tensors.SymmetricTensor{4, 3, Float64}) =
    map(x -> round(x, sigdigits = 12), Tuple(C))

"""
    _isotropic_moduli(crack, C₀) -> (μ, ν)

Shear modulus and Poisson ratio of the reference medium, refusing anything
that is not isotropic.

The test is on the **content**, not on the TensND type: a self-consistent or
differential iterate arrives as a `TensCanonical` even when its content is
isotropic, and rejecting it on the type alone would rule out perfectly valid
uses (an isotropically-symmetrized crack family, for instance).
"""
function _isotropic_moduli(crack, C₀::TensND.AbstractTens{4, 3}; rtol = 1.0e-8)
    C_iso = Core.isotropify(C₀)
    if !(C₀ isa TensND.TensISO{4, 3})
        A, Aiso = Core._C_array(C₀), Core._C_array(C_iso)
        dev = maximum(abs, A .- Aiso)
        dev ≤ rtol * max(maximum(abs, Aiso), eps()) || throw(
            ArgumentError(
                "`FEEllipticCrack` supports an isotropic reference medium only — " *
                    "the corrected boundary condition uses the closed-form Kelvin " *
                    "dipole field, whose anisotropic counterpart (Pan-Chou / " *
                    "Barnett-Willis) is not implemented. The reference medium " *
                    "deviates from isotropy by $(round(dev / maximum(abs, Aiso) * 100, sigdigits = 3)) %.\n" *
                    "This is the normal situation for an iterative scheme " *
                    "(`SelfConsistent`, `DifferentialScheme`) on a *parallel* crack " *
                    "family, whose effective medium is genuinely anisotropic. Add " *
                    "`symmetrize = IsoSymmetrize()` to the phase — the scheme then " *
                    "hands the kernel an isotropic reference — or use the " *
                    "closed-form `EllipticCrack`."
            )
        )
    end
    E, ν = Core.extract_iso_moduli(C_iso)
    return Float64(E / (2 * (1 + ν))), Float64(ν)
end

# ─── The seam ────────────────────────────────────────────────────────────────

function _fe_cod_tensor(
        crack::FEEllipticCrack, C₀::TensND.AbstractTens{4, 3}; kw...
    )
    μ, ν = _isotropic_moduli(crack, C₀)
    C_loc = _local_stiffness(crack, C₀)
    key = _cache_key(C_loc)

    B_loc = get!(crack.cache.tensors, key) do
        _, _, B = _solve_cod_local(crack, C_loc, μ, ν)
        B
    end

    R = _frame_matrix(crack)
    return TensND.TensCanonical(R * B_loc * R')
end

"""
    fe_cod_breakdown(crack, C₀) -> (; B_s, B_u, B_inf, B_s_glob, B_inf_glob)

Diagnostic view of the corrected solve: the COD tensor of the **finite** cell
`B_s`, the response `B_u` to the crack's own dipole far field, and the
infinite-medium result ``\\boldsymbol{B}_\\infty = (\\boldsymbol{1} - \\boldsymbol{B}_u)^{-1}\\boldsymbol{B}_s``, all in the crack's local
frame, plus `B_s` and `B_inf` rotated back to the global frame.

`norm(B_u)` measures how much work the boundary correction is doing; it should
fall like ``(a/R)^3``, and `B_inf` — unlike `B_s` — should be insensitive to
`radius_ratio`. That contrast is the practical proof that the correction is
wired correctly.

Bypasses the cache.
"""
function fe_cod_breakdown(
        crack::FEEllipticCrack, C₀::TensND.AbstractTens{4, 3}
    )

    μ, ν = _isotropic_moduli(crack, C₀)
    C_loc = _local_stiffness(crack, C₀)
    B_s, B_u, B_inf = _solve_cod_local(crack, C_loc, μ, ν)
    R = _frame_matrix(crack)
    return (;
        B_s, B_u, B_inf,
        B_s_glob = TensND.TensCanonical(R * B_s * R'),
        B_inf_glob = TensND.TensCanonical(R * B_inf * R'),
    )
end

"""
    fe_mesh_report(crack) -> NamedTuple

Mesh diagnostics: cell, node and dof counts, the two lip facet counts and their
measured areas against the exact ``\\pi a b``. Builds the discretization if it does
not exist yet, and caches it.

Both lip areas equaling ``\\pi a b`` is what says the `Crack` plugin split the
surface cleanly *and* the front weld did not glue the lips back together.
"""
function fe_mesh_report(crack::FEEllipticCrack)
    s = _crack_setup(crack)
    c = fe_crack_counts(s.backend, s.grid)
    ndofs, _, _ = fe_crack_dof_split(s.backend, s.space)
    return (;
        backend = s.backend, c..., ndofs,
        area_exact = π * Float64(crack.a) * Float64(crack.b),
    )
end
