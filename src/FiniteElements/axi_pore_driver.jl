# =============================================================================
#  axi_pore_driver.jl — the corrected solve on the axisymmetric cavity.
#
#  Same three assemblies and eight two-dimensional solves as the core-shell
#  cell, and the same modal fixed point. One thing changes, and it changes the
#  averaging step: **a cavity has no interior to integrate over**.
#
#  `fe_axi_average` is a volume integral over meshed cells, and the core-shell
#  driver hands it the two sets that make up the inclusion. There are none here,
#  so `⟨ε⟩_D` comes from the wall — a line integral on the meridian trace,
#  `fe_axi_pore_boundary`, exactly as `fe_cell_mean_strain` does in three
#  dimensions.
#
#  ## The route that looks cleverer and is not
#
#  The divergence identity on the matrix `M = Ω \ D`,
#
#      ∫_M ε(u) dV = ∮_∂Ω (u ⊗ n)ˢ dS − ∮_∂D (u ⊗ n_D)ˢ dS ,
#
#  gives `⟨ε⟩_D` from the volume average over the matrix plus an outer boundary
#  term that is **analytic**, the datum there being imposed. No new generic, no
#  quadrature on a spline. It is algebraically exact and numerically hopeless:
#  it computes `V_D` as the difference of `V_Ω` and `V_M`, whose ratio is
#  `(R/a)³`. Two digits go at `R/a = 4`, and more as the cell grows — which is
#  the opposite of what one wants, the whole point of a large cell being
#  accuracy.
#
#  Measured, before it was abandoned: the implied cavity volume is 2.9 % off at
#  `V_Ω/V_D = 8`, 5.8 % at 64 and 9.2 % at 216, and the localization error
#  tracks it (2.0e-2 → 6.1e-2). Convergence in `nradial` was `O(h)` where the
#  core-shell cell gets `O(h²)`. Recorded here because the trap is attractive:
#  it reuses only tested code and adds nothing.
#
#  ## The fixed point is the existing one
#
#  For a cavity `𝔹ᴱ = 𝔹ᵖ = 0`, so `_axi_correct` degenerates *identically* into
#  the pore form `𝕃 = (𝕀 − 𝕃_u 𝔽)⁻¹ 𝕃_s` of `_cell_close_dipole` with
#  `𝔽 = −P₀`, block by modal block. It is reused rather than rewritten.
#
#  `𝔽 = −P₀` and not `−V_D P₀`: the cavity volume is already carried by the
#  imposed dipole moment, the boundary datum being the field of moment `V_D Π`.
#  Putting `V_D` in twice weakens the correction by that factor, which reads as
#  a mesh that will not converge.
# =============================================================================

"""
    _axi_pore_solve_mode(backend, mode, Dmap, Bop, proj, nload, bc_E, bc_p,
                         dofmap, V_D) -> (L_s, L_u)

One modal block of a cavity's localization, both families, on a single
factorization.

The normalization is the **meridian** measure of the volume the *meshed* wall
encloses, matching what `fe_axi_average` divides by, and the sign is the facet
normal's: Ferrite reports it outward from the matrix, hence into the cavity, so
the accumulated integral is `−∮(u ⊗ n_D)ˢ`.

`V_D` is passed in only as the fallback for a wall with no facets, which would
be a bug elsewhere; the value actually used is measured.
"""
function _axi_pore_solve_mode(
        backend, mode, Dmap, Bop, proj, nload::Int, bc_E, bc_p, dofmap, V_D::Real
    )
    K = fe_axi_stiffness(backend, mode, Dmap, Bop)
    ndofs, free, presc = fe_axi_dof_split(backend, mode)
    F = LinearAlgebra.lu(K[free, free])
    Kfp = K[free, presc]
    function solve_with(f)
        u = zeros(ndofs)
        fe_axi_set_dirichlet!(backend, mode, u, f)
        u[free] .= F \ Vector(-Kfp * u[presc])
        return u
    end

    wall(u) = fe_axi_pore_boundary(
        backend, mode, u, dofmap, proj, AXI_SET_INCLUSION
    )

    L_s = zeros(nload, nload)
    L_u = zeros(nload, nload)
    Vmesh = V_D
    for j in 1:nload
        a, Vmesh = wall(solve_with(bc_E(j)))
        L_s[:, j] .= a
        b, _ = wall(solve_with(bc_p(j)))
        L_u[:, j] .= b
    end
    # Normalized by the volume the meshed wall encloses, not by the closed form:
    # the integral is over that wall, and mixing the two leaves a systematic
    # error of the geometry's own size that refinement does not remove.
    L_s ./= -Vmesh / (2π)
    L_u ./= -Vmesh / (2π)
    return L_s, L_u, Vmesh
end
