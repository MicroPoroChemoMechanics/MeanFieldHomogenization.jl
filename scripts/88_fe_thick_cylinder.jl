# # A thick-walled cylinder whose material is a microstructure
#
# The canonical MFront/FEniCS coupling demonstration, run the other way round:
# instead of a constitutive law fitted to data, every Gauss point of the mesh
# carries a **representative volume element**, and the scheme that upscales it
# supplies the stress and the consistent tangent.
#
# Two runs on the same geometry:
#
# 1. a **linear** composite (Mori-Tanaka, spherical inclusions), checked against
#    the closed-form Lamé solution — this validates the coupling itself, since
#    any discrepancy can only come from the plumbing;
# 2. a **microcracked** solid whose cracks close under compression, which has no
#    closed form and is what the coupling is actually for.
#
# The geometry is a quarter annulus under internal pressure, plane strain:
#
# ```math
# \sigma_{rr}(R_i) = -p, \qquad \sigma_{rr}(R_o) = 0,
# \qquad u_\theta = 0 \ \text{on}\ \theta = 0, \tfrac{\pi}{2}.
# ```
#
# The Lamé solution in plane strain, for the homogenized moduli ``(E, \nu)``:
#
# ```math
# \sigma_{\theta\theta}(r) = \frac{p R_i^2}{R_o^2 - R_i^2}\left(1 + \frac{R_o^2}{r^2}\right),
# \qquad
# u_r(r) = \frac{(1+\nu)\,p\,R_i^2}{E\,(R_o^2 - R_i^2)}
#          \left[(1-2\nu)\,r + \frac{R_o^2}{r}\right].
# ```

import Pkg                                                                #jl
Pkg.activate(joinpath(@__DIR__, "fe"); io = devnull)                      #jl
Pkg.instantiate(; io = devnull)                                           #jl

using MeanFieldHomogenization
using TensND
using Ferrite
using LinearAlgebra
using Printf
using Plots
gr()

# `import Ferrite` alone activates the Gauss-point glue — no gmsh needed.
const EXT = Base.get_extension(
    MeanFieldHomogenization, :MeanFieldHomogenizationFerriteMaterialExt
)

# ## Geometry and discretization
#
# A quarter annulus, meshed by bending a structured rectangle — no gmsh, so this
# stays cheap enough to re-run at will.

const Ri, Ro = 0.1, 1.0
const PRESSURE = 20.0

grid = EXT.annulus_grid(Ri, Ro, 24, 24)

ip = Lagrange{RefQuadrilateral, 1}()^2
qr = QuadratureRule{RefQuadrilateral}(2)
qr_face = FacetQuadratureRule{RefQuadrilateral}(2)
cv = CellValues(qr, ip)
fv = FacetValues(qr_face, ip)

dh = DofHandler(grid)
add!(dh, :u, ip)
close!(dh)

# Symmetry: the radial edges carry no tangential displacement. `"bottom"` lies
# on ``\theta = 0`` (so ``u_y = 0``) and `"top"` on ``\theta = \pi/2``
# (so ``u_x = 0``).
ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getfacetset(grid, "bottom"), (x, t) -> 0.0, 2))
add!(ch, Dirichlet(:u, getfacetset(grid, "top"), (x, t) -> 0.0, 1))
close!(ch)

# ## The microstructures
#
# Both share the same matrix; only the second phase differs.

const C_MATRIX = TensISO{3}(3 * 30.0, 2 * 18.0)        # k = 30 GPa, μ = 18 GPa

"Stiff spherical inclusions — an isotropic composite, so Lamé applies."
function composite_rve(f = 0.25)
    rve = RVE(:M)
    add_matrix!(rve, Ellipsoid(1.0), Dict(:C => C_MATRIX))
    add_phase!(
        rve, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(3 * 120.0, 2 * 80.0));
        fraction = f
    )
    return rve
end

"Two crack families, normal to `e₁` and `e₂`, which close under compression."
function cracked_rve(d = 0.15)
    rve = RVE(:M)
    add_matrix!(rve, Ellipsoid(1.0), Dict(:C => C_MATRIX))
    add_phase!(
        rve, :Fx, PennyCrack(1.0; euler_angles = (π / 2, 0.0)),
        Dict(:C => C_MATRIX); density = d
    )
    add_phase!(
        rve, :Fy, PennyCrack(1.0; euler_angles = (π / 2, π / 2)),
        Dict(:C => C_MATRIX); density = d
    )
    return rve
end

# ## The solver
#
# One Newton loop, material-agnostic: it only ever calls `mfh_element!`.

function solve!(material, nsteps; cache = MaterialCache(), tol = 1.0e-9, maxit = 25)
    nqp = getnquadpoints(cv)
    states = EXT.mfh_states(material, getncells(grid), nqp)
    states_old = EXT.mfh_states(material, getncells(grid), nqp)

    u = zeros(ndofs(dh))
    K = allocate_matrix(dh, ch)
    nbf = getnbasefunctions(cv)
    Ke, re = zeros(nbf, nbf), zeros(nbf)

    for step in 1:nsteps
        p = PRESSURE * step / nsteps
        for it in 1:maxit
            assembler = start_assemble(K, zeros(ndofs(dh)))
            r = zeros(ndofs(dh))
            for cell in CellIterator(dh)
                reinit!(cv, cell)
                fill!(Ke, 0); fill!(re, 0)
                eldofs = celldofs(cell)
                EXT.mfh_element!(
                    Ke, re, cv, material, u[eldofs],
                    states[cellid(cell)], states_old[cellid(cell)], 0.0; cache = cache
                )
                assemble!(assembler, eldofs, Ke, zeros(nbf))
                r[eldofs] .+= re
            end
            r .-= external_force(p)
            apply_zero!(K, r, ch)
            norm(r) < tol && break
            u .-= K \ r
            apply!(u, ch)
        end
        for c in eachindex(states), q in eachindex(states[c])
            states_old[c][q] = states[c][q]
        end
    end
    return u, states
end

"Consistent nodal forces of the internal pressure on ``r = R_i``."
function external_force(p)
    f = zeros(ndofs(dh))
    inner = getfacetset(grid, "left")
    for fc in FacetIterator(dh, inner)
        reinit!(fv, fc)
        fe = zeros(getnbasefunctions(fv))
        for q in 1:getnquadpoints(fv)
            dΓ = getdetJdV(fv, q)
            n = getnormal(fv, q)
            for i in 1:getnbasefunctions(fv)
                fe[i] += (shape_value(fv, q, i) ⋅ (-p * n)) * dΓ
            end
        end
        f[celldofs(fc)] .+= fe
    end
    return f
end

# ## Run 1 — linear composite against Lamé

rve_lin = composite_rve()
mat_lin = HomogenizedElastic(rve_lin, MoriTanaka())
k_h, μ_h = k_mu(stiffness(mat_lin))
E_h = 9k_h * μ_h / (3k_h + μ_h)
ν_h = (3k_h - 2μ_h) / (2 * (3k_h + μ_h))
@printf("homogenized moduli: E = %.4f GPa, ν = %.4f\n", E_h, ν_h)

u_lin, _ = solve!(mat_lin, 1)

lame_ur(r) = (1 + ν_h) * PRESSURE * Ri^2 / (E_h * (Ro^2 - Ri^2)) *
    ((1 - 2ν_h) * r + Ro^2 / r)
lame_σθθ(r) = PRESSURE * Ri^2 / (Ro^2 - Ri^2) * (1 + Ro^2 / r^2)

# Radial displacement of every node, against the closed form.
#
# `evaluate_at_grid_nodes` and not `u[2i-1], u[2i]`: a `DofHandler` numbers
# degrees of freedom by cell traversal, **not** in node order, so indexing the
# solution vector by node is wrong. On a structured mesh most nodes happen to
# line up, which makes the mistake look like a plausible discretization error
# instead of the nonsense it is.
coords = [grid.nodes[i].x for i in 1:getnnodes(grid)]
radii = [norm(x) for x in coords]
u_nodes = evaluate_at_grid_nodes(dh, u_lin, :u)
ur_fe = [
    (u_nodes[i][1] * coords[i][1] + u_nodes[i][2] * coords[i][2]) / radii[i]
        for i in 1:getnnodes(grid)
]
err = maximum(abs, ur_fe .- lame_ur.(radii)) / maximum(abs, lame_ur.(radii))
@printf("max relative error on u_r vs Lamé : %.3e   (Q1, 24x24)\n", err)

# ## Run 2 — microcracked solid
#
# The cracks are normal to ``e_1`` and ``e_2``. Near the bore the hoop stress is
# strongly tensile and the radial stress compressive, so the two families do not
# see the same loading: the coupling produces an evolving, spatially varying
# anisotropy that no fitted law would reproduce.

rve_cr = cracked_rve()
mat_cr = MicrocrackedMaterial(rve_cr, MoriTanaka(); ω₀ = (2.0e-3, 2.0e-3))
cache = MaterialCache()
u_cr, states_cr = solve!(mat_cr, 8; cache = cache)
@printf(
    "cracked run: %d scheme solves for %d quadrature points\n",
    cache_stats(cache).entries, getncells(grid) * getnquadpoints(cv)
)

# Fraction of quadrature points where each family has closed.
closed_x = count(st -> !open_set(st)[1], Iterators.flatten(states_cr))
closed_y = count(st -> !open_set(st)[2], Iterators.flatten(states_cr))
ntot = getncells(grid) * getnquadpoints(cv)
@printf("closed: family ⟂e₁ %.1f%%, family ⟂e₂ %.1f%%\n",
    100closed_x / ntot, 100closed_y / ntot)

nothing
