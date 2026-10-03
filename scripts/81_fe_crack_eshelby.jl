# # A crack whose COD tensor comes out of a finite-element solve
#
# The custom-inclusion contract exists so that a morphology with no closed form
# can still take part in the schemes. This script exercises it on a case where
# the closed form *does* exist — an elliptical crack — precisely so the
# finite-element answer can be checked against it.
#
# The method is the **finite Eshelby cell with a first-order corrected boundary
# condition** of Adessina, Barthélémy, Lavergne & Ben Fraj (2017), in the crack
# declination used in the `SifAniso` study. The infinite matrix is truncated to
# a ball of radius ``R = 5a``; the truncation bias is removed by adding, to the
# imposed boundary displacement, the **dipole far field of the open crack**:
#
# ```math
# \underline u(\underline x) \;\underset{\|\underline x\|\to\infty}{\approx}\;
#   (\mathbb S_0 : \boldsymbol\Sigma)\cdot\underline x
#   \;-\; b\,S_f\,\bigl(\nabla\boldsymbol G(\underline x):\mathbb C_0\cdot\underline n\bigr)
#         \cdot\underline U ,
# \qquad \underline U = \frac{\langle[\![\underline u]\!]\rangle}{b},\quad S_f = \pi a b .
# ```
#
# Three "traction" solves give the COD tensor ``\boldsymbol B_s`` of the truncated
# cell, three "dipole" solves give its response ``\boldsymbol B_u`` to that far
# field, and the fixed point closes in one step:
#
# ```math
# \boldsymbol B_\infty = (\boldsymbol 1 - \boldsymbol B_u)^{-1}\cdot\boldsymbol B_s .
# ```
#
# Requires `Ferrite`, `FerriteGmsh` and `Gmsh`.

import Pkg                                                          #jl
Pkg.activate(joinpath(@__DIR__, "fe"); io = devnull)                 #jl
Pkg.instantiate(; io = devnull)                                      #jl

using MeanFieldHomogenization
using TensND
using LinearAlgebra
using Printf

import Ferrite, FerriteGmsh, Gmsh                                    #jl

# Dimensionless isotropic matrix, `E = 1`, `ν = 0.3`.
const νm = 0.3
const Em = 1.0
const C₀ = iso_stiffness(Em / (3 * (1 - 2νm)), Em / (2 * (1 + νm)))

diag3(B) = diag(Matrix(get_array(B)))

# ## The mesh
#
# A ball of matrix around an elliptical slit, refined in a torus hugging the
# crack front. Two details are worth stating because they cost real debugging
# time: the crack disc is `embed`ed in the ball (`fragment` would cut it into
# two half-balls), and the nodes that gmsh's `Crack` plugin duplicates **on the
# front** are welded back together — the plugin duplicates them despite
# `OpenBoundaryPhysicalGroup`, which would make the crack half an element too
# large and overestimate the opening by 10-20 %.

crack = FEEllipticCrack(1.0, 0.25; htipdiv = 6.0)
rep = fe_mesh_report(crack)

println("="^78)
@printf "mesh: %d cells, %d nodes, %d dofs\n" rep.ncells rep.nnodes rep.ndofs
@printf "lips: %d / %d facets;  areas %.6f / %.6f  vs exact πab = %.6f  (%.2f %%)\n" rep.nfacets_up rep.nfacets_dn rep.area_up rep.area_dn rep.area_exact 100abs(rep.area_up - rep.area_exact) / rep.area_exact

# The two lips carry the same number of facets and the same area: the slit is
# discretized symmetrically, and the residual gap to `πab` is just the
# polygonal approximation of the ellipse.

# ## The correction at work
#
# `fe_cod_breakdown` exposes the three tensors. The magnitude of ``B_u`` is the
# size of the truncation bias being removed — and it must fall like
# ``(a/R)^3``.

println("\n", "="^78)
println("Boundary correction vs domain radius")
println("-"^78)
@printf "%-8s %12s %14s %14s\n" "R/a" "‖B_u‖" "diag(B_s)₃₃" "diag(B_inf)₃₃"
prev = nothing
for rr in (5.0, 10.0)
    d = fe_cod_breakdown(FEEllipticCrack(1.0, 0.25; htipdiv = 6.0, radius_ratio = rr), C₀)
    @printf "%-8.1f %12.3e %14.5f %14.5f" rr norm(d.B_u) d.B_s[3, 3] d.B_inf[3, 3]
    prev === nothing ? println() : @printf "     ratio ‖B_u‖ = %.2f  (expected 2³ = 8)\n" prev / norm(d.B_u)
    global prev = norm(d.B_u)
end

# The ratio comes out at ≈ 8 for a doubling of `R`: the boundary term really is
# the ``O((a/R)^3)`` dipole, which is what pins both its magnitude and its sign.
# Its *effect* at `R = 5a` is small (≈ 0.1 % on `B`), which is the point — the
# correction is what makes such a small domain legitimate in the first place.

# ## Convergence to the closed form
#
# The crack front carries a square-root displacement field, so a fixed-order
# element converges slowly. The error turns out to be **first order in the
# element size** `h ∝ 1/htipdiv`, which makes Richardson extrapolation to
# `h → 0` both legitimate and informative.

function convergence(a, b, htipdivs)
    B_ana = diag3(cod_tensor(EllipticCrack(a, b), C₀))
    println("\n", "="^78)
    @printf "Crack a = %.2f, b = %.2f  (η = %.2f)\n" a b b / a
    @printf "analytical diag(B) = [%.5f %.5f %.5f]\n" B_ana[1] B_ana[2] B_ana[3]
    println("-"^78)
    @printf "%-8s %9s %-26s %-22s\n" "h_tip" "ndof" "diag(B_fe)" "rel. error (%)"
    Bs = Vector{Float64}[]
    for hd in htipdivs
        c = FEEllipticCrack(a, b; htipdiv = hd)
        B = diag3(cod_tensor(c, C₀))
        push!(Bs, B)
        e = 100 .* (B .- B_ana) ./ B_ana
        @printf "b/%-6g %9d [%.5f %.5f %.5f]   [%+6.2f %+6.2f %+6.2f]\n" hd fe_mesh_report(c).ndofs B[1] B[2] B[3] e[1] e[2] e[3]
        flush(stdout)
    end
    ## Richardson, first order in h = 1/htipdiv, on the two finest meshes.
    h1, h2 = 1 / htipdivs[end - 1], 1 / htipdivs[end]
    B1, B2 = Bs[end - 1], Bs[end]
    B0 = B2 .+ (B2 .- B1) .* (h2 / (h1 - h2))
    e0 = 100 .* (B0 .- B_ana) ./ B_ana
    @printf "%-8s %9s [%.5f %.5f %.5f]   [%+6.2f %+6.2f %+6.2f]\n" "h→0" "—" B0[1] B0[2] B0[3] e0[1] e0[2] e0[3]
    return nothing
end

convergence(1.0, 1.0, (4, 6, 9, 12))       # penny-shaped
convergence(1.0, 0.25, (6, 9, 12))         # elongated, η = 1/4

# On the raw meshes the finite-element opening sits a few percent **below** the
# closed form — a truncated cell is stiffer than an infinite medium and a
# fixed-order element under-resolves the front. The extrapolated values land
# within about a percent, which is the real validation: the residual is
# discretization, not a modeling error.
#
# For reference, the FEniCSx implementation of the same scheme in `SifAniso`
# reports ±5 % on ``B_\infty`` at `htipdiv = 12` with P3 elements; `Ferrite`
# provides no cubic Lagrange element on tetrahedra, so this port runs P2 on
# straight tetrahedra and lands in the same band.
