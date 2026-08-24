# =============================================================================
#  50_visco_law_basics.jl
#
#  Basic walkthrough of the ALV pipeline:
#    * build a Maxwell relaxation `ViscoLaw` for an iso 4-tensor matrix,
#    * discretize the Stieltjes integral on a time grid (`trapezoidal_matrix`),
#    * compute its Volterra inverse to obtain the corresponding creep
#      compliance matrix,
#    * extract scalar Volterra responses (uniaxial relaxation / creep)
#      from the iso block matrix,
#    * plot the four kernels on a log-scaled axis.
#
#  Usage : julia --project scripts/50_visco_law_basics.jl
#  Output : scripts/figures/50_visco_law_basics.png
# =============================================================================

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "docs"); io = devnull)

using MeanFieldHomogenization
using TensND
using LinearAlgebra
using Printf
using Plots

# ─── Setup ────────────────────────────────────────────────────────────────────

# Iso Maxwell relaxation : 3 k e^{-(t-t')/η_k} 𝕁 + 2 μ e^{-(t-t')/η_μ} 𝕂.
const k0 = 10.0           # bulk modulus (instantaneous)
const μ0 = 4.0            # shear modulus (instantaneous)
const η_k = 1.0           # bulk relaxation time
const η_μ = 0.5           # shear relaxation time

law = maxwell_iso(k0, μ0, η_k, η_μ)
println("Maxwell iso law : k=$k0, μ=$μ0, η_k=$η_k, η_μ=$η_μ")
println("law(0, 0)   = ", law(0.0, 0.0))
println("law(1, 0)   = ", law(1.0, 0.0))
println("law(0, 1)   = ", law(0.0, 1.0), "  (causality: t<t' ⇒ 0)")
println()

# ─── Trapezoidal discretization ──────────────────────────────────────────────

const T_grid = collect(range(0.0, 5.0; length = 41))
const n = length(T_grid)
println("Time grid : n = $n points, t ∈ [$(T_grid[1]), $(T_grid[end])]")

R_M = trapezoidal_matrix(law, T_grid)
@printf "Block matrix R̃ size : %d × %d (= 6n × 6n)\n" size(R_M, 1) size(R_M, 2)

# Volterra inverse : the discrete creep compliance matrix.
J_M = volterra_inverse(R_M; block_size = 6)
@printf "Round-trip ‖R̃ J̃ - I‖_F = %.3e\n" norm(R_M * J_M - I)
println()

# Iso parameter extraction : (3K, 2μ) trapezoidal matrices, n×n each.
α, β = iso_params_from_blocks(R_M)
@printf "α[1,1] = 3K = %.4f  (expected 3·%.0f = %.0f)\n" α[1, 1] k0 3 * k0
@printf "β[1,1] = 2μ = %.4f  (expected 2·%.0f = %.0f)\n" β[1, 1] μ0 2 * μ0

# ─── Scalar Volterra inverses (longitudinal & shear) ────────────────────────

# Longitudinal modulus  k + 4μ/3  and shear μ.
M_long = @. (α + 2 * β) / 3
M_shear = β ./ 2
J_long = volterra_inverse(M_long; block_size = 1)
J_shear = volterra_inverse(M_shear; block_size = 1)

# Discrete relaxation / creep responses to a unit step at t = 0.
# `M * 1` gives the response to a Heaviside step (`1 = ones(n)`).
unit_step = ones(n)
R_long_response = M_long * unit_step
J_long_response = J_long * unit_step
R_shear_response = M_shear * unit_step
J_shear_response = J_shear * unit_step

# ─── Plot ────────────────────────────────────────────────────────────────────

p1 = plot(;
    xlabel = "t", ylabel = "modulus",
    title = "Maxwell iso — relaxation & creep responses",
    legend = :outerright, grid = true
)
plot!(
    p1, T_grid, R_long_response; lw = 2, color = :red,
    label = "M_long(t) = ⟨k+4μ/3⟩ · H(t)"
)
plot!(
    p1, T_grid, R_shear_response; lw = 2, color = :blue,
    label = "M_shear(t) = μ · H(t)"
)
plot!(
    p1, T_grid, 1 ./ R_long_response; lw = 1, color = :red,
    linestyle = :dot, label = "1 / M_long  (instantaneous)"
)
plot!(
    p1, T_grid, 1 ./ R_shear_response; lw = 1, color = :blue,
    linestyle = :dot, label = "1 / M_shear (instantaneous)"
)
plot!(
    p1, T_grid, J_long_response; lw = 2, color = :darkred,
    linestyle = :dash, label = "J_long(t) = ⟨k+4μ/3⟩^{-vol} · H(t)"
)
plot!(
    p1, T_grid, J_shear_response; lw = 2, color = :darkblue,
    linestyle = :dash, label = "J_shear(t) = ⟨μ⟩^{-vol} · H(t)"
)

const figdir = joinpath(@__DIR__, "figures")
isdir(figdir) || mkdir(figdir)
figpath = joinpath(figdir, "50_visco_law_basics.png")
savefig(p1, figpath)
display(p1)
@printf "\nSaved : %s\n" figpath
