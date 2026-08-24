# # A creeping laminate: the multilayer in ageing viscoelasticity
#
# The laminate solution is pure algebra — products of Kelvin-Mandel matrices
# and one inversion restricted to the out-of-plane subspace. Replacing each
# scalar by a discretized Volterra operator therefore transposes it verbatim
# to ageing linear viscoelasticity: the *same* kernel runs, with the
# ``3\times3`` cofactor inversion swapped for `volterra_inverse` on the
# out-of-plane restriction.
#
# This script checks that transposition against the elastic answer, then uses
# it: a stack alternating a creeping binder with an elastic reinforcement
# relaxes very differently along the layers and across them.

import Pkg                                                          #jl
Pkg.activate(joinpath(@__DIR__, "..", "docs"); io = devnull)                 #jl

using MeanFieldHomogenization
using TensND
using Printf
using LinearAlgebra
using Plots

default(; left_margin = 5Plots.mm, bottom_margin = 5Plots.mm)

iso(k, μ) = TensISO{3}(3k, 2μ)

# The creeping phase is the COMPLIANT one — a binder between stiff, elastic
# reinforcing layers. That is the configuration in which the two directions
# behave most differently: across the layers (a series arrangement) the binder
# dominates, in their plane (a parallel one) the reinforcement does.
C_binder = iso(0.5, 0.2)    # creeping binder, instantaneous moduli
C_reinf = iso(2.0, 0.8)     # elastic reinforcement
f_binder, f_reinf = 0.4, 0.6

# Diagonal time block of an ALV operator, as a 6×6 Mandel matrix.
blk(M, i) = M[(6 * (i - 1) + 1):(6 * i), (6 * (i - 1) + 1):(6 * i)]

# ## The elastic limit
#
# Feeding each layer a Heaviside law built on its elastic stiffness must give
# back, in every diagonal time block, the elastic laminate — and nothing
# off-diagonal, an elastic material having no memory. That single check pins
# the whole Volterra transposition against a result already validated in
# closed form against Backus (1962).

times = [0.0, 1.0, 3.0, 7.0]

lam_el = Laminate(; normal = (0, 0, 1))
add_layer!(lam_el, :BINDER, Dict(:C => C_binder); thickness = f_binder)
add_layer!(lam_el, :REINF, Dict(:C => C_reinf); thickness = f_reinf)
C_elastic = Matrix(KM(homogenize(lam_el, Laminated(), :C)))

lam_h = Laminate(; normal = (0, 0, 1))
add_layer!(lam_h, :BINDER, Dict(:C => heaviside_law(C_binder)); thickness = f_binder)
add_layer!(lam_h, :REINF, Dict(:C => heaviside_law(C_reinf)); thickness = f_reinf)
M_h = homogenize_alv(lam_h, Laminated(), :C; times = times)

err_diag = maximum(maximum(abs, blk(M_h, i) - C_elastic) for i in eachindex(times))
err_off = maximum(
    maximum(abs, M_h[(6 * (i - 1) + 1):(6 * i), (6 * (j - 1) + 1):(6 * j)])
        for i in 2:length(times) for j in 1:(i - 1)
)
@printf "elastic limit : max |diag block − ℂ_hom| = %.2e\n" err_diag
@printf "                max |off-diagonal block| = %.2e   (no memory)\n" err_off

# ## A genuinely creeping stack
#
# The binder is a Maxwell chain that loses half its stiffness with a
# relaxation time of 3; the reinforcement stays elastic.

τ = 3.0
law_binder = maxwell_relaxation(iso(0.25, 0.1), [iso(0.25, 0.1)], [τ])
law_reinf = heaviside_law(C_reinf)

ts = collect(range(0.0, 20.0; length = 40))
lam = Laminate(; normal = (0, 0, 1))
add_layer!(lam, :BINDER, Dict(:C => law_binder); thickness = f_binder)
add_layer!(lam, :REINF, Dict(:C => law_reinf); thickness = f_reinf)

M = homogenize_alv(lam, Laminated(), :C; times = ts)
Mv = homogenize_alv(lam, Voigt(), :C; times = ts)
Mr = homogenize_alv(lam, Reuss(), :C; times = ts)

C33 = [blk(M, i)[3, 3] for i in eachindex(ts)]        # across the layers
C12 = [blk(M, i)[6, 6] / 2 for i in eachindex(ts)]    # in the plane
C33v = [blk(Mv, i)[3, 3] for i in eachindex(ts)]
C33r = [blk(Mr, i)[3, 3] for i in eachindex(ts)]
C12v = [blk(Mv, i)[6, 6] / 2 for i in eachindex(ts)]

@printf "\nrelaxation (τ = %.1f)\n" τ
@printf "  C₃₃₃₃ : %.6f (t=0) → %.6f (t=%.0f),  %.1f %% lost\n" C33[1] C33[end] ts[end] 100 * (1 - C33[end] / C33[1])
@printf "  C₁₂₁₂ : %.6f (t=0) → %.6f (t=%.0f),  %.1f %% lost\n" C12[1] C12[end] ts[end] 100 * (1 - C12[end] / C12[1])

# The two directions relax by very different amounts, and for a structural
# reason: across the layers the stack is a *series* arrangement, so the
# creeping binder dominates and the effective stiffness follows it closely; in
# the plane it is a *parallel* one, so the elastic reinforcement carries the
# load and the loss is only the volume-weighted loss of the binder.

# ## The exact saturations survive the transposition
#
# In-plane Voigt and out-of-plane Reuss are equalities at *every* time.

@printf "\nat every time step :\n"
@printf "  max |C₁₂₁₂ − Voigt| = %.2e\n" maximum(abs, C12 .- C12v)
@printf "  max |C₃₃₃₃ − Reuss| = %.2e\n" maximum(abs, C33 .- C33r)
@printf "  and Reuss ≤ exact ≤ Voigt : %s\n" all(C33r .- 1.0e-12 .≤ C33 .≤ C33v .+ 1.0e-12)

p1 = plot(
    ts, C33; lw = 2, label = "exact (= Reuss)", xlabel = "t", ylabel = "C₃₃₃₃",
    title = "across the layers", legend = :topright
)
plot!(p1, ts, C33v; lw = 2, ls = :dash, label = "Voigt")

p2 = plot(
    ts, C12; lw = 2, label = "exact (= Voigt)", xlabel = "t", ylabel = "C₁₂₁₂",
    title = "in the plane of the layers", legend = :topright
)

plot(p1, p2; layout = (1, 2), size = (900, 350))

# ## Transport
#
# The same transposition at order 2, where the out-of-plane subspace is
# one-dimensional and `volterra_inverse` runs with `block_size = 1`.

K_A, K_B = TensISO{3}(2.0), TensISO{3}(0.3)
lam_k = Laminate(; normal = (0, 0, 1))
add_layer!(lam_k, :A, Dict(:K => heaviside_law(K_A)); thickness = 0.3)
add_layer!(lam_k, :B, Dict(:K => heaviside_law(K_B)); thickness = 0.7)
Mk = homogenize_alv(lam_k, Laminated(), :K; times = [0.0, 1.0, 3.0])
k_perp = Mk[3, 3]
@printf "\ntransport (elastic limit) : 1/k_⊥ = %.10f   series = %.10f\n" 1 / k_perp (0.3 / 2.0 + 0.7 / 0.3)

# ## Interfaces
#
# Elastic interfaces carry over unchanged: a spring compliance still adds to
# the out-of-plane series law, at every time.

lam_i = Laminate(; normal = (0, 0, 1))
add_layer!(
    lam_i, :BINDER, Dict(:C => law_binder); thickness = f_binder,
    interface = SpringInterface(5.0e-2, 5.0e-2)
)
add_layer!(lam_i, :REINF, Dict(:C => law_reinf); thickness = f_reinf)
Mi = homogenize_alv(lam_i, Laminated(), :C; times = ts)
C33i = [blk(Mi, i)[3, 3] for i in eachindex(ts)]
@printf "\nwith a spring interface : C₃₃₃₃ %.6f → %.6f (was %.6f → %.6f)\n" C33i[1] C33i[end] C33[1] C33[end]
@printf "  the interface softens the stack at every time : %s\n" all(C33i .< C33)
