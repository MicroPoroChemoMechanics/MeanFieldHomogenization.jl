# # n-layer sphere: pointwise fields
#
# The strain, stress and displacement **at a point** inside — and outside —
# an n-layer composite sphere, under an arbitrary remote loading, with
# perfect or imperfect interfaces.
#
# Where [`layer_strain_average`](@ref) gives one tensor per layer, the
# pointwise API gives the field itself:
#
# - [`LayeredSphereFields`](@ref)`(sphere, C₀)` — solve the recurrence once;
# - [`local_strain_strain_loc`](@ref)`(sol, x)` — ``\mathbb{A}(x)`` with
#   ``\varepsilon(x) = \mathbb{A}(x):\varepsilon^\infty``, and its three
#   siblings `local_stress_strain_loc`, `local_strain_stress_loc`,
#   `local_stress_stress_loc` for a remote stress;
# - [`local_strain`](@ref), [`local_stress`](@ref),
#   [`local_displacement`](@ref) — the fields for one loading;
# - [`get_layer`](@ref)`(sphere, r; side)` — which region a radius belongs
#   to, and which limit is meant exactly on an interface.
#
# ``\mathbb{A}(x)`` is transversely isotropic about ``\underline n = x/r``
# and carries no major symmetry, so it is a `TensTI{4,T,6}`: six Walpole
# scalars and an axis, not an 81-component array.

import Pkg                                                          #jl
Pkg.activate(joinpath(@__DIR__, "..", "docs"); io = devnull)        #jl

using MeanFieldHomogenization
using TensND
using LinearAlgebra
using Printf
using Plots
gr()

default(; left_margin = 5Plots.mm, bottom_margin = 5Plots.mm)

# ## Setup — a stiff core in a compliant ITZ shell
#
# The classic three-phase model of a cement paste: a stiff aggregate, a
# compliant interfacial transition zone, and the matrix at infinity.

_stiff_Enu(E, ν) = TensISO{3}(3E / (3 * (1 - 2ν)), 2E / (2 * (1 + ν)))

const C₀ = _stiff_Enu(30.0, 0.3)      # matrix
const Cᵢ = _stiff_Enu(100.0, 0.2)     # core (aggregate)
const Cₛ = _stiff_Enu(10.0, 0.35)      # ITZ shell

const R = 1.0                          # core radius
const Rₛ = 1.5                         # shell outer radius

const sphere = LayeredSphere((R, Rₛ), (Cᵢ, Cₛ))
const sol = LayeredSphereFields(sphere, C₀)

# ## Radial profiles under a hydrostatic far field
#
# ``\varepsilon^\infty = \varepsilon_v\,\mathbf 1``. The displacement is
# purely radial, ``u_r = A_k r + B_k/r^2``, so ``\sigma_{rr}`` is continuous
# across every perfect interface while ``\sigma_{\theta\theta}`` jumps with
# the modulus.

const εᵥ = 1.0
const ε∞_hyd = εᵥ * TensISO{3}(1.0)

"Radial and hoop stress at radius `r` under the remote loading `ε∞`."
function radial_hoop(sol, r, ε∞; side = :outer)
    σ = local_stress(sol, [r, 0.0, 0.0], ε∞; side)
    return σ[1, 1], σ[2, 2]              # e₁ is radial here, e₂ is a hoop axis
end

rs = range(1.0e-6, 4 * Rₛ; length = 600)
σrr_h = [radial_hoop(sol, r, ε∞_hyd)[1] for r in rs]
σθθ_h = [radial_hoop(sol, r, ε∞_hyd)[2] for r in rs]

σ_far = 3 * (30.0 / (3 * (1 - 2 * 0.3))) * εᵥ   # 3K₀ εᵥ

println("Hydrostatic far field, σ∞ᵢᵢ = 3K₀εᵥ = ", round(σ_far; digits = 4))
@printf "  %-9s  %12s  %12s\n" "r" "σ_rr" "σ_θθ"
for r in (0.5R, R, 0.5(R + Rₛ), Rₛ, 2Rₛ)
    σrr, σθθ = radial_hoop(sol, r, ε∞_hyd)
    @printf "  %-9.4f  %+12.6f  %+12.6f\n" r σrr σθθ
end

# Continuity of the radial traction is a property of the solution, not
# something imposed on the plot — `side = :inner` and `side = :outer` are
# the two limits at an interface radius.
for r in (R, Rₛ)
    Δ = abs(
        radial_hoop(sol, r, ε∞_hyd; side = :outer)[1] -
            radial_hoop(sol, r, ε∞_hyd; side = :inner)[1]
    )
    @printf "  σ_rr continuity at r = %.2f :  Δ = %.3e\n" r Δ
end

# ## Radial profiles under a deviatoric far field
#
# The deviatoric (``Y_2``) part is what the averaged API could not reach.
# Under a uniaxial ``\varepsilon^\infty = \mathrm{diag}(0,0,1)`` the field is
# no longer radial: it depends on ``\theta`` as well, and both the radial
# and hoop stresses vary along the interface.

const ε∞_dev = Tens(diagm([0.0, 0.0, 1.0]))

# Note the two spellings of a point: a Cartesian vector, as here, or the three
# separate arguments `(r, θ, φ)`.  `local_stress(sol, a, b, c, ε∞)` is the
# SPHERICAL form — passing Cartesian components that way silently evaluates
# somewhere else entirely.
"σ in the meridian plane φ = 0, at colatitude θ, in the canonical basis."
σ_at(sol, r, θ, ε∞; side = :outer) =
    local_stress(sol, [r * sin(θ), 0.0, r * cos(θ)], ε∞; side)

σzz_pole = [σ_at(sol, r, 0.0, ε∞_dev)[3, 3] for r in rs]      # along the load axis
σzz_eq = [σ_at(sol, r, π / 2, ε∞_dev)[3, 3] for r in rs]      # perpendicular to it

# ## A spring interface debonds the core
#
# `SpringInterface(kn, kt)` takes the interface **stiffnesses**; the softer
# the interface, the larger the displacement jump it allows. The traction
# stays continuous.

const sphere_spring = LayeredSphere(
    (R, Rₛ), (Cᵢ, Cₛ);
    interfaces = (SpringInterface(20.0, 12.0), PerfectInterface{Float64}())
)
const sol_spring = LayeredSphereFields(sphere_spring, C₀)

let n = [1.0, 0.0, 0.0], x = R * n
    u⁻ = local_displacement(sol_spring, x, ε∞_hyd; side = :inner)
    u⁺ = local_displacement(sol_spring, x, ε∞_hyd; side = :outer)
    t = local_stress(sol_spring, x, ε∞_hyd; side = :inner)[1, 1]
    @printf "\nSpring interface at r = R:  [u_r] = %.6f,  σ_rr/kn = %.6f\n" (u⁺[1] - u⁻[1]) (t / 20.0)
end

σrr_s = [radial_hoop(sol_spring, r, ε∞_hyd)[1] for r in rs]
σθθ_s = [radial_hoop(sol_spring, r, ε∞_hyd)[2] for r in rs]

# ## Figure 1 — radial stress profiles
#
# Left: hydrostatic loading, perfect interfaces. Middle: the same with a
# spring interface at the core boundary. Right: the deviatoric response,
# along and across the loading axis.

p1 = plot(;
    xlabel = "r", ylabel = "σ / (3K₀εᵥ)", legend = :bottomright, grid = true,
    title = "hydrostatic, perfect interfaces"
)
plot!(p1, rs, σrr_h ./ σ_far; lw = 2, color = :crimson, label = "σ_rr")
plot!(p1, rs, σθθ_h ./ σ_far; lw = 2, color = :navy, label = "σ_θθ")
hline!(p1, [1.0]; lw = 1, ls = :dot, color = :black, label = "σ∞")
vline!(p1, [R, Rₛ]; lw = 1, ls = :dash, color = :grey, label = "")

p2 = plot(;
    xlabel = "r", ylabel = "σ / (3K₀εᵥ)", legend = :bottomright, grid = true,
    title = "hydrostatic, spring at r = R"
)
plot!(p2, rs, σrr_s ./ σ_far; lw = 2, color = :crimson, label = "σ_rr")
plot!(p2, rs, σθθ_s ./ σ_far; lw = 2, color = :navy, label = "σ_θθ")
hline!(p2, [1.0]; lw = 1, ls = :dot, color = :black, label = "σ∞")
vline!(p2, [R, Rₛ]; lw = 1, ls = :dash, color = :grey, label = "")

p3 = plot(;
    xlabel = "r", ylabel = "σ_zz", legend = :bottomright, grid = true,
    title = "deviatoric ε∞ = diag(0,0,1)"
)
plot!(p3, rs, σzz_pole; lw = 2, color = :darkorange, label = "θ = 0 (pole)")
plot!(p3, rs, σzz_eq; lw = 2, color = :seagreen, label = "θ = π/2 (equator)")
vline!(p3, [R, Rₛ]; lw = 1, ls = :dash, color = :grey, label = "")

fig1 = plot(
    p1, p2, p3; layout = (1, 3), size = (1500, 460),
    left_margin = 8Plots.mm, bottom_margin = 8Plots.mm,
    plot_title = "Pointwise stress in a two-layer sphere"
)

# ## Figure 2 — a meridian map of the von Mises stress
#
# The deviatoric loading breaks the spherical symmetry, so the field is
# genuinely two-dimensional. A meridian slice ``(x, z)`` shows the ITZ
# concentrating the shear and the core shielding.

function von_mises(σ)
    s = σ - (tr(σ) / 3) * TensISO{3}(1.0)
    return sqrt(3 / 2 * sum(s[i, j]^2 for i in 1:3, j in 1:3))
end

xs = range(-2.6Rₛ, 2.6Rₛ; length = 220)
zs = range(-2.6Rₛ, 2.6Rₛ; length = 220)
vm = [
    let r = hypot(x, z)
        r < 1.0e-9 ? von_mises(local_stress(sol, [0.0, 0.0, 0.0], ε∞_dev)) :
            von_mises(local_stress(sol, [x, 0.0, z], ε∞_dev))
    end
        for z in zs, x in xs
]

fig2 = heatmap(
    xs, zs, vm;
    aspect_ratio = 1, c = :viridis, xlabel = "x", ylabel = "z",
    title = "von Mises stress, ε∞ = diag(0,0,1)",
    size = (620, 560), right_margin = 6Plots.mm
)
let θ = range(0, 2π; length = 400)
    plot!(fig2, R .* cos.(θ), R .* sin.(θ); lw = 1.5, color = :white, label = "")
    plot!(fig2, Rₛ .* cos.(θ), Rₛ .* sin.(θ); lw = 1.5, color = :white, ls = :dash, label = "")
end

# ## The pointwise field reproduces the layer averages
#
# Averaging ``\mathbb{A}(x)`` over a layer must return the
# ``(\alpha_k, \beta_k)`` the averaged API reports —
# [`shell_localization`](@ref) exposes that identity from the same cached
# amplitudes, so the two routes cannot drift apart.

println("\nlayer   α_k (pointwise)   α_k (averaged)   β_k (pointwise)   β_k (averaged)")
for k in 1:layer_count(sphere)
    αp, βp = shell_localization(sol, k)
    A = strain_strain_loc(sphere, C₀; layer = k)
    αa, βa = TensND.get_data(A)
    @printf "  %d      %14.10f   %14.10f   %14.10f   %14.10f\n" k αp αa βp βa
end

const figdir = joinpath(@__DIR__, "figures")                        #jl
isdir(figdir) || mkdir(figdir)                                      #jl
savefig(fig1, joinpath(figdir, "31_local_nlayers.png"))             #jl
savefig(fig2, joinpath(figdir, "31_local_nlayers_map.png"))         #jl
println("\nSaved : ", joinpath(figdir, "31_local_nlayers.png"))     #jl
println("Saved : ", joinpath(figdir, "31_local_nlayers_map.png"))   #jl
