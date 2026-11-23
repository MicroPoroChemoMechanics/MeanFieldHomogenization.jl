# # [Coated spheres in an aging viscoelastic matrix](@id app-aging-coated-spheres)
#
# !!! info "Before this page"
#     [Layered inclusions](@ref man-layered) for the layered sphere, and
#     [Aging viscoelasticity](@ref man-viscoelasticity) for the operators of the aging
#     case.
#
# A mortar or a concrete at early age is a composite of elastic particles in a
# matrix that creeps and whose properties still change with its age. This page
# takes the material of [honorioIJSS2017](@cite): elastic spheres, possibly
# coated, in a matrix whose relaxation functions are those of a generalized
# Maxwell model multiplied by a function of the age of loading. It computes its
# effective relaxation functions by the Mori-Tanaka scheme and by the
# generalized self-consistent scheme, in which the exact solution of the
# layered sphere is used with operators of time in place of the moduli.

import Pkg                                                          #jl
Pkg.activate(joinpath(@__DIR__, "..", "docs"); io = devnull)                 #jl

using MeanFieldHomogenization
using TensND
using LinearAlgebra
using JSON
using Printf
using Plots
gr()
nothing #hide

# ## The phases
#
# The bulk and the shear relaxation functions of an aging phase are
#
# ```math
# k(t,t')=\Bigl(k^0+\sum_{j=1}^{3}k^j\,\mathrm{e}^{-(t-t')/\tau^j}\Bigr)f_{\mathrm{age}}(t'),
# \qquad f_{\mathrm{age}}(t')=\tfrac34\bigl(1-\mathrm{e}^{-t'}\bigr),
# ```
#
# and the same with ``\mu``, ``t'`` being the age at loading, in days. The
# parameters are read from the article.

data = JSON.parsefile(joinpath(pkgdir(MeanFieldHomogenization), "data", "literature", "honorioIJSS2017.json"))
parameters(phase) = only(row for row in data["tables"]["maxwell"] if row["phase"] == phase)
quantity(name) = data["quantities"][name]["value"]
nothing #hide

#md Main.show_literature("honorioIJSS2017") # hide

f_age(t′) = quantity("aging_amplitude") * (1 - exp(-t′))
function relaxation(p, prefix, t, t′)
    return (p[prefix * "0"] + sum(p["$prefix$j"] * exp(-(t - t′) / p["tau$j"]) for j in 1:3)) * f_age(t′)
end
## A law of the package: the relaxation tensor at (t, t′), zero before the loading.
aging_law(p) = ViscoLaw(
    (t, t′) -> t ≥ t′ ? iso_stiffness(relaxation(p, "k", t, t′), relaxation(p, "mu", t, t′)) : iso_stiffness(0.0, 0.0),
    :relaxation,
)
elastic_law(C) = ViscoLaw((t, t′) -> t ≥ t′ ? C : 0 * C, :relaxation)

matrix = aging_law(parameters("matrix"))
coating = aging_law(parameters("coating"))
core = elastic_law(iso_stiffness_young_poisson(quantity("E_core"), quantity("nu_core")))
nothing #hide

# ## The instants
#
# The operators are discretized on a grid of instants. The relaxation
# function at an age ``t'`` is the response to a unit step of strain applied
# at that age: each age is entered twice in the grid, so that the step is a
# jump, and is followed by instants in geometric progression up to 20 days.

ages = (0.5, 1.0, 2.0, 5.0, 10.0)
times = sort(vcat([[a; a; a .+ 10 .^ range(-3, log10(20.0 - a); length = 14)] for a in ages]...))
step_at(age) = [i ≥ findlast(==(age), times) ? 1.0 : 0.0 for i in eachindex(times)]
## The bulk and the shear relaxation functions at an age, from the blocks 3k and 2μ of an operator.
function relaxation_functions(R, age)
    α, β = iso_params_from_blocks(Matrix(R))
    later = findlast(==(age), times):length(times)
    return times[later], (α * step_at(age))[later] ./ 3, (β * step_at(age))[later] ./ 2
end
length(times)

# ## Two schemes
#
# With 40 % of spheres, the Mori-Tanaka estimate is that of the scheme of the
# package on laws of time:

function two_phases(fraction)
    rve = RVE()
    add_phase!(rve, :matrix, Ellipsoid(1.0), Dict(:C => matrix); fraction = :rest)
    add_phase!(rve, :spheres, Ellipsoid(1.0), Dict(:C => core); fraction)
    return rve
end
fraction = quantity("inclusion_fraction")
mori_tanaka = homogenize_alv(two_phases(fraction), MoriTanaka(), :C; times)
nothing #hide

# In the generalized self-consistent scheme the composite sphere, a core in a
# shell of matrix in the proportions of the material, is embedded in the
# effective medium. Under a remote strain history its mean strain and its mean
# stress are given by two operators, ``\widetilde{\mathbb{A}}`` and
# ``\widetilde{\mathbb{B}}``, which depend on the effective relaxation
# operator ``\widetilde{\mathbb{C}}``; the scheme requires
#
# ```math
# \widetilde{\mathbb{C}}=\widetilde{\mathbb{B}}(\widetilde{\mathbb{C}})\circ\widetilde{\mathbb{A}}(\widetilde{\mathbb{C}})^{-1} ,
# ```
#
# the product and the inverse being those of Volterra, which are the product
# and the inverse of the matrices of the operators. The two operators of a
# layered sphere in an aging reference are functions of the package; the fixed
# point is iterated from the Mori-Tanaka estimate.

function generalized_self_consistent(sphere, start; tolerance = 1.0e-10)
    R = Matrix(start)
    for iteration in 1:100
        A = gradient_gradient_concentration_alv(sphere, R, times)
        B = flux_gradient_concentration_alv(sphere, R, times)
        change = norm(B / A - R) / norm(R)
        R = B / A
        change < tolerance && return R, iteration
    end
    error("the generalized self-consistent iteration did not converge")
end
radii(fractions) = Tuple(cbrt.(cumsum(collect(fractions))))
gsc, iterations = generalized_self_consistent(LayeredSphere(radii((fraction, 1 - fraction)), (core, matrix)), mori_tanaka)
iterations

# The relaxation functions at the five ages (the case of Fig. 7 of the
# article), the generalized self-consistent scheme in solid lines and the
# Mori-Tanaka scheme in dashed lines:

function draw(estimates; title)
    panels = map((2, 3), ("k_eff (GPa)", "μ_eff (GPa)")) do component, label
        panel = plot(; xlabel = "time (days)", ylabel = label, legend = component == 2 ? :bottomright : false, xlims = (0, 20), ylims = (0, 30))
        for (i, age) in enumerate(ages), (name, R, style) in estimates
            curve = relaxation_functions(R, age)
            plot!(panel, curve[1], curve[component]; color = i, ls = style, lw = 2, label = age == last(ages) ? name : "")
        end
        panel
    end
    return plot(panels...; layout = (1, 2), size = (860, 360), plot_title = title, plot_titlefontsize = 10, left_margin = 4Plots.mm, bottom_margin = 4Plots.mm)
end
draw((("generalized self-consistent", gsc, :solid), ("Mori-Tanaka", mori_tanaka, :dash)); title = "40 % of elastic spheres, loading at 0.5, 1, 2, 5 and 10 days")

# The two bulk relaxation functions coincide: under a hydrostatic loading the
# composite sphere in its matrix is the assemblage of Hashin, whose bulk
# modulus the two schemes share, in the aging case as in elasticity. The shear
# functions differ, here at the last age:

let (_, k₁, μ₁) = relaxation_functions(gsc, 10.0), (_, k₂, μ₂) = relaxation_functions(mori_tanaka, 10.0)
    @printf "largest relative difference at 10 days: bulk %.1e, shear %.1e\n" maximum(abs, k₁ ./ k₂ .- 1) maximum(abs, μ₁ ./ μ₂ .- 1)
end

# ## A check: the instant of loading
#
# At the instant of loading the response is elastic, with the moduli of that
# age: the first value of each relaxation function is the generalized
# self-consistent estimate of the elastic composite whose matrix has the
# moduli ``k(t',t')`` and ``\mu(t',t')``, computed by the elastic scheme on
# the layered sphere.

function elastic_gsc(age)
    C_matrix = visco_eval(matrix, age, age)
    sphere = LayeredSphere(radii((fraction, 1 - fraction)), (visco_eval(core, age, age), C_matrix))
    rve = RVE()
    add_phase!(rve, :composite, sphere, Dict(:C => C_matrix); fraction = 1.0)
    return bulk_shear(isotropify(homogenize(rve, SelfConsistent(; init = C_matrix), :C)))
end
for age in ages
    _, k, μ = relaxation_functions(gsc, age)
    kₑ, μₑ = elastic_gsc(age)
    @printf "age %4.1f d:  k %.4f (elastic %.4f)   μ %.4f (elastic %.4f)\n" age first(k) kₑ first(μ) μₑ
end

# ## Coated spheres
#
# With a coating, itself aging, the composite sphere has three layers: 25 % of
# core, 15 % of coating and the matrix (the case of Figs. 10 and 11 of the
# article). The same iteration applies; it starts from the estimate without
# coating.

coated = LayeredSphere(radii((0.25, 0.15, 0.6)), (core, coating, matrix))
gsc_coated, iterations = generalized_self_consistent(coated, gsc)
draw((("coated spheres", gsc_coated, :solid), ("spheres without coating", gsc, :dash)); title = "Generalized self-consistent scheme: 25 % of core and 15 % of coating, or 40 % of core")
