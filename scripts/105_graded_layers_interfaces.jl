# # [Graded isotropic layers with interfaces at given radii](@id tut-graded-layers-interfaces)
#
# !!! info "Before this page"
#     [Layered inclusions](@ref man-layered), for the syntax of a layered
#     sphere and of its interfaces, and
#     [Layered spheres and cylinders: the radial problem](@ref th-radial-problems),
#     section 7, for the integration of a layer whose moduli vary.
#
# Around a particle the matrix is often altered over a zone where its bulk and
# shear moduli vary with the distance to the particle, and the particle may be
# imperfectly bonded to that zone. This page enters such a pattern as it is: a
# sphere, then a fiber, whose layers have moduli ``\kappa(r)`` and ``\mu(r)``
# given as functions, with interfaces at chosen radii. The functions are
# integrated across each layer; nothing is cut into thin layers, except to
# check the result.

import Pkg                                                          #jl
Pkg.activate(joinpath(@__DIR__, "..", "docs"); io = devnull)                 #jl

using MeanFieldHomogenization
using TensND
using LinearAlgebra
using JSON
using ForwardDiff
using Plots
gr()
nothing #hide

# ## 1. A particle and its zone
#
# The profile is that of [Lutz and Zimmerman](@cite lutzIJSS2005). Around a
# particle of radius ``a``, the moduli of the matrix are ``\kappa_m`` and
# ``\mu_m`` far from it and ``(1 - D)`` times these at its surface:
#
# ```math
# \kappa(r) = \kappa_m\,\Bigl[1 - D\,\Bigl(\frac{a}{r}\Bigr)^{b}\Bigr] ,
# \qquad
# \mu(r) = \mu_m\,\Bigl[1 - D\,\Bigl(\frac{a}{r}\Bigr)^{b}\Bigr] .
# ```
#
# The data are those of their example: an exponent ``b = 10``, a particle five
# times stiffer than the matrix, a Poisson ratio of ``1/4`` everywhere, and a
# zone softer than the matrix, ``D = 0.5``.

data = JSON.parsefile(joinpath(pkgdir(MeanFieldHomogenization), "data", "literature", "lutzIJSS2005.json"))["quantities"]
val(name) = data[name]["value"]
b, contrast, ν = val("profile_exponent"), val("stiffness_contrast"), val("poisson_ratio")
nothing #hide

#md Main.show_literature("lutzIJSS2005") # hide

#-

a, κ_m = 1.0, 1.0
μ_m = 3κ_m * (1 - 2ν) / (2 * (1 + ν))
C_m, C_p = iso_stiffness(κ_m, μ_m), iso_stiffness(contrast * κ_m, contrast * μ_m)
softening(r, D) = 1 - D * (a / r)^b
zone(D) = r -> iso_stiffness(κ_m * softening(r, D), μ_m * softening(r, D))
nothing #hide

# The profile reaches the matrix at infinity. The zone is ended at the radius
# ``r_2 = 2a``, where it differs from the matrix by ``D\,2^{-10}``, about
# ``5\times10^{-4}``: beyond, the matrix is uniform. A layered sphere takes a
# function of the radius as the modulus of a layer:

r₂ = 2a
bonded = LayeredSphere((a, r₂), (C_p, zone(0.5)))

# The second layer is a [`GradedLayer`](@ref): the system of the first order
# of each loading mode is integrated across it, with step control. The sphere
# is used as any other inclusion; its stiffness contribution tensor
# ``\mathbb{N}`` in the matrix, per unit volume of the sphere of radius
# ``r_2``, is isotropic, ``\mathbb{N} = 3n_\kappa\,\mathbb{J} + 2n_\mu\,\mathbb{K}``:

contribution(sphere) = k_mu(property_contribution(sphere, C_m, C_m))          # (n_κ, n_μ)
contribution(bonded)

# ## 2. Interfaces at given radii
#
# An interface is given at the outer radius of each layer, the last one being
# the boundary with the matrix. Here the particle is bonded to its zone by
# springs, of normal and tangential stiffnesses ``k_n`` and ``k_t``
# (`SpringInterface`): the displacement jumps at ``r = a`` in proportion to
# the traction.

kₙ, kₜ = 20κ_m / a, 8κ_m / a
sprung = LayeredSphere((a, r₂), (C_p, zone(0.5)); interfaces = (SpringInterface(kₙ, kₜ), PerfectInterface()))
contribution(sprung)

# An interface may also lie inside the zone, at any radius: the zone is then
# two layers, each one given by the same function, with the interface between
# them. Below, a membrane of surface moduli ``\kappa^{\mathrm{s}}``,
# ``\mu^{\mathrm{s}}`` (`MembraneInterface`), across which the traction jumps,
# at the radius ``r_1 = 1.2a``:

r₁ = 1.2a
sheet = MembraneInterface(0.5κ_m * a, 0.2κ_m * a)
sphere = LayeredSphere((a, r₁, r₂), (C_p, zone(0.5), zone(0.5)); interfaces = (SpringInterface(kₙ, kₜ), sheet, PerfectInterface()))
contribution(sphere)

# The two layers need not follow the same function, and the moduli may jump
# at the radius that separates them. The profile and the three radii:

rs = range(0, 2.5a; length = 600)
κ_of(r) = r < a ? contrast * κ_m : r < r₂ ? κ_m * softening(r, 0.5) : κ_m
plot(rs, κ_of.(rs); lw = 2, label = "κ(r) / κₘ", xlabel = "r / a", ylabel = "bulk modulus", legend = :topright,
    size = (600, 320), left_margin = 4Plots.mm, bottom_margin = 4Plots.mm)
vline!([a, r₁, r₂]; linestyle = :dash, color = :gray, label = "interfaces: spring, membrane, perfect")

# ## 3. Two checks
#
# **Thin uniform layers.** `graded_layers` cuts a zone into uniform layers that
# follow its profile. With the same interfaces at the same radii, their answer
# tends to that of the integrated layers as the square of their thickness:

function layered(n)
    inner_radii, inner_moduli = graded_layers(zone(0.5), a, r₁, n)
    outer_radii, outer_moduli = graded_layers(zone(0.5), r₁, r₂, 4n)
    perfect(m) = ntuple(_ -> PerfectInterface(), m)
    interfaces = (SpringInterface(kₙ, kₜ), perfect(n - 1)..., sheet, perfect(4n)...)
    return LayeredSphere((a, inner_radii..., outer_radii...), (C_p, inner_moduli..., outer_moduli...); interfaces)
end
ns = [2, 4, 8, 16, 32]
exact = contribution(sphere)
errors = [abs(contribution(layered(n))[2] / exact[2] - 1) for n in ns]
plot(5 .* ns, errors; xscale = :log10, yscale = :log10, marker = :circle, label = "uniform layers",
    xticks = (5 .* ns, string.(5 .* ns)), xlabel = "number of layers in the zone",
    ylabel = "relative difference on the shear part", size = (600, 320), left_margin = 4Plots.mm, bottom_margin = 4Plots.mm)
plot!(5 .* ns, errors[1] .* (ns[1] ./ ns) .^ 2; linestyle = :dash, label = "n⁻²")

# **A layer crossed exactly.** When the moduli follow a power of the radius, a
# [`PowerLawLayer`](@ref) is crossed without integration, by the power of a
# matrix. The same zone entered as a function gives the same tensor, to the
# tolerance of the integration, with the same interfaces:

β = -1.5
interfaces = (SpringInterface(kₙ, kₜ), PerfectInterface())
as_function = LayeredSphere((a, r₂), (C_p, r -> (r / r₂)^β * C_m); interfaces)
as_power = LayeredSphere((a, r₂), (C_p, PowerLawLayer(C_m, β; at = r₂)); interfaces)
contribution(as_function), contribution(as_power)

# ## 4. Effective moduli, and how they depend on the bond
#
# In a scheme the phase is the particle with its zone. For a volume fraction
# ``c`` of particles its fraction is ``c\,(r_2/a)^3``, the zone being counted
# up to ``r_2``. With the Mori-Tanaka scheme:

function effective(sphere; c = 0.05)
    rve = RVE()
    add_phase!(rve, :matrix, Ellipsoid(1.0), Dict(:C => C_m); fraction = :rest)
    add_phase!(rve, :particles, sphere, Dict(:C => C_p); fraction = c * (r₂ / a)^3)
    return k_mu(homogenize(rve, MoriTanaka(), :C))
end
effective(bonded), effective(sprung)

# The stiffness of the bond is swept from a debonded particle to a perfectly
# bonded one, for a zone softer than the matrix, absent, and stiffer
# (``D = 0.5``, ``0`` and ``-0.5``, three of the values of the article). Each
# point is one integration across the zone.

stiffnesses = 10 .^ range(-2, 3; length = 26)
plt = plot(; xscale = :log10, xticks = 10.0 .^ (-2:3), xlabel = "kₙ a / κₘ   (kₜ = 0.4 kₙ)",
    ylabel = "effective shear modulus μ / μₘ", legend = :bottomright, size = (640, 380),
    left_margin = 4Plots.mm, bottom_margin = 4Plots.mm)
for D in (0.5, 0.0, -0.5)
    μs = [effective(LayeredSphere((a, r₂), (C_p, zone(D)); interfaces = (SpringInterface(s, 0.4s), PerfectInterface())))[2] for s in stiffnesses]
    plot!(plt, stiffnesses, μs ./ μ_m; lw = 2, label = "D = $D")
end
plt

# Dual numbers go through the integration: the derivative of an effective
# modulus with respect to a parameter of the profile, or of an interface, is
# obtained by automatic differentiation.

shear(D, s) = effective(LayeredSphere((a, r₂), (C_p, zone(D)); interfaces = (SpringInterface(s, 0.4s), PerfectInterface())))[2]
ForwardDiff.derivative(D -> shear(D, kₙ), 0.5), ForwardDiff.derivative(s -> shear(0.5, s), kₙ)

# ## 5. Conduction
#
# The conductivity of the zone follows the same profile, and the particle is
# five times more conductive than the matrix, as in the second example of the
# article. A Kapitza resistance ``\rho`` at the surface of the particle makes
# the temperature jump there:

k_m, k_p = 1.0, val("conductivity_contrast") * 1.0
conducting_zone(D) = r -> TensISO{3}(k_m * softening(r, D))
resistant = LayeredSphere((a, r₂), (TensISO{3}(k_p), conducting_zone(0.5)); interfaces = (KapitzaInterface(0.3a / k_m), PerfectInterface()))
property_contribution(resistant, TensISO{3}(k_m), TensISO{3}(k_m))

# ## 6. A fiber and its graded coating
#
# A [`LayeredCylinder`](@ref) takes the same layers and the same interfaces.
# Below, a fiber transversely isotropic about its axis, given by Hill's moduli
# ``(k, l, m, n, p)``, in a coating whose isotropic moduli follow the profile
# above, with a spring between them; the composite of aligned fibers is
# transversely isotropic about their axis.

ez = (0.0, 0.0, 1.0)
carbon = ti_from_constants(15κ_m, 6κ_m, 5κ_m, 150κ_m, 8κ_m, ez; form = :hill)
fiber = LayeredCylinder((a, r₂), (carbon, zone(0.5)); interfaces = (SpringInterface(kₙ, kₜ), PerfectInterface()))
composite = RVE()
add_phase!(composite, :matrix, LayeredCylinder((1.0,), (C_m,)), Dict(:C => C_m); fraction = :rest)
add_phase!(composite, :fibers, fiber, Dict(:C => carbon); fraction = 0.1 * (r₂ / a)^2)
hill_moduli(best_fit_ti(homogenize(composite, MoriTanaka(), :C), ez))

# ## Where to go next
#
# - [A graded interphase around a spherical particle](@ref app-graded-interphase):
#   the same zone without interface, as thin layers, against the exact series
#   of the article, and its two figures recomputed.
# - [Layered spheres and cylinders: the radial problem](@ref th-radial-problems):
#   the systems that are integrated, and the layers crossed exactly.
# - [Layered sphere: the systems of the first order, symbolically](@ref tut-layered-sphere-first-order-symbolic):
#   their matrices for moduli ``\kappa(r)`` and ``\mu(r)``, built from the field
#   equations.
# - [Layered inclusions](@ref man-layered): the syntax, and the notations of
#   the moduli of a layer.
