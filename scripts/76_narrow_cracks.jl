# # [Narrow cracks: rings, rectangles, bands](@id app-narrow-cracks)
#
# !!! info "Before this page"
#     The manual page [Planar cracks of any contour](@ref man-planar-cracks), and
#     the theory page of the [same name](@ref th-gaussian-cracks), §5, for the
#     normalization of the compliance of a crack of any contour.
#
# A crack much longer than wide opens, away from its ends, as a crack in plane
# strain of the same width. [Pronina, Markov and Kachanov](@cite pronina2026)
# build on this a *plane strain model* of the compliance of narrow cracks, and
# correct it by factors fitted on computations for a ring and for a rectangle.
# This page computes the same cracks with [`PlanarCrack`](@ref) and sets the
# results against the model, against the published fits, and against the cracks
# for which the package has another solution: the annular crack and the ellipse.

import Pkg                                                          #jl
Pkg.activate(joinpath(@__DIR__, "..", "docs"); io = devnull)                 #jl

using MeanFieldHomogenization
using TensND
using LinearAlgebra
using Printf
using Plots
gr()

# ## What is compared
#
# The contribution of a flat crack of area ``S`` to the compliance of a solid is
# set by ``S\,B_N``, the integral of its normal opening per unit normal
# traction. In an isotropic matrix it is obtained from the conduction problem,
# which has one unknown per node
# ([Cracks with contacts between their faces](@ref app-crack-contacts)): with
# ``E' = E/(1-\nu^{2})``, ``k_0`` the conductivity and ``\ell`` the reference
# length of the crack,
#
# ```math
# S\,B_N\,E' = 2\pi\,\ell^{3}\,k_0\,b ,
# ```
#
# where ``b`` is the thermal COD scalar that `cod_tensor` returns. All the
# values below are ``S\,B_N\,E'``, a volume, in units of the cube of a length of
# the crack.

const K₀ = TensISO{3}(1.0)
SBN(crack) = 2π * semi_minor(crack)^3 * cod_tensor(crack, K₀)      ## semi_minor: the reference length
gaussian(region; n) = SBN(PlanarCrack(region; size = 1.0, n))
nothing #hide

# The check on a disk of radius 1, for which ``S\,B_N\,E' = 16/3``:

gaussian(Disk(1.0); n = 128), 16 / 3

# ## The plane strain model
#
# A line crack of half-width ``l`` under a unit traction has the mean opening
# ``\pi l/E'``. A narrow crack of midline ``\Gamma`` and local half-width
# ``l(\tau)`` is seen as a succession of such line cracks:
#
# ```math
# S\,B_N^{\mathrm{PS}}\,E' = 2\pi\int_{\Gamma} l^{2}(\tau)\,\mathrm{d}\tau .
# ```
#
# It ignores the ends of the crack and the curvature of its midline.

# ## The ring
#
# A ring of radii ``R = 1`` and ``r = \xi R``: half-width ``(R-r)/2``, midline
# of length ``\pi(R+r)``, hence
# ``S\,B_N^{\mathrm{PS}}E' = \tfrac{\pi^{2}}{2}R^{3}(1-\xi)^{2}(1+\xi)``. The
# paper corrects the model by a polynomial ``y_1(\xi)`` fitted on its
# computations. The package has the [`AnnularCrack`](@ref), solved by a
# Galerkin method.

plane_strain_ring(ξ) = π^2 / 2 * (1 - ξ)^2 * (1 + ξ)
y₁(ξ) = -0.0739ξ^3 + 0.2169ξ^2 - 0.2211ξ + 1.0778
ξs = 0.2:0.1:0.9
rings = map(ξs) do ξ
    (
        ps = plane_strain_ring(ξ), fit = plane_strain_ring(ξ) * y₁(ξ),
        galerkin = SBN(AnnularCrack(1.0, ξ)), gauss = gaussian(Annulus(1.0, ξ); n = ξ > 0.75 ? 320 : 128),
    )
end
for (ξ, r) in zip(ξs, rings)
    @printf(
        "ξ = %.1f:  plane strain %.4f   fit of the paper %.4f   Galerkin %.4f   Gaussian %.4f\n",
        ξ, r.ps, r.fit, r.galerkin, r.gauss
    )
end

#-

plt = plot(; xlabel = "ξ = r / R", ylabel = "S B_N / plane strain model", legend = :topright)
plot!(plt, ξs, [r.galerkin / r.ps for r in rings]; lw = 2, label = "annular crack (Galerkin)")
scatter!(plt, ξs, [r.gauss / r.ps for r in rings]; label = "Gaussian approximation")
plot!(plt, ξs, y₁.(ξs); ls = :dash, lw = 2, label = "fit y₁ of the paper")
plt

# The three agree: the largest gaps over the range are

@printf(
    "Gaussian against Galerkin: %.2f %%     fit of the paper against Galerkin: %.2f %%\n",
    100 * maximum(abs(r.gauss / r.galerkin - 1) for r in rings),
    100 * maximum(abs(r.fit / r.galerkin - 1) for r in rings)
)

# The plane strain model alone, below the three, is off by

@printf(
    "%.1f %% at ξ = 0.3, %.1f %% at ξ = 0.5, %.1f %% at ξ = 0.8\n",
    (100 * (1 - rings[i].ps / rings[i].galerkin) for i in (2, 4, 7))...
)

# ## The rectangle
#
# A rectangle of half-sides ``a = \lambda b`` and ``b``: the model gives
# ``S\,B_N^{\mathrm{PS}}E' = 4\pi\lambda\,b^{3}``. The ends are not narrow, and
# what they take away from the model is the point of the comparison. The paper
# gives a fit ``y_2(\lambda)`` of its computations in two pieces, and recalls
# two estimates by an equivalent circular crack: that of Budiansky and O'Connell,
# built on the area and the perimeter, and that of Fabrikant.

y₂(λ) = λ ≤ 5 ? 8.208e-3 * λ^3 - 9.782e-2 * λ^2 + 3.998e-1 * λ + 0.231 :
    3.939e-6 * λ^3 - 3.032e-4 * λ^2 + 1.036e-2 * λ + 0.766
budiansky(λ) = 128λ^2 / (3π * (λ + 1)) / (4π * λ)
fabrikant(λ) = 32λ^2 / (3sqrt(λ^2 + 1)) / (4π * λ)

λs = (1, 2, 3, 5, 10, 20, 30)
across = (64, 64, 48, 48, 32, 24, 24)                 ## nodes across the width
rectangles = [gaussian(Rectangle(float(λ), 1.0); n = m * λ) for (λ, m) in zip(λs, across)]
for (λ, v) in zip(λs, rectangles)
    @printf(
        "λ = %2d:  S B_N E' = %8.3f   over the model: Gaussian %.3f   fit y₂ %.3f   B.-O'C. %.3f   Fabrikant %.3f\n",
        λ, v, v / (4π * λ), y₂(λ), budiansky(λ), fabrikant(λ)
    )
end

#-

plt = plot(;
    xscale = :log10, xlabel = "λ = a / b", ylabel = "S B_N / plane strain model", legend = :bottomright,
    ylims = (0.5, 1.08)
)
plot!(plt, collect(λs), rectangles ./ (4π .* λs); marker = :circle, lw = 2, label = "Gaussian approximation")
fine = 10 .^ range(0, log10(30); length = 60)
plot!(plt, fine, y₂.(fine); ls = :dash, lw = 2, label = "fit y₂ of the paper")
plot!(plt, fine, budiansky.(fine); ls = :dot, label = "Budiansky and O'Connell")
plot!(plt, fine, fabrikant.(fine); ls = :dashdot, label = "Fabrikant")
plt

# What the two ends remove from the model, ``4\pi\lambda\,b^{3} - S\,B_N E'``,
# varies little once the rectangle is three times longer than wide: the ends no
# longer see each other.

for (λ, v) in zip(λs, rectangles)
    @printf("λ = %2d:  end deficit %.2f b³\n", λ, 4π * λ - v)
end

# The values computed here lie above the fit of the paper, by

@printf(
    "%.0f %% at λ = 1, %.0f %% at λ = 10, %.0f %% at λ = 30\n",
    (100 * (rectangles[i] / (4π * λs[i]) / y₂(λs[i]) - 1) for i in (1, 5, 7))...
)

# whereas the two agree on the ring. Two checks support the present values. They
# do not depend on the grid: the square, with 32 to 128 nodes across,

[round(gaussian(Rectangle(1.0, 1.0); n) / 4π; digits = 4) for n in (32, 64, 128)]

# and an ellipse as elongated as these rectangles is within two thousandths of
# its closed form (``\ell^{3}b = a\,b^{2}\,b^{\mathcal{E}}``):

for a in (10.0, 30.0)
    exact = 2π * a * cod_tensor(EllipticCrack(a, 1.0), K₀)
    @printf("ellipse %2.0f:1   Gaussian / closed form - 1 = %.4f\n", a, gaussian(Ellipse(a, 1.0); n = round(Int, 24a)) / exact - 1)
end

# With these values, the rectangle reaches 95 % of the plane strain model at
# ``\lambda = 10`` and 98 % at ``\lambda = 30``. It is close to Fabrikant's
# estimate at ``\lambda = 2`` and to that of Budiansky and O'Connell at
# ``\lambda = 5``; neither follows it over the whole range.

# ## The ellipse: a crack that narrows gradually
#
# For an ellipse of semi-axes ``a`` and ``b = \gamma a`` the half-width is
# ``b\sqrt{1-x^{2}/a^{2}}`` and the model gives
# ``S\,B_N^{\mathrm{PS}}E' = \tfrac{8\pi}{3}a\,b^{2}``. The closed form of the
# package is exact. Without blunt ends, the model holds much earlier than for
# the rectangle:

γs = 0.05:0.05:1.0
ratio = [2π * cod_tensor(EllipticCrack(1.0, γ), K₀) / (8π / 3) for γ in γs]   ## a = 1: ℓ³ b = b² bᴱ
plot(
    γs, ratio; lw = 2, xlabel = "γ = b / a", ylabel = "S B_N / plane strain model", label = "ellipse, closed form",
    ylims = (0.6, 1.02)
)

#-

@printf(
    "model within 1 %% for γ ≤ %.2f, within 5 %% for γ ≤ %.2f\n",
    γs[findlast(>(0.99), ratio)], γs[findlast(>(0.95), ratio)]
)

# The paper reports the model essentially exact below ``\gamma = 0.1`` and
# within 5 % below 0.2.

# ## A band of any midline
#
# The regions combine: a band along an arc of circle, with rounded or blunt
# ends, is a ring cut by a sector. The half-angle ``\alpha`` of the arc goes
# from a short band to the closed ring.

function arc_band(R, r, α)
    sector = Region(x -> abs(atan(x[2], x[1])) ≤ α, (-R, -R), (R, R))
    return Annulus(R, r) ∩ sector
end
αs = (π / 6, π / 3, π / 2, 2π / 3, 5π / 6)
bands = [gaussian(arc_band(1.0, 0.7, α); n = 192) for α in αs]
for (α, v) in zip(αs, bands)
    model = 2π * 0.15^2 * (2α * 0.85)                 ## half-width 0.15, midline of radius 0.85
    @printf("2α = %3.0f°:  S B_N E' = %.4f   plane strain model %.4f   ratio %.3f\n", rad2deg(2α), v, model, v / model)
end

# The model, which ignores the two blunt ends, overestimates a short band and
# is approached as the band lengthens, exactly as for the rectangle.

# ## Where to go next
#
# - [Cracks with contacts between their faces](@ref app-crack-contacts): the
#   same solver on a disk with islands.
# - [Planar cracks of any contour](@ref man-planar-cracks): the ways to describe
#   a contour.
