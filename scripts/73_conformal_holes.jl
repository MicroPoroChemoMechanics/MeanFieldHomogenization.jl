# # [Prismatic pores with corners](@id app-conformal-holes)
#
# !!! info "Before this page"
#     [Holes given by a conformal map](@ref th-conformal-mapping), which derives
#     the tensors used here, and [Conformal holes](@ref man-conformal-holes),
#     which gives the syntax.
#
# The pores of a ceramic or of a rock are seldom elliptic in section. This page
# takes prismatic pores whose section is a hypotrochoid or a polygon, compares
# their compressibility and their shear compliance with the values published by
# [Ekneligoda and Zimmerman](@cite ekneligodaPRSA2006) for maps of two to four
# terms, carries the polygons to the limit of the map, and ends with the
# stiffness and the conductivity of a matrix that contains such pores.

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

# ## The pores and the two numbers compared
#
# The section of the pore is the image of the unit circle by
# ``z = \zeta + \sum_k m_k\,\zeta^{-k}``. A single term ``m_n`` gives a
# hypotrochoid, with ``n + 1`` rounded corners; the map of a regular polygon,
# cut after a few terms, gives a polygon with rounded corners
# ([`polygon_map`](@ref)).

contour(hole) = [ζ + sum(m * ζ^(-k) for (k, m) in pairs(hole.coefficients)) for ζ in cis.(range(0, 2π; length = 721))]
function draw(holes; kw...)
    panels = [
        plot(
            real.(contour(h)), imag.(contour(h)); aspect_ratio = 1, fill = (0, 0.35, :orange), color = :black,
            legend = false, axis = false, grid = false, title = name, titlefontsize = 8
        ) for (name, h) in holes
    ]
    return plot(panels...; layout = (1, length(holes)), size = (170 * length(holes), 190), kw...)
end
draw(
    [
        "m₂ = 1/9" => ConformalHole(2 => 1 / 9), "m₂ = 1/3" => ConformalHole(2 => 1 / 3),
        "m₂ = 1/2" => ConformalHole(2 => 1 / 2), "m₂ = 1/3, m₅ = 1/15" => ConformalHole(2 => 1 / 3, 5 => 1 / 15),
        "m₂ = 1/9, m₅ = 7/45" => ConformalHole(2 => 1 / 9, 5 => 7 / 45),
    ]
)

# The matrix is isotropic, of shear modulus ``\mu`` and Poisson ratio ``\nu``,
# in plane strain in the plane of the section. Two numbers measure the pore:
#
# - its *compressibility*, the relative change of its area under a unit
#   hydrostatic stress in the plane;
# - its *shear compliance*, its shear strain, in the engineering sense, under a
#   unit shear stress in the plane.
#
# Both are proportional to ``(1 - \nu)/\mu``
# ([Holes given by a conformal map](@ref th-conformal-mapping), section 4), and
# both equal 2 for the circle once divided by this factor. They are read on the
# compliance contribution tensor ``\mathbb H`` of the pore, whose axis is
# ``\underline{e}_1``:

const K₀, μ₀ = 1.7, 0.6                                  ## bulk and shear moduli of the matrix
const ν₀ = (3K₀ - 2μ₀) / (2 * (3K₀ + μ₀))
const C₀ = iso_stiffness(K₀, μ₀)
const void = 0 * C₀
compliance(hole) = canonical_components(inverse_property_contribution(hole, void, C₀))
function compressibility(hole)
    H = compliance(hole)
    Σ = (2ν₀, 1.0, 1.0)                                  ## hydrostatic in the plane; the axial stress is the reaction of plane strain
    return sum(H[i, i, j, j] * Σ[j] for i in 2:3, j in 1:3) * μ₀ / (1 - ν₀)
end
function shear_compliance(hole)                          ## averaged over the directions of the shear in the plane
    H = compliance(hole)
    return μ₀ * (H[2, 3, 2, 3] + (H[2, 2, 2, 2] - 2H[2, 2, 3, 3] + H[3, 3, 3, 3]) / 4) / (1 - ν₀)
end
compressibility(ConformalHole(())), shear_compliance(ConformalHole(()))      ## the circle

# A pore with a threefold axis is isotropic in its plane and the average does
# nothing; the square is not, and the published values are averaged.

# ## Published values: maps of two and three terms
#
# [Ekneligoda and Zimmerman](@cite ekneligodaPRSA2006) solve the hydrostatic
# loading for maps of up to four terms, and the shear in a
# [second article](@cite ekneligodaPRSA2008). Their tables give pores with a
# threefold axis, ``z = \zeta + m_2\,\zeta^{-2}`` and
# ``z = \zeta + m_2\,\zeta^{-2} + m_5\,\zeta^{-5}`` in the notation of this
# page.

literature(key) = JSON.parsefile(joinpath(pkgdir(MeanFieldHomogenization), "data", "literature", key * ".json"))["tables"]
const published_2006 = literature("ekneligodaPRSA2006")
const published_2008 = literature("ekneligodaPRSA2008")
nothing #hide

#md Main.show_literature("ekneligodaPRSA2006") # hide

#md Main.show_literature("ekneligodaPRSA2008") # hide

pore(row) = haskey(row, "m2") ? ConformalHole(2 => row["m1"], 5 => row["m2"]) : ConformalHole(2 => row["m1"])
name(row) = haskey(row, "m2") ? @sprintf("m₂ = %.4f, m₅ = %.4f", row["m1"], row["m2"]) : @sprintf("m₂ = %.4f             ", row["m1"])
println("compressibility                 published   computed")
for row in [published_2006["two_terms"]; published_2006["three_terms"]]
    @printf "%s   %8.4f   %8.4f\n" name(row) row["analytical"] compressibility(pore(row))
end
println("\nshear compliance                published   computed")
for row in [published_2008["two_terms"]; published_2008["three_terms"]]
    @printf "%s   %8.4f   %8.4f\n" name(row) row["analytical"] shear_compliance(pore(row))
end

# The compressibilities agree to the digits printed, and so do the shear
# compliances of the maps of two terms and of the last map of three terms. For
# the two other maps of three terms the values computed here are 1.3 % and
# 5.6 % above the published ones. The test suite of the package checks the
# potentials of such maps against a direct solution of the condition on the
# contour, by collocation.
#
# For the hypotrochoid the system of the potentials is solved by hand, which
# gives closed forms: the compressibility ``2\,(1 + n\,m^2)/(1 - n\,m^2)``,
# recalled in the article, and, for ``n \ge 4``, the shear compliance
# ``2/\bigl((1 - (n-2)\,m^2)(1 - n\,m^2)\bigr)``:

n, m = 4, 1 / 10
shear_compliance(ConformalHole(n => m)), 2 / ((1 - (n - 2) * m^2) * (1 - n * m^2))

# ## Regular polygons
#
# The map of a regular polygon has infinitely many terms. The articles keep
# two to four of them and extrapolate; the package solves the system for any
# number of terms, here up to 400, a system of size 800.

polygon(sides, terms) = ConformalHole(polygon_map(sides, terms)...)
draw(["$terms term(s)" => polygon(4, terms) for terms in (1, 2, 4, 30)])

# The two numbers against the number of terms, for the square:

terms = [1, 2, 3, 4, 6, 10, 20, 50, 100, 200, 400]
square_values = [(compressibility(polygon(4, t)), shear_compliance(polygon(4, t))) for t in terms]
plot(
    terms, [first.(square_values) last.(square_values)]; xscale = :log10, marker = :circle, lw = 2,
    xticks = ([1, 10, 100], ["1", "10", "100"]), label = ["compressibility" "shear compliance"],
    xlabel = "number of terms of the map", ylabel = "normalized value", legend = :right,
    title = "Prismatic pore of square section", titlefontsize = 10, size = (600, 340)
)

# The quasi-polygons of the second article, the map cut after its first or
# second term (which the article, counting ``\zeta``, calls two and three
# terms):

println("shear compliance      1 term: published, computed     2 terms: published, computed")
for row in published_2008["quasi_polygons"]
    s = row["sides"]
    @printf "%d sides                    %.4f    %.4f               %.4f    %.4f\n" s row["two_terms"] shear_compliance(polygon(s, 1)) row["three_terms"] shear_compliance(polygon(s, 2))
end

# The values agree, except for the pentagon with one term, which is the
# hypotrochoid ``n = 4``, ``m = 1/10`` of the closed form above.
#
# At the limit of the map, beside the values the first article extrapolates
# from three and four terms and those it computes by boundary elements:

println("compressibility   400 terms   extrapolated (article)   boundary elements (article)   shear compliance, 400 terms")
for row in published_2006["polygons"]
    s = row["sides"]
    @printf "%d sides           %.4f          %.3f                      %.3f                         %.4f\n" s compressibility(polygon(s, 400)) row["extrapolated"] row["boundary_elements"] shear_compliance(polygon(s, 400))
end

# The extrapolation of the article is confirmed to within 0.1 %. The
# compressibility of the square tends to a value close to ``1 + \sqrt2``, from
# which it differs, at 400 terms, by

compressibility(polygon(4, 400)) - (1 + sqrt(2))

# ## A matrix with prismatic pores
#
# Aligned pores of the same section, at the porosity ``f``, by the scheme of
# Mori and Tanaka. The stiffness is compared through two moduli of the plane
# of the sections: the plane-strain bulk modulus
# ``k = (C_{2222} + C_{2233})/2`` and the shear modulus averaged over the
# directions of the plane.

function plane_moduli(hole, f)
    rve = RVE()
    add_phase!(rve, :matrix, Ellipsoid(1.0), Dict(:C => C₀); fraction = :rest)
    add_phase!(rve, :pores, hole, Dict(:C => void); fraction = f)
    C = canonical_components(homogenize(rve, MoriTanaka(), :C))
    return (; k = (C[2, 2, 2, 2] + C[2, 2, 3, 3]) / 2, μ = (C[2, 3, 2, 3] + (C[2, 2, 2, 2] - C[2, 2, 3, 3]) / 2) / 2)
end
sections = ["circle" => ConformalHole(()), "hexagon" => polygon(6, 100), "square" => polygon(4, 100), "triangle" => polygon(3, 100)]
fs = range(0, 0.3; length = 16)
k_matrix, μ_matrix = K₀ + μ₀ / 3, μ₀
bulk = plot(; xlabel = "porosity f", ylabel = "k / k₀", title = "plane-strain bulk modulus", titlefontsize = 9, legend = :bottomleft)
shear = plot(; xlabel = "porosity f", ylabel = "μ / μ₀", title = "shear modulus in the plane", titlefontsize = 9, legend = false)
for (name, hole) in sections
    moduli = [plane_moduli(hole, f) for f in fs]
    plot!(bulk, fs, [x.k / k_matrix for x in moduli]; lw = 2, label = name)
    plot!(shear, fs, [x.μ / μ_matrix for x in moduli]; lw = 2)
end
plot(bulk, shear; layout = (1, 2), size = (760, 340), left_margin = 4Plots.mm, bottom_margin = 4Plots.mm)

# The fewer the sides, the softer the porous material: at equal porosity a
# triangular section lowers the plane bulk modulus markedly more than a
# circular one, and a hexagonal section is hardly told from it.
#
# In conduction the answer is one number. A regular polygon has no term
# ``m_1``, so that the gradient concentration tensor of an insulating prismatic
# pore is isotropic in its plane, ``2/(1 - \sum k\,|m_k|^2)``: twice the ratio
# of the areas of the unit disk and of the section.

const k₀ = TensISO{3}(1.0)
for (name, hole) in sections
    A = canonical_components(gradient_gradient_concentration(hole, 0 * k₀, k₀))
    @printf "%-8s   along the axis %.4f   in the plane %.4f\n" name A[1, 1] A[2, 2]
end

# The conductivity across aligned pores then follows from the scheme as for
# any inclusion; along the pores it is that of the mixture law, the pores being
# prisms.
