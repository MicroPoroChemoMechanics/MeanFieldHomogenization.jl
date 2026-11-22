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
# stiffness and the conductivity of a matrix that contains such pores, a
# perforated plate included.

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
# 5.6 % above the tabulated ones. The closed form of the
# [article](@cite ekneligodaPRSA2008) settles the matter: its equations (3.22)
# to (3.30) and (4.9), written for a threefold axis, reduce to

closed_form(m₂, m₅) = 2 / ((1 - 3m₅^2) * (1 - 2m₂^2 - 5m₅^2))
println("shear compliance                closed form   computed   2/(1 - 2m₂² - 5m₅²)")
for row in published_2008["three_terms"]
    @printf "%s   %8.4f   %8.4f   %8.4f\n" name(row) closed_form(row["m1"], row["m2"]) shear_compliance(pore(row)) 2 / (1 - 2 * row["m1"]^2 - 5 * row["m2"]^2)
end

# The formula of the article gives the values computed here for the three maps.
# The two tabulated values that differ are those of the last column, the same
# formula without its factor ``1/(1 - 3\,m_5^2)``.
#
# For the hypotrochoid the system of the potentials is solved by hand, which
# gives closed forms: the compressibility ``2\,(1 + n\,m^2)/(1 - n\,m^2)``,
# recalled in the article, and, for ``n \ge 4``, the shear compliance
# ``2/\bigl((1 - (n-2)\,m^2)(1 - n\,m^2)\bigr)``, its equation (4.8):

n, m = 4, 1 / 10
shear_compliance(ConformalHole(n => m)), 2 / ((1 - (n - 2) * m^2) * (1 - n * m^2))

# ## The shear compliance against the direction of the shear
#
# A section with a fourfold axis is not isotropic in its plane: its shear
# compliance depends on the angle ``\theta`` between the shear and the pore.
# For the hypotrochoid ``n = 3`` the [article](@cite ekneligodaPRSA2008) finds
# (its equation (5.5)) the mean value above multiplied by
# ``1 - m\cos4\theta``. Turning the pore by ``-\theta`` is turning the shear
# by ``\theta``; a coefficient ``m_k`` of a pore turned by ``\alpha`` is
# ``m_k\,\mathrm{e}^{\mathrm{i}(k+1)\alpha}``.

turned(hole, α) = ConformalHole(Tuple(m * cis((k + 1) * α) for (k, m) in pairs(hole.coefficients)))
pure_shear(hole) = 2 * μ₀ * compliance(hole)[2, 3, 2, 3] / (1 - ν₀)
m₃ = 0.2
square_like = ConformalHole(3 => m₃)
θs = range(0, π; length = 73)
plot(
    θs, [pure_shear(turned(square_like, -θ)) for θ in θs]; lw = 2, label = "computed",
    xlabel = "angle θ of the shear", ylabel = "normalized shear compliance", size = (600, 340),
    xticks = ([0, π / 4, π / 2, 3π / 4, π], ["0", "π/4", "π/2", "3π/4", "π"]), title = "Hypotrochoid n = 3, m = $m₃", titlefontsize = 10,
)
plot!(θs, [2 * (1 - m₃ * cos(4θ)) / ((1 - m₃^2) * (1 - 3 * m₃^2)) for θ in θs]; ls = :dash, lw = 2, label = "equation (5.5) of the article")

# ## A scaling law on the perimeter
#
# The same article proposes to estimate the shear compliance of a pore from
# its perimeter ``P`` and its area ``A`` alone, by ``1 + P^2/(4\pi A)`` (its
# equation (6.1)), exact for the circle. The perimeter of the image of the unit
# circle is the integral of ``|\omega'|``:

perimeter(hole; n = 4096) = 2π / n * sum(abs(1 - sum(k * m * ζ^(-k - 1) for (k, m) in pairs(hole.coefficients))) for ζ in cis.(2π .* (0:(n - 1)) ./ n))
area(hole) = π * (1 - sum(k * abs2(m) for (k, m) in pairs(hole.coefficients)))
scaling_law(hole) = 1 + perimeter(hole)^2 / (4π * area(hole))
println("                                scaling law: published, computed   exact shear compliance")
for row in [published_2008["two_terms"]; published_2008["three_terms"]]
    @printf "%s   %8.4f   %8.4f            %8.4f\n" name(row) row["scaling_law"] scaling_law(pore(row)) shear_compliance(pore(row))
end

# The published values of the law are recovered to within 0.002. Against the
# exact compliance, the law is within 1 % up to ``m_2 = 1/3`` and 6 % above it
# for the section with cusps:

family = [ConformalHole(2 => m) for m in range(0, 0.5; length = 26)]
scatter(
    [perimeter(h)^2 / (2π * area(h)) for h in family], shear_compliance.(family); label = "hypotrochoids n = 2, exact",
    xlabel = "P² / (2π A)", ylabel = "normalized shear compliance", legend = :topleft, size = (600, 340), markersize = 3,
)
scatter!([perimeter(pore(r))^2 / (2π * area(pore(r))) for r in published_2008["three_terms"]], [shear_compliance(pore(r)) for r in published_2008["three_terms"]]; label = "maps of three terms, exact", marker = :diamond)
plot!([2, 6.6], [2, 4.3]; ls = :dash, color = :black, label = "1 + P²/(4π A)")

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

# The values agree, except for the pentagon with one term. It is the
# hypotrochoid ``n = 4``, ``m = 1/10`` computed above: equation (4.8) of the
# article gives 2.1259 as well, and its Table 1 quotes 2.1260 from
# [Kachanov, Tsukrov and Shafiro](@cite kachanovAMR1994).
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
# ## The schemes whose reference is the estimate
#
# The scheme of Mori and Tanaka solves each pore in the matrix. The
# self-consistent and the differential schemes solve it in their own estimate,
# which aligned pores make transversely isotropic about their axis when the
# section has an axis of symmetry of order 3, or of order 5 and more. Such a
# medium is isotropic in the plane of the section, and the pore is solved in it
# as in an isotropic one
# ([Holes given by a conformal map](@ref th-conformal-mapping), section 7).
# The medium is here seen in its plane: the cell holds the stiffness of the
# plane of the matrix, a tensor of dimension 2, and the pores as their
# sections ([Plane problems](@ref man-plane-problems)).

const disk = Ellipsoid(1.0; dim = 2)
function plane_cell(c, hole, f; kw...)
    rve = RVE()
    add_phase!(rve, :matrix, disk, Dict(:C => c); fraction = :rest)
    add_phase!(rve, :pores, hole, Dict(:C => 0 * c); fraction = f, kw...)
    return rve
end
const c_strain = plane_strain_stiffness(C₀)
schemes = ["Mori-Tanaka" => MoriTanaka(), "differential" => DifferentialScheme(), "self-consistent" => SelfConsistent()]
fs = range(0, 0.25; length = 11)
plane_bulk(t) = (c = canonical_components(t); (c[1, 1, 1, 1] + c[1, 1, 2, 2]) / 2)
panels = map(["circle" => ConformalHole(()), "triangle" => polygon(3, 100)]) do (name, hole)
    panel = plot(; xlabel = "porosity f", ylabel = "k / k₀", title = name, titlefontsize = 9, legend = name == "circle" ? :bottomleft : false, ylims = (0, 1))
    for (label, scheme) in schemes
        plot!(panel, fs, [plane_bulk(homogenize(plane_cell(c_strain, hole, f), scheme, :C)) / k_matrix for f in fs]; lw = 2, label)
    end
    panel
end
plot(panels...; layout = (1, 2), size = (760, 340), left_margin = 4Plots.mm, bottom_margin = 4Plots.mm)

# The three schemes share their tangent at the origin, which the compressibility
# of the pore fixes, and part as the porosity grows; the self-consistent
# estimate of the triangular pores vanishes at a porosity below that of the
# circular ones.
#
# A square section has a fourfold axis only, and aligned square pores leave the
# plane anisotropic. Oriented at random in the plane, by the average declared
# on the phase, they give an isotropic plane again:

at_random = plane_cell(c_strain, polygon(4, 100), 0.1; symmetrize = IsoSymmetrize())
bulk_shear(isotropify(homogenize(at_random, SelfConsistent(), :C))) ./ (k_matrix, μ_matrix)

# ## A perforated plate
#
# A thin plate loaded in its plane is in plane stress. Its mean fields satisfy
# the equations of plane strain with another bulk modulus, and the cell is the
# same with the plane-stress stiffness of the plate. Young's modulus of a plate
# with holes of four sections, by the self-consistent scheme. The estimate is a
# general tensor of the plane, which these sections leave isotropic: its moduli
# are read on its isotropic part.

const c_stress = plane_stress_stiffness(C₀)
plate_modulus(cell) = first(plane_strain_to_plane_stress(young_poisson(isotropify(homogenize(cell, SelfConsistent(), :C)))...))
E₀ = young_modulus(C₀)
shapes = [
    "circle" => f -> plane_cell(c_stress, ConformalHole(()), f), "hexagon" => f -> plane_cell(c_stress, polygon(6, 100), f),
    "square, at random" => f -> plane_cell(c_stress, polygon(4, 100), f; symmetrize = IsoSymmetrize()),
    "triangle" => f -> plane_cell(c_stress, polygon(3, 100), f),
]
plate = plot(; xlabel = "porosity f", ylabel = "E / E₀", title = "Perforated plate, self-consistent scheme", titlefontsize = 10, legend = :topright, size = (600, 340))
for (name, cell) in shapes
    plot!(plate, fs, [plate_modulus(cell(f)) / E₀ for f in fs]; lw = 2, label = name)
end
plate

# For circular holes the estimate is the line ``1 - 3f``, whatever Poisson's
# ratio of the plate:

plate_modulus(plane_cell(c_stress, ConformalHole(()), 0.1)) / E₀

# ## Conduction
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
