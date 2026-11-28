# # [The elliptic hole, from an isotropic to an anisotropic plate](@id app-elliptic-hole)
#
# !!! info "Before this page"
#     [Holes given by a conformal map](@ref th-conformal-mapping) and the
#     appendices [Complex potentials](@ref th-complex-potentials) and
#     [The elliptic hole in an anisotropic matrix](@ref th-lekhnitskii-ellipse),
#     which derive what is computed here, and
#     [Conformal holes](@ref man-conformal-holes), which gives the syntax.
#
# A plate pulled in its plane carries, along the edge of an elliptic hole, a
# stress several times the applied one. This page computes that stress in an
# isotropic plate, against the closed form of
# [Muskhelishvili](@cite muskhelishvili1977), then in an orthotropic plywood
# sheet, against the formulas and the tables of
# [Lekhnitskii](@cite lekhnitskii1981), follows the stress away from the edge,
# and ends with the same hole as an inclusion of a homogenized sheet and as an
# insulator.

import Pkg                                                          #jl
Pkg.activate(joinpath(@__DIR__, "..", "docs"); io = devnull)                 #jl

using MeanFieldHomogenization
using TensND
using LinearAlgebra
using ForwardDiff
using JSON
using Printf
using Plots
gr()
nothing #hide

# ## The hole and the load
#
# The edge of the hole is the image of the unit circle by
# ``z = \omega(\zeta) = \zeta + m/\zeta``, with ``z = x_1 + \mathrm{i}\,x_2``
# and ``0 \le m < 1``. The point of parameter ``\theta`` is
# ``x_1 = a\cos\theta``, ``x_2 = b\sin\theta``: the hole is an ellipse of
# semi-axes ``a = 1 + m`` and ``b = 1 - m``, its major axis along
# ``\underline{e}_1``; ``\theta = 0`` is the end of the major axis and
# ``\theta = \pi/2`` that of the minor one
# ([Holes given by a conformal map](@ref th-conformal-mapping), section 2).
# The ratio ``c = a/b`` gives ``m = (c - 1)/(c + 1)``.
#
# The plate is pulled at infinity by the uniform stress
# ``p\,\underline{n}\otimes\underline{n}``, where ``\underline{n}`` makes the
# angle ``\varphi`` with the major axis. The edge is free: the only stress
# left there is the normal stress ``\sigma_{tt}`` on the facets normal to the
# tangent ``\underline{t}``, the hoop stress, which
# [`contour_stress`](@ref) returns for a unit pull.

ellipse(c) = ConformalHole(((c - 1) / (c + 1),))                ## a/b = c, the major axis along e₁
uniaxial(φ) = (d = [cos(φ), sin(φ)]; Tens(d * transpose(d)))   ## unit pull along (cos φ, sin φ)
const across = Tens([0.0 0.0; 0.0 1.0])                         ## unit pull along e₂, across the major axis
const θs = range(0, 2π; length = 7201)                          ## the points where the extremes are sought
nothing #hide

# ## The isotropic plate
#
# In an isotropic plate the hoop stress is
# ``\sigma_{tt} = 4\operatorname{Re}\bigl(\Phi'(\sigma)/\omega'(\sigma)\bigr)``,
# where ``\Phi`` is the first complex potential written in the variable
# ``\zeta``: the moduli do not enter, and plane stress and plane strain give
# the same value ([Holes given by a conformal map](@ref th-conformal-mapping),
# section 9). [`contour_stress`](@ref) is then called without a stiffness.
#
# For the circle the classical result is
# ``\sigma_{tt}/p = 1 - 2\cos2(\theta - \varphi)``: three times the pull at the
# points where the edge is parallel to it, minus the pull where it is normal.

circle = ConformalHole(())
contour_stress(circle, across, [0.0, π / 2])                     ## 3 and -1

# [Muskhelishvili](@cite muskhelishvili1977), section 82a, example 1, gives the
# hoop stress around the ellipse ``\zeta + m/\zeta`` under a pull at the angle
# ``\alpha`` from the major axis,
#
# ```math
# \frac{\sigma_{tt}}{p} = \frac{1 - m^2 + 2m\cos2\alpha - 2\cos2(\theta - \alpha)}{1 - 2m\cos2\theta + m^2} .
# ```
#
# Pulled across the major axis, ``\alpha = \pi/2``, the stress at the end of
# that axis is ``(3 + m)/(1 - m) = 1 + 2a/b``: the flatter the hole, the
# larger the concentration.

muskhelishvili(m, α, θ) = (1 - m^2 + 2m * cos(2α) - 2cos(2(θ - α))) / (1 - 2m * cos(2θ) + m^2)
const ratios = (1, 2, 3, 5)
println("a/b   stress at the tip   1 + 2a/b   largest deviation from the closed form")
for c in ratios
    m = (c - 1) / (c + 1)
    σ = contour_stress(ellipse(c), across, θs)
    @printf "%d     %10.6f        %6.3f     %.1e\n" c first(σ) 1 + 2c maximum(abs, σ - muskhelishvili.(m, π / 2, θs))
end

# The stress along the edge, from the end of the major axis (``\theta = 0``) to
# the other one (``\theta = \pi``): the computed curves, and the closed form as
# markers. The hole and the load are unchanged by a half turn, so that
# ``\sigma_{tt}(\theta + \pi) = \sigma_{tt}(\theta)``.

θh = range(0, π; length = 361)
iso_curves = plot(;
    xlabel = "θ", ylabel = "σtt / p", xticks = ([0, π / 2, π], ["0", "π/2", "π"]), legend = :top,
    title = "Isotropic plate pulled across the major axis", titlefontsize = 10, size = (600, 340)
)
for c in ratios
    plot!(iso_curves, θh, contour_stress(ellipse(c), across, θh); lw = 2, label = "a/b = $c")
end
scatter!(
    iso_curves, θh[1:12:end], [muskhelishvili((c - 1) / (c + 1), π / 2, θ) for θ in θh[1:12:end], c in ratios];
    color = :black, markersize = 2, label = ["Muskhelishvili" "" "" ""]
)
iso_curves

# The largest tension sits at the ends of the major axis; the edge is
# compressed, by the pull itself, at the ends of the minor one.
#
# ## Lekhnitskii's plywood plate
#
# [Lekhnitskii](@cite lekhnitskii1981), section 31, solves the same plate
# made of a three-ply birch plywood: an orthotropic sheet, of Young moduli
# ``E_1`` along the fibers of its outer plies and ``E_2`` across them, shear
# modulus ``G`` and Poisson ratio ``\nu_1`` (a pull ``\sigma`` along the
# fibers shortens the sheet across them by ``\nu_1\sigma/E_1``). The hole has
# ``a/b = 2``. The constants are those of his Table 6, the ratio and the
# results those of his section 31:

literature(key) = JSON.parsefile(joinpath(pkgdir(MeanFieldHomogenization), "data", "literature", key * ".json"))
const book = literature("lekhnitskii1981")
quantity(name) = book["quantities"][name]["value"]
const ply = (E₁ = quantity("E1"), E₂ = quantity("E2"), G = quantity("G"), ν₁ = quantity("nu1"))
const c = quantity("axis_ratio")
const hole = ellipse(c)
nothing #hide

#md Main.show_literature("lekhnitskii1981") # hide

# The sheet is in plane stress. In its own axes, ``\underline{e}_1'`` along
# the fibers, its compliance gives
# ``\varepsilon_{11} = (\sigma_{11} - \nu_1\sigma_{22})/E_1``,
# ``\varepsilon_{22} = (\sigma_{22} - \nu_2\sigma_{11})/E_2`` and
# ``\varepsilon_{12} = \sigma_{12}/(2G)``, with ``\nu_2 = \nu_1E_2/E_1``
# by the symmetry of the compliance (0.0355, the 0.036 of the table to its
# rounding). Inverted, its stiffness is the orthotropic tensor of the plane
#
# ```math
# C_{1111} = \frac{E_1}{1 - \nu_1\nu_2},\qquad
# C_{2222} = \frac{E_2}{1 - \nu_1\nu_2},\qquad
# C_{1122} = \frac{\nu_1E_2}{1 - \nu_1\nu_2},\qquad
# C_{1212} = G ,
# ```
#
# written in the frame ``(\underline{e}_1',\underline{e}_2')``, whose first
# vector makes the angle ``\psi`` with the major axis of the hole, counted in
# the same sense as ``\varphi``.

function plywood(E₁, E₂, G, ν₁, frame)
    ν₂ = ν₁ * E₂ / E₁                                          ## the symmetry of the compliance
    d = 1 - ν₁ * ν₂
    return TensOrtho(E₁ / d, E₂ / d, ν₁ * E₂ / d, G, frame)    ## C₁₁₁₁, C₂₂₂₂, C₁₁₂₂, C₁₂₁₂ in the frame
end
plywood(ψ) = plywood(ply.E₁, ply.E₂, ply.G, ply.ν₁, Basis(ψ))
ε = canonical_components(inv(plywood(0.0)) ⊡ uniaxial(0.0))    ## a unit pull along the fibers
ply.E₁ * ε[1, 1], -ε[2, 2] / ε[1, 1]                            ## 1 and ν₁

# ### The characteristic parameters
#
# In an anisotropic plate the stress derives from two functions of the
# variables ``z_k = x_1 + \mu_k x_2``, where ``\mu_1`` and ``\mu_2`` are the
# roots of positive imaginary part of a quartic built on the compliances
# ([The elliptic hole in an anisotropic matrix](@ref th-lekhnitskii-ellipse)).
# In the axes of an orthotropic sheet, with ``a_{11} = 1/E_1``,
# ``a_{22} = 1/E_2``, ``a_{12} = -\nu_1/E_1`` and ``a_{66} = 1/G``, it is
# biquadratic,
#
# ```math
# a_{11}\,\mu^4 + (2a_{12} + a_{66})\,\mu^2 + a_{22} = 0 ,
# ```
#
# and its roots are ``\mu_k = \mathrm{i}\,\beta_k`` with
# ``\beta_1\beta_2 = \sqrt{a_{22}/a_{11}}`` and
# ``\beta_1^2 + \beta_2^2 = (2a_{12} + a_{66})/a_{11}``. With Lekhnitskii's
# parameters ([Lekhnitskii](@cite lekhnitskii1981), Eq. (31.1)),
#
# ```math
# k = \sqrt{E_1/E_2} = \beta_1\beta_2,\qquad
# n = \sqrt{2k + E_1/G - 2\nu_1} = \beta_1 + \beta_2 ,
# ```
#
# so that ``\beta_{1,2} = \bigl(n \pm \sqrt{n^2 - 4k}\bigr)/2``. The
# compliances are read on the tensor, in Kelvin-Mandel form, whose third
# diagonal term is ``a_{66}/2``:

S = TensND.mandel(inv(plywood(0.0)), CanonicalBasis{2, Float64}())
a₁₁, a₂₂, a₁₂, a₆₆ = S[1, 1], S[2, 2], S[1, 2], 2S[3, 3]
q = (2a₁₂ + a₆₆) / a₁₁
β = sqrt.((q .+ (1, -1) .* sqrt(q^2 - 4a₂₂ / a₁₁)) ./ 2)          ## μ₁ = iβ₁, μ₂ = iβ₂
k = sqrt(ply.E₁ / ply.E₂)
n = sqrt(2k + ply.E₁ / ply.G - 2ply.ν₁)
println("            computed   Lekhnitskii, Eq. (31.17)")
for (name, computed, key) in (("k", k, "k"), ("n", n, "n"), ("Im μ₁", β[1], "mu1_imag"), ("Im μ₂", β[2], "mu2_imag"))
    @printf "%-8s   %8.4f   %8.3f\n" name computed quantity(key)
end
@printf "β₁β₂ - k = %.1e   β₁ + β₂ - n = %.1e\n" (β[1] * β[2] - k) (β[1] + β[2] - n)

# ### The hoop stress, four configurations
#
# The stress along the edge for the four pairs (pull ``\varphi``, fibers
# ``\psi``) that Lekhnitskii draws, against the isotropic plate under the same
# pull. The stiffness is the last argument of [`contour_stress`](@ref).

configurations = [(0, 0), (90, 45), (90, 0), (30, 60)]
ply_curves = plot(
    [
        plot(
            θh, [contour_stress(hole, uniaxial(deg2rad(φ)), θh, plywood(deg2rad(ψ))) contour_stress(hole, uniaxial(deg2rad(φ)), θh)];
            lw = 2, ls = [:solid :dash], label = ["plywood" "isotropic"], legend = φ == 0 ? :topright : false,
            xlabel = "θ", ylabel = "σtt / p", xticks = ([0, π / 2, π], ["0", "π/2", "π"]),
            title = "φ = $(φ)°, ψ = $(ψ)°", titlefontsize = 9
        ) for (φ, ψ) in configurations
    ]...; layout = (2, 2), size = (760, 560), left_margin = 4Plots.mm, bottom_margin = 4Plots.mm
)

# With the fibers along the major axis and the pull across it,
# ``(\varphi, \psi) = (90^\circ, 0)``, the stress at the tip of the hole is 7.3
# times the pull, against 5 in the isotropic plate.
#
# ### The ends of the axes
#
# Pulled across the major axis, the sheet has the stress at the ends of the
# axes in closed form ([Lekhnitskii](@cite lekhnitskii1981), Eqs. (31.13) to
# (31.16)): with the fibers along the major axis, ``1 + c\,n/k`` at the end of
# that axis and ``-k`` at the end of the minor one; with the fibers across it,
# ``1 + c\,n`` and ``-1/k``, ``k`` and ``n`` keeping the first axis along the
# fibers.

println("fibers         point          computed   closed form")
for (fibers, ψ, at_tip, at_side) in (("along  (ψ = 0)", 0.0, 1 + c * n / k, -k), ("across (ψ = 90°)", π / 2, 1 + c * n, -1 / k))
    @printf "%-16s θ = 0     %9.6f   %9.6f\n" fibers contour_stress(hole, across, 0.0, plywood(ψ)) at_tip
    @printf "%-16s θ = π/2   %9.6f   %9.6f\n" "" contour_stress(hole, across, π / 2, plywood(ψ)) at_side
end

# ## Tables 8, 9 and 10, recomputed
#
# Lekhnitskii tabulates the largest tension (Table 8) and the largest
# compression (Table 9, as a positive number) along the edge of the plywood
# hole, for five angles ``\varphi`` of the pull and nine angles ``\psi`` of the
# fibers, and the largest tension in the isotropic plate (Table 10). They are
# recomputed on the 7201 points of ``\theta``:

const φs = [row["phi_deg"] for row in book["tables"]["max_tension_plywood"]]
const ψs = [0, 30, 45, 60, 90, 120, 135, 150, 180]
printed(table) = [row["psi_$ψ"] for row in book["tables"][table], ψ in ψs]
extremes = [extrema(contour_stress(hole, uniaxial(deg2rad(φ)), θs, plywood(deg2rad(ψ)))) for φ in φs, ψ in ψs]
tension, compression = last.(extremes), -first.(extremes)
nothing #hide

# The book keeps two decimals. A cell is marked by a star when the computed
# value differs from the printed one by more than one unit of the second
# decimal:

const tolerance = 0.01
function print_table(title, computed, published)
    println(title)
    println(rpad("φ \\ ψ", 7), join(lpad.(ψs, 7)))
    for (i, φ) in pairs(φs)
        cells = [@sprintf("%6.2f%s", computed[i, j], abs(computed[i, j] - published[i, j]) > tolerance ? "*" : " ") for j in eachindex(ψs)]
        println(rpad(φ, 7), join(cells))
    end
    return nothing
end
print_table("largest tension / p (Table 8)", tension, printed("max_tension_plywood"))
print_table("\nlargest compression / p (Table 9)", compression, printed("max_compression_plywood"))

# The marked cells, with both values:

function print_differences(name, computed, published)
    for I in CartesianIndices(computed)
        abs(computed[I] - published[I]) > tolerance &&
            @printf "%-12s φ = %2d°  ψ = %3d°   computed %.3f   printed %.2f\n" name φs[I[1]] ψs[I[2]] computed[I] published[I]
    end
    return nothing
end
print_differences("tension", tension, printed("max_tension_plywood"))
print_differences("compression", compression, printed("max_compression_plywood"))

# In the isotropic plate, beside the largest values of Muskhelishvili's closed
# form on the same points:

isotropic_printed = [row["sigma_max_over_p"] for row in book["tables"]["max_tension_isotropic"]]
println("φ     largest tension: computed   closed form   printed     largest compression: computed")
for (φ, published) in zip(φs, isotropic_printed)
    σ = contour_stress(hole, uniaxial(deg2rad(φ)), θs)
    closed_form = maximum(muskhelishvili((c - 1) / (c + 1), deg2rad(φ), θ) for θ in θs)
    @printf "%2d°                      %6.3f        %6.3f      %5.2f                           %6.3f\n" φ maximum(σ) closed_form published -minimum(σ)
end

# Most cells agree with the printed values. The values computed here meet the
# closed forms of the same book, Eqs. (31.13) to (31.16), to the last digits,
# and the closed form of Muskhelishvili in the isotropic plate; the section on
# the hole as an inclusion checks that it has the mean strain the Hill tensor
# of the ellipse gives, and the tests of the package check that the traction
# vanishes on the edge. On the cell ``\varphi = 90^\circ``, ``\psi = 0`` (and ``180^\circ``),
# Table 8 prints 7.20 where Eq. (31.13) gives ``1 + c\,n/k = 7.298``, the value
# computed. Several cells of the columns ``\psi = 45^\circ`` and ``135^\circ``, and a few
# others, differ by more than the rounding; Table 10 differs at ``45^\circ`` and
# ``60^\circ``. The largest compression in the isotropic plate is the pull itself
# when the pull is along an axis of the hole, and slightly more at the
# intermediate angles.
#
# The largest stresses for the pull across the major axis, against the angle of
# the fibers, with the published points:

ψc = 0:2:180
sweep = [extrema(contour_stress(hole, across, θs, plywood(deg2rad(ψ)))) for ψ in ψc]
i90 = findfirst(==(90), φs)
fiber_sweep = plot(
    plot(
        ψc, last.(sweep); lw = 2, label = "computed", xlabel = "ψ (degrees)", ylabel = "largest tension / p",
        title = "Pull across the major axis: tension", titlefontsize = 9, xticks = 0:30:180
    ),
    plot(
        ψc, -first.(sweep); lw = 2, label = false, xlabel = "ψ (degrees)", ylabel = "largest compression / p",
        title = "Pull across the major axis: compression", titlefontsize = 9, xticks = 0:30:180
    ); layout = (1, 2), size = (760, 320), left_margin = 4Plots.mm, bottom_margin = 4Plots.mm
)
scatter!(fiber_sweep[1], ψs, printed("max_tension_plywood")[i90, :]; color = :black, label = "Table 8")
scatter!(fiber_sweep[2], ψs, printed("max_compression_plywood")[i90, :]; color = :black, label = "Table 9")
fiber_sweep

# and against the angle of the pull, for the fibers along and across the
# major axis and for the isotropic plate:

φc = 0:2:90
pull_sweep = plot(;
    xlabel = "φ (degrees)", ylabel = "largest tension / p", xticks = 0:15:90, legend = :topleft,
    title = "Largest tension against the angle of the pull", titlefontsize = 10, size = (600, 340)
)
for (label, C, column) in (("fibers along, ψ = 0", plywood(0.0), 1), ("fibers across, ψ = 90°", plywood(π / 2), findfirst(==(90), ψs)))
    plot!(pull_sweep, φc, [maximum(contour_stress(hole, uniaxial(deg2rad(φ)), θs, C)) for φ in φc]; lw = 2, label)
    scatter!(pull_sweep, φs, printed("max_tension_plywood")[:, column]; color = :black, markersize = 3, label = column == 1 ? "Table 8" : false)
end
plot!(pull_sweep, φc, [maximum(contour_stress(hole, uniaxial(deg2rad(φ)), θs)) for φ in φc]; lw = 2, ls = :dash, label = "isotropic")
scatter!(pull_sweep, φs, isotropic_printed; color = :black, marker = :diamond, markersize = 3, label = "Table 10")

# ## Away from the edge
#
# The stress is largest on the edge; how fast it comes back to the pull
# decides whether a neighboring hole or an edge of the plate feels the hole.
# [`matrix_stress`](@ref) gives the stress tensor at a point of the plate, in
# the axes and at the scale of the map, with the stiffness as its last
# argument in the anisotropic plate
# ([The stress in the matrix](@ref th-matrix-stress)). Ahead of the tip, on
# the major axis at the distance ``d`` from the end ``x_1 = a`` of the hole,
# under the pull across that axis, the stress ``\sigma_{22}`` is computed for
# the circle, for the ellipse ``a/b = 2`` in the isotropic plate, and for the
# same ellipse in the plywood plate with the fibers along (``\psi = 0``) and
# across (``\psi = 90^\circ``) the hole. For the circle, of radius 1, the
# closed form of Kirsch at ``r = 1 + d``, derived in the same appendix, is
# ``\sigma_{22}/p = 1 + 1/(2r^2) + 3/(2r^4)``.

semi_axes = (2c / (c + 1), 2 / (c + 1))                          ## 1 + m and 1 - m
distance = range(0, 8; length = 801)                             ## from the tip, along e₁
ahead(hole, a, C...) = [canonical_components(σ)[2, 2] for σ in matrix_stress(hole, across, [[a + d, 0.0] for d in distance], C...)]
kirsch(r) = 1 + 1 / (2r^2) + 3 / (2r^4)
curves = [
    "circle" => ahead(circle, 1.0),
    "a/b = 2, isotropic" => ahead(hole, semi_axes[1]),
    "a/b = 2, plywood ψ = 0" => ahead(hole, semi_axes[1], plywood(0.0)),
    "a/b = 2, plywood ψ = 90°" => ahead(hole, semi_axes[1], plywood(π / 2)),
]
maximum(abs, last(curves[1]) - kirsch.(1 .+ distance))           ## the circle against Kirsch

#-

away = plot(;
    xlabel = "distance to the tip", ylabel = "σ₂₂ / p", xlims = (0, 4), legend = :topright,
    title = "Ahead of the tip, pull across the major axis", titlefontsize = 10, size = (600, 340)
)
for (label, σ) in curves
    plot!(away, distance, σ; lw = 2, label)
end
scatter!(away, distance[1:20:201], kirsch.(1 .+ distance[1:20:201]); color = :black, markersize = 2, label = "Kirsch")
hline!(away, [1.1]; color = :gray, ls = :dot, label = "1.1 p")
away

# The distance beyond which ``\sigma_{22}`` stays within 10 % of the pull, on
# the points of the curves. For the circle,
# ``1/(2r^2) + 3/(2r^4) = 1/10`` gives ``1/r^2 = (\sqrt{85} - 5)/30``:

println("hole                        stress at the tip   within 10 % beyond the distance")
for (label, σ) in curves
    i = findlast(s -> abs(s - 1) > 0.1, σ)                        ## the last point outside the band
    @printf "%-27s %10.3f          %5.2f\n" label first(σ) distance[i + 1]
end
@printf "circle, closed form                              %6.3f\n" 1 / sqrt((sqrt(85) - 5) / 30) - 1

# ## The same hole as an inclusion
#
# The potentials that give the stress along the edge also give the
# displacement of the edge, hence the mean strain of the hole per unit remote
# strain: its strain concentration tensor. The ellipse has besides a Hill
# tensor in any matrix of the plane
# ([Hill polarization tensors](@ref th-hill-plane)), from which the package
# computes the concentration of `Ellipsoid(a, b)`. The two routes, for several
# angles of the fibers:

for ψ in (0, 30, 45, 90)
    C = plywood(deg2rad(ψ))
    A_map = canonical_components(gradient_gradient_concentration(hole, 0 * C, C))
    A_hill = canonical_components(gradient_gradient_concentration(Ellipsoid(semi_axes...), 0 * C, C))
    @printf "ψ = %2d°   largest component %.4f   largest difference %.1e\n" ψ maximum(abs, A_hill) maximum(abs, A_map - A_hill)
end

# A plywood sheet with aligned holes, at the area fraction ``f = 0.05``, the
# fibers along the major axes: its Young moduli along and across the holes,
# divided by those of the sheet, by the dilute and the Mori-Tanaka schemes,
# with the holes as `ConformalHole` and as `Ellipsoid`:

function perforated(C, f; shape = hole)
    rve = RVE()
    add_phase!(rve, :matrix, Ellipsoid(1.0; dim = 2), Dict(:C => C); fraction = :rest)
    add_phase!(rve, :holes, shape, Dict(:C => 0 * C); fraction = f)
    return rve
end
young_moduli(C) = (S = canonical_components(inv(C)); (1 / S[1, 1, 1, 1], 1 / S[2, 2, 2, 2]))   ## along e₁ and e₂
E_sheet = young_moduli(plywood(0.0))
for (name, scheme) in ("dilute" => Dilute(), "Mori-Tanaka" => MoriTanaka())
    E_map = young_moduli(homogenize(perforated(plywood(0.0), 0.05), scheme, :C)) ./ E_sheet
    E_hill = young_moduli(homogenize(perforated(plywood(0.0), 0.05; shape = Ellipsoid(semi_axes...)), scheme, :C)) ./ E_sheet
    @printf "%-12s ConformalHole %.5f %.5f   Ellipsoid %.5f %.5f\n" name E_map[1] E_map[2] E_hill[1] E_hill[2]
end

# ## The same hole as an insulator
#
# In conduction the hole is an insulator, and the elliptic hole takes any
# symmetric conductivity of the plane
# ([Conduction around the insulating ellipse](@ref th-ellipse-conduction)). The
# conductivities below are illustrative, not taken from the book: an
# orthotropic sheet that conducts three times better along its fibers than
# across them, the fibers at the angle ``\psi`` from the major axis. The
# gradient concentration tensor of the hole, by its map and by the Hill tensor
# of `Ellipsoid(a, b)`, for several angles; it is not symmetric when the
# fibers are oblique:

conductor(ψ) = TensOrtho{2}(3.0, 1.0, Basis(ψ))                    ## illustrative: along and across the fibers
for ψ in (0, 30, 45, 90)
    K = conductor(deg2rad(ψ))
    A_map = canonical_components(gradient_gradient_concentration(hole, 0 * K, K))
    A_hill = canonical_components(gradient_gradient_concentration(Ellipsoid(semi_axes...), 0 * K, K))
    @printf "ψ = %2d°   A₁₁ %.4f   A₂₂ %.4f   A₁₂ - A₂₁ %+.1e   largest difference %.1e\n" ψ A_map[1, 1] A_map[2, 2] (A_map[1, 2] - A_map[2, 1]) maximum(abs, A_map - A_hill)
end

# With the fibers along the major axis the tensor is diagonal,
# ``A_{11} = 1 + (b/a)\sqrt{k_1/k_2}`` and ``A_{22} = 1 + (a/b)\sqrt{k_2/k_1}``,
# ``k_1 = 3`` and ``k_2 = 1`` the conductivities along and across the fibers:

1 + sqrt(3.0) / c, 1 + c / sqrt(3.0)

# ## Derivatives
#
# The moduli of the sheet are differentiated through the solution. The stress
# at the tip of the hole, fibers along the major axis, is ``1 + c\,n/k``; with
# ``k' = k/(2E_1)`` and ``n' = (2k' + 1/G)/(2n)``, its derivative with respect
# to ``E_1``, the other constants held, is ``c\,(n'k - n\,k')/k^2``:

tip(E₁) = contour_stress(hole, across, 0.0, plywood(E₁, ply.E₂, ply.G, ply.ν₁, Basis(0.0)))
k′ = k / (2ply.E₁)
n′ = (2k′ + 1 / ply.G) / (2n)
ForwardDiff.derivative(tip, ply.E₁), c * (n′ * k - n * k′) / k^2

# The second derivative, by nested duals, against that of the closed form:

closed_tip(E₁) = (κ = sqrt(E₁ / ply.E₂); 1 + c * sqrt(2κ + E₁ / ply.G - 2ply.ν₁) / κ)
second(f, x) = ForwardDiff.derivative(y -> ForwardDiff.derivative(f, y), x)
second(tip, ply.E₁), second(closed_tip, ply.E₁)

# ## The tip stress on symbols
#
# The orthotropic class has its characteristic roots in closed form: on a
# `TensOrtho` of symbolic constants, given in its own axes, the hoop stress is
# an expression of the constants.

import Symbolics
Symbolics.@variables E₁ E₂ G ν₁
sheet = plywood(E₁, E₂, G, ν₁, CanonicalBasis{2, Symbolics.Num}())
σ_tip = contour_stress(hole, across, 0.0, sheet)
nothing #hide

# At the constants of the plywood it is Eq. (31.13). With the two moduli
# exchanged, and ``\nu_2 = \nu_1E_2/E_1`` in place of ``\nu_1``, the same
# expression describes the sheet whose fibers lie across the hole, and gives
# Eq. (31.14), ``1 + c\,n``:

value(x, v) = Symbolics.value(Symbolics.substitute(x, v; fold = Val(true)))
along_hole = Dict(E₁ => ply.E₁, E₂ => ply.E₂, G => ply.G, ν₁ => ply.ν₁)
across_hole = Dict(E₁ => ply.E₂, E₂ => ply.E₁, G => ply.G, ν₁ => ply.ν₁ * ply.E₂ / ply.E₁)
(value(σ_tip, along_hole), 1 + c * n / k), (value(σ_tip, across_hole), 1 + c * n)

# ## Where to go next
#
# - [The elliptic hole in an anisotropic matrix](@ref th-lekhnitskii-ellipse):
#   the derivation of the potentials used here, and why the ellipse is the one
#   hole solved in an anisotropic plate.
# - [The stress in the matrix](@ref th-matrix-stress) and
#   [Conduction around the insulating ellipse](@ref th-ellipse-conduction):
#   the stress away from the edge, and the insulating hole.
# - [Holes given by a conformal map](@ref th-conformal-mapping): the hole of
#   several terms in an isotropic matrix, and the stress along its edge.
# - [Conformal holes](@ref man-conformal-holes): the syntax.
# - [Prismatic pores with corners](@ref app-conformal-holes): holes with
#   corners, against published values.
# - [Plane problems](@ref man-plane-problems): cells of the plane with an
#   anisotropic matrix.
# - [Elliptical holes and cracks in an anisotropic plate](@ref app-elliptical-holes-anisotropic):
#   the next page, the compliance of such holes and of cracks, and the anisotropy
#   of a plate with many of them.
