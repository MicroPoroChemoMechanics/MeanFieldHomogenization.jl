# # [Elliptical holes and cracks in an anisotropic plate](@id app-elliptical-holes-anisotropic)
#
# !!! info "Before this page"
#     [The elliptic hole, from an isotropic to an anisotropic plate](@ref app-elliptic-hole),
#     which computes the stress around one hole, [Plane problems](@ref man-plane-problems)
#     for the cells of the plane, and [Crack opening displacement and compliance](@ref th-cod-tensors)
#     for the tensors of a crack.
#
# A plate of orthotropic material holds elliptical holes and cracks, without
# interaction. This page computes the compliance contribution of one hole and
# of one crack in such a plate, against the closed forms of
# [Tsukrov and Kachanov](@cite tsukrov2000), checks that an ellipse
# contributes as a circle plus two cracks, and recomputes the effective
# anisotropy of a plate with randomly oriented ellipses.

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

# ## The plate
#
# The plate is in plane stress, of Young moduli ``E_1`` and ``E_2`` along the
# axes ``\underline{e}_1`` and ``\underline{e}_2`` of its orthotropy, shear
# modulus ``G_{12}`` and Poisson ratio ``\nu_{12}``. Its compliance gives
# ``\varepsilon_{11} = \sigma_{11}/E_1 - \nu_{12}\sigma_{22}/E_1``,
# ``\varepsilon_{22} = \sigma_{22}/E_2 - \nu_{12}\sigma_{11}/E_1`` and
# ``\varepsilon_{12} = \sigma_{12}/(2G_{12})``. The constants are those of
# the figures of [Tsukrov and Kachanov](@cite tsukrov2000), ratios only;
# ``E_2`` is the unit of stress:

literature(key) = JSON.parsefile(joinpath(pkgdir(MeanFieldHomogenization), "data", "literature", key * ".json"))
const paper = literature("tsukrov2000")
quantity(name) = paper["quantities"][name]["value"]
const E₂ = 1.0
const E₁ = quantity("E1_over_E2") * E₂
const G₁₂ = E₁ / quantity("E1_over_G12")
const ν₁₂ = quantity("nu12")
(E₁ = E₁, E₂ = E₂, G₁₂ = G₁₂, ν₁₂ = ν₁₂, ν₂₁ = ν₁₂ * E₂ / E₁)

#-
#md Main.show_literature("tsukrov2000") # hide

# The stiffness of the plane is the inverse of that compliance, an
# orthotropic tensor of the plane in the axes of the plate
# ([Plane problems](@ref man-plane-problems)):

function plate(E₁, E₂, G₁₂, ν₁₂)
    ν₂₁ = ν₁₂ * E₂ / E₁
    d = 1 - ν₁₂ * ν₂₁
    return TensOrtho(E₁ / d, E₂ / d, ν₁₂ * E₂ / d, G₁₂, CanonicalBasis{2, Float64}())   ## C₁₁₁₁, C₂₂₂₂, C₁₁₂₂, C₁₂₁₂
end
const C₀ = plate(E₁, E₂, G₁₂, ν₁₂)
young_moduli(C) = (S = canonical_components(inv(C)); (1 / S[1, 1, 1, 1], 1 / S[2, 2, 2, 2]))
young_moduli(C₀)

# ## The compliance contribution
#
# Holes of area fraction ``f`` add ``f\,\mathbb{H}`` to the compliance of the
# plate; cracks of half-length ``l``, ``N`` per unit area, add
# ``\pi\rho\,\mathbb{H}`` with ``\rho = Nl^2``. The article writes the
# contributions per unit area of the plate, ``\pi ab/A`` times the first
# ``\mathbb{H}`` for a hole of semi-axes ``a`` and ``b`` and ``\pi l^2/A``
# times the second for a crack. Three combinations of the constants enter:
#
# ```math
# L = \sqrt{\frac{1}{G_{12}} - \frac{2\nu_{12}}{E_1} + \frac{2}{\sqrt{E_1E_2}}},
# \qquad
# C = \frac{\sqrt{E_1} + \sqrt{E_2}}{2\sqrt{E_1E_2}}\,L,
# \qquad
# D = \frac{\sqrt{E_1} - \sqrt{E_2}}{\sqrt{E_1} + \sqrt{E_2}} .
# ```

const L = sqrt(1 / G₁₂ - 2ν₁₂ / E₁ + 2 / sqrt(E₁ * E₂))
const Cc = (sqrt(E₁) + sqrt(E₂)) / (2sqrt(E₁ * E₂)) * L
const D = (sqrt(E₁) - sqrt(E₂)) / (sqrt(E₁) + sqrt(E₂))
(L = L, C = Cc, D = D)

# ### The circular hole
#
# A hole is an inclusion of zero stiffness, here `Ellipsoid(a, b)` of the
# plane; [`compliance_contribution`](@ref) gives ``\mathbb{H}``. For the
# circle, Eq. (27) of the article reads
#
# ```math
# H_{1111} = \frac{1 + L\sqrt{E_1}}{E_1},\quad
# H_{2222} = \frac{1 + L\sqrt{E_2}}{E_2},\quad
# H_{1122} = -\frac{1}{\sqrt{E_1E_2}},\quad
# H_{1212} = \frac{L}{4}\Bigl(\frac{1}{\sqrt{E_1}} + \frac{1}{\sqrt{E_2}} + L\Bigr) .
# ```

hole_H(shape, C) = compliance_contribution(shape, 0 * C, C)
circle_closed_form(E₁, E₂, L) = (
    H₁₁₁₁ = (1 + L * sqrt(E₁)) / E₁, H₂₂₂₂ = (1 + L * sqrt(E₂)) / E₂,
    H₁₁₂₂ = -1 / sqrt(E₁ * E₂), H₁₂₁₂ = L / 4 * (1 / sqrt(E₁) + 1 / sqrt(E₂) + L),
)
components(H) = (h = canonical_components(H); (H₁₁₁₁ = h[1, 1, 1, 1], H₂₂₂₂ = h[2, 2, 2, 2], H₁₁₂₂ = h[1, 1, 2, 2], H₁₂₁₂ = h[1, 2, 1, 2]))
disk = Ellipsoid(1.0; dim = 2)
components(hole_H(disk, C₀)), circle_closed_form(E₁, E₂, L)

# The circle lowers the stiffness more along the stiff axis:
# ``E_1 H_{1111} = 1 + L\sqrt{E_1}`` exceeds ``E_2 H_{2222}``. In an
# isotropic plate, ``L = 2/\sqrt{E}`` and
# ``(H_{1111}, H_{1122}, H_{1212}) = (3, -1, 2)/E``:

C_iso = plate(1.0, 1.0, 1 / (2 * (1 + 0.3)), 0.3)
components(hole_H(disk, C_iso)), circle_closed_form(1.0, 1.0, 2.0)

# The article states the isotropic case in words below Eq. (27) with
# ``H_{1122} = \pi a^2/(AE_0)``; Eq. (27) itself, its Eq. (33) for
# ``\nu_{12}``, and the computation give the negative sign: the hole raises the
# Poisson ratio over Young's modulus, ``\nu_{12}/E_1``.
#
# ### The crack
#
# `RibbonCrack(l; angle = φ)` is a crack of the plane of half-length ``l``,
# whose line makes the angle ``\varphi`` with ``\underline{e}_1``; its unit
# vectors are ``\underline{t} = (\cos\varphi, \sin\varphi)`` along it and
# ``\underline{n} = (-\sin\varphi, \cos\varphi)`` normal to it. Its opening
# under the traction ``\underline{n}\cdot\boldsymbol{\sigma}`` is the tensor
# ``\boldsymbol{B}`` ([`cod_tensor`](@ref)), and
# ``\mathbb{H} = \frac{2}{\pi}\,\underline{n}\otimes^s\boldsymbol{B}\otimes^s\underline{n}``.
# [Tsukrov and Kachanov](@cite tsukrov2000), after Mauge and Kachanov, find
# ``\boldsymbol{B}`` independent of the orientation of the crack in an
# orthotropic plate, with the eigenvalues ``\pi C(1 \pm D)/2`` along the axes
# of the plate. The computed tensor, for several orientations:

crack(φ) = RibbonCrack(1.0; angle = φ)
println("φ (degrees)   B₁₁        B₂₂        B₁₂")
for φ in (0, 30, 45, 60, 90, 135)
    b = canonical_components(cod_tensor(crack(deg2rad(φ)), C₀))
    @printf "%5d        %.6f   %.6f   %+.1e\n" φ b[1, 1] b[2, 2] b[1, 2]
end
(πC_1_minus_D_over_2 = π * Cc * (1 - D) / 2, πC_1_plus_D_over_2 = π * Cc * (1 + D) / 2)

# The tensor does not depend on ``\varphi``, and
#
# ```math
# \boldsymbol{B} = \frac{\pi C}{2}\Bigl[(1 - D)\,\underline{e}_1\otimes\underline{e}_1
#   + (1 + D)\,\underline{e}_2\otimes\underline{e}_2\Bigr] :
# ```
#
# a crack opens more under a traction along the soft axis ``\underline{e}_2``.
# The article prints its Eq. (25) with ``1 + D`` on ``\underline{e}_1`` and
# ``1 - D`` on ``\underline{e}_2``, the sign of ``D`` reversed with respect
# to the computation and to its own Eq. (24).
#
# The components of ``\mathbb{H}`` in the frame of the crack follow from
# ``\boldsymbol{B}``: with ``B_{nn} = \underline{n}\cdot\boldsymbol{B}\cdot\underline{n}``
# and the like, ``H_{nnnn} = \frac{2}{\pi}B_{nn}``,
# ``H_{tnnn} = \frac{1}{\pi}B_{tn}`` and ``H_{tntn} = \frac{1}{2\pi}B_{tt}``,
# that is
#
# ```math
# H_{nnnn} = C(1 + D\cos2\varphi),\qquad
# H_{tnnn} = \frac{CD}{2}\sin2\varphi,\qquad
# H_{tntn} = \frac{C}{4}(1 - D\cos2\varphi) .
# ```

frame(φ) = (Tens([cos(φ), sin(φ)]), Tens([-sin(φ), cos(φ)]))
function crack_components(H, φ)
    t, n = frame(φ)
    return (nnnn = (n ⊗ n) ⊡ H ⊡ (n ⊗ n), tnnn = (t ⊗ n) ⊡ H ⊡ (n ⊗ n), tntn = (t ⊗ n) ⊡ H ⊡ (t ⊗ n))
end
from_B(φ) = (nnnn = Cc * (1 + D * cos(2φ)), tnnn = Cc * D / 2 * sin(2φ), tntn = Cc / 4 * (1 - D * cos(2φ)))
println("φ (degrees)   H_nnnn                H_tnnn                 H_tntn")
println("              computed   formula    computed   formula     computed   formula")
for φ in (0, 30, 60, 90)
    h, r = crack_components(compliance_contribution(crack(deg2rad(φ)), C₀), deg2rad(φ)), from_B(deg2rad(φ))
    @printf "%5d        %.6f   %.6f   %+.6f  %+.6f    %.6f   %.6f\n" φ h.nnnn r.nnnn h.tnnn r.tnnn h.tntn r.tntn
end

# These are the first two lines of Eq. (24) of the article; its third line
# prints ``H_{tntn} = C(1 - D\cos2\varphi)``, without the factor ``1/4``
# that its Eq. (9) gives. The effective moduli derived further on, Eqs. (39)
# and (44) below, agree with the computation. A crack normal to the stiff
# axis, ``\varphi = 90^\circ``, opens least.
#
# In an isotropic plate ``\boldsymbol{B} = (\pi/E)\,\boldsymbol{1}``
# ([Plane problems](@ref man-plane-problems)), so that ``C = 2/E`` and
# ``D = 0``:

canonical_components(cod_tensor(crack(0.3), C_iso)), π

# ## An ellipse is a circle and two cracks
#
# Eqs. (28) and (29) of the article split the contribution of an elliptical
# hole of semi-axes ``a \ge b``, its major axis at the angle ``\varphi``, into
# that of a circular hole of radius ``\sqrt{ab}``, plus that of a crack of
# half-length ``\sqrt{a(a-b)}`` along the major axis, minus that of a crack of
# half-length ``\sqrt{b(a-b)}`` along the minor axis. Per unit area of the
# plate, with the weights ``\pi ab``, ``\pi a(a-b)`` and ``\pi b(a-b)``:
#
# ```math
# ab\,\mathbb{H}^{\text{ellipse}} = ab\,\mathbb{H}^{\text{circle}}
#   + a(a-b)\,\mathbb{H}^{\text{crack}}(\varphi)
#   - b(a-b)\,\mathbb{H}^{\text{crack}}(\varphi + \pi/2) .
# ```
#
# The two sides, computed with the four tensors of the package, for several
# shapes and orientations:

ellipse(a, b, φ) = Ellipsoid(a, b; angle = φ)
function split_defect(a, b, φ, C)
    left = a * b * hole_H(ellipse(a, b, φ), C)
    right = a * b * hole_H(disk, C) + a * (a - b) * compliance_contribution(crack(φ), C) -
        b * (a - b) * compliance_contribution(crack(φ + π / 2), C)
    return norm(left - right) / norm(left)
end
println("b/a    φ (degrees)   relative difference, orthotropic plate   isotropic plate")
for (β, φ) in ((0.5, 0), (0.5, 30), (0.2, 75), (0.05, 120))
    @printf "%.2f   %5d         %.1e                                %.1e\n" β φ split_defect(1.0, β, deg2rad(φ), C₀) split_defect(1.0, β, deg2rad(φ), C_iso)
end

# The identity holds, to the accuracy of the numerical crack of the
# anisotropic plate, whatever the shape, the orientation and the matrix. Since ``\boldsymbol{B}`` does not depend on ``\varphi``, the
# contribution of a crack, hence of any ellipse, depends on its orientation
# only through ``\underline{n}\otimes\underline{n}``, that is through
# ``\cos2\varphi`` and ``\sin2\varphi``.
#
# ## Holes at random orientations
#
# Ellipses of one shape, their major axes spread uniformly over the
# orientations, add the average of ``\mathbb{H}(\varphi)`` over ``\varphi``.
# Since ``\mathbb{H}`` holds no harmonic of ``\varphi`` beyond the second,
# three families at ``0``, ``60^\circ`` and ``120^\circ`` give that average
# exactly. The cell of the plane holds the plate and the families, each of
# one third of the area fraction ``p``, and the scheme without interactions,
# [`DiluteDual`](@ref), adds their contributions to the compliance:

function plate_with_holes(β, p; families = 3, C = C₀)
    rve = RVE()
    add_phase!(rve, :plate, disk, Dict(:C => C); fraction = :rest)
    for j in 1:families
        add_phase!(rve, Symbol(:holes, j), ellipse(1.0, β, (j - 1) * π / families), Dict(:C => 0 * C); fraction = p / families)
    end
    return rve
end
anisotropy(C) = ((e₁, e₂) = young_moduli(C); e₁ / e₂)
[anisotropy(homogenize(plate_with_holes(0.25, 0.2; families = n), DiluteDual(), :C)) for n in (3, 4, 12)]

# The option `symmetrize = IsoSymmetrize()` of a phase averages over the
# rotations of the whole problem, matrix included, and asks for an isotropic
# matrix; here the plate stays in place while the holes turn, and the families
# give the average.

# The average holds two scalars of the holes, their area fraction
# ``p = \frac{\pi}{A}\sum ab`` and ``q = \frac{\pi}{A}\sum(a-b)^2``, and the
# ratio of the Young moduli of the plate is the article's Eq. (44):
#
# ```math
# \frac{E_1}{E_2} = \frac{E_1^0}{E_2^0}\,
#   \frac{1 + (1 + L\sqrt{E_2^0})\,p + \frac{L}{2}\sqrt{E_2^0}\,q}
#        {1 + (1 + L\sqrt{E_1^0})\,p + \frac{L}{2}\sqrt{E_1^0}\,q} ,
# ```
#
# the superscript 0 marking the plate without holes. For ellipses of aspect
# ratio ``\beta = b/a``, ``q = p\,(1-\beta)^2/\beta``.

eq44(p, q) = E₁ / E₂ * (1 + (1 + L * sqrt(E₂)) * p + L / 2 * sqrt(E₂) * q) / (1 + (1 + L * sqrt(E₁)) * p + L / 2 * sqrt(E₁) * q)
q_of(p, β) = p * (1 - β)^2 / β
computed(β, p) = anisotropy(homogenize(plate_with_holes(β, p), DiluteDual(), :C))
println("b/a    p      E₁/E₂ computed   Eq. (44)")
for β in (1.0, 0.5, 0.25), p in (0.1, 0.4)
    @printf "%.2f   %.1f    %.6f         %.6f\n" β p computed(β, p) eq44(p, q_of(p, β))
end

# ### Against the area fraction
#
# The anisotropy of the plate, ``E_1/E_2 = 4.1`` without holes, against the
# area fraction of randomly oriented ellipses of three aspect ratios (the
# setting of Fig. 5 of the article): the curves are computed, the markers are
# Eq. (44).

ps = range(0, 0.4; length = 21)
by_porosity = plot(;
    xlabel = "area fraction p", ylabel = "E₁ / E₂", legend = :topright, size = (620, 360),
    title = "Randomly oriented ellipses: anisotropy against p", titlefontsize = 10
)
for (β, color) in ((1.0, 1), (0.5, 2), (0.25, 3))
    plot!(by_porosity, ps, [computed(β, p) for p in ps]; lw = 2, color, label = "b/a = $β")
    scatter!(by_porosity, ps[1:4:end], [eq44(p, q_of(p, β)) for p in ps[1:4:end]]; color, markersize = 3, label = false)
end
by_porosity

# The holes reduce the anisotropy, and flat ones more than round ones at the
# same area fraction: they lower the stiffness more along the stiff axis of
# the plate.
#
# ### From cracks to circles, at a fixed size
#
# Keep the half major axis ``a`` and the number of holes, so that the density
# of their major axes ``\rho = \frac1A\sum a^2`` stays fixed, and let the
# ellipses open from cracks (``b = 0``) to circles (``b = a``). Then
# ``p = \pi\rho\beta`` and ``q = \pi\rho(1-\beta)^2`` (the setting of Fig. 6 of
# the article). At ``\beta = 0`` the families are cracks, of density
# ``\rho/3`` each:

function plate_with_cracks(ρ; families = 3)
    rve = RVE()
    add_phase!(rve, :plate, disk, Dict(:C => C₀); fraction = :rest)
    for j in 1:families
        add_phase!(rve, Symbol(:cracks, j), crack((j - 1) * π / families), Dict(:C => 0 * C₀); density = ρ / families)
    end
    return rve
end
at_fixed_size(ρ, β) = iszero(β) ? anisotropy(homogenize(plate_with_cracks(ρ), DiluteDual(), :C)) : computed(β, π * ρ * β)
βs = range(0, 1; length = 21)
by_shape = plot(;
    xlabel = "aspect ratio b/a", ylabel = "E₁ / E₂", legend = :topright, size = (620, 360),
    title = "Ellipses opening at a fixed major axis", titlefontsize = 10
)
for (ρ, color) in ((0.05, 1), (0.1, 2))
    plot!(by_shape, βs, [at_fixed_size(ρ, β) for β in βs]; lw = 2, color, label = "ρ = $ρ")
    scatter!(by_shape, βs[1:4:end], [eq44(π * ρ * β, π * ρ * (1 - β)^2) for β in βs[1:4:end]]; color, markersize = 3, label = false)
end
by_shape

# While the cracks open into circles and the area of the holes grows from zero
# to ``\pi\rho``, the anisotropy changes by about 6 % only: the major
# axes, more than the area, set the effect.

[(ρ = ρ, cracks = at_fixed_size(ρ, 0.0), circles = at_fixed_size(ρ, 1.0)) for ρ in (0.05, 0.1)]

# At ``\beta = 0`` Eq. (44) reduces to the result for cracks at random,
# ``E_i = E_i^0\big/\bigl(1 + \frac{\pi L}{2}\sqrt{E_i^0}\,\rho\bigr)``
# (Eq. (39) of the article, whose second line repeats the subscript 1 where the
# direction 2 is meant):

[(ρ = ρ, computed = at_fixed_size(ρ, 0.0), closed_form = E₁ / E₂ * (1 + π * L / 2 * sqrt(E₂) * ρ) / (1 + π * L / 2 * sqrt(E₁) * ρ)) for ρ in (0.05, 0.1)]

# ### Against the aspect ratio, at a fixed area fraction
#
# At a fixed area fraction (the setting of Fig. 7 of the article), the
# anisotropy against the aspect ratio of the ellipses:

βf = range(0.1, 1; length = 19)
by_aspect = plot(;
    xlabel = "aspect ratio b/a", ylabel = "E₁ / E₂", legend = :bottomright, size = (620, 360),
    title = "Randomly oriented ellipses at a fixed area fraction", titlefontsize = 10
)
for (p, color) in ((0.2, 1), (0.3, 2))
    plot!(by_aspect, βf, [computed(β, p) for β in βf]; lw = 2, color, label = "p = $p")
    scatter!(by_aspect, βf[1:3:end], [eq44(p, q_of(p, β)) for β in βf[1:3:end]]; color, markersize = 3, label = false)
end
by_aspect

# Between ``b/a = 0.5`` and ``1`` the anisotropy changes by less than 4 %,
# and much faster below:

[(p = p, b_over_a_05 = computed(0.5, p), b_over_a_1 = computed(1.0, p)) for p in (0.2, 0.3)]

# ## Where to go next
#
# - [The elliptic hole, from an isotropic to an anisotropic plate](@ref app-elliptic-hole):
#   the stress around one such hole.
# - [Plane problems](@ref man-plane-problems): cells of the plane, their
#   inclusions and their schemes.
# - [Crack opening displacement and compliance](@ref th-cod-tensors): the
#   tensors ``\boldsymbol{B}`` and ``\mathbb{H}`` of a crack.
# - [Elastic and conductive properties of porous and cracked materials](@ref app-cross-property):
#   the same contributions of pores and cracks, set against those of conduction.
