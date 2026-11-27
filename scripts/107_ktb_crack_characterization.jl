# # [Cracks of the KTB gneiss read on its velocities](@id app-ktb-cracks)
#
# !!! info "Before this page"
#     [Cracks seen as fractures](@ref th-fracture-compliance), sections 2 and 3,
#     which define the compliances of a set of fractures and invert three
#     orthogonal sets, [Seismic signatures](@ref man-seismic) for the syntax,
#     and the [Plane waves](@extref TensND man-plane-waves) of TensND.
#
# A gneiss cored in the pilot hole of the German deep drilling program (KTB)
# was measured by ultrasonic waves at two confining pressures, 400 MPa, where
# its cracks are closed, and 10 MPa, where they are open
# [rasolofosaonGJI2000](@cite). The difference of the two compliances is the
# compliance of the cracks. This page reads on it the frame of the cracks, the
# normal and tangential compliances of three orthogonal sets of fractures, and
# what their ratio says about the cracks, and rebuilds the two models of the
# article.

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

# ## The two measured stiffnesses
#
# The article prints the two stiffness matrices in the two-index (Voigt)
# notation, in GPa, in the frame ``(X, Y, Z)`` of the sample, ``Z`` vertical
# (its Figs 5a and 5b). [`stiffness_from_voigt`](@extref TensND TensND.stiffness_from_voigt)
# turns each into a tensor ``\mathbb C``; [`voigt_compliance`](@extref TensND TensND.voigt_compliance)
# writes a compliance back in the same notation, with ``S_{44} = 4\,S_{2323}``.
# Compliances are printed in ``10^{-12}\,\mathrm{Pa}^{-1}``, that is
# ``10^{-3}\,\mathrm{GPa}^{-1}``.

lit = JSON.parsefile(joinpath(pkgdir(MeanFieldHomogenization), "data", "literature", "rasolofosaonGJI2000.json"))
voigt(name) = [lit["tables"][name][i]["c$j"] for i in 1:6, j in 1:6]       ## GPa
axes_of(name) = [lit["tables"][name][j][c] for c in ("x", "y", "z"), j in 1:3]   ## one axis per column
C_400 = stiffness_from_voigt(voigt("stiffness_400MPa"))
C_10 = stiffness_from_voigt(voigt("stiffness_10MPa"))
ρ = lit["quantities"]["density"]["value"] / 1000                            ## g/cm³: velocities in km/s
per_TPa(S) = round.(voigt_compliance(S) * 1.0e3; digits = 3)                ## 1e-12 Pa⁻¹
nothing #hide

#md Main.show_literature("rasolofosaonGJI2000") # hide

# The excess compliance of the cracks is
#
# ```math
# \Delta\mathbb S = \mathbb S(10\ \mathrm{MPa}) - \mathbb S(400\ \mathrm{MPa}),
# \qquad \mathbb S = \mathbb C^{-1} ,
# ```
#
# the rock at 400 MPa being taken as free of open cracks. In the frame of the
# sample:

ΔS = inv(C_10) - inv(C_400)
per_TPa(ΔS)

# ## How far from a symmetry class
#
# The figures of the article give the distance of each stiffness to the
# isotropic, transversely isotropic and orthotropic classes. The relative
# Frobenius distance of ``\mathbb C`` to its best approximation in a class,
# ``\|\mathbb C - \mathbb C^{\mathrm{class}}\|/\|\mathbb C\|``, the best axis or
# frame being searched, is the third output of
# [`project`](@extref TensND TensND.project). It gives back the printed
# percentages (MON, the monoclinic class, is not computed here):

distances(C) = [100 * project(C, class)[3] for class in (:ISO, :TI, :ORTHO)]
printed(fig) = (r = only(filter(r -> r["figure"] == fig, lit["tables"]["deviations_from_higher_symmetries"]));
    [r["ISO_percent"], r["TI_percent"], r["ORT_percent"]])
for (fig, C) in (("5a", C_400), ("5b", C_10))
    @printf("Fig. %s  computed  ISO %5.2f  TI %5.2f  ORT %5.2f %%\n", fig, distances(C)...)
    @printf("         printed   ISO %5.1f  TI %5.1f  ORT %5.1f %%\n", printed(fig)...)
end

# The intact rock is nearly transversely isotropic about ``Z``, its foliation
# being horizontal; the open cracks double its anisotropy.

# ## The cracks in an isotropic rock
#
# To separate the cracks from the fabric of the minerals, the article adds
# ``\Delta\mathbb S`` to the compliance of an isotropic rock equivalent to the
# intact one. Its eq. (1) is the isotropic projection of the compliance
# ``\mathbb S(400\ \mathrm{MPa})``, which [`best_fit_iso`](@extref TensND TensND.best_fit_iso)
# computes. The medium of its Fig. 6a, the *isotropic cracked* rock, is then
#
# ```math
# \mathbb C^{\mathrm{ic}} = \bigl(\mathbb S^{\mathrm{iso}} + \Delta\mathbb S\bigr)^{-1} .
# ```

S_iso = best_fit_iso(inv(C_400))                     ## eq. (1): projection of the compliance
C_ic = inv(S_iso + ΔS)
round.(voigt_stiffness(C_ic); digits = 2)

# Against the matrix printed in Fig. 6a, component by component (GPa):

maximum(abs, voigt_stiffness(C_ic) - voigt("stiffness_iso_cracked_10MPa"))

# that is, to the rounding of the figure. The text of the article quotes for the
# isotropic intact rock ``C_{11} = 110.45`` GPa and ``C_{44} = 34.69`` GPa: these
# are the moduli of the isotropic projection of the *stiffness*
# ``\mathbb C(400\ \mathrm{MPa})``, while the projection of the compliance,
# used for Fig. 6a, has slightly lower ones. Both are computed below; the
# medium built on the projection of the stiffness differs from Fig. 6a by up to
# 2.4 GPa.

C_iso = best_fit_iso(C_400)                          ## projection of the stiffness
(stiffness_projection = (C11 = pwave_modulus(C_iso), C44 = shear_modulus(C_iso)),
    compliance_projection = (C11 = pwave_modulus(inv(S_iso)), C44 = shear_modulus(inv(S_iso))),
    printed = (C11 = lit["quantities"]["iso_C11_intact"]["value"], C44 = lit["quantities"]["iso_C44_intact"]["value"]),
    Fig6a_with_stiffness_projection = maximum(abs, voigt_stiffness(inv(inv(C_iso) + ΔS)) - voigt("stiffness_iso_cracked_10MPa")))

# The distances of the isotropic cracked rock to the symmetry classes are those
# printed in Fig. 6a:

@printf("Fig. 6a  computed  ISO %5.2f  TI %5.2f  ORT %5.2f %%\n", distances(C_ic)...)
@printf("         printed   ISO %5.1f  TI %5.1f  ORT %5.1f %%\n", printed("6a")...)

# ## The frame of the cracks
#
# The cracks are described in the frame of the best orthotropic approximation
# of ``\mathbb C^{\mathrm{ic}}``, the cracks being then the only source of its
# anisotropy. [`project`](@extref TensND TensND.project) with `:ORTHO` searches
# that frame over all rotations; `TensND.frame` returns
# it. Its axes are put in the order and the sense of the frame
# ``(X_c, Y_c, Z_c)`` printed in the article (p. 421), with which they are
# compared:

F_found = vecbasis(TensND.frame(first(project(C_ic, :ORTHO))), :cov)
F_printed = axes_of("crack_frame")
nearest = [argmax(abs.(transpose(F_found) * F_printed)[:, j]) for j in 1:3]   ## the found axis nearest each printed one
F = F_found[:, nearest] .* transpose(sign.(diag(transpose(F_found[:, nearest]) * F_printed)))
(found = round.(F; digits = 3), printed = F_printed,
    angle_deg = round.(acosd.(min.(1, abs.(diag(transpose(F) * F_printed)) ./ norm.(eachcol(F_printed)))); digits = 1))

# ``Z_c`` is nearly ``X``, ``Y_c`` nearly the vertical ``Z``: the most compliant
# cracks, normal to ``Y_c``, lie in the foliation. The axes found here are
# within 3.5 to 6 degrees of the printed ones.
#
# ## The excess compliance in the frame of the cracks
#
# The components of ``\Delta\mathbb S`` in the frame are those of the tensor
# expressed in the basis of its axes ([`express`](@extref TensND TensND.express)).
# The printed frame is rounded to three decimals and is left-handed: it is made
# orthonormal by its polar factor and right-handed by turning one axis, which
# changes no component of a tensor of order four.

function orthonormal(A)                              ## polar factor, made right-handed
    f = svd(A)
    Q = f.U * f.Vt
    return Q * Diagonal([1, 1, sign(det(Q))])
end
in_frame(t, A) = Tens(get_array(express(t, RotatedBasis(orthonormal(A), Val(3)))))   ## components in the frame, as canonical ones
excess = [lit["tables"]["excess_compliance_ort_10MPa"][i]["value"] for i in 1:9]       ## 11 22 33 44 55 66 12 23 13
ort_components(S) = (M = voigt_compliance(S) * 1.0e3; [M[1, 1], M[2, 2], M[3, 3], M[4, 4], M[5, 5], M[6, 6], M[1, 2], M[2, 3], M[1, 3]])
labels = ("11", "22", "33", "44", "55", "66", "12", "23", "13")
dS_found, dS_printed_frame = ort_components(in_frame(ΔS, F)), ort_components(in_frame(ΔS, F_printed))
println("ΔS_IJ   found frame   printed frame   printed (p. 421)")
for (l, a, b, c) in zip(labels, dS_found, dS_printed_frame, excess)
    @printf("  %s     %7.3f       %7.3f        %7.3f\n", l, a, b, c)
end

# In the frame found here, the nine orthotropic components agree with the
# printed ones to within ``0.05\times10^{-12}\,\mathrm{Pa}^{-1}``, with one exchange of labels: the value
# printed as ``\Delta S_{23}`` is the computed ``\Delta S_{13}``, and the value
# printed as ``\Delta S_{13}`` the computed ``\Delta S_{23}``, in either frame.
# In the printed frame, rounded, the shear components move by up to
# ``0.5\times10^{-12}\,\mathrm{Pa}^{-1}``, a change of frame of a few degrees
# being enough for that.
#
# ## Three orthogonal sets of fractures
#
# Three sets of parallel fractures normal to the axes ``\underline e_i`` of the
# frame, each invariant by rotation about its normal, change the compliance by
# (eq. 3 of the article, §3 of [Cracks seen as fractures](@ref th-fracture-compliance))
#
# ```math
# \Delta S_{iiii} = Z_N^i,
# \qquad
# 4\,\Delta S_{ijij} = Z_T^i + Z_T^j \quad (i \neq j),
# ```
#
# where ``Z_N^i`` and ``Z_T^i`` are the normal and tangential compliances of set
# ``i`` per unit length. [`orthogonal_fracture_compliances`](@ref) inverts these
# six relations, the couplings ``\Delta S_{iijj}`` being left out:

sets(A) = orthogonal_fracture_compliances(ΔS, orthonormal(A))
g = sets(F)
Z_N, Z_T = g.Z_N, g.Z_T
pub = lit["tables"]["fracture_compliances_10MPa"]
println("set   Z_N found  printed frame  printed    Z_T found  printed frame  printed")
for i in 1:3
    @printf(" %d    %6.3f     %6.3f       %5.2f      %6.3f     %6.3f       %5.2f\n", i,
        1.0e3 * Z_N[i], 1.0e3 * sets(F_printed).Z_N[i], pub[i]["S_N"],
        1.0e3 * Z_T[i], 1.0e3 * sets(F_printed).Z_T[i], pub[i]["S_T"])
end

# The frame found here gives back the six printed compliances to within
# ``0.03\times10^{-12}\,\mathrm{Pa}^{-1}``. Set 2, normal to ``Y_c``, holds most
# of the compliance: these are the cracks of the foliation.
#
# The article normalizes them by the moduli of the isotropic intact rock,
# ``E_N^i = Z_N^i\,C_{11}`` and ``E_T^i = Z_T^i\,C_{44}`` (its eq. 4), with the
# values 110.45 and 34.69 GPa quoted above:

C11, C44 = pwave_modulus(C_iso), shear_modulus(C_iso)
E_N, E_T = 100 .* Z_N .* C11, 100 .* Z_T .* C44           ## per cent
nor = lit["tables"]["normalized_compliances_10MPa"]
for i in 1:3
    @printf("set %d   E_N %5.1f %% (printed %4.1f)   E_T %5.1f %% (printed %4.1f)\n",
        i, E_N[i], nor[i]["E_N_percent"], E_T[i], nor[i]["E_T_percent"])
end

# ## The intact rock with three sets of fractures
#
# The model of Fig. 6b of the article puts the three sets back in the intact
# rock, the measured ``\mathbb C(400\ \mathrm{MPa})``, without the couplings
# left out above:
#
# ```math
# \mathbb S^{\mathrm{model}} = \mathbb C(400\ \mathrm{MPa})^{-1}
#   + \sum_{i=1}^{3}\operatorname{sym}\bigl(\underline e_i\otimes\boldsymbol Z_i\otimes\underline e_i\bigr),
# \qquad
# \boldsymbol Z_i = Z_N^i\,\underline e_i\otimes\underline e_i + Z_T^i\,(\boldsymbol 1 - \underline e_i\otimes\underline e_i) .
# ```
#
# [`fractured_stiffness`](@ref) builds it from pairs `normal => (Z_N, Z_T)`.
# Built with the frame and the compliances found here, then with the printed
# frame and the printed compliances:

model(A, ZN, ZT) = (Q = orthonormal(A); fractured_stiffness(C_400, (Q[:, i] => (ZN[i], ZT[i]) for i in 1:3)...))
C_model = model(F, Z_N, Z_T)
C_model_printed = model(F_printed, [1.0e-3 * r["S_N"] for r in pub], [1.0e-3 * r["S_T"] for r in pub])
round.(voigt_stiffness(C_model); digits = 2)

# Against the matrix of Fig. 6b, the largest difference of a component (GPa) and
# the largest on the diagonal:

fig6b = voigt("stiffness_model_10MPa")
compare(V) = (largest = maximum(abs, V - fig6b), diagonal = maximum(abs, diag(V - fig6b)))
(found = compare(voigt_stiffness(C_model)), printed_data = compare(voigt_stiffness(C_model_printed)))

# The diagonal is rebuilt to about a tenth of a GPa. The off-diagonal
# components differ by up to 2.4 GPa, and the differences have a pattern: the
# components ``C_{14}``, ``C_{15}``, ``C_{24}``, ``C_{34}``, ``C_{35}``, ``C_{46}``,
# ``C_{56}``, those with an odd number of indices 3, have opposite signs. These
# are the components that change sign in the mirror image of the medium
# through the horizontal plane, ``Z \to -Z``, whose Voigt matrix is
# ``\boldsymbol D\,\boldsymbol C\,\boldsymbol D`` with
# ``\boldsymbol D = \operatorname{diag}(1, 1, 1, -1, -1, 1)``. The mirror image
# of the model built in the frame found here is Fig. 6b to within 0.15 GPa,
# component by component:

mirror(V) = (D = Diagonal([1, 1, 1, -1, -1, 1]); D * V * D)
(found = compare(mirror(voigt_stiffness(C_model))), printed_data = compare(mirror(voigt_stiffness(C_model_printed))))

# The mirror image is the same medium written in a left-handed frame, and the
# frame of the cracks printed in the article is left-handed. The distances of
# the model to the symmetry classes, which a reflection does not change:

@printf("Fig. 6b  computed  ISO %5.2f  TI %5.2f  ORT %5.2f %%\n", distances(C_model)...)
@printf("         printed   ISO %5.1f  TI %5.1f  ORT %5.1f %%\n", printed("6b")...)

# ## What the ratio of the compliances says
#
# The ratio ``R_i = E_N^i/E_T^i`` does not depend on the density of cracks: it
# depends on their shape and on what fills them. For dry flat cracks, isolated
# in an isotropic rock, the stress-driven dilute scheme gives the compliances of
# a family of penny-shaped cracks of density ``\varepsilon`` (number per unit
# volume times the cube of the radius) [kachanov1992](@cite),
#
# ```math
# \frac{Z_N}{\varepsilon} = \frac{16\,(1 - \nu^2)}{3E},
# \qquad
# \frac{Z_T}{\varepsilon} = \frac{32\,(1 - \nu^2)}{3E\,(2 - \nu)},
# \qquad
# \frac{Z_N}{Z_T} = 1 - \frac{\nu}{2} .
# ```
#
# [`fracture_compliance`](@ref) reads them on a cell, here the isotropic intact
# rock of the article with one family of unit density:

function dry_cracks(C)
    rve = RVE()
    add_phase!(rve, :rock, Ellipsoid(1.0), Dict(:C => C); fraction = :rest)
    add_phase!(rve, :cracks, PennyCrack(1.0), Dict(:C => C); density = 1.0)
    return fracture_compliance(rve, DiluteDual(), :cracks)
end
dry = dry_cracks(C_iso)
E, ν = young_modulus(C_iso), poisson_ratio(C_iso)
(Z_N = dry.Z_N, closed_form_N = 16 * (1 - ν^2) / (3E), Z_T = dry.Z_T, closed_form_T = 32 * (1 - ν^2) / (3E * (2 - ν)), ν = ν)

# Then ``R = (1 - \nu/2)\,C_{11}/C_{44}``. With ``\nu = 1/4``, the value of
# the article, ``C_{11}/C_{44} = 2(1 - \nu)/(1 - 2\nu) = 3`` and
# ``R = 2.625``, the 2.63 it quotes for dry cracks; with the moduli of the
# isotropic intact rock, ``\nu = 0.27``:

quarter = iso_stiffness(5 / 3, 1.0)                  ## k = 5μ/3: ν = 1/4
R_dry(C) = (d = dry_cracks(C); d.Z_N / d.Z_T * pwave_modulus(C) / shear_modulus(C))
(ν_quarter = R_dry(quarter), printed = lit["quantities"]["R_dry_cracks"]["value"], ν_rock = R_dry(C_iso))

# The three sets of the gneiss:

R = E_N ./ E_T
(R = round.(R; digits = 2), printed = Tuple(r["R"] for r in lit["tables"]["ratio_R_10MPa"]))

# Set 2 has a ratio above that of dry flat cracks; the two others below it,
# set 3 close to 1, the value the article associates with cracks filled with a
# soft material.
#
# ## The density of the cracks of the foliation
#
# The compliances of set 2, divided by those of a family of unit density in the
# isotropic intact rock, give its density of cracks, once from the normal
# compliance and once from the tangential one:

(from_normal = Z_N[2] / dry.Z_N, from_tangential = Z_T[2] / dry.Z_T,
    printed = lit["quantities"]["crack_density_family2"]["value"])

# The two readings differ because ``R_2`` differs from the ratio of dry penny
# cracks. The density of 0.07 the article gives is the one read on the
# tangential compliance.
#
# Its eq. (5) writes the coefficients ``E/\varepsilon`` with
# ``\gamma = C_{44}/C_{11} = 0.314``: ``4/[3\gamma(1 - \gamma)]`` for dry cracks and
# ``16/[3(3 - 2\gamma)]`` for dry and water-saturated ones, both named
# ``C^{(N)}`` in the text. Written with ``\gamma``, the dilute compliances of a
# penny-shaped crack give the first as the normal coefficient
# ``C_{11}Z_N/\varepsilon`` and the second as the tangential one
# ``C_{44}Z_T/\varepsilon``, which a fluid in the crack does not change:
#
# ```math
# \frac{C_{11}Z_N}{\varepsilon} = \frac{4}{3\gamma(1 - \gamma)},
# \qquad
# \frac{C_{44}Z_T}{\varepsilon} = \frac{16}{3(3 - 2\gamma)} ,
# ```
#
# for ``\nu = (1 - 2\gamma)/(2(1 - \gamma))``. The computed coefficients and the
# two densities of set 2 they give:

γ = lit["quantities"]["gamma"]["value"]
iso_γ = iso_stiffness(1 - 4γ / 3, γ)                 ## C₁₁ = k + 4μ/3 = 1, C₄₄ = μ = γ
d = dry_cracks(iso_γ)
coefficients = (normal = d.Z_N * pwave_modulus(iso_γ), printed_normal = 4 / (3γ * (1 - γ)),
    tangential = d.Z_T * shear_modulus(iso_γ), printed_tangential = 16 / (3 * (3 - 2γ)))

#-

(from_normal = E_N[2] / 100 / coefficients.printed_normal, from_tangential = E_T[2] / 100 / coefficients.printed_tangential)

# ## The velocities
#
# The waves along the axes of the sample, from the Christoffel problem of
# [`plane_waves`](@extref TensND TensND.plane_waves), in km/s, at both pressures
# and in the model:

directions = ("X" => [1.0, 0.0, 0.0], "Y" => [0.0, 1.0, 0.0], "Z" => [0.0, 0.0, 1.0])
println("         400 MPa (P, S1, S2)      10 MPa (P, S1, S2)       model (P, S1, S2)")
for (name, n) in directions
    v(C) = plane_waves(C, ρ, n).velocity
    @printf("%s    %5.2f %5.2f %5.2f        %5.2f %5.2f %5.2f        %5.2f %5.2f %5.2f\n", name, v(C_400)..., v(C_10)..., v(C_model)...)
end

# The P wave along ``Z``, across the foliation, loses a quarter of its velocity
# when the cracks open, and 7 to 8 per cent along ``X`` and ``Y``. The model
# of the intact rock with three sets, which ignores the couplings
# ``\Delta S_{iijj}`` and the components outside the orthotropic pattern,
# recovers the S waves to within 0.06 km/s and the P waves less well,
# the largest gap being along ``Z``. The article reports a deviation of the model
# from the measurements below 5 per cent; the relative differences of the P
# velocities along the three axes are:

[name => round(100 * (first(plane_waves(C_model, ρ, n).velocity) / first(plane_waves(C_10, ρ, n).velocity) - 1); digits = 1)
    for (name, n) in directions]

# In the vertical plane ``(X, Z)``:

θs = range(0, 180; length = 181)
vP(C, θ) = first(plane_waves(C, ρ, [sind(θ), 0.0, cosd(θ)]).velocity)
plot(θs, [vP.(Ref(C_400), θs) vP.(Ref(C_10), θs) vP.(Ref(C_model), θs)];
    label = ["400 MPa, measured" "10 MPa, measured" "10 MPa, intact rock + three sets"],
    lw = [2 2 2], ls = [:solid :solid :dash],
    xlabel = "angle from the vertical Z, in the plane (X, Z) (°)", ylabel = "P velocity (km/s)",
    xticks = 0:30:180, legend = :bottom, size = (700, 420))

# ## Where to go next
#
# - [Cracks seen as fractures](@ref th-fracture-compliance): the compliance of a
#   set of fractures, its inversion, and the excess compliance of a crack family
#   under each scheme.
# - [Seismic signatures](@ref man-seismic): the syntax of the functions used
#   here, and the density of a cell.
# - [Plane waves](@extref TensND man-plane-waves): velocities, polarizations and
#   Thomsen's parameters.
# - [Squirt flow in a cracked rock](@ref app-squirt-dispersion): the same cracks
#   filled with a fluid, whose response depends on the frequency.
# - [Interacting cracks](@ref app-interacting-cracks): cracks close enough to
#   load each other, beyond the sets of non-interacting fractures used here.
# - [Viscoelastic complex modulus of a bituminous mixture](@ref app-bituminous):
#   the next application, which turns to time-dependent behavior.
