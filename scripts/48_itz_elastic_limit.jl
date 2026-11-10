# # [The elastic limit of concrete: ITZ failure and ITZ–aggregate separation](@id app-itz-elastic-limit)
#
# !!! info "Before this page"
#     [The interfacial transition zone in concrete](@ref app-itz-concrete), which
#     introduces the coated aggregate, and
#     [Periodic multilayer — the laminate cell](@ref th-laminate) §2–4, where the
#     jump across a plane interface is written with the flat Hill tensors
#     ``\mathbb P`` and ``\mathbb Q`` used below.
#
# Concrete cracks long before it fails. Under a growing load, microcracks appear
# first around the aggregates, and the macroscopic stress at that moment is the
# **elastic limit** of the material. Two mechanisms can start them: the
# interfacial transition zone (ITZ) separates from the aggregate along their
# common surface, or the ITZ itself breaks within its thickness. This page
# follows the two companion papers of [konigsberger2014stress](@citet) and
# [konigsberger2014strength](@citet), and recomputes their results with the
# library:
#
# 1. **Part I, stress concentration**: from the macroscopic stress
#    ``\boldsymbol\Sigma`` down to the stress of the aggregates, to the
#    tractions on their surface, and to the three-dimensional stress inside the
#    ITZ (§§2–4);
# 2. **Part II, strength upscaling**: a tension criterion for each of the two
#    mechanisms, raised to the scale of concrete, and their combination set
#    against the elastic limits measured on concrete (§§5–8).
#
# ![The two scales of the model and the two ways a crack can start next to an aggregate](../../assets/applications/itz_scales.svg)
#
# The ITZ is a few tens of micrometers thick, much thinner than an aggregate
# (left). At the scale of concrete it is therefore a surface, which does not
# change how the load is shared between paste and aggregates. At the scale of a
# few micrometers (right) it is a layer of material, bonded to the aggregate
# across a plane of normal ``\underline n``, and its stress
# ``\boldsymbol\sigma_{\mathrm{ITZ}}`` follows from that of the aggregate through
# the continuity conditions of a perfectly bonded interface.

import Pkg                                                          #jl
Pkg.activate(joinpath(@__DIR__, "..", "docs"); io = devnull)        #jl

using MeanFieldHomogenization, TensND, LinearAlgebra, JSON, Printf, Plots
gr()  # headless backend; GKSwstype is set to "100" in make.jl
nothing #hide

# ## 1. The typical concrete
#
# The elastic constants are those of the "typical concrete" of
# [konigsberger2014stress; Table I](@citet): quartz aggregates at a volume
# fraction of 0.65, a mature cement paste, and an ITZ whose Young modulus is
# 85 % of that of the paste, with the same Poisson ratio. They are read from
# `data/literature/konigsberger2014stress.json`, together with the values of
# Part I the page checks against; those of Part II are read from
# `data/literature/konigsberger2014strength.json`. Both files are shown, folded,
# under the block.

literature(key) = JSON.parsefile(
    joinpath(pkgdir(MeanFieldHomogenization), "data", "literature", "$key.json")
)["quantities"]
part1 = literature("konigsberger2014stress")
part2 = literature("konigsberger2014strength")
val(q, name) = q[name]["value"]

C_agg = iso_stiffness_E_nu(val(part1, "E_aggregate"), val(part1, "nu_aggregate"))
C_cp = iso_stiffness_E_nu(val(part1, "E_paste"), val(part1, "nu_paste"))
C_itz = iso_stiffness_E_nu(val(part1, "E_itz"), val(part1, "nu_itz"))
f_agg = val(part1, "f_aggregate")
nothing #hide

#md Main.show_literature("konigsberger2014stress") # hide

#-

#md Main.show_literature("konigsberger2014strength") # hide

# Every comparison below prints the computed value next to the published one,
# and checks that they agree to the digits the paper gives.

## Within half a unit of the last digit the paper prints, or `units` of it.
function check(label, x, published; units = 0.5)
    str = string(published)
    digits = occursin('.', str) ? length(str) - findfirst('.', str) : 0
    @printf("%-44s %9.4f   paper %s\n", label, x, str)
    abs(x - published) ≤ units * 10.0^(-digits) * (1 + 1.0e-9) ||
        error("$label: computed $x, published $published")
    return nothing
end
nothing #hide

# ## 2. From concrete to the aggregate
#
# Spherical aggregates in a paste matrix call for the Mori–Tanaka scheme. The
# aggregate strain ``\boldsymbol\varepsilon_{\mathrm{agg}} = \mathbb A_{\mathrm{agg}}:\boldsymbol\varepsilon^{\infty}`` is
# uniform, ``\mathbb A_{\mathrm{agg}}`` being the dilute concentration tensor in the
# paste, and the remote strain ``\boldsymbol\varepsilon^{\infty}`` follows from the
# macroscopic strain ``\boldsymbol E = \langle\mathbb A\rangle:\boldsymbol\varepsilon^{\infty}``. Both
# aggregate fields are therefore linear in the macroscopic stress ``\boldsymbol\Sigma``:
#
# ```math
# \begin{aligned}
# \boldsymbol\varepsilon_{\mathrm{agg}} &= \mathbb A_{\mathrm{agg}}:\langle\mathbb A\rangle^{-1}:(\mathbb C^{\mathrm{hom}})^{-1}:\boldsymbol\Sigma,
# \\
# \boldsymbol\sigma_{\mathrm{agg}} &= \mathbb C_{\mathrm{agg}}:\boldsymbol\varepsilon_{\mathrm{agg}}
#   = \mathbb B_{\mathrm{agg}}:\boldsymbol\Sigma,
# \qquad
# \mathbb B_{\mathrm{agg}} = B_{\mathrm{vol}}\,\mathbb J + B_{\mathrm{dev}}\,\mathbb K.
# \end{aligned}
# ```
#
# ``\mathbb J`` and ``\mathbb K`` are the spherical and deviatoric projectors, so
# that ``B_{\mathrm{vol}}`` scales the mean stress and ``B_{\mathrm{dev}}`` the
# deviator.

A_agg = gradient_gradient_concentration(Ellipsoid(1.0), C_agg, C_cp)
A_mean = (1 - f_agg) * TensISO{3}(1.0, 1.0) + f_agg * A_agg
C_hom = ((1 - f_agg) * C_cp + f_agg * C_agg ⊡ A_agg) ⊡ inv(A_mean)
A_ε = A_agg ⊡ inv(A_mean) ⊡ inv(C_hom)    # ε_agg = A_ε : Σ
B_σ = C_agg ⊡ A_ε                         # σ_agg = B_σ : Σ
B_vol, B_dev = get_data(B_σ)
check("B_vol", B_vol, val(part1, "Bvol"))
check("B_dev", B_dev, val(part1, "Bdev"))
check("B_vol − B_dev", B_vol - B_dev, val(part2, "Bvol_minus_Bdev"))

# ## 3. Tractions on the aggregate surface
#
# A point of the aggregate surface is labeled by its zenith angle ``\theta``,
# measured from the axis ``\underline e_z``, and its azimuth ``\varphi``; the
# spherical frame there is ``(\underline e_r, \underline e_\theta, \underline e_\varphi)``,
# with ``\underline n = \underline e_r`` the outer normal. Since
# ``\boldsymbol\sigma_{\mathrm{agg}}`` is uniform, the traction the aggregate
# exerts across its surface is Cauchy's
# ``\underline T = \boldsymbol\sigma_{\mathrm{agg}}\cdot\underline e_r``, and its
# components in the spherical frame are projections:
#
# ```math
# T_r = \underline e_r\cdot\boldsymbol\sigma_{\mathrm{agg}}\cdot\underline e_r,
# \qquad
# T_\theta = \underline e_\theta\cdot\boldsymbol\sigma_{\mathrm{agg}}\cdot\underline e_r,
# \qquad
# T_\varphi = \underline e_\varphi\cdot\boldsymbol\sigma_{\mathrm{agg}}\cdot\underline e_r .
# ```

e_r(θ, φ) = Tens([sin(θ) * cos(φ), sin(θ) * sin(φ), cos(θ)])
e_θ(θ, φ) = Tens([cos(θ) * cos(φ), cos(θ) * sin(φ), -sin(θ)])
e_φ(θ, φ) = Tens([-sin(φ), cos(φ), zero(φ)])

e₃ = 𝐞(3, 3, Float64)
uz = e₃ ⊗ e₃                  # uniaxial load along e_z
bxy = TensISO{3}(1.0) - uz     # equibiaxial load in the plane (e_x, e_y)

traction(Σ, θ, φ = zero(θ)) = (B_σ ⊡ Σ) ⋅ e_r(θ, φ)
T_r(Σ, θ) = e_r(θ, zero(θ)) ⋅ traction(Σ, θ)
T_θ(Σ, θ) = e_θ(θ, zero(θ)) ⋅ traction(Σ, θ)
nothing #hide

# Under a uniaxial load ``\boldsymbol\Sigma = \Sigma\,\underline e_z\otimes\underline e_z``
# the problem is symmetric about ``\underline e_z``: ``T_\varphi = 0``, and ``T_r``,
# ``T_\theta`` depend on ``\theta`` only. The poles are the points ``\theta = 0`` and
# ``\pi``, the equator the circle ``\theta = \pi/2``; the curves are symmetric about
# the equator, so the upper hemisphere suffices.

θs = range(0, π / 2; length = 181)
plot(
    θs ./ π, [T_r.(Ref(uz), θs) T_θ.(Ref(uz), θs)];
    label = ["T_r / Σ" "T_θ / Σ"], lw = 2, xlabel = "θ / π",
    ylabel = "traction per unit macroscopic stress",
    title = "Tractions on the aggregate, uniaxial load", legend = :right
)
hline!([0]; color = :black, lw = 0.5, label = "")

# The radial traction is largest at the poles, where it amplifies a tensile
# load by ``(B_{\mathrm{vol}} + 2B_{\mathrm{dev}})/3``, and smallest at the
# equator, where it equals ``(B_{\mathrm{vol}} - B_{\mathrm{dev}})\,\Sigma/3``. That
# value is slightly negative: a **compressive** load therefore pulls the equator
# of the aggregate, because the aggregate, stiffer than the paste, takes more
# than its share of the deviatoric load.

check("T_r(pole) / Σ, tension", T_r(uz, 0.0), val(part1, "traction_pole_tension"))
check("T_r(equator) / |Σ|, compression", T_r(-uz, π / 2), val(part1, "traction_equator_compression"))

# ## 4. The stress in the ITZ
#
# The traction gives three components of the stress of the ITZ at the interface,
# those acting on the plane of normal ``\underline n``. The three others, acting
# on planes normal to the interface, are not continuous across it: they depend
# on the stiffness of the ITZ. [konigsberger2014stress](@citet) obtain them
# component by component in the spherical frame, from the continuity of the
# tractions and from the continuity of the strains along the interface, and
# gather the result in an aggregate-to-ITZ concentration tensor. The same
# conditions can be written in intrinsic form, which gives that tensor at once
# for any normal.
#
# Across a perfectly bonded interface of normal ``\underline n``, the displacement is
# continuous, so the strain can only jump by a rank-one symmetric term, and the
# traction is continuous:
#
# ```math
# \boldsymbol\varepsilon_{\mathrm{ITZ}} = \boldsymbol\varepsilon_{\mathrm{agg}} + \underline a\stackrel{s}{\otimes}\underline n,
# \qquad
# \boldsymbol\sigma_{\mathrm{ITZ}}\cdot\underline n = \boldsymbol\sigma_{\mathrm{agg}}\cdot\underline n .
# ```
#
# These are the two conditions of the laminate cell. Solving for ``\underline a``
# with the acoustic tensor ``\boldsymbol K = \underline n\cdot\mathbb C_{\mathrm{ITZ}}\cdot\underline n``
# gives the Hadamard jump
#
# ```math
# \boldsymbol\varepsilon_{\mathrm{ITZ}} = \boldsymbol\varepsilon_{\mathrm{agg}}
#   + \mathbb P(\underline n):(\boldsymbol\sigma_{\mathrm{agg}} - \mathbb C_{\mathrm{ITZ}}:\boldsymbol\varepsilon_{\mathrm{agg}}),
# \qquad
# \mathbb P(\underline n) = \underline n\stackrel{s}{\otimes}\boldsymbol K^{-1}\stackrel{s}{\otimes}\underline n ,
# ```
#
# the elastic counterpart of the interfacial operator that
# [barthelemyBignonnetIJES2020](@citet) (their Appendix E) write for conduction.
# ``\mathbb P`` is the Hill tensor of a flat inclusion of the ITZ. Applying
# ``\mathbb C_{\mathrm{ITZ}}`` and introducing the second flat Hill tensor
# ``\mathbb Q = \mathbb C_{\mathrm{ITZ}} - \mathbb C_{\mathrm{ITZ}}:\mathbb P:\mathbb C_{\mathrm{ITZ}}`` separates what each condition
# transmits:
#
# ```math
# \boldsymbol\sigma_{\mathrm{ITZ}}(\underline n)
#   = \mathbb Q(\underline n):\boldsymbol\varepsilon_{\mathrm{agg}}
#   + \mathbb C_{\mathrm{ITZ}}:\mathbb P(\underline n):\boldsymbol\sigma_{\mathrm{agg}}
#   = \mathbb M(\underline n):\boldsymbol\Sigma .
# ```
#
# The in-plane part of the ITZ stress comes from the in-plane strain of the
# aggregate, through ``\mathbb Q``; the out-of-plane part comes from the traction,
# through ``\mathbb C_{\mathrm{ITZ}}:\mathbb P``. `TensND` writes the operator as it reads:

function itz_stress_operator(n)
    n = Tens(n)
    K = n ⋅ C_itz ⋅ n                     # acoustic tensor of the ITZ
    P = n ⊗ˢ inv(K) ⊗ˢ n                  # flat Hill tensor
    Q = C_itz - C_itz ⊡ P ⊡ C_itz         # second flat Hill tensor
    return Q ⊡ A_ε + C_itz ⊡ P ⊡ B_σ      # σ_ITZ = 𝕄(n) : Σ
end
σ_itz(Σ, θ, φ = zero(θ)) = itz_stress_operator(e_r(θ, φ)) ⊡ Σ
nothing #hide

# The six conditions of Part I hold, at any point and for any load: the three
# components of the traction, and the three strains along the interface,
# ``\underline t\cdot\boldsymbol\varepsilon\cdot\underline t'`` for ``\underline t, \underline t'``
# in ``\{\underline e_\theta, \underline e_\varphi\}``. Here at a point off every
# symmetry plane, under a load without symmetry:

Σ_any = Tens([0.3 0.5 -0.2; 0.5 -1.0 0.4; -0.2 0.4 0.7])
θ₀, φ₀ = 0.3π, 0.2π
n, t, s = e_r(θ₀, φ₀), e_θ(θ₀, φ₀), e_φ(θ₀, φ₀)
σ_I, σ_a = σ_itz(Σ_any, θ₀, φ₀), B_σ ⊡ Σ_any
ε_I, ε_a = inv(C_itz) ⊡ σ_I, A_ε ⊡ Σ_any
traction_gap = norm(σ_I ⋅ n - σ_a ⋅ n)
strain_gap = maximum(abs(a ⋅ (ε_I - ε_a) ⋅ b) for (a, b) in ((t, t), (s, s), (t, s)))
(traction_gap, strain_gap)

# The same stress is also the limit of a coated sphere whose ITZ shell gets thin,
# an independent approach through the exact fields of
# [Layered sphere — pointwise fields](@ref th-layered-sphere-pointwise). With a
# shell of relative thickness ``10^{-6}``, read on the outer side of the
# aggregate:

coated = LayeredSphere((1.0, 1.0 + 1.0e-6), (C_agg, C_itz))
B_coated = flux_gradient_concentration(coated, C_cp, C_cp)
ε∞ = inv((1 - f_agg) * C_cp + f_agg * B_coated) ⊡ uz   # Mori–Tanaka remote strain
fields = LayeredSphereFields(coated, C_cp)
σ_sphere(θ, φ = 0.0) = local_flux(fields, 1.0, θ, φ, ε∞; side = :outer)
norm(σ_sphere(θ₀, φ₀) - σ_itz(uz, θ₀, φ₀))

# The difference is of the order of the shell thickness, as it should be: the
# jump conditions are the thin-shell limit of the layered sphere.
#
# Under the uniaxial load, the stress of the ITZ in the spherical frame has four
# nonzero components, functions of ``\theta`` only. The curves below are those of
# the jump formula; the dots, at a few angles, those of the coated sphere.

comp(a, σ, b) = a ⋅ σ ⋅ b
function spherical_components(σ, θ)
    r, h, f = e_r(θ, 0.0), e_θ(θ, 0.0), e_φ(θ, 0.0)
    return [comp(r, σ, r), comp(h, σ, h), comp(f, σ, f), comp(r, σ, h)]
end
principal(σ) = reverse(eigvals(Symmetric(canonical_components(σ))))   # σ_I ≥ σ_II ≥ σ_III

S_jump = reduce(hcat, spherical_components(σ_itz(uz, θ), θ) for θ in θs)'
P_jump = reduce(hcat, principal(σ_itz(uz, θ)) for θ in θs)'
θdots = range(0, π / 2; length = 7)
S_dots = reduce(hcat, spherical_components(σ_sphere(θ), θ) for θ in θdots)'
P_dots = reduce(hcat, principal(σ_sphere(θ)) for θ in θdots)'

component_names = ["σ_rr" "σ_θθ" "σ_φφ" "σ_rθ"]
p1 = plot(
    θs ./ π, S_jump; label = component_names, lw = 2, xlabel = "θ / π", ylabel = "stress / Σ",
    title = "ITZ stress, spherical components", legend = :topright
)
scatter!(p1, θdots ./ π, S_dots; label = "", color = [1 2 3 4], ms = 4, markerstrokewidth = 0)
p2 = plot(
    θs ./ π, P_jump; label = ["σ_I" "σ_II" "σ_III"], lw = 2, xlabel = "θ / π",
    ylabel = "stress / Σ", title = "ITZ stress, principal values", legend = :topright
)
scatter!(p2, θdots ./ π, P_dots; label = "", color = [1 2 3], ms = 4, markerstrokewidth = 0)
for p in (p1, p2)
    hline!(p, [0]; color = :black, lw = 0.5, label = "")
end
plot(p1, p2; layout = (1, 2), size = (900, 400), left_margin = 6Plots.mm, bottom_margin = 6Plots.mm)

# The radial and shear components are the tractions of §3, continuous across the
# interface. The two tangential components ``\sigma_{\theta\theta}`` and
# ``\sigma_{\varphi\varphi}`` are not: they are set by the in-plane strain of the
# aggregate acting on the softer ITZ. The largest principal stress, which
# combines all four, peaks **off** the poles, where the shear of the tractions
# rotates the principal directions.

## The largest principal ITZ stress at the zenith angle θ, under an axisymmetric Σ.
peak_at(Σ, θ) = first(principal(σ_itz(Σ, θ)))
## Its maximum over the hemisphere: a grid locates it, a golden-section search
## refines it within the grid step. Comparisons of numbers: not a symbolic path.
function itz_peak_axisymmetric(Σ; n = 721)
    θg = range(0, π / 2; length = n)
    i = argmax([peak_at(Σ, θ) for θ in θg])
    a, b = θg[max(i - 1, 1)], θg[min(i + 1, n)]
    g = (sqrt(5) - 1) / 2
    for _ in 1:60
        c, d = b - g * (b - a), a + g * (b - a)
        peak_at(Σ, c) > peak_at(Σ, d) ? (b = d) : (a = c)
    end
    θ = (a + b) / 2
    return peak_at(Σ, θ), θ
end
pt, θt = itz_peak_axisymmetric(uz)
pc, θc = itz_peak_axisymmetric(-uz)
check("max σ_I(ITZ) / Σ, tension (Part I, 20 %)", pt, val(part1, "itz_peak_tension"))
check("max σ_I(ITZ) / |Σ|, compression (Part I, 27 %)", pc, val(part1, "itz_peak_compression"))
check("max σ_I(ITZ) / Σ, tension (Part II)", pt, val(part2, "itz_peak_tension"))
check("max σ_I(ITZ) / Σ, compression (Part II, signed)", -pc, val(part2, "itz_peak_compression"))

# Under uniaxial tension the peak is at ``\theta \approx 0.11\pi``, about
# ``\pi/9`` from the poles; under uniaxial compression, at
# ``\theta \approx 0.33\pi``, about ``\pi/6`` from the equator. Part I gives
# these angles roughly (``\pi/8`` and ``\pi/6``), Part II to three digits:

check("θ / π of the peak, tension", θt / π, val(part2, "theta_tension"))
check("θ / π of the peak, compression", θc / π, val(part2, "theta_compression"))

# ## 5. Part II: two ways for a crack to start
#
# The stresses of §§3–4 are elastic, so they hold up to the first crack. Part II
# assigns a tensile strength to each of the two places where that crack can
# appear, and asks which macroscopic stresses reach either strength first.
#
# ![The interface criterion compares the normal traction on the aggregate with the bond strength; the volume criterion compares the largest principal stress of the ITZ with its tensile strength](../../assets/applications/itz_failure_modes.svg)
#
# - **ITZ–aggregate separation** is a failure *of the interface*: the bond
#   between aggregate and ITZ opens where the normal traction ``T_r`` first
#   exceeds a bond strength ``T^{\mathrm{ult}}``. Only the traction enters, and
#   the crack runs along the aggregate surface.
# - **ITZ failure** is a failure *of the volume* of the ITZ: a Rankine criterion
#   in the ITZ, the largest principal stress
#   ``\sigma_{\mathrm I}(\boldsymbol\sigma_{\mathrm{ITZ}})`` reaching the tensile
#   strength ``\sigma^{\mathrm{ult}}_{\mathrm{ITZ}}``. All six components enter,
#   and the crack opens normal to the principal direction, wherever it points.
#
# Both stresses are linear in ``\boldsymbol\Sigma``, so each criterion bounds a
# domain of macroscopic stresses whose boundary is reached, along any loading
# direction ``\boldsymbol N``, at ``\boldsymbol\Sigma = s\,\boldsymbol N`` with ``s``
# the strength divided by the peak stress per unit load: the largest principal
# stress is positively homogeneous of degree one. When the peak stress is not
# tensile, there is no limit in that direction.
#
# ## 6. The interface criterion: ITZ–aggregate separation
#
# The criterion reads
#
# ```math
# f_{\mathrm{sep}}(\boldsymbol\Sigma)
#   = \max_{\underline n}\, \underline n\cdot\boldsymbol\sigma_{\mathrm{agg}}\cdot\underline n
#   - T^{\mathrm{ult}} \le 0 .
# ```
#
# The maximum over the unit sphere of the quadratic form of a symmetric tensor is
# its largest eigenvalue, reached on its principal axis. With
# ``\boldsymbol\sigma_{\mathrm{agg}} = B_{\mathrm{vol}}\,\Sigma_m\boldsymbol 1 + B_{\mathrm{dev}}\,\boldsymbol\Sigma^{\mathrm d}``,
# ``\Sigma_m = \operatorname{tr}\boldsymbol\Sigma/3`` and ``\boldsymbol\Sigma^{\mathrm d}`` its
# deviator, the separation criterion is therefore explicit:
#
# ```math
# f_{\mathrm{sep}}(\boldsymbol\Sigma)
#   = B_{\mathrm{vol}}\,\Sigma_m + B_{\mathrm{dev}}\,\sigma_{\mathrm I}(\boldsymbol\Sigma^{\mathrm d}) - T^{\mathrm{ult}},
# ```
#
# and the crack starts where ``\underline n`` is the principal direction of the
# largest macroscopic principal stress. In principal stress space, each of the
# three principal stresses can be the largest, which gives three planes
# ``B_{\mathrm{vol}}\Sigma_m + B_{\mathrm{dev}}(\Sigma_k - \Sigma_m) = T^{\mathrm{ult}}``:
# the domain is a three-sided pyramid around the hydrostatic axis, open towards
# hydrostatic compression.

σI(σ) = eigmax(Symmetric(canonical_components(σ)))
## A sign test on the peak stress: a numerical path.
separation_limit(N) = (λ = σI(B_σ ⊡ N); λ > 0 ? 1 / λ : Inf)   # in units of T^ult
nothing #hide

# Along a uniaxial load the two signs differ. In tension, the largest principal
# stress of the deviator is ``2\Sigma/3``, and the limit is
# ``\Sigma^{\mathrm{sep}}_{\mathrm{ut}} = 3T^{\mathrm{ult}}/(B_{\mathrm{vol}} + 2B_{\mathrm{dev}})``,
# reached at the poles. In compression it is ``|\Sigma|/3``, and
# ``\Sigma^{\mathrm{sep}}_{\mathrm{uc}} = 3T^{\mathrm{ult}}/(B_{\mathrm{vol}} - B_{\mathrm{dev}})``,
# reached at the equator: a compressive limit exists only because
# ``B_{\mathrm{dev}} > B_{\mathrm{vol}}``, and it is large, since that difference
# is small. Under equibiaxial compression the deviator has the largest
# principal stress ``2|\Sigma|/3`` along ``\underline e_z``, so separation starts at
# the poles, under half the uniaxial load.

sep_ut, sep_uc, sep_bc = separation_limit(uz), -separation_limit(-uz), -separation_limit(-bxy)
check("Σ_ut / T^ult, separation", sep_ut, val(part2, "sep_uniaxial_tension"))
check("Σ_uc / T^ult, separation", sep_uc, val(part2, "sep_uniaxial_compression"))
check("Σ_bc / T^ult, separation", sep_bc, val(part2, "sep_biaxial_compression"))
check("3/(B_vol + 2B_dev), closed form", 3 / (B_vol + 2B_dev), val(part2, "sep_uniaxial_tension"))
check("3/(B_vol − B_dev), closed form", 3 / (B_vol - B_dev), val(part2, "sep_uniaxial_compression"))

# The pyramid in principal stress space, cut by the deviatoric plane
# ``\xi = \sqrt 3\,\Sigma_m = -40`` (in units of ``T^{\mathrm{ult}}``). Its apex lies on the
# hydrostatic axis, at ``\Sigma_m = T^{\mathrm{ult}}/B_{\mathrm{vol}}``, and its
# edges point to the three uniaxial compressions, far away:

include(joinpath(pkgdir(MeanFieldHomogenization), "scripts", "common", "docviz.jl"))
nothing #hide

#md # ```@raw html
#md # <details class="source-file"><summary>Show the code of <code>scripts/common/docviz.jl</code>, included above</summary>
#md # ```
#md #
#md # ```@eval
#md # Main.show_source("scripts/common/docviz.jl")
#md # ```
#md #
#md # ```@raw html
#md # </details>
#md # ```

N3 = [1, 1, 1] / sqrt(3)       # hydrostatic axis
N1 = [-1, 1, 0] / sqrt(2)      # two deviatoric directions
N2 = [-1, -1, 2] / sqrt(6)
direction(u, v) = cos(u) * N3 + sin(u) * (cos(v) * N1 + sin(v) * N2)   # u: angle from N3

## The surface `radius(u, v)` of a limit, from its apex (u = 0) to the deviatoric
## plane ξ = ξcut, one meridian per v. Bisection on the angle at which the
## meridian crosses that plane: the limit is infinite near hydrostatic
## compression, where the product below is -Inf.
function u_cut(radius, v, ξcut)
    lo, hi = π / 2, π
    for _ in 1:40
        mid = (lo + hi) / 2
        radius(mid, v) * cos(mid) > ξcut ? (lo = mid) : (hi = mid)
    end
    return lo
end
meridian(radius, v, ξcut; n = 41) = (
    uc = u_cut(radius, v, ξcut);
    [radius(t * uc, v) * direction(t * uc, v) for t in range(0, 1; length = n)]
)
envelope(radius, ξcut; vs = range(0, 2π; length = 73)) = reduce(hcat, [meridian(radius, v, ξcut) for v in vs])

sep_radius(u, v) = separation_limit(Tens(diagm(direction(u, v))))
Psep = envelope(sep_radius, -40.0)
plotly_scene(
    [
        surface_trace(getindex.(Psep, 1), getindex.(Psep, 2), getindex.(Psep, 3); color = "#e08a3c", opacity = 0.85),
        line_trace(Psep[end, :]; color = FIG_VECTOR, width = 5),
        line_trace([-45 * N3, 3 * N3]; color = FIG_GUIDE, width = 3),
    ];
    uid = "itz-separation-3d", height = 520, axes_visible = true, aspectmode = "cube",
    camera = (1.5, -1.5, 0.9),
    xlabel = "Σ₁ / T^ult", ylabel = "Σ₂ / T^ult", zlabel = "Σ₃ / T^ult",
    title = "ITZ–aggregate separation: elastic domain in principal stress space, cut at ξ = −40",
)

# ## 7. The volume criterion: ITZ failure
#
# The criterion reads
#
# ```math
# f_{\mathrm{ITZ}}(\boldsymbol\Sigma)
#   = \max_{\underline n}\, \sigma_{\mathrm I}\big(\mathbb M(\underline n):\boldsymbol\Sigma\big)
#   - \sigma^{\mathrm{ult}}_{\mathrm{ITZ}} \le 0 ,
# ```
#
# a maximum over the aggregate surface of the largest principal stress of the
# ITZ. Unlike the traction, ``\mathbb M(\underline n):\boldsymbol\Sigma`` has no
# extremum in closed form, so the maximum is sought on a grid of normals. Under
# the uniaxial loads, by symmetry, the zenith angle alone is swept (§4); the
# limits follow, together with the direction along which the first crack opens,
# the principal direction of ``\sigma_{\mathrm I}`` at the peak.

itz_ut, itz_uc = 1 / pt, -1 / pc
pb, θb = itz_peak_axisymmetric(-bxy)
itz_bc = -1 / pb
check("Σ_ut / σ_ITZ^ult, ITZ failure", itz_ut, val(part2, "itz_uniaxial_tension"))
check("Σ_uc / σ_ITZ^ult, ITZ failure", itz_uc, val(part2, "itz_uniaxial_compression"))
check("Σ_bc / σ_ITZ^ult, ITZ failure", itz_bc, val(part2, "itz_biaxial_compression"))
## The paper's angle differs from the refined maximum by one unit of its last
## digit, the size of a step of a sweep over the angle.
check("θ of the peak (rad), equibiaxial compression", θb, val(part2, "theta_biaxial_compression"); units = 1)

## The direction of the largest principal stress of a symmetric 3×3 stress.
principal_direction(σ) = eigvecs(Symmetric(canonical_components(σ)))[:, end]
## The angle, in units of π, between e_z and that direction. A direction has no
## sign, so the angle is defined up to its supplement; it is folded to [0, 1/2].
crack_angle(σ) = (a = acos(clamp(principal_direction(σ)[3], -1, 1)) / π; min(a, 1 - a))
fold(a) = min(a, 1 - a)
check("crack normal ∠ e_z (/π), tension", crack_angle(σ_itz(uz, θt)), fold(val(part2, "crack_angle_tension")))
check("crack normal ∠ e_z (/π), compression", crack_angle(σ_itz(-uz, θc)), fold(val(part2, "crack_angle_compression")))

# Under tension the first ITZ crack is nearly normal to the load; under
# compression its normal is nearly orthogonal to the load, i.e. the crack runs
# along it. Both criteria thus agree on the orientation of the first cracks,
# not on their place. The meridian section below shows, for each load, where
# each criterion places the first crack and how it is oriented: on the surface
# of the aggregate for separation, along that surface; inside the ITZ for ITZ
# failure, normal to the principal direction.

## The two mechanisms keep the colors of the drawings above.
const SEP_COLOR = "#E07B39"
const ITZ_COLOR = "#9558B2"
function crack_sites!(p, Σ, θpeak)
    φs = (0.0, π)                                     # the two halves of the x–z section
    θsep = T_r(Σ, 0.0) ≥ T_r(Σ, π / 2) ? (0.0, π) : (π / 2,)
    for φ in φs, θ in θsep                           # separation: arcs along the surface
        θarc = range(θ - 0.14, θ + 0.14; length = 15)
        plot!(p, sin.(θarc) .* cos(φ), cos.(θarc); color = SEP_COLOR, lw = 5, label = "")
    end
    for φ in φs, θ in (θpeak, π - θpeak)             # ITZ failure: segments normal to e_I
        x = 1.05 * e_r(θ, φ)
        d = principal_direction(σ_itz(Σ, θ, φ))
        c = [-d[3], d[1]] / hypot(d[1], d[3])         # the crack trace, normal to e_I in the section
        plot!(p, x[1] .+ 0.16 .* [-c[1], c[1]], x[3] .+ 0.16 .* [-c[2], c[2]]; color = ITZ_COLOR, lw = 4, label = "")
    end
    return p
end
function aggregate_section(title, Σ, θpeak, arrow)
    ψ = range(0, 2π; length = 200)
    p = plot(
        cos.(ψ), sin.(ψ); seriestype = :shape, color = :lightgray, linecolor = :gray, label = "",
        aspect_ratio = :equal, axis = false, grid = false, title = title, xlims = (-1.9, 1.9), ylims = (-1.9, 1.9)
    )
    plot!(p, 1.1 .* cos.(ψ), 1.1 .* sin.(ψ); color = :gray, lw = 1, ls = :dot, label = "")
    for y in (1.45, -1.45)                           # the load, along e_z
        s = arrow * sign(y)
        plot!(p, [0, 0], [y - 0.2s, y + 0.2s]; arrow = true, color = :black, lw = 2, label = "")
    end
    crack_sites!(p, Σ, θpeak)
    plot!(p, [NaN], [NaN]; color = SEP_COLOR, lw = 5, label = "separation (interface)")
    plot!(p, [NaN], [NaN]; color = ITZ_COLOR, lw = 4, label = "ITZ failure (volume)")
    return p
end
plot(
    aggregate_section("uniaxial tension", uz, θt, 1), aggregate_section("uniaxial compression", -uz, θc, -1);
    layout = (1, 2), size = (900, 470), legend = :outerbottom, legendfontsize = 9,
)

# The dotted circle stands for the outer face of the ITZ, its thickness much
# enlarged. The ITZ cracks are drawn in the four symmetric places of the section;
# in space each is a circle of latitude.

# ## 8. The elastic limit: the first criterion reached
#
# Both mechanisms are possible at every point, so the elastic limit is reached
# when the first of the two criteria is:
#
# ```math
# f_{\mathrm{comb}}(\boldsymbol\Sigma)
#   = \max\big(f_{\mathrm{sep}}(\boldsymbol\Sigma),\, f_{\mathrm{ITZ}}(\boldsymbol\Sigma)\big) \le 0 .
# ```
#
# In units of ``\sigma^{\mathrm{ult}}_{\mathrm{ITZ}}``, the separation limits are
# multiplied by the strength ratio ``\rho = T^{\mathrm{ult}}/\sigma^{\mathrm{ult}}_{\mathrm{ITZ}}``,
# and along each loading direction the elastic limit is the smaller of the two:

limit_ut(ρ) = min(ρ * sep_ut, itz_ut)
limit_uc(ρ) = min(ρ * abs(sep_uc), abs(itz_uc))
limit_bc(ρ) = min(ρ * abs(sep_bc), abs(itz_bc))
nothing #hide

# Neither strength is measured directly, but ratios of elastic limits are: Part
# II builds, from tests in the literature, a range for the ratio of the tensile
# to the compressive uniaxial limit, and one for the ratio of the equibiaxial to
# the uniaxial compressive limit. The model gives each as a function of ``\rho``,
# with three regimes, whose boundaries are the values of ``\rho`` at which the two
# criteria give the same limit:

ρ_uc = abs(itz_uc / sep_uc)    # above, ITZ failure governs uniaxial compression
ρ_ut = itz_ut / sep_ut         # above, ITZ failure also governs uniaxial tension
ρ_bc = abs(itz_bc / sep_bc)    # above, ITZ failure governs equibiaxial compression
check("ρ: compression changes mode", ρ_uc, val(part2, "ratio_ut_uc_bound_low"))
check("ρ: tension changes mode", ρ_ut, val(part2, "ratio_ut_uc_bound_high"))
check("ρ: equibiaxial compression changes mode", ρ_bc, val(part2, "ratio_bc_uc_bound_high"))
check("Σ_ut/|Σ_uc|, separation in both", sep_ut / abs(sep_uc), val(part2, "ratio_ut_uc_separation"))
check("Σ_ut/|Σ_uc| per unit ρ, mixed", sep_ut / abs(itz_uc), val(part2, "ratio_ut_uc_mixed_slope"))
check("Σ_ut/|Σ_uc|, ITZ failure in both", itz_ut / abs(itz_uc), val(part2, "ratio_ut_uc_itz"))
check("Σ_bc/Σ_uc, separation in both", sep_bc / sep_uc, val(part2, "ratio_bc_uc_separation"))
check("Σ_bc/Σ_uc per unit ρ, mixed", abs(sep_bc / itz_uc), val(part2, "ratio_bc_uc_mixed_slope"))
check("Σ_bc/Σ_uc, ITZ failure in both", itz_bc / itz_uc, val(part2, "ratio_bc_uc_itz"))

# The ratios against ``\rho``, with the experimental ranges of
# [konigsberger2014strength; Eqs. 23 and 28](@cite) shaded and the regime
# boundaries dashed:

ρs = exp10.(range(-2.5, 0.7; length = 400))
band(p, lo, hi) = hspan!(p, [lo, hi]; color = :green, alpha = 0.15, label = "experimental range")
q1 = plot(
    ρs, limit_ut.(ρs) ./ limit_uc.(ρs); xscale = :log10, lw = 2, label = "model",
    xlabel = "ρ = T^ult / σ_ITZ^ult", ylabel = "Σ_ut / |Σ_uc|", title = "tension / compression", legend = :topleft
)
band(q1, val(part2, "experimental_ratio_ut_uc_min"), val(part2, "experimental_ratio_ut_uc_max"))
vline!(q1, [ρ_uc, ρ_ut]; color = :gray, ls = :dash, label = "")
q2 = plot(
    ρs, limit_bc.(ρs) ./ limit_uc.(ρs); xscale = :log10, lw = 2, label = "model",
    xlabel = "ρ = T^ult / σ_ITZ^ult", ylabel = "Σ_bc / Σ_uc", title = "equibiaxial / uniaxial compression", legend = :bottomright
)
band(q2, val(part2, "experimental_ratio_bc_uc_min"), val(part2, "experimental_ratio_bc_uc_max"))
vline!(q2, [ρ_uc, ρ_bc]; color = :gray, ls = :dash, label = "")
plot(q1, q2; layout = (1, 2), size = (900, 400), left_margin = 6Plots.mm, bottom_margin = 6Plots.mm)

# ![The three regimes of the strength ratio and the mechanism that governs each load](../../assets/applications/itz_regimes.svg)
#
# - For ``\rho`` below ``\rho_{\mathrm{uc}} \approx 0.05``, separation governs
#   every load. The tension-to-compression ratio is then about 0.012, an order
#   of magnitude below the measured range: concrete is not that weak in tension.
# - Between ``0.05`` and ``\rho_{\mathrm{ut}} \approx 0.98``, the ITZ breaks first
#   in compression and separation still starts first in tension; the ratio grows
#   linearly with ``\rho`` and enters the measured range from
#   ``\rho \approx 0.5``.
# - Above ``0.98``, the ITZ breaks first under every load, and the ratio is
#   0.225, near the middle of the measured range.
#
# The equibiaxial ratio confirms that, in compression, the ITZ fails first: the
# model lies in the measured range only above ``\rho_{\mathrm{bc}} \approx 0.10``,
# where it equals 1.02. Together, the two comparisons ask for a bond strength at
# least half the strength of the ITZ.
#
# Under a general biaxial load, the combined envelope is the inner of the two,
# and each arc of it belongs to one mechanism. For two strength ratios, one in
# each of the two upper regimes:

ψs = range(0, 2π; length = 241)
radial(limit) = [limit(diagm([cos(ψ), sin(ψ), 0.0])) for ψ in ψs]

θgrid = range(0, π / 2; length = 46)
φgrid = range(0, π / 2; length = 46)
## The operators 𝕄(n) on an octant of normals, stacked into one matrix, so that
## the ITZ stresses at all the normals come out of a single product. A diagonal
## Σ is invariant under the reflections of the three principal planes, so one
## octant suffices.
stacked = reduce(
    vcat, [
        reshape(canonical_components(itz_stress_operator(e_r(θ, φ))), 9, 9)
            for θ in θgrid for φ in φgrid
    ]
)

## Largest eigenvalue of the symmetric matrix [a d e; d b f; e f c], in closed
## form (trigonometric solution of the characteristic cubic). The guard and the
## clamp compare values: a numerical path, not a symbolic one.
function eigmax3(a, b, c, d, e, f)
    q = (a + b + c) / 3
    p = sqrt(((a - q)^2 + (b - q)^2 + (c - q)^2 + 2 * (d^2 + e^2 + f^2)) / 6)
    iszero(p) && return q
    A, B, C, D, E, F = (a - q) / p, (b - q) / p, (c - q) / p, d / p, e / p, f / p
    r = (A * B * C + 2 * D * F * E - A * F^2 - B * E^2 - C * D^2) / 2
    return q + 2p * cos(acos(clamp(r, -1, 1)) / 3)
end

function itz_peak(S)                       # S a 3×3 matrix
    σ = stacked * vec(S)                   # 9 components per normal, column-major
    return maximum(
        eigmax3(σ[k + 1], σ[k + 5], σ[k + 9], σ[k + 2], σ[k + 3], σ[k + 6])
            for k in 0:9:(length(σ) - 1)
    )
end
## A sign test on the peak stress: a numerical path.
itz_limit(S) = (p = itz_peak(S); p > 0 ? 1 / p : Inf)    # units of σ_ITZ^ult

r_itz = radial(itz_limit)
r_sep = radial(N -> separation_limit(Tens(N)))
function combined_panel(ρ)
    r = min.(ρ .* r_sep, r_itz)
    sep = ρ .* r_sep .< r_itz
    ## Each arc keeps its neighbors, so that the two arcs meet where the mode changes.
    widen(g) = g .| circshift(g, 1) .| circshift(g, -1)
    by_sep = ifelse.(widen(sep), r, NaN)
    by_itz = ifelse.(widen(.!sep), r, NaN)
    p = plot(
        by_itz .* cos.(ψs), by_itz .* sin.(ψs); color = ITZ_COLOR, lw = 3, label = "ITZ failure",
        aspect_ratio = :equal, xlabel = "Σxx / σ_ITZ^ult", ylabel = "Σyy / σ_ITZ^ult",
        title = @sprintf("ρ = %.1f", ρ), legend = :bottomright, xlims = (-4.5, 1.5), ylims = (-4.5, 1.5)
    )
    plot!(p, by_sep .* cos.(ψs), by_sep .* sin.(ψs); color = SEP_COLOR, lw = 3, label = "separation")
    hline!(p, [0]; color = :black, lw = 0.5, label = "")
    vline!(p, [0]; color = :black, lw = 0.5, label = "")
    return p
end
plot(
    combined_panel(0.6), combined_panel(1.5); layout = (1, 2), size = (900, 430),
    left_margin = 6Plots.mm, bottom_margin = 6Plots.mm
)

# With ``\rho = 0.6``, separation cuts the tensile corner of the ITZ envelope and
# the ITZ governs as soon as one principal stress is compressive enough; with
# ``\rho = 1.5`` the ITZ envelope is the elastic limit everywhere.
#
# ## 9. The ITZ envelope in principal stress space
#
# The ITZ criterion alone, now for general loads. Under biaxial loading
# ``\boldsymbol\Sigma = \Sigma_{xx}\,\underline e_x\otimes\underline e_x + \Sigma_{yy}\,\underline e_y\otimes\underline e_y``
# the two envelopes, in the units of their own strength:

p1 = plot(
    r_itz .* cos.(ψs), r_itz .* sin.(ψs); lw = 2, aspect_ratio = :equal, legend = false,
    xlabel = "Σxx / σ_ITZ^ult", ylabel = "Σyy / σ_ITZ^ult", title = "ITZ failure"
)
p2 = plot(
    r_sep .* cos.(ψs), r_sep .* sin.(ψs); lw = 2, aspect_ratio = :equal, legend = false,
    xlabel = "Σxx / T^ult", ylabel = "Σyy / T^ult", title = "ITZ–aggregate separation"
)
for p in (p1, p2)
    hline!(p, [0]; color = :black, lw = 0.5)
    vline!(p, [0]; color = :black, lw = 0.5)
end
plot!(p1, [-4, 1], [-4, 1]; color = :gray, ls = :dash, lw = 1)   # Σxx = Σyy
plot(p1, p2; layout = (1, 2), size = (900, 420))

# The ITZ envelope is symmetric about the diagonal ``\Sigma_{xx} = \Sigma_{yy}`` (dashed), as it
# must be for an isotropic material. Its edges are slightly curved: each corresponds
# to a family of points of the aggregate surface where the maximum moves as the
# load turns.
#
# Sections of the ITZ envelope by deviatoric planes, at fixed values of the
# coordinate ``\xi = \sqrt 3\,\Sigma_m`` along the hydrostatic axis, show the shape of the
# full surface. Each point is found by bisection on the deviatoric radius:

function section_radius(ξ, ψ; xmax = 8.0)
    S(x) = diagm(ξ * N3 + x * (cos(ψ) * N1 + sin(ψ) * N2))
    itz_peak(S(0.0)) ≥ 1 && return NaN      # beyond the hydrostatic limit
    lo, hi = 0.0, xmax
    for _ in 1:30
        mid = (lo + hi) / 2
        itz_peak(S(mid)) < 1 ? (lo = mid) : (hi = mid)
    end
    return lo
end
ψd = range(0, 2π; length = 73)
p = plot(;
    aspect_ratio = :equal, xlabel = "along N₁", ylabel = "along N₂",
    legend = :outerright, title = "Deviatoric sections of the ITZ envelope"
)
for ξ in -2.0:0.5:1.0
    r = [section_radius(ξ, ψ) for ψ in ψd]
    plot!(p, r .* cos.(ψd), r .* sin.(ψd); lw = 2, label = "ξ = $ξ")
end
p

# The sections shrink to a point as the hydrostatic tension grows, and widen
# under hydrostatic compression: a Rankine criterion in the ITZ makes concrete
# sensitive to the mean stress, with no friction law assumed anywhere. Under
# hydrostatic compression both the tractions and the ITZ stresses are
# compressive everywhere, and neither criterion is ever reached.
#
# The whole surface, cut by the deviatoric plane ``\xi = -5`` since it does not
# close under hydrostatic compression. Its six curved faces surround the
# hydrostatic axis (gray line); each meridian is followed from the apex up to the
# angle at which it crosses the cutting plane, and the rim of the surface is the
# section of the envelope by that plane.

ξcut = -5.0
itz_radius(u, v) = itz_limit(diagm(direction(u, v)))
ξapex = itz_radius(0.0, 0.0)                            # hydrostatic tension
P = envelope(itz_radius, ξcut)
surface = surface_trace(
    getindex.(P, 1), getindex.(P, 2), getindex.(P, 3);
    color = "#4a90d9", opacity = 0.85
)
rim = line_trace(P[end, :]; color = FIG_VECTOR, width = 5)
axis = line_trace([(ξcut - 1) * N3, (ξapex + 1) * N3]; color = FIG_GUIDE, width = 3)
nothing #hide

# Three families of planes cut it. The deviatoric planes ``\xi = \text{const}``
# give back the sections of the previous figure. The planes ``\Sigma_3 = s``
# contain the hydrostatic point ``(s, s, s)``, inside the envelope as long as it
# lies below the apex, so that the section is found by bisection along the rays
# issued from it; ``s = 0`` gives back the biaxial envelope drawn above. Under
# lateral compression these sections reach beyond the cut of the surface, and
# that part of them is drawn in the panel only. The meridian planes contain the
# hydrostatic axis and the deviatoric direction
# ``\cos\psi\,\underline N_1 + \sin\psi\,\underline N_2``, where
# ``\underline N_1 = (-1, 1, 0)/\sqrt 2`` and ``\underline N_2 = (-1, -1, 2)/\sqrt 6``
# are the in-plane axes of the previous figure; their sections are two meridians
# of the surface itself, and ``\psi = 90^{\circ}`` is the plane
# ``\Sigma_1 = \Sigma_2`` of axisymmetric triaxial tests.

ψc = range(0, 2π; length = 73)
points(cols) = reduce(hcat, cols)
dirN(ψ) = cos(ψ) * N1 + sin(ψ) * N2

## Deviatoric planes ξ = const.
ξs = filter(<(ξapex), -5.0:0.5:2.0)
rdev = [[section_radius(ξ, ψ; xmax = 20.0) for ψ in ψc] for ξ in ξs]
hdev = 1.15 * maximum(maximum(r) for r in rdev)
deviatoric = CutFamily(
    "Deviatoric plane", "ξ", string.(ξs),
    [plane_patch(ξ * N3, N1, N2, hdev, hdev) for ξ in ξs],
    [points(ξ * N3 + ri * dirN(ψ) for (ri, ψ) in zip(r, ψc)) for (ξ, r) in zip(ξs, rdev)],
    [points([ri * cos(ψ), ri * sin(ψ)] for (ri, ψ) in zip(r, ψc)) for r in rdev],
    ("along N₁", "along N₂"); active = findfirst(==(0.0), ξs),
)

## Planes Σ₃ = s, by bisection along the rays issued from (s, s, s).
function biaxial_point(s, ψ; xmax = 20.0)
    S(x) = [s + x * cos(ψ), s + x * sin(ψ), s]
    lo, hi = 0.0, xmax
    for _ in 1:40
        mid = (lo + hi) / 2
        itz_peak(diagm(S(mid))) < 1 ? (lo = mid) : (hi = mid)
    end
    return S(lo)
end
sbi = [s for s in -2.0:0.25:1.0 if sqrt(3) * s < ξapex]
cbi = [points(biaxial_point(s, ψ) for ψ in ψc) for s in sbi]
## On the surface, the part of a section beyond the cut is left out.
onsurface(C) = ifelse.(sum(C; dims = 1) ./ sqrt(3) .≥ ξcut, C, NaN)
lo, hi = extrema(vcat(getindex.(P, 1), getindex.(P, 2)))
biaxial = CutFamily(
    "Plane Σ₃ = s", "s", string.(sbi),
    [plane_patch([(lo + hi) / 2, (lo + hi) / 2, s], [1, 0, 0], [0, 1, 0], 0.55 * (hi - lo), 0.55 * (hi - lo)) for s in sbi],
    onsurface.(cbi), [C[1:2, :] for C in cbi],
    ("Σ₁ / σ_ITZ^ult", "Σ₂ / σ_ITZ^ult"); active = findfirst(==(0.0), sbi),
)

## Meridian planes: two meridians of the surface, joined at the apex.
ψm = 0:15:165
cme = [points(vcat(reverse(meridian(itz_radius, deg2rad(ψ) + π, ξcut)), meridian(itz_radius, deg2rad(ψ), ξcut))) for ψ in ψm]
hme = 1.15 * maximum(maximum(abs, transpose(dirN(deg2rad(ψ))) * C) for (ψ, C) in zip(ψm, cme))
meridional = CutFamily(
    "Meridian plane", "ψ", ["$(ψ)°" for ψ in ψm],
    [plane_patch((ξcut + ξapex) / 2 * N3, N3, dirN(deg2rad(ψ)), 0.55 * (ξapex - ξcut), hme) for ψ in ψm],
    cme, [vcat(transpose(dirN(deg2rad(ψ))) * C, transpose(N3) * C) for (ψ, C) in zip(ψm, cme)],
    ("along cos ψ N₁ + sin ψ N₂", "ξ = √3 Σm"); active = findfirst(==(90), ψm),
)

section_figure(
    [surface, rim, axis], [deviatoric, biaxial, meridional];
    uid = "itz-envelope-3d", height = 640, camera = (1.6, -1.6, 1.0),
    xlabel = "Σ₁ / σ_ITZ^ult", ylabel = "Σ₂ / σ_ITZ^ult", zlabel = "Σ₃ / σ_ITZ^ult",
    title = "Elastic limit of concrete by ITZ failure, principal stress space, cut at ξ = −5"
)

# The buttons above the surface choose a family of planes, or none, and the slider
# below moves the plane; on the right, the section is drawn in coordinates of its
# plane, which is where the two-dimensional envelopes above are read.
#
# ## Where to go next
#
# The jump conditions used here are those of
# [Periodic multilayer — the laminate cell](@ref th-laminate), applied to a single
# interface, and the pointwise fields of the coated sphere that converge to them
# are described in [Layered sphere — pointwise fields](@ref th-layered-sphere-pointwise).
# The next application, [A lamellar porous material: swelling clays and C-S-H](@ref app-lamellar),
# moves to the geomaterials, whose grains interact through interfaces that slide.
