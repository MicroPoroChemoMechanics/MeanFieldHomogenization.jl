# [The elastic limit of concrete: ITZ failure and ITZ–aggregate separation](@id app-itz-elastic-limit)

!!! info "Before this page"
    [The interfacial transition zone in concrete](@ref app-itz-concrete), which
    introduces the coated aggregate, and
    [Periodic multilayer — the laminate cell](@ref th-laminate) §2–4, where the
    jump across a plane interface is written with the flat Hill tensors
    ``\mathbb P`` and ``\mathbb Q`` used below.

Concrete cracks long before it fails. Under a growing load, microcracks appear
first around the aggregates, either because the interfacial transition zone
(ITZ) separates from the aggregate or because the ITZ itself breaks, and the
macroscopic stress at that moment is the **elastic limit** of the material.
This page reproduces the strength upscaling of
[konigsberger2014strength](@citet), built on the stress concentration of the
companion paper [konigsberger2014stress](@citet). Both failure modes are
tension-driven: a normal traction exceeds a bond strength
``T^{\mathrm{ult}}``, or the largest principal stress of the ITZ exceeds its tensile
strength ``\sigma^{\mathrm{ult}}_{\mathrm{ITZ}}`` (a Rankine criterion).

The ITZ is a few tens of micrometers thick, much thinner than an aggregate, so
it does not change how the load is shared between paste and aggregates. What it
changes is the stress **at** the aggregate surface, and there the jump
conditions of a perfectly bonded interface give it in closed form, as a tensor
equation rather than component by component.

## 1. The typical concrete

The elastic constants are those of Table I of
[konigsberger2014stress](@citet): quartz aggregates at a volume fraction of
0.65, a mature cement paste, and an ITZ whose Young modulus is 85 % of that of
the paste. They are read from `data/literature/konigsberger2014stress.json`,
and the published results the page checks against from
`data/literature/konigsberger2014strength.json`.

```@example itzlim
using MeanFieldHomogenization, TensND, LinearAlgebra, JSON, Printf, Plots
gr()  # headless backend; GKSwstype is set to "100" in make.jl

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
nothing # hide
```

## 2. From concrete to the aggregate

Spherical aggregates in a paste matrix call for the Mori–Tanaka scheme. The
aggregate strain ``\boldsymbol\varepsilon_{\mathrm{agg}} = \mathbb A_{\mathrm{agg}}:\boldsymbol\varepsilon^{\infty}`` is
uniform, ``\mathbb A_{\mathrm{agg}}`` being the dilute concentration tensor in the
paste, and the remote strain ``\boldsymbol\varepsilon^{\infty}`` follows from the
macroscopic strain ``\boldsymbol E = \langle\mathbb A\rangle:\boldsymbol\varepsilon^{\infty}``. Both
aggregate fields are therefore linear in the macroscopic stress ``\boldsymbol\Sigma``:

```math
\boldsymbol\varepsilon_{\mathrm{agg}} = \mathbb A_{\mathrm{agg}}:\langle\mathbb A\rangle^{-1}:(\mathbb C^{\mathrm{hom}})^{-1}:\boldsymbol\Sigma,
\qquad
\boldsymbol\sigma_{\mathrm{agg}} = \mathbb C_{\mathrm{agg}}:\boldsymbol\varepsilon_{\mathrm{agg}}
  = \mathbb B_{\mathrm{agg}}:\boldsymbol\Sigma,
\qquad
\mathbb B_{\mathrm{agg}} = B_{\mathrm{vol}}\,\mathbb J + B_{\mathrm{dev}}\,\mathbb K.
```

```@example itzlim
A_agg = strain_strain_loc(Ellipsoid(1.0), C_agg, C_cp)
A_mean = (1 - f_agg) * TensISO{3}(1.0, 1.0) + f_agg * A_agg
C_hom = ((1 - f_agg) * C_cp + f_agg * C_agg ⊡ A_agg) ⊡ inv(A_mean)
A_ε = A_agg ⊡ inv(A_mean) ⊡ inv(C_hom)    # ε_agg = A_ε : Σ
B_σ = C_agg ⊡ A_ε                         # σ_agg = B_σ : Σ
B_vol, B_dev = get_data(B_σ)
@printf("B_vol = %.5f, B_dev = %.5f, B_vol − B_dev = %.5f (published %.5f)\n",
    B_vol, B_dev, B_vol - B_dev, val(part2, "Bvol_minus_Bdev"))
```

## 3. Separation: the normal traction on the aggregate

At a point of normal ``\underline n``, the aggregate surface carries the traction
``\boldsymbol\sigma_{\mathrm{agg}}\cdot\underline n``, continuous across the bonded interface, and its
normal component ``T_n = \underline n\cdot\boldsymbol\sigma_{\mathrm{agg}}\cdot\underline n``. Since
``\boldsymbol\sigma_{\mathrm{agg}}`` is uniform, the largest ``T_n`` over the surface is the largest
principal stress of the aggregate:

```math
\max_{\underline n}\,\underline n\cdot\boldsymbol\sigma_{\mathrm{agg}}\cdot\underline n
  = \sigma_{\mathrm I}(\mathbb B_{\mathrm{agg}}:\boldsymbol\Sigma)
  = B_{\mathrm{vol}}\,\Sigma_m + B_{\mathrm{dev}}\,\sigma_{\mathrm I}(\boldsymbol\Sigma^{\mathrm d}) \le T^{\mathrm{ult}} .
```

The separation domain is bounded by three planes in principal stress space,
one per principal direction, which is why it is a pyramid. Along a loading
direction ``\boldsymbol N``, the elastic limit is ``\boldsymbol\Sigma = s\,\boldsymbol N`` with
``s = T^{\mathrm{ult}}/\sigma_{\mathrm I}(\mathbb B_{\mathrm{agg}}:\boldsymbol N)`` when that principal
stress is positive, and no limit otherwise.

```@example itzlim
σI(σ) = eigmax(Symmetric(components_canon(σ)))
separation_limit(N) = (λ = σI(B_σ ⊡ N); λ > 0 ? 1 / λ : Inf)   # in units of T^ult

uz = Tens(Float64[0 0 0; 0 0 0; 0 0 1])      # uniaxial
bxy = Tens(Float64[1 0 0; 0 1 0; 0 0 0])     # equibiaxial
nothing # hide
```

## 4. ITZ failure: the stress across the interface

Across a perfectly bonded interface of normal ``\underline n``, the displacement is
continuous, so the strain can only jump by a rank-one symmetric term, and the
traction is continuous:

```math
\boldsymbol\varepsilon_{\mathrm{ITZ}} = \boldsymbol\varepsilon_{\mathrm{agg}} + \underline a\stackrel{s}{\otimes}\underline n,
\qquad
\boldsymbol\sigma_{\mathrm{ITZ}}\cdot\underline n = \boldsymbol\sigma_{\mathrm{agg}}\cdot\underline n .
```

These are the two conditions of the laminate cell. Solving for ``\underline a``
with the acoustic tensor ``\boldsymbol K = \underline n\cdot\mathbb C_{\mathrm{ITZ}}\cdot\underline n``
gives the Hadamard jump

```math
\boldsymbol\varepsilon_{\mathrm{ITZ}} = \boldsymbol\varepsilon_{\mathrm{agg}}
  + \mathbb P(\underline n):(\boldsymbol\sigma_{\mathrm{agg}} - \mathbb C_{\mathrm{ITZ}}:\boldsymbol\varepsilon_{\mathrm{agg}}),
\qquad
\mathbb P(\underline n) = \underline n\stackrel{s}{\otimes}\boldsymbol K^{-1}\stackrel{s}{\otimes}\underline n ,
```

the elastic counterpart of the interfacial operator that
[barthelemyBignonnetIJES2020](@citet) (their Appendix E) write for conduction.
``\mathbb P`` is the Hill tensor of a flat inclusion of the ITZ. Applying
``\mathbb C_{\mathrm{ITZ}}`` and introducing the second flat Hill tensor
``\mathbb Q = \mathbb C_{\mathrm{ITZ}} - \mathbb C_{\mathrm{ITZ}}:\mathbb P:\mathbb C_{\mathrm{ITZ}}`` separates what each condition
transmits:

```math
\boldsymbol\sigma_{\mathrm{ITZ}}(\underline n)
  = \mathbb Q(\underline n):\boldsymbol\varepsilon_{\mathrm{agg}}
  + \mathbb C_{\mathrm{ITZ}}:\mathbb P(\underline n):\boldsymbol\sigma_{\mathrm{agg}}
  = \mathbb M(\underline n):\boldsymbol\Sigma .
```

The in-plane part of the ITZ stress comes from the in-plane strain of the
aggregate, through ``\mathbb Q``; the out-of-plane part comes from the traction,
through ``\mathbb C_{\mathrm{ITZ}}:\mathbb P``. `TensND` writes the operator as it reads:

```@example itzlim
function itz_stress_operator(n)
    n = Tens(n)
    K = n ⋅ C_itz ⋅ n                     # acoustic tensor of the ITZ
    P = n ⊗ˢ inv(K) ⊗ˢ n                  # flat Hill tensor
    Q = C_itz - C_itz ⊡ P ⊡ C_itz         # second flat Hill tensor
    return Q ⊡ A_ε + C_itz ⊡ P ⊡ B_σ      # σ_ITZ = 𝕄(n) : Σ
end
nothing # hide
```

The same stress is the limit of a coated sphere whose ITZ shell gets thin. With
a shell of relative thickness ``10^{-6}``, read on the outer side of the
aggregate:

```@example itzlim
coated = LayeredSphere((1.0, 1.0 + 1.0e-6), (C_agg, C_itz))
B_coated = stress_strain_loc(coated, C_cp, C_cp)
ε∞ = inv((1 - f_agg) * C_cp + f_agg * B_coated) ⊡ uz   # Mori–Tanaka remote strain
fields = LayeredSphereFields(coated, C_cp)
θ, φ = 0.3π, 0.2π
σ_sphere = local_stress(fields, 1.0, θ, φ, ε∞; side = :outer)
σ_jump = itz_stress_operator([sin(θ) * cos(φ), sin(θ) * sin(φ), cos(θ)]) ⊡ uz
maximum(abs, components_canon(σ_sphere) - components_canon(σ_jump))
```

The difference is of the order of the shell thickness, as it should be: the
jump conditions are the thin-shell limit of the layered sphere.

## 5. The published values

Under a uniaxial or an equibiaxial load the problem is symmetric about the
loading axis, and the largest principal ITZ stress is found by sweeping the
zenith angle ``\theta`` alone.

```@example itzlim
θfine = range(0, π / 2; length = 4001)
function itz_peak_axisymmetric(Σ)
    v = [σI(itz_stress_operator([sin(θ), 0.0, cos(θ)]) ⊡ Σ) for θ in θfine]
    i = argmax(v)
    return v[i], θfine[i]
end
pt, θt = itz_peak_axisymmetric(uz)
pc, θc = itz_peak_axisymmetric(-uz)
pb, θb = itz_peak_axisymmetric(-bxy)

rows = [
    ("B_vol − B_dev", B_vol - B_dev, "Bvol_minus_Bdev"),
    ("max σ_I(ITZ)/Σ, uniaxial tension", pt, "itz_peak_tension"),
    ("θ/π, uniaxial tension", θt / π, "theta_tension"),
    ("max σ_I(ITZ)/Σ, uniaxial compression", -pc, "itz_peak_compression"),
    ("θ/π, uniaxial compression", θc / π, "theta_compression"),
    ("ITZ limit, uniaxial tension", 1 / pt, "itz_uniaxial_tension"),
    ("ITZ limit, uniaxial compression", -1 / pc, "itz_uniaxial_compression"),
    ("ITZ limit, equibiaxial compression", -1 / pb, "itz_biaxial_compression"),
    ("separation limit, uniaxial tension", separation_limit(uz), "sep_uniaxial_tension"),
    ("separation limit, uniaxial compression", -separation_limit(-uz), "sep_uniaxial_compression"),
    ("separation limit, equibiaxial compression", -separation_limit(-bxy), "sep_biaxial_compression"),
]
for (label, x, key) in rows
    @printf("%-44s %9.4f   published %9.4f\n", label, x, val(part2, key))
end
```

Every value of the article is recovered. The ITZ limits are in units of
``\sigma^{\mathrm{ult}}_{\mathrm{ITZ}}``, the separation limits in units of ``T^{\mathrm{ult}}``; the
"compression" peaks are signed as in the article, a positive principal stress
per unit negative load. Uniaxial compression thus puts the ITZ in tension, at
about a third of the way from the poles, and its limit ratio
``\Sigma_{\mathrm{ut}}/|\Sigma_{\mathrm{uc}}| = 0.225`` is the one the article finds within the
experimental range.

## 6. Elastic limit envelopes

For a general load, the largest principal ITZ stress is sought over the whole
aggregate surface. The operator ``\mathbb M(\underline n)`` does not depend on the load,
so it is computed once on a grid of normals and reused for every
``\boldsymbol\Sigma``. A diagonal ``\boldsymbol\Sigma`` is invariant under the reflections of the
three principal planes, so one octant of normals suffices. The operators are
stacked into one matrix, so that the ITZ stresses at all the normals come out
of a single product, and the largest eigenvalue of each symmetric 3×3 stress is
taken in closed form; the sweeps below call this function tens of thousands of
times.

```@example itzlim
θs = range(0, π / 2; length = 46)
φs = range(0, π / 2; length = 46)
stacked = reduce(vcat, [
    reshape(components_canon(itz_stress_operator([sin(θ) * cos(φ), sin(θ) * sin(φ), cos(θ)])), 9, 9)
    for θ in θs for φ in φs
])

# Largest eigenvalue of the symmetric matrix [a d e; d b f; e f c].
function eigmax3(a, b, c, d, e, f)
    q = (a + b + c) / 3
    p = sqrt(((a - q)^2 + (b - q)^2 + (c - q)^2 + 2 * (d^2 + e^2 + f^2)) / 6)
    p == 0 && return q
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
itz_limit(S) = (p = itz_peak(S); p > 0 ? 1 / p : Inf)    # units of σ_ult
nothing # hide
```

Under biaxial loading ``\boldsymbol\Sigma = \Sigma_{xx}\,\underline e_x\otimes\underline e_x + \Sigma_{yy}\,\underline e_y\otimes\underline e_y``,
the two envelopes are those of Figure 7 of the article:

```@example itzlim
ψs = range(0, 2π; length = 241)
radial(limit) = [limit(diagm([cos(ψ), sin(ψ), 0.0])) for ψ in ψs]
r_itz = radial(itz_limit)
r_sep = radial(N -> separation_limit(Tens(N)))
p1 = plot(r_itz .* cos.(ψs), r_itz .* sin.(ψs); lw = 2, aspect_ratio = :equal, legend = false,
    xlabel = "Σxx / σ_ITZ^ult", ylabel = "Σyy / σ_ITZ^ult", title = "ITZ failure")
p2 = plot(r_sep .* cos.(ψs), r_sep .* sin.(ψs); lw = 2, aspect_ratio = :equal, legend = false,
    xlabel = "Σxx / T^ult", ylabel = "Σyy / T^ult", title = "ITZ–aggregate separation")
for p in (p1, p2)
    hline!(p, [0]; color = :black, lw = 0.5)
    vline!(p, [0]; color = :black, lw = 0.5)
end
plot!(p1, [-4, 1], [-4, 1]; color = :gray, ls = :dash, lw = 1)   # Σxx = Σyy
plot(p1, p2; layout = (1, 2), size = (900, 420))
```

The ITZ envelope is symmetric about the diagonal ``\Sigma_{xx} = \Sigma_{yy}`` (dashed), as it
must be for an isotropic material. Its edges are slightly curved: each corresponds
to a family of points of the aggregate surface where the maximum moves as the
load turns.

Sections of the ITZ envelope by deviatoric planes, at fixed values of the
coordinate ``\xi = \sqrt 3\,\Sigma_m`` along the hydrostatic axis, show the shape of the
full surface in principal stress space. Each point is found by bisection on the
deviatoric radius:

```@example itzlim
N3 = [1, 1, 1] / sqrt(3)       # hydrostatic axis
N1 = [-1, 1, 0] / sqrt(2)      # two deviatoric directions
N2 = [-1, -1, 2] / sqrt(6)
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
p = plot(; aspect_ratio = :equal, xlabel = "along N₁", ylabel = "along N₂",
    legend = :outerright, title = "Deviatoric sections of the ITZ envelope")
for ξ in -2.0:0.5:1.0
    r = [section_radius(ξ, ψ) for ψ in ψd]
    plot!(p, r .* cos.(ψd), r .* sin.(ψd); lw = 2, label = "ξ = $ξ")
end
p
```

The sections shrink to a point as the hydrostatic tension grows, and widen
under hydrostatic compression: a Rankine criterion in the ITZ makes concrete
sensitive to the mean stress, with no friction law assumed anywhere. The
surface they describe, in interactive form:

```@example itzlim
include(joinpath(pkgdir(MeanFieldHomogenization), "scripts", "common", "docviz.jl"))
us = range(0, π; length = 31)              # angle from the hydrostatic axis
vs = range(0, 2π; length = 61)
direction(u, v) = cos(u) * N3 + sin(u) * (cos(v) * N1 + sin(v) * N2)
R = [itz_limit(diagm(direction(u, v))) for u in us, v in vs]
R[R .> 6] .= NaN                           # open towards hydrostatic compression
X = [R[i, j] * direction(us[i], vs[j])[1] for i in eachindex(us), j in eachindex(vs)]
Y = [R[i, j] * direction(us[i], vs[j])[2] for i in eachindex(us), j in eachindex(vs)]
Z = [R[i, j] * direction(us[i], vs[j])[3] for i in eachindex(us), j in eachindex(vs)]
surface = replace(surface_trace(X, Y, Z; color = "#4a90d9", opacity = 0.85), "NaN" => "null")
axis = line_trace([[-3.5, -3.5, -3.5], [1.0, 1.0, 1.0]]; color = "#555555", width = 3)
plotly_scene([surface, axis]; uid = "itz-envelope-3d", height = 520,
    xlabel = "Σ₁ / σ_ITZ^ult", ylabel = "Σ₂ / σ_ITZ^ult", zlabel = "Σ₃ / σ_ITZ^ult",
    title = "Elastic limit of concrete by ITZ failure, principal stress space")
```

The six curved faces of the article's Figure 4(b) appear around the hydrostatic
axis (gray line); the surface is cut where the elastic limit exceeds six times
the ITZ strength, since it does not close under compression.

## Where to go next

The combined criterion of the article takes the lower of the two limits; which
mode governs then depends only on the ratio ``T^{\mathrm{ult}}/\sigma^{\mathrm{ult}}_{\mathrm{ITZ}}``,
and the two functions of this page give it for any load. The jump conditions
used here are those of [Periodic multilayer — the laminate cell](@ref th-laminate),
applied to a single interface, and the pointwise fields of the coated sphere
that converge to them are described in
[Layered sphere — pointwise fields](@ref th-layered-sphere-pointwise). The
next application, [A lamellar porous material: swelling clays and C-S-H](@ref app-lamellar),
moves to the geomaterials, whose grains interact through interfaces that slide.
