# # [Cracks from their intensity factors](@id tut-rice-cracks)
#
# !!! info "Before this page"
#     The tutorial [Cracks and crack density](@ref tut-cracks), and the theory page
#     [Contributions from intensity factors](@ref th-rice-contributions), whose
#     formulas (1) to (4) are used below.
#
# Three cracks whose contribution is obtained by integrating their intensity
# factors along their growth: the annular crack (a penny bonded on a central
# disc), the circular arc crack and the Saturn ring (a pore with an annular
# crack). Then a crack defined by the user from its own intensity factors.

import Pkg                                                          #jl
Pkg.activate(joinpath(@__DIR__, "..", "docs"); io = devnull)                 #jl

using MeanFieldHomogenization
using TensND
using LinearAlgebra
using Printf
using Plots
gr()

## Interactive 3-D helpers; `pkgdir` rather than `@__DIR__` so the include also
## resolves inside a Documenter `@example` block.
include(joinpath(pkgdir(MeanFieldHomogenization), "scripts", "common", "docviz.jl"))

# The matrix is isotropic, of Young modulus ``E``, Poisson ratio ``\nu`` and
# conductivity ``k_0``; ``E^{*} = E/(1-\nu^2)``.

const E, ν, k₀ = 1.0, 0.25, 1.0
const C₀ = iso_stiffness_E_nu(E, ν)
const K₀ = TensISO{3}(k₀)
const Es = E / (1 - ν^2)
nothing #hide

# ## The annular crack
#
# A penny of outer radius ``a`` bonded on the disc of radius ``c``: the crack
# occupies ``c < r < a``.

plotly_scene(
    shape_traces(AnnularCrack(1.0, 0.4)); uid = "rc-annulus", height = 420,
    title = "Annular crack, c/a = 0.4 (bonded disc in gray)"
)

# ### Intensity factors
#
# Per unit traction, scaled by those of a straight crack of the same width
# ``2h = a - c``, ``\sqrt{\pi h}``: all six tend to 1 for a narrow ring. The
# dashed line is the fit of the inner opening factor read from the tables of
# Smetanin by Kachanov and Sevostianov,
# ``\sqrt{1-\lambda}\,K_I/\sqrt{\pi h} = 1 + (4\sqrt2/\pi^2 - 1)\lambda``, with
# ``\lambda = (a-c)/a``.

ratios = range(0.05, 0.95; length = 19)
sifs = map(ratios) do r
    crack = AnnularCrack(1.0, r)
    Σn = TensND.Tens([0 0 0; 0 0 0; 0 0 1.0], crack_basis(crack))     # normal traction
    Σt = TensND.Tens([0 0 1.0; 0 0 0; 1.0 0 0], crack_basis(crack))   # tangential traction along ℓ
    sq = sqrt(π * (1 - r) / 2)
    out = (
        sif(crack, C₀, Σn; front = :outer)[2][1], sif(crack, C₀, Σt; front = :outer)[2][2],
        sif(crack, C₀, TensND.Tens([0 0 0; 0 0 1.0; 0 1.0 0], crack_basis(crack)); front = :outer)[2][3],
    )
    inn = (
        sif(crack, C₀, Σn; front = :inner)[2][1], sif(crack, C₀, Σt; front = :inner)[2][2],
        sif(crack, C₀, TensND.Tens([0 0 0; 0 0 1.0; 0 1.0 0], crack_basis(crack)); front = :inner)[2][3],
    )
    (abs.(out) ./ sq, abs.(inn) ./ sq)
end
A = 4sqrt(2) / π^2 - 1
fig = plot(; xlabel = "c/a", ylabel = "K / sqrt(π h)", legend = :topright, size = (720, 420))
for (k, name) in enumerate(("I", "II", "III"))
    plot!(fig, ratios, [s[2][k] for s in sifs]; label = "K_$name inner", lw = 2)
    plot!(fig, ratios, [s[1][k] for s in sifs]; label = "K_$name outer", lw = 2, ls = :dot)
end
plot!(fig, ratios, (1 .+ A .* (1 .- ratios)) ./ sqrt.(ratios); label = "fit, inner K_I", ls = :dash, color = :black)
fig

# ### Compliance and resistivity contributions
#
# ``\boldsymbol{B}`` is normalized by the outer radius, as for the penny of
# radius ``a``, so its ratio to the penny's says how much the contact stiffens
# the crack. Each point of the curves solves the final crack once
# (`method = :opening`); the markers integrate the intensity factors of the
# inner front along its growth from ``r = a`` (Rice's formula, the default), and
# fall on the curves.

Bp = TensND.canonical_components(cod_tensor(PennyCrack(1.0), C₀))
bp = cod_tensor(PennyCrack(1.0), K₀)
Bopen = [TensND.canonical_components(cod_tensor(AnnularCrack(1.0, r), C₀; method = :opening)) for r in ratios]
rice_pts = (0.2, 0.5, 0.8)
Brice = [TensND.canonical_components(cod_tensor(AnnularCrack(1.0, r), C₀)) for r in rice_pts]
brice = [cod_tensor(AnnularCrack(1.0, r), K₀) for r in rice_pts]
fig = plot(; xlabel = "c/a", ylabel = "contribution / penny", legend = :topright, size = (720, 420), ylims = (0, 1.02))
plot!(fig, ratios, [B[3, 3] / Bp[3, 3] for B in Bopen]; label = "B_nn (opening)", lw = 2)
plot!(fig, ratios, [B[1, 1] / Bp[1, 1] for B in Bopen]; label = "B_tt (sliding)", lw = 2)
scatter!(fig, collect(rice_pts), [B[3, 3] / Bp[3, 3] for B in Brice]; label = "Rice, inner front", color = :black)
scatter!(fig, collect(rice_pts), [B[1, 1] / Bp[1, 1] for B in Brice]; label = "", color = :black)
scatter!(fig, collect(rice_pts), brice ./ bp; label = "b (conduction), Rice", marker = :diamond, color = :gray)
fig

# The conduction scalar is the opening one, ``b = E^{*}B_{nn}/(2k_0)``:

for (r, B, b) in zip(rice_pts, Brice, brice)
    @printf "c/a = %.1f   b = %.10f   E* B_nn/(2k₀) = %.10f\n" r b Es * B[3, 3] / (2k₀)
end

# A crack of radius ``R_{\mathrm{eq}}`` without contact has the same normal
# compliance when ``R_{\mathrm{eq}}^3 B^{\mathrm{penny}}_{nn} = a^3 B_{nn}``:

plot(
    ratios .^ 2, [cbrt(B[3, 3] / Bp[3, 3]) for B in Bopen];
    xlabel = "area of contact / area of the penny, (c/a)²", ylabel = "R_eq / a",
    legend = false, lw = 2, size = (620, 380)
)

# ### Effective properties
#
# Annular cracks of random orientation, density ``\varepsilon = N a^3``, in the
# non-interaction approximation ``\mathbb{S} = \mathbb{S}_0 + \tfrac{4\pi}{3}
# \varepsilon\,\langle\mathbb{H}\rangle``: the isotropic part of
# ``\mathbb{H}`` is computed once per geometry.

densities = range(0, 0.3; length = 7)
fig = plot(; xlabel = "crack density ε = N a³", ylabel = "E/E₀ (solid), k/k₀ (dashed)", size = (720, 420))
for (j, r) in enumerate((0.0, 0.4, 0.7))
    crack = r == 0 ? PennyCrack(1.0) : AnnularCrack(1.0, r)
    H = MeanFieldHomogenization.Core.isotropify(inverse_property_contribution(crack, C₀))
    R = MeanFieldHomogenization.Core.isotropify(inverse_property_contribution(crack, K₀))
    Eh = [1 / TensND.canonical_components(inv(C₀) + 4π / 3 * ε * H)[3, 3, 3, 3] for ε in densities]
    kh = [1 / TensND.canonical_components(inv(K₀) + 4π / 3 * ε * R)[3, 3] for ε in densities]
    plot!(fig, densities, Eh ./ E; label = "c/a = $r", lw = 2, color = j)
    plot!(fig, densities, kh ./ k₀; label = "", lw = 2, ls = :dash, color = j)
end
fig

# The crack enters every scheme like any other crack, by its density; a
# self-consistent estimate needs an isotropic reference, hence the isotropic
# distribution:

rve = RVE()
add_phase!(rve, :M, Spheroid(1.0), Dict(:C => C₀); fraction = :rest)
add_phase!(rve, :F, AnnularCrack(1.0, 0.4), Dict(:C => C₀); density = 0.2, symmetrize = IsoSymmetrize())
k_sc, μ_sc = k_mu(homogenize(rve, SelfConsistent(), :C))
@printf "self-consistent, ε = 0.2, c/a = 0.4:  k = %.5f  μ = %.5f  (intact: %.5f, %.5f)\n" k_sc μ_sc k_mu(C₀)...

# ## The circular arc crack
#
# A crack along an arc of circle, unbounded along the axis of the circle. It is
# normalized as a ribbon of half-width ``R\alpha`` (density
# ``\varepsilon = N (R\alpha)^2``).

plotly_scene(
    shape_traces(CircularArcCrack(1.0, 2π / 3)); uid = "rc-arc", height = 420,
    title = "Circular arc crack, α = 2π/3"
)

# The components of ``V\Delta\mathbb{S}`` per unit length, in units of
# ``\pi R^2/E^{*}``: the closed forms (lines) and Rice's formula integrated
# numerically from the intensity factors (markers). At ``\alpha = \pi`` they are
# the plane-strain hole, ``(3, 3, -1, 2)``.

αs = range(0.02, π; length = 60)
comp(α) = π * (α)^2 .* TensND.canonical_components(inverse_property_contribution(CircularArcCrack(1.0, α), C₀)) ./ (π / Es)
idx = ((2, 2, 2, 2), (3, 3, 3, 3), (2, 2, 3, 3), (2, 3, 2, 3), (1, 3, 1, 3))
names = ("mmmm", "nnnn", "mmnn", "mnmn", "ℓnℓn")
vals = [comp(α) for α in αs]
fig = plot(;
    xlabel = "half-angle α", ylabel = "V ΔS / (π R²/E*)", legend = :topleft, size = (720, 420),
    xticks = ([0, π / 4, π / 2, 3π / 4, π], ["0", "π/4", "π/2", "3π/4", "π"])
)
for (i, nm) in zip(idx, names)
    plot!(fig, αs, [v[i...] for v in vals]; label = nm, lw = 2)
end
for α in (π / 4, π / 2, 3π / 4)
    Hr = π * α^2 .* TensND.canonical_components(inverse_property_contribution(CircularArcCrack(1.0, α), C₀; method = :rice)) ./ (π / Es)
    scatter!(fig, fill(α, length(idx)), [Hr[i...] for i in idx]; label = "", color = :black, ms = 3)
end
fig

# ## The Saturn ring
#
# A spherical pore of radius ``a`` with an annular crack of length ``c`` in its
# equatorial plane. Only the opening mode of the ring is published, hence only
# ``H_{nnnn}``, compared here with the penny of radius ``a + c``.

plotly_scene(saturn_traces(1.0, 0.8); uid = "rc-saturn", height = 420, title = "Saturn ring, c/a = 0.8")

#-

λs = range(0.05, 4; length = 30)
S = [saturn_ring_compliance(1.0, λ, C₀) for λ in λs]
penny(λ) = 16 * (1 + λ)^3 / (3Es)
plot(
    λs, [s.total / penny(λ) for (s, λ) in zip(S, λs)];
    label = "pore + ring", lw = 2, xlabel = "c/a", ylabel = "V H_nnnn / penny of radius a + c",
    size = (720, 400)
)
plot!(λs, [s.ring / penny(λ) for (s, λ) in zip(S, λs)]; label = "ring alone", lw = 2, ls = :dash)
hline!([1]; color = :gray, label = "")

# ## A crack defined by its intensity factors
#
# [`RiceCrack`](@ref) takes the intensity factors per unit traction at a
# circular front of radius ``\rho``, as functions of ``(\rho, \nu)``, and the
# radii between which the front sweeps. Here a penny grown from its center;
# its tensor is that of `PennyCrack`.

grown = RiceCrack(
    1.0;
    K_I = (ρ, ν) -> 2sqrt(ρ / π),
    K_II = (ρ, ν) -> 4sqrt(ρ / π) / (2 - ν),
    K_III = (ρ, ν) -> 4(1 - ν) * sqrt(ρ / π) / (2 - ν)
)
display(TensND.canonical_components(cod_tensor(grown, C₀)) .- TensND.canonical_components(cod_tensor(PennyCrack(1.0), C₀)))

# For any flat front, [`rice_cod_tensor`](@ref) takes the front point by point:
# its weight, its outer normal and the matrix ``\boldsymbol{M}`` with
# ``\underline{K} = \boldsymbol{M}\cdot(\boldsymbol{\Sigma}\cdot\underline{n})``.
