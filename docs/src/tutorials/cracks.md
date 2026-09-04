# [Cracks and crack density](@id tut-cracks)

Cracks are inclusions with **zero volume** — a pore flattened to a
surface. `MeanFieldHomogenization` handles them with the same [`RVE`](@ref)/
[`homogenize`](@ref) machinery as ordinary inclusions, but parameterized
by **crack density** instead of volume fraction, and characterized by a
**crack-opening-displacement (COD) tensor** rather than an Eshelby
tensor.

## The crack-opening-displacement tensor

For a flat crack in an infinite matrix ``\mathbb{C}_0``, the
displacement jump across the crack faces under remote stress is linear
in that stress through the COD tensor **B**:

```@example tutcracks
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using Plots
gr()  # headless backend; GKSwstype is set to "100" in make.jl

E, ν = 30.0, 0.2
k_m, μ_m = E / (3 * (1 - 2ν)), E / (2 * (1 + ν))
C0 = iso_stiffness(k_m, μ_m)

pc = PennyCrack(1.0)
B = cod_tensor(pc, C0)
B[3, 3]   # normal-opening component, crack-local frame
```

`B` depends only on the matrix stiffness and the crack's *shape*
(circular penny, elliptic, or ribbon) — not on its size, which enters
separately through the **crack density**.

## From size to density: the compliance contribution

A single crack contributes an increment to the macroscopic compliance
proportional to its size. For a population of ``N`` non-interacting
penny cracks per unit volume, each of radius ``a``, the (dimensionless)
**crack density** is ``\varepsilon = N a^3`` (Budiansky–O'Connell), and
the dilute compliance correction is

```math
\Delta\mathbb{S} = \frac{4\pi}{3}\,\varepsilon\,\mathbb{H}, \qquad
\mathbb{H} = \tfrac{3}{4}\,\underline{n}\otimes^{\!s}\boldsymbol{B}\otimes^{\!s}\underline{n},
```

[kachanov1992](@cite), [kachanov1993](@cite) — ``\mathbb{H}`` is the
**size-independent compliance contribution tensor**
([`compliance_contribution`](@ref)), computed once from `B` and the
crack normal ``\underline{n}``, and reused unchanged across an entire
crack population that shares the same orientation and shape.

```@example tutcracks
H = compliance_contribution(pc, C0)
ΔS = delta_compliance(pc, H, 0.05)   # ε = 0.05
nothing # hide
```

## Cracks in an RVE — the `density` keyword

Rather than assembling `ΔS` by hand, [`add_phase!`](@ref) accepts a
crack geometry with a `density` keyword in place of `fraction`; the
scheme machinery then computes the appropriate contribution internally
for whichever scheme is requested:

```@example tutcracks
function build(ρ)
    r = RVE()
    add_phase!(r, :M, Ellipsoid(1.0), Dict(:C => C0); fraction = :rest)
    add_phase!(r, :CRACK, PennyCrack(1.0), Dict(:C => C0); density = ρ, symmetrize = IsoSymmetrize())
    return r
end
crack_density(build(0.05), :CRACK)
```

`symmetrize = IsoSymmetrize()` declares a **uniform spatial
distribution of crack orientations** (see the
[porous benchmark tutorial](porous_benchmark.md) for the same
keyword on ordinary inclusions), so the macroscopic effect of an
isotropically-oriented crack population is itself isotropic.

Note: [`SelfConsistent`](@ref) does not support cracks (its
strain-localization tensor becomes singular for zero-volume
inclusions); use [`AsymmetricSelfConsistent`](@ref) instead. See the
[cracks manual page](../manual/cracks.md) for the full API, including
finite interface stiffness (Sevostianov springs).

## Effective modulus vs. crack density

```@example tutcracks
ρs = collect(range(0.0, 0.3; length = 20))
schemes = [
    (MoriTanaka(), "MoriTanaka", :black),
    (AsymmetricSelfConsistent(; select_best = true), "AsymmetricSelfConsistent", :purple),
    (DifferentialScheme(; nsteps = 100), "Differential", :orange),
    (PonteCastanedaWillis(), "PCW", :green),
]

plt = plot(;
    xlabel = "crack density ρ", ylabel = "k_eff",
    legend = :bottomleft, framestyle = :box, size = (760, 480),
)
for (sch, label, color) in schemes
    ks = [k_mu(best_fit_iso(homogenize(build(ρ), sch, :C)))[1] for ρ in ρs]
    plot!(plt, ρs, ks; label = label, color = color, lw = 2)
end
plt
```

Stiffness decreases monotonically with crack density for every scheme,
as expected — but at different rates: `AsymmetricSelfConsistent`
degrades fastest because it lets cracks interact through a softening
effective medium, exactly as the self-consistent porous estimate did
in [an earlier tutorial](porous_materials.md). Crack density plays
the same qualitative role here that porosity played there; what differs
is the *shape* (a surface rather than a volume) entering the
localization tensor.
