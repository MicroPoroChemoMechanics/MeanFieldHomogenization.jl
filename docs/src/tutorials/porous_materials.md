# [Porous materials and the self-consistent trap](@id tut-porous-materials)

Porosity is the simplest microstructure that breaks naive intuition: a
pore is not "just a very soft inclusion" — it is an inclusion whose
stiffness may vanish, and several schemes are not built to handle that
limit gracefully. This page shows why, and introduces the scheme that
was designed to.

## Modeling a pore

A pore is modeled as an ordinary phase carrying a **near-zero
stiffness**. Strictly zero stiffness makes the Reuss bound singular
(it inverts ``\mathbb{C}_i``), so a small positive regularization is
used instead:

```@example tutporous
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using Plots
gr()  # headless backend; GKSwstype is set to "100" in make.jl

C_solid = iso_stiffness(90.0, 30.0)
C_void = iso_stiffness(0.01, 0.005)   # soft, not exactly zero
nothing # hide
```

## Why the standard self-consistent scheme fails

[`SelfConsistent`](@ref) embeds every phase directly in the *effective*
medium — including the pores. As porosity grows, the Picard iteration
that solves the implicit self-consistent condition (see the
[previous tutorial](bounds_and_schemes.md)) has to locate an
effective stiffness soft enough to be consistent with soft, connected
voids; near the percolation threshold the iteration becomes numerically
unstable and can converge to an unphysical branch.

[`AsymmetricSelfConsistent`](@ref) fixes this by switching to the
**compliance-form** iteration when the contrast calls for it — solving
the dual condition on ``\mathbb{S}_{\text{eff}} = \mathbb{C}_{\text{eff}}^{-1}``
instead — which remains well-posed for soft inclusions. The
`select_best = true` keyword additionally keeps the best iterate seen
during the loop, guarding against the Picard noise that can otherwise
cross to the wrong branch near percolation. [`DiluteDual`](@ref) is the
analogous *dilute* estimate for a soft phase: the natural dual
counterpart of [`Dilute`](@ref) when it is the compliance, not the
stiffness, that varies smoothly.

## Schemes against the bounds

```@example tutporous
function bulk_at(f, scheme)
    r = RVE()
    add_phase!(r, :M, Ellipsoid(1.0), Dict(:C => C_solid); fraction = :rest)
    f > 0 && add_phase!(r, :V, Ellipsoid(1.0), Dict(:C => C_void); fraction = f)
    k, _ = k_mu(homogenize(r, scheme, :C))
    return k
end

fs = collect(range(0.0, 0.5; length = 61))
plt = plot(;
    xlabel = "porosity f", ylabel = "k_eff",
    legend = :topright, framestyle = :box, size = (760, 480),
)
plot!(plt, fs, [bulk_at(f, Voigt()) for f in fs]; label = "Voigt", color = :red, ls = :dash, lw = 2)
plot!(plt, fs, [bulk_at(f, Reuss()) for f in fs]; label = "Reuss", color = :gray, ls = :dash, lw = 2)
plot!(plt, fs, [bulk_at(f, MoriTanaka()) for f in fs]; label = "MoriTanaka", color = :green, lw = 2)
plot!(plt, fs, [bulk_at(f, DiluteDual()) for f in fs]; label = "DiluteDual", color = :blue, lw = 2)
plot!(
    plt, fs, [bulk_at(f, AsymmetricSelfConsistent(; abstol = 1.0e-10, maxiters = 200, select_best = true)) for f in fs];
    label = "AsymmetricSelfConsistent", color = :purple, lw = 2,
)
plot!(plt, fs, [bulk_at(f, DifferentialScheme(; nsteps = 100)) for f in fs]; label = "Differential", color = :orange, lw = 2)
plt
```

`AsymmetricSelfConsistent` and `Differential` sit comfortably between
the bounds across the whole porosity range, while a naive
`SelfConsistent` call on this same problem (try it — replace
`AsymmetricSelfConsistent` above) becomes unreliable as `f` grows. The
next tutorial pushes this benchmark to its canonical form — porosity up
to 1 — and runs *every* scheme side by side.
