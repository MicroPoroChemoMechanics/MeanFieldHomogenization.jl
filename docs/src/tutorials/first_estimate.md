# [A first homogenization](@id tut-first-estimate)

Every `MeanFieldHomogenization` computation starts from the same three ingredients:
a **representative volume element** (RVE) describing the phases, their
**geometry**, and a **scheme** that turns the RVE into a single
effective property. This page builds the simplest possible RVE — a
matrix with one spherical inclusion phase — and computes its effective
stiffness two different ways.

## Building an RVE

```@example tut1st
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using Plots
gr()  # headless backend; GKSwstype is set to "100" in make.jl

rve = RVE()
add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => iso_stiffness(30.0, 10.0)); fraction = :rest)
add_phase!(rve, :I, Ellipsoid(1.0), Dict(:C => iso_stiffness(60.0, 20.0)); fraction = 0.2)
rve
```

`RVE()` creates an empty container; [`add_phase!`](@ref) then attaches a
**geometry** (here `Ellipsoid(1.0)`, a unit sphere), a **property dictionary**
and an **amount** to each phase. `fraction = 0.2` means 20 % of the RVE by
volume; `fraction = :rest` means "whatever the others leave", here 80 %.

Note that nothing so far says which phase is a matrix — that is the scheme's
to decide, and `MoriTanaka()` below takes the `:rest` phase because it is the
only candidate. See [Who is the matrix?](@ref man-who-is-the-matrix).

### A storage convention worth knowing

`iso_stiffness(k, mu)` builds the isotropic stiffness tensor
from the physical bulk and shear moduli ``k`` and ``\mu``. Internally
`TensND` stores an isotropic 4th-order tensor as the *raw* pair
``(3k, 2\mu)`` — so `TensISO{3}(a, b)` interprets its two arguments as
that pair directly, **not** as ``(k, \mu)``:

```@example tut1st
C1 = iso_stiffness(30.0, 10.0)      # from physical (k, μ) = (30, 10)
C2 = TensISO{3}(3 * 30.0, 2 * 10.0)  # same tensor, built from the raw pair
C1 == C2
```

Recover the physical moduli from either with `k_mu`:

```@example tut1st
k_mu(C1)
```

To avoid this trap, every example in these tutorials builds isotropic
stiffnesses with `iso_stiffness(k, mu)` and reads results back with
`k_mu`.

## The dilute estimate

The simplest scheme, [`Dilute`](@ref), assumes each inclusion sits in
an *infinite* matrix, ignoring every other inclusion around it. The
effective stiffness is a sum of independent contributions:

```math
\mathbb{C}_{\text{eff}} = \mathbb{C}_0 + \sum_i f_i\,(\mathbb{C}_i-\mathbb{C}_0):\mathbb{A}_i^{\text{dil}},
\qquad
\mathbb{A}_i^{\text{dil}} = \big[\mathbb{I}+\mathbb{P}_i:(\mathbb{C}_i-\mathbb{C}_0)\big]^{-1},
```

where ``\mathbb{P}_i`` is the Hill polarization tensor of inclusion
``i`` in the matrix ``\mathbb{C}_0`` ([`hill_tensor`](@ref)) and
``\mathbb{A}_i^{\text{dil}}`` is its **dilute strain-localization
tensor**: the linear map from the macroscopic strain to the strain
inside inclusion ``i``. This is exact only in the dilute limit
``f_i \to 0`` — at finite volume fraction, inclusions interact and the
estimate drifts.

## Mori–Tanaka: accounting for interaction

[`MoriTanaka`](@ref) corrects for this by localizing not on the
macroscopic strain, but on the *average strain in the matrix*:

```math
\mathbb{A}_i^{\text{MT}} = \mathbb{A}_i^{\text{dil}}:\Big(\sum_j f_j\,\mathbb{A}_j^{\text{dil}}\Big)^{-1}.
```

Both schemes are called the same way — only the scheme argument to
[`homogenize`](@ref) changes:

```@example tut1st
k_dil, _ = k_mu(homogenize(rve, Dilute(), :C))
k_mt, _ = k_mu(homogenize(rve, MoriTanaka(), :C))
(k_dil, k_mt)
```

## Comparing the two across volume fraction

```@example tut1st
function build(f)
    r = RVE()
    add_phase!(r, :M, Ellipsoid(1.0), Dict(:C => iso_stiffness(30.0, 10.0)); fraction = :rest)
    add_phase!(r, :I, Ellipsoid(1.0), Dict(:C => iso_stiffness(60.0, 20.0)); fraction = f)
    return r
end

fs = exp10.(range(-4, log10(0.6); length = 30))
k_dil = [k_mu(homogenize(build(f), Dilute(), :C))[1] for f in fs]
k_mt = [k_mu(homogenize(build(f), MoriTanaka(), :C))[1] for f in fs]

plt = plot(;
    xlabel = "inclusion volume fraction f", ylabel = "k_eff",
    xscale = :log10, legend = :topleft, framestyle = :box, size = (760, 480),
)
plot!(plt, fs, k_dil; label = "Dilute", lw = 2)
plot!(plt, fs, k_mt; label = "MoriTanaka", lw = 2)
hline!(plt, [k_dil[1]]; label = "matrix", lw = 1, color = :gray, ls = :dash)
plt
```

The two curves coincide as ``f \to 0`` — both schemes agree in the
dilute limit, as they must — and diverge as ``f`` grows: at finite
volume fraction the inclusions interact, and Mori–Tanaka, which accounts
for that, departs from the naive dilute sum.
