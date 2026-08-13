# [The differential scheme and path dependence](@id tut-differential-paths)

The differential (or incremental) scheme builds a composite the way
some real materials are actually made: by adding inclusions a little
at a time, re-homogenizing after every increment. That construction
history turns out to matter as soon as there is more than one
inclusion phase.

## The differential scheme

[`DifferentialScheme`](@ref) integrates the **Norris differential
equation** [norris1985](@cite) over a fictitious incorporation time
``\tau \in [0, 1]``:

```math
\frac{d\mathbb{C}^{hom}}{d\tau} = \sum_i \dot\varphi_i \, \mathbb{N}_i\big(\mathbb{C}^{hom}\big),
\qquad
\dot\varphi_i = \dot f_i + \frac{f_i}{f_0}\sum_j \dot f_j,
```

starting from ``\mathbb{C}^{hom}(\tau=0) = \mathbb{C}_0`` (the matrix) and
adding inclusions at each step *into the current effective medium*
rather than into the original matrix. Each increment is itself a dilute
estimate — so the scheme is, in effect, an infinite sequence of
infinitesimal dilute corrections. The increments ``\dot\varphi_i`` differ
from the prescribed ``\dot f_i`` because replacing part of the current
medium also removes the inclusions it already contained; the correction
is the Sherman-Morrison factor above, derived in
[The differential scheme](../theory/differential_scheme.md).

## Trajectories for multi-phase RVEs

With a single inclusion phase, the trajectory is unambiguous: the
fraction grows monotonically from 0 to its target value. With **two or
more** inclusion phases, the *order* in which they are added along the
way becomes a modeling choice, exposed through the `trajectory` keyword:

- `Proportional()` (default) — every phase grows in lock-step,
  proportionally to its target fraction.
- `Sequential(:A, :B)` — phase `:A` is added to completion first, then
  `:B` is added into the resulting effective medium.
- `Path(:A => τ -> ..., ...)` — an arbitrary fraction-of-`τ` schedule per
  phase, given as callables (`CustomPath` takes tabulated values
  instead).

## Seeing the path dependence

```@example tutdiff
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using Plots
gr()  # headless backend; GKSwstype is set to "100" in make.jl

C_m = iso_stiffness(30.0, 10.0)
C_stiff = iso_stiffness(60.0, 20.0)
C_soft = iso_stiffness(5.0, 2.0)

function build(f_total)
    r = RVE(:M)
    add_matrix!(r, Ellipsoid(1.0), Dict(:C => C_m))
    add_phase!(r, :STIFF, Ellipsoid(1.0), Dict(:C => C_stiff); fraction = f_total / 2)
    add_phase!(r, :SOFT, Ellipsoid(1.0), Dict(:C => C_soft); fraction = f_total / 2)
    return r
end

trajectories = [
    (Proportional(), "Proportional", :black),
    (Sequential(:STIFF, :SOFT), "Sequential(STIFF, SOFT)", :blue),
    (Sequential(:SOFT, :STIFF), "Sequential(SOFT, STIFF)", :red),
]

fs = collect(range(0.005, 0.3; length = 25))
plt = plot(;
    xlabel = "total inclusion fraction f₁+f₂", ylabel = "k_eff",
    legend = :topleft, framestyle = :box, size = (760, 480),
)
for (traj, label, color) in trajectories
    ks = [k_mu(homogenize(build(f), DifferentialScheme(; trajectory = traj, nsteps = 100), :C))[1] for f in fs]
    plot!(plt, fs, ks; label = label, color = color, lw = 2)
end
plt
```

The three curves coincide in the dilute limit (small total fraction,
where increments barely interact regardless of order) and fan out as
the total fraction grows. Adding the stiff phase first leaves it
embedded in a matrix that later softens further when the soft phase is
added; adding the soft phase first has the opposite effect. Neither
order is "more correct" in the abstract — the differential scheme
encodes a **construction history**, and for a real composite the
mixing or loading order is itself a physical modeling choice, not a
numerical one.

[Comparing loading-path trajectories](differential_loading_paths.md)
takes this further: instead of reading off only the endpoint `k_eff` of
a fraction sweep, it plots ``\mathbb C^{\mathrm{hom}}(\tau)`` for several trajectories —
including a `Path` schedule — over the whole incorporation history.

## Following the construction history

The scheme knows the effective property at every `τ`, not just at the
end. [`differential_path`](@ref) returns the saved states, which is both
cheaper and more informative than re-running `homogenize` for a series
of target fractions:

```@example tutdiff
rve = build(0.3)
τ, Cs = differential_path(rve, DifferentialScheme(; nsteps = 200), :C)

plt2 = plot(;
    xlabel = "incorporation time τ", ylabel = "modulus",
    legend = :topright, framestyle = :box, size = (760, 480),
)
plot!(plt2, τ, [k_mu(C)[1] for C in Cs]; label = "k_eff", lw = 2)
plot!(plt2, τ, [k_mu(C)[2] for C in Cs]; label = "μ_eff", lw = 2, color = :red)
plt2
```

The curve is the actual construction history: reading it at `τ = 0.5`
gives the composite that has reached half of each target fraction, which
is *not* the same as the composite built directly at those fractions
unless the trajectory is proportional.

## Stiffness or compliance

The same ODE can be integrated on the compliance
``\mathbb{S} = \mathbb{C}^{-1}``, using the compliance contribution
``\mathbb{H}_i = -\mathbb{S}:\mathbb{N}_i:\mathbb{S}`` of each phase. The
two forms are analytically equivalent, so any difference is pure
numerics — which makes the comparison a free accuracy check:

```@example tutdiff
rve = build(0.3)
tol = (abstol = 1e-12, reltol = 1e-10)
C_stiff = homogenize(rve, DifferentialScheme(; tol...), :C)
C_compl = homogenize(rve, DifferentialScheme(; formulation = :compliance, tol...), :C)
(k_mu(C_stiff), k_mu(C_compl))
```

The choice matters when the effective medium approaches a degenerate
state: integrating `C → 0` (porous or cracked media near percolation) is
stiff in the stiffness variable and gentle in the compliance one.

## Cracks

Cracks are the one case where the volume-replacement picture breaks
down: their volume fraction vanishes (``f_c \sim \varepsilon_c X \to 0``
as the aspect ratio ``X \to 0``) while their density stays finite. They
therefore contribute through ``\varepsilon_c``, are absent from the
volume balance, and are merely *diluted* by the solid increments — see
the [theory page](../theory/differential_scheme.md). In practice they
are declared with `density` and follow the trajectory like any other
phase:

```@example tutdiff
function cracked(ε)
    r = RVE(:M)
    add_matrix!(r, Ellipsoid(1.0), Dict(:C => C_m))
    add_phase!(r, :CR, PennyCrack(1.0), Dict(:C => C_m); density = ε, symmetrize = :iso)
    return r
end

εs = collect(range(0.0, 0.6; length = 25))
ks = [k_mu(homogenize(cracked(ε), DifferentialScheme(), :C))[1] for ε in εs]

plot(εs, ks ./ k_mu(C_m)[1];
    xlabel = "crack density ε", ylabel = "k_eff / k₀",
    label = "differential", lw = 2, legend = :topright,
    framestyle = :box, size = (760, 480),
)
```

The differential scheme drives the stiffness towards zero without ever
reaching it at finite density — no percolation threshold, unlike the
self-consistent scheme, because each infinitesimal crack increment is
introduced into an already-degraded medium.
