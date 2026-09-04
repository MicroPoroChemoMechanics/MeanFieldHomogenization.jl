# [Comparing loading-path trajectories](@id tut-differential-loading-paths)

[The differential scheme and path dependence](differential_paths.md) shows
*that* the incorporation order matters, through the endpoint `k_eff` of a
fraction sweep. This page shows *how* it matters: several trajectories
racing to the same target fractions, watched the whole way through
``\tau \in [0, 1]`` rather than read off only at ``\tau = 1``.

## A two-phase RVE with a strong contrast

To make the divergence between trajectories visible at a glance, the
inclusions here are a genuinely stiff and a genuinely soft phase — a
5× contrast on each side of the matrix, rather than the milder pair used
in the previous tutorial:

```@example loadpath
using MeanFieldHomogenization
using TensND
using Plots
gr()  # headless backend; GKSwstype is set to "100" in make.jl

C_M = TensISO{3}(3 * 5.0, 2 * 2.0)          # matrix   : k = 5,  μ = 2
C_I1 = TensISO{3}(3 * 30.0, 2 * 12.0)       # phase I1 : stiff, 6× the matrix
C_I2 = TensISO{3}(3 * 0.5, 2 * 0.2)         # phase I2 : compliant, 0.1× the matrix
F1, F2 = 0.2, 0.2                            # target volume fractions

function build_rve()
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C_M); fraction = :rest)
    add_phase!(rve, :I1, Ellipsoid(1.0), Dict(:C => C_I1); fraction = F1)
    add_phase!(rve, :I2, Ellipsoid(1.0), Dict(:C => C_I2); fraction = F2)
    return rve
end
nothing # hide
```

## Four trajectories to the same target

Every trajectory below reaches `f₁ = f₂ = 0.2` at `τ = 1` — only the path
taken to get there differs:

- [`Proportional`](@ref) — both phases grow in lock-step.
- `Sequential(:I1, :I2)` — the stiff phase is incorporated to completion
  first, the soft phase second, into the resulting (already stiffened)
  medium.
- `Sequential(:I2, :I1)` — the reverse order.
- `Path(:I1 => τ -> τ^2, :I2 => τ -> 2τ - τ^2)` — a continuous schedule
  where `:I1` is *backloaded* (``\tau^2`` starts flat, catches up near
  ``\tau = 1``) and `:I2` is *frontloaded* (``2\tau - \tau^2 = 1-(1-\tau)^2``
  grows fast then flattens) — both curves are monotone on `[0, 1]` with
  the required endpoints, but neither is proportional.

[`differential_path`](@ref) returns the full trajectory ``\mathbb C^{\mathrm{hom}}(\tau)``
in one solve, rather than the single value `homogenize` returns at
`τ = 1`:

```@example loadpath
const NSTEPS = 200

function eval_path(traj)
    scheme = DifferentialScheme(; trajectory = traj, nsteps = NSTEPS, abstol = 1e-9, reltol = 1e-7)
    τ, Cs = differential_path(build_rve(), scheme, :C)
    kμ = k_mu.(Cs)
    return τ, first.(kμ), last.(kμ)
end

paths_to_run = (
    ("Proportional", Proportional(), :black),
    ("Sequential(I1 → I2)", Sequential(:I1, :I2), :blue),
    ("Sequential(I2 → I1)", Sequential(:I2, :I1), :red),
    ("Path (I1 backloaded, I2 frontloaded)", Path(:I1 => τ -> τ^2, :I2 => τ -> 2τ - τ^2), :green),
)

results = [(name, eval_path(traj)...) for (name, traj, _) in paths_to_run]
nothing # hide
```

## Watching the curves diverge

```@example loadpath
p_k = plot(; xlabel = "τ", ylabel = "k_eff(τ)", title = "Bulk modulus", legend = :right, framestyle = :box)
p_μ = plot(; xlabel = "τ", ylabel = "μ_eff(τ)", title = "Shear modulus", legend = false, framestyle = :box)
for ((name, _, color), (_, τ, k, μ)) in zip(paths_to_run, results)
    plot!(p_k, τ, k; label = name, color = color, lw = 2)
    plot!(p_μ, τ, μ; label = name, color = color, lw = 2)
end
plot(p_k, p_μ; layout = (1, 2), left_margin = 8Plots.mm, bottom_margin = 8Plots.mm, size = (1400, 600))
```

All four curves start together — in the dilute limit, the order in which
two vanishingly small increments are added cannot matter — and pull apart
as ``\tau`` grows. `Sequential(I1, I2)` ends up the stiffest of the four:
the soft phase, added last, is diluted into an already-reinforced medium
and does comparatively less damage than it would in a softer host.
`Sequential(I2, I1)` is the mirror image and ends up the softest. The two
`Proportional` and `Path` curves sit strictly between the two sequential
extremes — expected, since every intermediate schedule interpolates
between the two orderings in some sense — but they are not equal to each
other, which is the point: the trajectory itself is a modeling choice,
not just its start and end.

```@example loadpath
using Printf
for (name, _, k, μ) in results
    @printf "%-38s k_eff(1) = %.5f   μ_eff(1) = %.5f\n" name k[end] μ[end]
end
```

A standalone, plain-Julia version of this comparison — same RVE, same
four trajectories, writing a PNG instead of embedding it in the docs —
lives in
[`scripts/24_differential_loading_paths.jl`](https://github.com/MicroPoroChemoMechanics/MeanFieldHomogenization.jl/blob/main/scripts/24_differential_loading_paths.jl).
