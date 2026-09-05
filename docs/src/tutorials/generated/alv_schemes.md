```@meta
EditURL = "../../../../scripts/62_alv_schemes.jl"
```

# Ageing viscoelastic schemes side by side

Every mean-field scheme `MeanFieldHomogenization` implements in elasticity also exists on
the **ageing linear viscoelastic** path, where each modulus becomes a Volterra
operator and each product a Volterra product. This page runs four of them on
one composite and reads the differences off a single scalar: the effective
uniaxial creep compliance under a stress step.

The morphology is the one of [barthelemyIJES2019](@cite) — a creeping matrix
reinforced by spheroids of controlled aspect ratio — and the two questions are
the ones that separate the schemes: **which reference medium** each one
polarizes against, and **what shape** it assumes for the spatial distribution
of the inclusions, which is not the same thing as the shape of an inclusion.

````@example alv_schemes
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using Printf
using Plots
gr()  # headless backend; GKSwstype is set to "100" before Literate runs
````

## §1 An ageing matrix and elastic reinforcements

The matrix is a Maxwell fluid whose stiffness **grows with the age** `t'` at
which it is loaded — the elementary model of a setting binder:

```math
\mathbb{R}^{M}(t, t') = a(t')\,
  \Big( 3k_0\,\mathbb{J} + 2\mu_0\,\mathbb{K} \Big)\,
  e^{-(t-t')/\left(\tau\,a(t')\right)},
\qquad
a(t') = 1 + 0.3\sqrt{t'} .
```

Because ``a`` depends on ``t'`` alone and the exponential on ``t-t'``, the
kernel is **not** a function of ``t-t'``: the material ages, the
correspondence principle does not apply, and only the time-domain route
[`homogenize_alv`](@ref) can reach the answer.

````@example alv_schemes
const k₀, μ₀, τ = 5.0, 2.0, 1.0

ageing(tp) = 1 + 0.3 * sqrt(max(tp, 0.0))

function R_matrix(t, tp)
    t < tp && return TensISO{3}(0.0, 0.0)
    a = ageing(tp)
    decay = exp(-(t - tp) / (τ * a))
    return TensISO{3}(3k₀ * a * decay, 2μ₀ * a * decay)
end

const law_M = ViscoLaw(R_matrix, :relaxation)
````

The inclusions are stiff and **elastic**: a relaxation kernel that is constant
for `t ≥ t'` is exactly how an elastic phase enters the ALV algebra.

````@example alv_schemes
const C_I = TensISO{3}(3 * 30.0, 2 * 12.0)     # k = 30, μ = 12
const FRACTION = 0.3

elastic_law(C) = ViscoLaw((t, tp) -> (t < tp ? zero(C) : C), :relaxation)

function build_rve(aspect_ratio)
    rve = RVE(; distribution_shape = Ellipsoid(1.0))
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => law_M); fraction = :rest)
    add_phase!(
        rve, :I, Ellipsoid(1.0, 1.0, aspect_ratio), Dict(:C => elastic_law(C_I));
        fraction = FRACTION, symmetrize = :iso
    )
    return rve
end

function matrix_only()
    rve = RVE(; distribution_shape = Ellipsoid(1.0))
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => law_M); fraction = :rest)
    return rve
end
````

`symmetrize = :iso` averages the inclusion contribution over all orientations,
so the composite stays isotropic however flat the spheroids are. The RVE's
`distribution_shape` is left at its default — a sphere — which matters in §4.

## §2 Reading a creep test off the Volterra matrix

`homogenize_alv` returns the effective **relaxation** operator as a
``6n \times 6n`` block matrix on the time grid. Its Volterra inverse is the
creep operator ``\tilde{\mathbb{J}}``, and the response to a unit uniaxial
stress step ``\sigma(t) = H(t)\,\underline{e}_1 \otimes \underline{e}_1`` is
the row sum of its ``(11,11)`` blocks:

```math
J^{E}_{\text{eff}}(t_i) = \sum_{j} \big[\tilde{\mathbb{J}}\big]_{11,\,ij} .
```

````@example alv_schemes
function uniaxial_creep(R)
    J = volterra_inverse(R; block_size = 6)
    n = size(J, 1) ÷ 6
    return [sum(J[6 * (i - 1) + 1, 6 * (j - 1) + 1] for j in 1:n) for i in 1:n]
end

const T = vcat(0.0, 10 .^ range(-2, log10(20.0); length = 60))

creep(rve, scheme) = uniaxial_creep(homogenize_alv(rve, scheme, :C; times = T))
````

## §3 Four schemes on spherical inclusions

| Scheme | Reference medium | Distribution shape |
|---|---|---|
| [`Dilute`](@ref) | the matrix | ignored — interactions neglected |
| [`MoriTanaka`](@ref) | the matrix | that of the inclusion |
| [`Maxwell`](@ref) | the matrix | `rve.distribution_shape` |
| [`PonteCastanedaWillis`](@ref) | the matrix | `rve.distribution_shape` |

````@example alv_schemes
const SCHEMES = (
    (Dilute(), "Dilute", :gray, :dot),
    (MoriTanaka(), "Mori-Tanaka", :blue, :solid),
    (Maxwell(), "Maxwell", :purple, :dash),
    (PonteCastanedaWillis(), "PCW", :green, :dashdot),
)

plt_sphere = plot(;
    xlabel = "t", ylabel = "J^E_eff(t)", xscale = :log10,
    title = "Spherical inclusions, f = $FRACTION", legend = :topleft,
    framestyle = :box, size = (760, 470),
)
let rve = build_rve(1.0)
    for (sch, lbl, col, ls) in SCHEMES
        plot!(
            plt_sphere, T[2:end], creep(rve, sch)[2:end];
            label = lbl, color = col, linestyle = ls, lw = 2
        )
    end
    plot!(
        plt_sphere, T[2:end], creep(matrix_only(), MoriTanaka())[2:end];
        label = "matrix alone", color = :black, lw = 1, alpha = 0.4
    )
end
plt_sphere
````

Four schemes, three visible curves. `Dilute` sits above the others — it
ignores the interaction between inclusions, so it under-reinforces — while
Mori-Tanaka, Maxwell and PCW fall exactly on top of each other:

````@example alv_schemes
let rve = build_rve(1.0)
    mt = creep(rve, MoriTanaka())[end]
    for (sch, lbl, _, _) in SCHEMES
        J = creep(rve, sch)[end]
        @printf("  %-12s J^E_eff(t_end) = %.8f   (MT %+.1e)\n", lbl, J, J - mt)
    end
end
````

That collapse is not a coincidence and not an approximation. Maxwell and PCW
polarize against the matrix and build their Hill operator on the
*distribution* shape; when the distribution shape equals the inclusion shape —
here both spherical — their formula reduces algebraically to Mori-Tanaka's.
The three schemes only have distinct content once the two shapes differ.

## §4 The aspect ratio

At fixed volume fraction, flattening the reinforcements makes them far more
effective: a disc intercepts the load over a much larger area than a sphere of
the same volume. Between spheres and 100:1 discs the long-term creep
compliance falls from ``0.771`` to ``0.240`` — a factor of ``3.2`` — at an
unchanged 30 % of stiff phase, against ``1.404`` for the bare matrix.

````@example alv_schemes
const RATIOS = (1.0, 0.1, 0.01)

plt_ratio = plot(;
    xlabel = "t", ylabel = "J^E_eff(t)", xscale = :log10,
    title = "Mori-Tanaka — aspect ratio at f = $FRACTION", legend = :topleft,
    framestyle = :box, size = (760, 470),
)
for (α, col) in zip(RATIOS, (:blue, :orange, :red))
    plot!(
        plt_ratio, T[2:end], creep(build_rve(α), MoriTanaka())[2:end];
        label = "α = $α", color = col, lw = 2
    )
end
plot!(
    plt_ratio, T[2:end], creep(matrix_only(), MoriTanaka())[2:end];
    label = "matrix alone", color = :black, lw = 1, alpha = 0.4
)
plt_ratio
````

## §5 Where Maxwell and PCW stop being admissible

Repeat the sweep with all four schemes and the schemes part company — then one
of them leaves the physical domain entirely:

````@example alv_schemes
@printf("\n  %-6s %10s %10s %10s %10s\n", "α", "Dilute", "MT", "Maxwell", "PCW")
for α in RATIOS
    rve = build_rve(α)
    @printf("  %-6.2f", α)
    for (sch, _, _, _) in SCHEMES
        @printf(" %10.5f", creep(rve, sch)[end])
    end
    println()
end
````

At `α = 0.01` the Maxwell/PCW estimate returns a **negative** creep
compliance. The cause is geometric, not numerical: the RVE still declares a
*spherical* distribution shape, and a sphere cannot host 30 % of 100:1 discs
without the enveloping spheres overlapping. This is the admissibility
restriction of [ponte1995](@cite), and outside it the estimate carries no
meaning. Mori-Tanaka is unaffected because it never uses a distribution shape
distinct from the inclusion's.

The remedy is to declare the distribution shape the morphology actually has:

````@example alv_schemes
let α = 0.01
    rve = RVE(; distribution_shape = UniformDistribution(Ellipsoid(1.0, 1.0, α)))
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => law_M); fraction = :rest)
    add_phase!(
        rve, :I, Ellipsoid(1.0, 1.0, α), Dict(:C => elastic_law(C_I));
        fraction = FRACTION, symmetrize = :iso
    )
    @printf(
        "\n  α = %.2f, oblate distribution shape:  Maxwell = %.5f\n",
        α, creep(rve, Maxwell())[end]
    )
end
````

!!! note "The distribution shape is a modeling choice, not a default"
    `RVE` leaves `distribution_shape` spherical unless told otherwise, which
    is harmless for [`MoriTanaka`](@ref) and [`Dilute`](@ref) and decisive for
    [`Maxwell`](@ref) and [`PonteCastanedaWillis`](@ref). Whenever the
    inclusions are markedly non-spherical, set it explicitly.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

