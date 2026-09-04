# [Bounds and classical schemes](@id tut-bounds-and-schemes)

The dilute and Mori–Tanaka estimates of the previous tutorial are two
points in a much larger family. This page places every classical
scheme on a single graph, bracketed by the two estimates that **must**
bound any physically admissible effective stiffness: the Voigt and
Reuss bounds.

## Rigorous bounds

For a composite with phases ``\mathbb{C}_i`` at volume fractions
``f_i``, the **Voigt** bound assumes a uniform strain throughout the
RVE, and the **Reuss** bound a uniform stress:

```math
\mathbb{C}_V = \sum_i f_i\,\mathbb{C}_i,
\qquad
\mathbb{C}_R = \Big(\sum_i f_i\,\mathbb{C}_i^{-1}\Big)^{-1}.
```

Both are exact bounds — no assumption on the microstructure's geometry
is needed — and order every physically realizable effective bulk
modulus [hill1963](@cite), [hill1965](@cite):

```math
k_R \le k_{\text{eff}} \le k_V.
```

## Estimates between the bounds

### Self-consistent

[`SelfConsistent`](@ref) drops the "matrix" altogether: every phase is
embedded directly in the *effective* medium itself, and the effective
stiffness must satisfy the implicit condition

```math
\mathbb{C}_{\text{eff}} = \sum_i f_i\,\mathbb{C}_i:\mathbb{A}_i(\mathbb{C}_{\text{eff}}),
```

[budiansky1976](@cite), solved by a damped Picard iteration internally
(`abstol`, `maxiters` control its convergence). This is the natural
model for an interpenetrating, polycrystal-like microstructure where no
phase plays the role of a continuous matrix.

### Differential

[`DifferentialScheme`](@ref) builds the composite incrementally,
re-homogenizing after each infinitesimal addition of inclusions — see
[the dedicated tutorial](differential_paths.md) for the full
picture.

## Putting them on one graph

```@example tutbounds
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using Plots
gr()  # headless backend; GKSwstype is set to "100" in make.jl

C_m = iso_stiffness(30.0, 10.0)   # matrix
C_i = iso_stiffness(60.0, 20.0)   # stiffer inclusion

function build(f)
    r = RVE()
    add_phase!(r, :M, Ellipsoid(1.0), Dict(:C => C_m); fraction = :rest)
    f > 0 && add_phase!(r, :I, Ellipsoid(1.0), Dict(:C => C_i); fraction = f)
    return r
end

fs = collect(range(0.0, 0.6; length = 25))
schemes = [
    (Voigt(), "Voigt", :red, :dash),
    (Reuss(), "Reuss", :red, :dot),
    (Dilute(), "Dilute", :blue, :solid),
    (MoriTanaka(), "MoriTanaka", :green, :solid),
    (SelfConsistent(; abstol = 1.0e-10, maxiters = 200), "SelfConsistent", :purple, :solid),
    (DifferentialScheme(; nsteps = 100), "Differential", :orange, :solid),
]

plt = plot(;
    xlabel = "inclusion volume fraction f", ylabel = "k_eff",
    legend = :topleft, framestyle = :box, size = (760, 480),
)
for (sch, label, color, ls) in schemes
    ks = [k_mu(homogenize(build(f), sch, :C))[1] for f in fs]
    plot!(plt, fs, ks; label = label, color = color, linestyle = ls, lw = 2)
end
plt
```

At `f = 0.3`, the numbers behind the plot:

```@example tutbounds
rve = build(0.3)
for (sch, label, _, _) in schemes
    k, _ = k_mu(homogenize(rve, sch, :C))
    println(rpad(label, 16), round(k, digits = 4))
end
```

Every estimate lies between the Voigt and Reuss bounds — Voigt above,
Reuss below, everything else in between. This is the central point of
mean-field homogenization: the choice of scheme is not a numerical
detail but a **modeling decision** about the microstructure's topology
(matrix-inclusion vs. interpenetrating, dilute vs. dense, aligned vs.
random).
