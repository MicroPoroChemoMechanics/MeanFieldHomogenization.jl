```@meta
EditURL = "../../../../scripts/79_graded_interphase.jl"
```

# [A graded interphase around a spherical particle](@id app-graded-interphase)

!!! tip "The whole page as one script"
    Every block of this page, in order, is [`scripts/79_graded_interphase.jl`](https://github.com/MicroPoroChemoMechanics/MeanFieldHomogenization.jl/blob/main/scripts/79_graded_interphase.jl).
    From a clone of the repository, `julia --project=docs -e 'using Pkg; Pkg.instantiate()'`
    once, then `julia scripts/79_graded_interphase.jl`, runs it end to end.

!!! info "Before this page"
    [Layered inclusions](@ref man-layered), for the layered sphere and
    `graded_layers`, and [Layered spheres](@ref th-layered-sphere) for the
    recurrence behind it.

Around a particle the matrix is often altered over a zone where its moduli
vary continuously: a diffusion zone around a fiber, the transition zone
around an aggregate in concrete. This page enters such a zone as thin uniform
layers, measures the error of the layers against the exact solution of
[Lutz and Zimmerman](@cite lutzIJSS2005) for the bulk modulus and for the
conductivity, and adds the shear modulus, which the layers give as well.

````@example graded_interphase
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using Printf
using Plots
gr()
nothing #hide
````

## The zone

A sphere of radius ``a``, of bulk and shear moduli ``K_i`` and ``\mu_i``, is
embedded in a matrix whose moduli are ``K_m`` and ``\mu_m`` far from it and
``K_{if}`` and ``\mu_{if}`` at its surface. Between the two,
[Lutz and Zimmerman](@cite lutzIJSS2005) take

```math
K(r) = K_m + (K_{if} - K_m)\left(\frac{a}{r}\right)^{b},
\qquad
\mu(r) = \mu_m + (\mu_{if} - \mu_m)\left(\frac{a}{r}\right)^{b},
```

with an integer ``b``: the larger ``b``, the thinner the zone. The data of
their example are a particle five times stiffer than the matrix, a Poisson
ratio of ``1/4`` everywhere and ``b = 10``; the zone is measured by
``D = (K_m - K_{if})/K_m``, positive when it is softer than the matrix.

````@example graded_interphase
const a, b = 1.0, 10
const K_m, ν = 1.0, 0.25
const μ_m = 3K_m * (1 - 2ν) / (2 * (1 + ν))
const K_i, μ_i = 5K_m, 5μ_m
const C_m, C_i = iso_stiffness(K_m, μ_m), iso_stiffness(K_i, μ_i)

zone(D) = r -> iso_stiffness(K_m * (1 - D * (a / r)^b), μ_m * (1 - D * (a / r)^b))
rs = range(a, 2a; length = 200)
plot(
    rs, [1 .- D .* (a ./ rs) .^ b for D in (0.5, -0.5)]; label = ["D = 0.5" "D = −0.5"], lw = 2,
    xlabel = "r / a", ylabel = "K(r) / Kₘ", title = "Profile of the zone, b = 10", titlefontsize = 10, size = (560, 320)
)
````

## The layers

The profile reaches the matrix at infinity. It is cut at the radius
``r_{out} = 5a``, where it differs from the matrix by ``5^{-10}``, about
``10^{-7}``. `graded_layers` then cuts the zone at given radii and gives each
layer the moduli of the profile at its mid-radius. The radii are taken here
in equal steps of ``1/r``, which makes the layers thin near the particle,
where the profile is steep.

````@example graded_interphase
const r_out = 5a
edges(n) = a ./ range(1, a / r_out; length = n + 1)
function particle(D, n)
    radii, moduli = graded_layers(zone(D), edges(n))
    return LayeredSphere((a, radii...), (C_i, moduli...))
end
layer_count(particle(0.5, 60))                 ## the particle and 60 layers
````

The particle and its zone are one inclusion. Its stiffness contribution
tensor ``\mathbb N``, per unit volume of the sphere of radius ``r_{out}``, is
isotropic: ``\mathbb N = 3\,n_K\,\mathbb J + 2\,n_\mu\,\mathbb K``
([Concentration and contribution tensors](@ref th-concentration)).

````@example graded_interphase
function contribution(D, n)
    N = canonical_components(property_contribution(particle(D, n), C_m, C_m))
    n_K = sum(N[i, i, j, j] for i in 1:3, j in 1:3) / 9
    return (; n_K, n_μ = N[1, 2, 1, 2])
end
contribution(0.5, 60)
````

## The exact solution under a hydrostatic loading

Under a remote hydrostatic strain the displacement is radial, ``u(r)``, and
the radial stress is ``\sigma_{rr} = M\,u' + 2\lambda\,u/r`` with
``\lambda = K - 2\mu/3`` and ``M = \lambda + 2\mu``. The equilibrium of a
medium whose moduli depend on ``r`` reads

```math
M\left(u'' + \frac{2}{r}\,u' - \frac{2}{r^2}\,u\right) + M'\,u' + 2\,\lambda'\,\frac{u}{r} = 0 .
```

With ``s = a/r``, the profile is ``M = M_m + \Delta M\,s^b`` and
``\lambda = \lambda_m + \Delta\lambda\,s^b``, and a term ``u = r\,s^n`` gives

```math
\Bigl[M_m\,n(n-3) + \bigl(\Delta M\,(n^2 + (b-3)\,n - b) - 2b\,\Delta\lambda\bigr)\,s^b\Bigr]\,\frac{s^n}{r} .
```

The first bracket vanishes for ``n = 0`` and ``n = 3``: the solution in a
homogeneous matrix, ``u = A_1\,r + A_2\,a^3/r^2``. In the zone each of the two
starts a series in powers of ``s^b``, the method of Frobenius used in the
article:

```math
u = A_1\,r\sum_{j\ge0} C_{jb}\,s^{jb} + A_2\,r\sum_{j\ge0} C_{jb+3}\,s^{jb+3},
\qquad
C_{n+b} = -\frac{\Delta M\,(n^2 + (b-3)\,n - b) - 2b\,\Delta\lambda}{(n+b)(n+b-3)\,M_m}\;C_n ,
```

with ``C_0 = C_3 = 1``. Inside the particle ``u = B\,r``. The continuity of
``u`` and of ``\sigma_{rr}`` at ``r = a`` gives the ratio of the two
amplitudes,

```math
f = -\frac{A_2}{A_1}
  = \frac{3\,(K_i - K_{if})\,S_1 + M_{if}\,S_1'}{3\,(K_i - K_{if})\,S_2 + M_{if}\,S_2'},
\qquad
S_1 = \sum_j C_{jb},\quad S_1' = \sum_j jb\;C_{jb},\quad
S_2 = \sum_j C_{jb+3},\quad S_2' = \sum_j (jb+3)\;C_{jb+3} ,
```

so that, far from the particle, ``u \simeq A_1\,(r - f\,a^3/r^2)``. Without a
zone, ``f = 3(K_i - K_m)/(3K_i + 4\mu_m)``.

````@example graded_interphase
function frobenius(K_i, λ_if, μ_if, λ_m, μ_m, b; terms = 400)
    M_m, M_if = λ_m + 2μ_m, λ_if + 2μ_if
    ΔM, Δλ = M_if - M_m, λ_if - λ_m
    ratio(n) = -(ΔM * (n^2 + (b - 3) * n - b) - 2b * Δλ) / ((n + b) * (n + b - 3) * M_m)
    S₁ = S₁′ = S₂ = S₂′ = 0.0
    c₁ = c₂ = 1.0
    for j in 0:terms
        S₁ += c₁; S₁′ += j * b * c₁; S₂ += c₂; S₂′ += (j * b + 3) * c₂
        c₁ *= ratio(j * b); c₂ *= ratio(j * b + 3)
    end
    K_if = λ_if + 2μ_if / 3
    return (3 * (K_i - K_if) * S₁ + M_if * S₁′) / (3 * (K_i - K_if) * S₂ + M_if * S₂′)
end
f_exact(D) = frobenius(K_i, (1 - D) * (K_m - 2μ_m / 3), (1 - D) * μ_m, K_m - 2μ_m / 3, μ_m, b)
f_exact(0.0), 3 * (K_i - K_m) / (3K_i + 4μ_m)
````

A homogeneous sphere of radius ``r_{out}`` with the same far field has the
contribution ``n_K`` such that

```math
f\,a^3 = \frac{3\,n_K}{3K_m + 4\mu_m}\;r_{out}^3 ,
```

which is how the layers are compared with the series:

````@example graded_interphase
f_layers(D, n) = 3 * contribution(D, n).n_K / (3K_m + 4μ_m) * (r_out / a)^3
f_layers(0.5, 60), f_exact(0.5)
````

## Convergence of the layers

The error of ``n`` layers decreases as ``1/n^2``:

````@example graded_interphase
ns = [10, 20, 40, 80, 160]
errors = [abs(f_layers(D, n) - f_exact(D)) for n in ns, D in (0.5, -0.5)]
plot(
    ns, errors; xscale = :log10, yscale = :log10, marker = :circle, lw = 2, label = ["D = 0.5" "D = −0.5"],
    xlabel = "number of layers n", ylabel = "| f(layers) − f(series) |", xticks = (ns, string.(ns)),
    title = "Error of the layers on the far field", titlefontsize = 10, size = (560, 340)
)
plot!(ns, errors[1, 1] .* (ns[1] ./ ns) .^ 2; color = :gray, ls = :dash, label = "1/n²")
````

Two computations then give a third order of accuracy at no cost: with
``n`` and ``2n`` layers, ``(4 f_{2n} - f_n)/3`` removes the term in
``1/n^2``.

````@example graded_interphase
extrapolated(g, n = 60) = (4 * g(2n) - g(n)) / 3
for D in (0.5, -0.5)
    @printf "D = %+.1f   series %.7f   120 layers %.7f   60 and 120 layers, extrapolated %.7f\n" D f_exact(D) f_layers(D, 120) extrapolated(n -> f_layers(D, n))
end
````

## Effective bulk modulus

The estimate of the article for a volume fraction ``c`` of particles is that
of Maxwell: the far field of a sphere of the composite, which contains the
particles and their zones, is the sum of theirs. It is

```math
\frac{K}{K_m} = \frac{1 + \dfrac{4\mu_m}{3K_m}\,f\,c}{1 - f\,c} ,
```

and is drawn below for the seven values of ``D`` of their Fig. 2, the lines
from the layers and the marks from the series.

````@example graded_interphase
bulk(f, c) = (1 + 4μ_m / (3K_m) * f * c) / (1 - f * c)
Ds = (-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75)
cs = range(0, 0.6; length = 61)
plt = plot(;
    xlabel = "volume fraction of particles c", ylabel = "K / Kₘ", legend = :topleft,
    title = "Effective bulk modulus, Kᵢ = 5 Kₘ, ν = 1/4, b = 10", titlefontsize = 10, size = (640, 400)
)
for (i, D) in enumerate(Ds)
    f = extrapolated(n -> f_layers(D, n))
    plot!(plt, cs, bulk.(f, cs); lw = 2, color = i, label = "D = $D")
    scatter!(plt, cs[11:10:end], bulk.(f_exact(D), cs[11:10:end]); color = i, markersize = 3, label = "")
end
plt
````

In the package the same estimate is that of a scheme whose phase is the
particle with its zone, of volume fraction ``c\,(r_{out}/a)^3``. This
fraction counts the zone up to ``r_{out}``, so that the phase fits in the
cell for small ``c`` only:

````@example graded_interphase
c = 0.004
rve = RVE()
add_phase!(rve, :matrix, Ellipsoid(1.0), Dict(:C => C_m); fraction = :rest)
add_phase!(rve, :particles, particle(0.5, 120), Dict(:C => C_i); fraction = c * (r_out / a)^3)
C = canonical_components(homogenize(rve, MoriTanaka(), :C))
sum(C[i, i, j, j] for i in 1:3, j in 1:3) / 9 / K_m, bulk(f_layers(0.5, 120), c)
````

## Conductivity

The temperature around the particle under a remote gradient is
``T = F(r)\cos\theta``, and ``F`` satisfies the equation of ``u`` above with
``\lambda = 0`` and ``M`` replaced by the conductivity ``k``: the article
notes that the conduction problem is the hydrostatic one for a Poisson ratio
equal to zero, ``3K`` and ``2\mu`` standing for ``k``. The same series then
gives the far field ``F \simeq G\,(r - f\,a^3/r^2)`` and

```math
\frac{k}{k_m} = \frac{1 + 2\,f\,c}{1 - f\,c} ,
```

while the layers are those of the conduction problem, with
``f\,a^3 = n_k\,r_{out}^3/(3k_m)`` for a conductivity contribution ``n_k``.

````@example graded_interphase
const k_m, k_i = 1.0, 5.0
function f_conduction(D, n)
    radii, moduli = graded_layers(r -> TensISO{3}(k_m * (1 - D * (a / r)^b)), edges(n))
    sphere = LayeredSphere((a, radii...), (TensISO{3}(k_i), moduli...))
    n_k = canonical_components(property_contribution(sphere, TensISO{3}(k_m), TensISO{3}(k_m)))[1, 1]
    return n_k / (3k_m) * (r_out / a)^3
end
f_conduction_exact(D) = frobenius(k_i / 3, 0.0, (1 - D) * k_m / 2, 0.0, k_m / 2, b)
for D in (0.5, -0.5)
    @printf "D = %+.1f   series %.7f   60 and 120 layers, extrapolated %.7f\n" D f_conduction_exact(D) extrapolated(n -> f_conduction(D, n))
end
````

````@example graded_interphase
plt = plot(;
    xlabel = "volume fraction of particles c", ylabel = "k / kₘ", legend = :topleft,
    title = "Effective conductivity, kᵢ = 5 kₘ, b = 10", titlefontsize = 10, size = (640, 400)
)
for (i, D) in enumerate(Ds)
    f = extrapolated(n -> f_conduction(D, n))
    plot!(plt, cs, (1 .+ 2 .* f .* cs) ./ (1 .- f .* cs); lw = 2, color = i, label = "D = $D")
    scatter!(plt, cs[11:10:end], [(1 + 2 * f_conduction_exact(D) * c) / (1 - f_conduction_exact(D) * c) for c in cs[11:10:end]]; color = i, markersize = 3, label = "")
end
plt
````

These two figures are those of the article (its Figs. 2 and 3), recomputed.

## The shear modulus

Under a remote shear the displacement around the particle is not radial and
the article gives no solution. The layers give the response all the same:
the shear part ``n_\mu`` of the contribution converges as the bulk one,

````@example graded_interphase
for n in (30, 60, 120)
    @printf "n = %3d   n_μ = %.7f\n" n contribution(0.5, n).n_μ
end
n_μ(D) = extrapolated(n -> contribution(D, n).n_μ)
nothing #hide
````

and the estimate of Maxwell for the shear modulus follows from it as for the
bulk modulus. With ``g = n_\mu\,(r_{out}/a)^3/(\mu_m + \mu^*)`` and
``\mu^* = \mu_m\,(9K_m + 8\mu_m)/\bigl(6\,(K_m + 2\mu_m)\bigr)``,

```math
\frac{\mu}{\mu_m} = \frac{1 + \dfrac{\mu^*}{\mu_m}\,g\,c}{1 - g\,c} .
```

````@example graded_interphase
const μ_star = μ_m * (9K_m + 8μ_m) / (6 * (K_m + 2μ_m))
shear(g, c) = (1 + μ_star / μ_m * g * c) / (1 - g * c)
plt = plot(;
    xlabel = "volume fraction of particles c", ylabel = "μ / μₘ", legend = :topleft,
    title = "Effective shear modulus, μᵢ = 5 μₘ, ν = 1/4, b = 10", titlefontsize = 10, size = (640, 400)
)
for (i, D) in enumerate(Ds)
    g = n_μ(D) * (r_out / a)^3 / (μ_m + μ_star)
    plot!(plt, cs, shear.(g, cs); lw = 2, color = i, label = "D = $D")
end
plt
````

Without a zone the contribution is that of the homogeneous sphere, which
checks the expression:

````@example graded_interphase
n_μ(0.0) * (r_out / a)^3, (μ_i - μ_m) * (μ_m + μ_star) / (μ_i + μ_star)
````

The zone acts on the two moduli in the same proportion, as the amplitudes
``f`` of the bulk estimate and ``g`` of the shear one show:

````@example graded_interphase
for D in Ds
    @printf "D = %+.2f   f = %.4f   g = %.4f\n" D extrapolated(n -> f_layers(D, n)) n_μ(D) * (r_out / a)^3 / (μ_m + μ_star)
end
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

