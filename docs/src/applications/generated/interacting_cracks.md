```@meta
EditURL = "../../../../scripts/109_interacting_cracks.jl"
```

# [Interacting cracks: the method of average tractions](@id app-interacting-cracks)

!!! tip "The whole page as one script"
    Every block of this page, in order, is [`scripts/109_interacting_cracks.jl`](https://github.com/MicroPoroChemoMechanics/MeanFieldHomogenization.jl/blob/main/scripts/109_interacting_cracks.jl).
    From a clone of the repository, `julia --project=docs -e 'using Pkg; Pkg.instantiate()'`
    once, then `julia scripts/109_interacting_cracks.jl`, runs it end to end.

!!! info "Before this page"
    [Interacting cracks](@ref th-interacting-cracks), which derives the method
    and the fields used here, and
    [Interacting cracks of a plane](@ref man-interacting-cracks) for the syntax.

Two cracks close to each other change each other's stress intensity
factors: collinear cracks amplify the factors at the tips that face each
other, stacked cracks shield each other. [kachanov1987](@citet) computes these
factors by keeping, in the tractions each crack sends onto the others, only
the part due to its mean traction. This page compares the method with the
exact solutions of two collinear cracks and of a periodic row of cracks
(Tables 1 and 2 of the article), shows how a stack of cracks shields its
members in an isotropic and in an orthotropic plane
[mauge1994](@cite), and ends with an arbitrary array.

````@example interacting_cracks
using MeanFieldHomogenization
using MeanFieldHomogenization.Elliptic: ell_K, ell_E
using TensND
using LinearAlgebra
using QuadGK
using Random
using JSON
using Printf
using Plots
gr()
nothing #hide
````

## The data

The two tables of the article give ``K_{\mathrm I}/K_{\mathrm I}^0``, the
factor of mode I divided by that of an isolated crack of the same length, in
two columns: the values of its method and *exact* values quoted from a
handbook.

````@example interacting_cracks
lit = JSON.parsefile(joinpath(pkgdir(MeanFieldHomogenization), "data", "literature", "kachanov1987.json"))
table1, table2 = lit["tables"]["two_collinear"], lit["tables"]["periodic_row"]
nothing #hide

Main.show_literature("kachanov1987") # hide
````

A crack of the plane is a [`RibbonCrack`](@ref) of half-length `b` whose
direction makes the angle `angle` with ``\underline e_1``, placed at a center;
[`sif`](@ref) takes a list of `center => crack` pairs, the stiffness of the
plane and the remote stress, and returns for each crack its factors
``(K_{\mathrm I}, K_{\mathrm{II}})`` at the tips ``+b`` (`plus`) and ``-b``
(`minus`) along its direction, and its mean tractions. In an isotropic plane
the stresses of a loaded crack do not depend on the moduli: any isotropic
stiffness serves.

````@example interacting_cracks
C_iso = TensISO{2}(2.0, 1.0)
tension = Tens([0.0 0.0; 0.0 1.0])                   ## unit tension normal to e₁
nothing #hide
````

## Two collinear cracks

The cracks occupy ``k < |x_1| < 1`` on the axis ``x_2 = 0``, under the remote
tension ``\sigma_{22}^\infty = p``. Each has the half-length ``l = (1 - k)/2``
and ``K_{\mathrm I}^0 = p\sqrt{\pi(1 - k)/2}``. The right crack has its inner
tip at ``x_1 = k``, its tip `minus`, and its outer tip at ``x_1 = 1``:

````@example interacting_cracks
pair(k) = (c = (1 + k) / 2; l = (1 - k) / 2;
    [[-c, 0.0] => RibbonCrack(l; angle = 0.0), [c, 0.0] => RibbonCrack(l; angle = 0.0)])
function collinear(k)
    right = sif(pair(k), C_iso, tension)[2]
    K₀ = sqrt(π * (1 - k) / 2)
    return (inner = right.minus[1] / K₀, outer = right.plus[1] / K₀)
end
nothing #hide
````

**The exact solution**, derived in [Interacting cracks](@ref th-interacting-cracks),
§4, from Westergaard's function of the two cracks: with ``\lambda^2 = E(m)/K(m)``
and ``m = 1 - k^2``, the factors at the outer and inner tips are
``(1 - \lambda^2)\sqrt{2/(1 - k)}/\sqrt{1 - k^2}`` and
``(\lambda^2 - k^2)\sqrt{2/(1 - k)}/(\sqrt{k}\sqrt{1 - k^2})`` times ``K_{\mathrm I}^0``.

````@example interacting_cracks
function collinear_exact(k)
    λ² = ell_E(1 - k^2) / ell_K(1 - k^2)
    f = sqrt(2 / (1 - k)) / sqrt(1 - k^2)
    return (inner = (λ² - k^2) / sqrt(k) * f, outer = (1 - λ²) * f)
end
nothing #hide
````

**The closed form of the method.** For two collinear cracks the method has
the closed form (6) of the article: each crack transmits to the other the
mean traction ``\Lambda\langle p\rangle``, with
``\Lambda = \sqrt{2(1 + k)}/(1 + \sqrt k) - 1`` (its eq. 4), so that
``\langle p\rangle = p/(1 - \Lambda)``, and the factors follow from eq. (14)
applied to the traction of eq. (3):

````@example interacting_cracks
function collinear_eq6(k)
    Λ = sqrt(2(1 + k)) / (1 + sqrt(k)) - 1
    K, E = ell_K(1 - k^2), ell_E(1 - k^2)
    a = 1 / ((1 - Λ) * π * (1 - k))
    return (inner = 1 + a * (-2E + (k + 1) * K - π / 2 * (1 - k)), outer = 1 + a * (2E - k * (k + 1) * K - π / 2 * (1 - k)))
end
maximum(r -> maximum(abs, values(collinear(r["k"])) .- values(collinear_eq6(r["k"]))), table1)
````

The general computation, quadratures and a linear system, equals the closed
form to the precision of the quadratures. Against Table 1:

````@example interacting_cracks
println("  k     inner: sif   eq.(6) printed  exact  printed     outer: sif  printed  exact  printed")
for r in table1
    m, e = collinear(r["k"]), collinear_exact(r["k"])
    @printf("%5.2f        %6.3f       %6.3f      %6.3f   %6.3f          %6.3f   %6.3f  %6.3f   %6.3f\n",
        r["k"], m.inner, r["inner_method"], e.inner, r["inner_exact"], m.outer, r["outer_method"], e.outer, r["outer_exact"])
end
````

The method gives back its printed column to the last digit, and the closed
form the exact column. The method underestimates the factor at the inner
tips, by 10 per cent when the gap ``2k`` is a fiftieth of the length of a
crack: the traction a crack sends onto its neighbor is concentrated near the
tip that faces it, and its mean misses that.

````@example interacting_cracks
gaps = exp10.(range(-2.3, 0.5; length = 60))          ## 2k/(1 - k), gap over length
k_of(g) = g / (2 + g)
plot(gaps, [getfield.(collinear.(k_of.(gaps)), :inner) getfield.(collinear_exact.(k_of.(gaps)), :inner)];
    xscale = :log10, lw = 2, ls = [:solid :dash], color = [1 1],
    label = ["inner tip, average tractions" "inner tip, exact"],
    xlabel = "gap / crack length", ylabel = "K_I / K_I⁰", size = (700, 420))
plot!(gaps, [getfield.(collinear.(k_of.(gaps)), :outer) getfield.(collinear_exact.(k_of.(gaps)), :outer)];
    lw = 2, ls = [:solid :dash], color = [2 2], label = ["outer tip, average tractions" "outer tip, exact"])
g1 = [2r["k"] / (1 - r["k"]) for r in table1]
scatter!(g1, [r["inner_exact"] for r in table1]; color = 1, marker = :circle, label = "Table 1, exact")
scatter!(g1, [r["outer_exact"] for r in table1]; color = 2, marker = :circle, label = "")
````

## A periodic row

An infinite row of cracks of half-length ``l`` and period ``h``, along
``\underline e_1``, under the tension ``p`` normal to it. With
``\zeta = h/(2l)``, the ratio ``h/(2l) - 1`` of Table 2 is the gap between two
cracks over their length.

**The method on the infinite row.** All cracks carry the same mean traction
``\langle p\rangle``. A crack under the uniform opening traction
``\langle p\rangle`` creates on its own line, at the distance ``d > l`` from
its center, the normal stress (Westergaard)

```math
\sigma_{22}(d) = \langle p\rangle\Bigl(\frac{d}{\sqrt{d^2 - l^2}} - 1\Bigr) .
```

Its mean over the crack at the distance ``|j|h`` is the transmission factor of
the article (p. 27),

```math
\Lambda^{(j)} = |j|\zeta\Bigl(\sqrt{1 + \tfrac{1}{|j|\zeta}} - \sqrt{1 - \tfrac{1}{|j|\zeta}}\Bigr) - 1
  = \frac{2u^2}{s^2\,(1 + \sqrt{1 + u})(1 + \sqrt{1 - u})},
\quad u = \frac{1}{|j|\zeta},\ s = \sqrt{1 + u} + \sqrt{1 - u},
```

the second form, obtained by rationalizing twice, being free of the
cancellation of the first when ``|j|`` is large (``\Lambda^{(j)} \simeq u^2/8``).
The mean of the traction on one crack then reads
``\langle p\rangle = p + 2\langle p\rangle\sum_{j \ge 1}\Lambda^{(j)}``, its
eq. (10):

````@example interacting_cracks
function Λ(j, ζ)
    u = 1 / (j * ζ)
    s = sqrt(1 + u) + sqrt(1 - u)
    return 2u^2 / (s^2 * (1 + sqrt(1 + u)) * (1 + sqrt(1 - u)))
end
mean_traction(ζ; J = 10^6) = 1 / (1 - 2 * sum(j -> Λ(j, ζ), 1:J))
(Λ(3, 1.5), 3 * 1.5 * (sqrt(1 + 1 / 4.5) - sqrt(1 - 1 / 4.5)) - 1, mean_traction(1.1), mean_traction(1.1; J = 10^7))
````

The two forms of ``\Lambda^{(j)}`` agree, and ``10^6`` terms fix the sum to
better than ``10^{-6}``.

The factor at a tip follows from eq. (14) of the article, applied to the
traction ``p + \sum_{j \neq 0}\sigma_{22}^{(j)}`` along the crack:

```math
\frac{K_{\mathrm I}}{K_{\mathrm I}^0} = 1 + \frac{\langle p\rangle}{p}\sum_{j \ge 1}\kappa_j,
\qquad
\kappa_j = \frac{1}{\pi}\int_0^\pi\Bigl[\tau(jh + l\cos\theta) + \tau(jh - l\cos\theta)\Bigr](1 - \cos\theta)\,\mathrm d\theta ,
```

``\tau(d) = d/\sqrt{d^2 - l^2} - 1 = l^2/\bigl(\sqrt{d^2 - l^2}\,(d + \sqrt{d^2 - l^2})\bigr)``
(the second form for far cracks, without cancellation), the pair of cracks ``\pm j`` being taken
together, and ``\xi = -l\cos\theta`` the abscissa along the crack. Far from the
crack the traction of the pair is uniform up to terms in ``1/j^4``, so that
``\kappa_j \simeq 2\Lambda^{(j)}``: the first 200 terms are integrated, the
others replaced by ``2\Lambda^{(j)}``; 100 integrated terms give the same value:

````@example interacting_cracks
τ(d, l) = (r = sqrt(d^2 - l^2); l^2 / (r * (d + r)))
κ(j, ζ; l = 0.5) = first(quadgk(θ -> (τ(2l * ζ * j + l * cos(θ), l) + τ(2l * ζ * j - l * cos(θ), l)) * (1 - cos(θ)), 0, π; rtol = 1.0e-10)) / π
row_eq10(ζ; n = 200, J = 10^6) = 1 + mean_traction(ζ; J) * (sum(j -> κ(j, ζ), 1:n) + 2 * sum(j -> Λ(j, ζ), (n + 1):J))
(n100 = row_eq10(1.1; n = 100), n200 = row_eq10(1.1))
````

The sums converge slowly, as ``1/J`` when they are stopped after ``J``
neighbors on each side. Stopped there, both of them:

````@example interacting_cracks
row_stopped(ζ, J) = 1 + sum(j -> κ(j, ζ), 1:J) / (1 - 2 * sum(j -> Λ(j, ζ), 1:J))
nothing #hide
````

**The exact solution** of the row, derived in
[Interacting cracks](@ref th-interacting-cracks), §4:
``K_{\mathrm I}/K_{\mathrm I}^0 = \sqrt{(h/\pi l)\tan(\pi l/h)}``.

````@example interacting_cracks
row_exact(ζ) = (x = π / (2ζ); sqrt(tan(x) / x))
println("gap/length   eq.(10)  stopped at 75   printed     exact  printed")
for r in table2
    ζ = 1 + r["gap_over_length"]
    @printf("   %5.3f      %6.3f      %6.3f       %6.3f    %6.3f   %6.3f\n",
        r["gap_over_length"], row_eq10(ζ), row_stopped(ζ, 75), r["method"], row_exact(ζ), r["exact"])
end
````

The closed form gives back the exact column of Table 2 to the last digit,
except at the smallest gap, where the printed value is 6.499 and the closed
form gives 6.430. The method, its sums carried to convergence, gives values
slightly above the printed column, by 0.1 per cent at the largest gap and
2 per cent at the smallest. Its sums stopped after 75 neighbors on each side
give back the first four printed values and come within 0.5 per cent of the
others; the article does not say where it stopped its sums. Either way, the method overestimates the factor
of the row, more and more as the gaps close: in a row each crack is amplified
by all the others.

**Finite rows.** The row is the limit of finite rows of ``2N + 1`` cracks, which
[`sif`](@ref) solves without assuming the mean tractions equal. The factor of
the central crack approaches that of the infinite row as ``N`` grows, as
slowly as the sums above:

````@example interacting_cracks
function finite_row(N, ζ; l = 0.5)
    cracks = [[2l * ζ * j, 0.0] => RibbonCrack(l; angle = 0.0) for j in -N:N]
    return sif(cracks, C_iso, tension)[N + 1].plus[1] / sqrt(π * l)
end
Ns = [1, 2, 4, 8, 16]
rows = [(ζ, [finite_row(N, ζ) for N in Ns], row_eq10(ζ)) for ζ in (1.5, 1.1)]
fig = plot(; xscale = :log2, xlabel = "N (row of 2N + 1 cracks)", ylabel = "K_I / K_I⁰ of the central crack",
    xticks = (Ns, string.(Ns)), legend = :right, size = (700, 420))
for (i, (ζ, K, K∞)) in enumerate(rows)
    plot!(fig, Ns, K; marker = :circle, lw = 2, color = i, label = @sprintf("gap/length = %.1f", ζ - 1))
    hline!(fig, [K∞]; ls = :dash, color = i, label = "infinite row, eq. (10)")
end
fig
````

## Stacked cracks shield each other

Two parallel cracks of half-length ``l = 1/2``, one above the other at the
distance ``s``, under the tension normal to them: each lowers the factor of
the other. In an anisotropic plane the field of a loaded crack is that of the
slit of the material, which [`sif`](@ref) uses when given an anisotropic
stiffness. The orthotropic plane below, given by its Kelvin–Mandel matrix,
has ``C_{1111} = 10`` and ``C_{2222} = 3``: it is stiffer along the cracks;
the second one, turned by 90°, is stiffer across them.

````@example interacting_cracks
C_ort = from_mandel([10.0 2.0 0.0; 2.0 3.0 0.0; 0.0 0.0 3.0])
C_ort90 = from_mandel([3.0 2.0 0.0; 2.0 10.0 0.0; 0.0 0.0 3.0])   ## the same turned by 90°
stacked(s) = [[0.0, 0.0] => RibbonCrack(0.5; angle = 0.0), [0.0, s] => RibbonCrack(0.5; angle = 0.0)]
shield(C, s) = sif(stacked(s), C, tension)[1].plus[1] / sqrt(π * 0.5)
ss = range(0.1, 3.0; length = 59)
plot(ss, [shield.(Ref(C), ss) for C in (C_iso, C_ort, C_ort90)];
    lw = 2, label = ["isotropic" "orthotropic, stiff along the cracks" "orthotropic, stiff across the cracks"],
    xlabel = "spacing s / crack length", ylabel = "K_I / K_I⁰", legend = :bottomright, size = (700, 420))
````

The factor tends to 1 as the cracks move apart. The shielding depends on the
anisotropy: it is weaker in the plane stiffer along the cracks, stronger in
the one stiffer across them. Collinear cracks do not see it: the stress a loaded slit
creates on its own line, ahead of its tips, is the same in every material
[mauge1994](@cite):

````@example interacting_cracks
(collinear = maximum(abs, sif(pair(0.1), C_ort, tension)[2].minus - sif(pair(0.1), C_iso, tension)[2].minus),
    stacked = shield(C_ort, 0.4) - shield(C_iso, 0.4))
````

## An arbitrary array

Five cracks of random positions, lengths and orientations, under a remote
stress with a shear part, show the general use. Each line gives the factors
at the two tips, divided by ``\sqrt{\pi b}``:

````@example interacting_cracks
rng = Xoshiro(2024)
centers, half_lengths, angles = [rand(rng, 2) .* [3.0, 2.0] for _ in 1:5], 0.25 .+ 0.25rand(rng, 5), π .* rand(rng, 5)
array = [c => RibbonCrack(b; angle = θ) for (c, b, θ) in zip(centers, half_lengths, angles)]
Σ = Tens([0.2 0.3; 0.3 1.0])
println("crack   K_I(+b)  K_II(+b)   K_I(-b)  K_II(-b)   isolated (K_I, K_II)")
for (i, ((c, crack), r)) in enumerate(zip(array, sif(array, C_iso, Σ)))
    _, (KI, KII) = sif(crack, C_iso, Σ)                ## the same crack alone
    @printf("  %d     %7.3f  %7.3f    %7.3f  %7.3f     %7.3f  %7.3f\n", i, (vcat(r.plus, r.minus, [KI, KII]) ./ sqrt(π * crack.b))...)
end
````

The cracks, numbered as above:

````@example interacting_cracks
plot(; aspect_ratio = :equal, legend = false, xlabel = "x₁", ylabel = "x₂", size = (600, 450))
for (i, (c, b, θ)) in enumerate(zip(centers, half_lengths, angles))
    m = [cos(θ), sin(θ)]
    plot!([c[1] - b * m[1], c[1] + b * m[1]], [c[2] - b * m[2], c[2] + b * m[2]]; lw = 3, color = :black)
    annotate!((c + 0.12 * [-m[2], m[1]])..., text(string(i), 10))          ## beside the crack, along its normal
end
current()
````

Against the factors of each crack alone, the last two columns of the table,
the neighbors raise or lower the factors according to their positions, and
make the two tips of a crack differ. A negative ``K_{\mathrm I}``, which a
crack shielded enough by its neighbors may show, would mean that its faces
come into contact near that tip: the method, linear, does not prevent it.

## Where to go next

- [Interacting cracks](@ref th-interacting-cracks): the derivation of the
  method, of the fields of a loaded crack and of the factors at the tips.
- [Interacting cracks of a plane](@ref man-interacting-cracks): the syntax.
- [Cracks seen as fractures](@ref th-fracture-compliance): many cracks at
  once, through the schemes, as sets of fractures.
- [Cracks of the KTB gneiss read on its velocities](@ref app-ktb-cracks): the
  compliance of the cracks of a rock, read on its wave velocities.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

