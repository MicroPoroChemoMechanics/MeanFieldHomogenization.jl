# [The strength of a sandstone: crushing grains and sliding contacts](@id app-sandstone-strength)

!!! info "Before this page"
    [Friction of a granular medium from its contacts](@ref app-granular-friction),
    the same polycrystal with rigid grains, and
    [Layered sphere §7](@ref th-layered-sphere-jumps) for the strain carried by
    the opening of an interface.

A sandstone is a sand whose grains have been cemented together. Its strength
involves two mechanisms: the grains can yield, and the cement between them can
slide. This page reproduces the micromechanical model of
[dormieux2010](@citet), in which the solid of the grains obeys a von Mises
criterion, ``\sigma_d \le K`` with ``\sigma_d = \sqrt{\boldsymbol\sigma^{\mathrm d}:\boldsymbol\sigma^{\mathrm d}/2}``, and the
cemented contacts a Mohr–Coulomb one, ``|\underline T_t| + \alpha\,T_n \le 0``. The
question is the macroscopic strength, in the plane of the mean stress
``\Sigma_m = \mathrm{tr}\,\boldsymbol\Sigma/3`` and of the deviatoric stress
``\Sigma_d = \sqrt{\boldsymbol\Sigma^{\mathrm d}:\boldsymbol\Sigma^{\mathrm d}/2}``.

!!! note "The normalization of this page"
    This page keeps the normalization of [dormieux2010](@citet), with
    ``\tfrac{1}{2}`` under the root, also for the deviatoric strain
    ``\bar\varepsilon_d`` below, so that its formulas and figures are those of the
    article. The norm used in the rest of this documentation carries no factor
    (see [Conventions](@ref th-notation-deviatoric)): it is ``\sqrt 2`` times the
    ``\sigma_d``, ``\Sigma_d`` and ``\bar\varepsilon_d`` of this page. In that norm the
    grains obey ``\sqrt{\boldsymbol\sigma^{\mathrm d}:\boldsymbol\sigma^{\mathrm d}} \le \sqrt 2\,K``,
    and every ratio ``\Sigma_d/|\Sigma_m|`` of this page, ``H`` and the slope of
    the Drucker–Prager limit, is multiplied by ``\sqrt 2``.

The model follows the modified secant method: each nonlinear behavior is
replaced by a linear one whose stiffness depends on the average strain it
undergoes, and the strength is reached when these stiffnesses vanish. The
linear problem behind it is therefore the heart of the model: an
incompressible grain of shear modulus ``\mu_{\mathrm s}``, bonded by an interface
rigid in the normal direction and of tangential stiffness ``k_t``, in a
self-consistent polycrystal with spherical pores of volume fraction ``\varphi``.

## 1. The grain and its interface

```@example sandstone
using MeanFieldHomogenization, TensND, ForwardDiff, Printf, Plots
gr()  # headless backend; GKSwstype is set to "100" in make.jl

brick(μs, kt) = LayeredSphere(
    (1.0,), (iso_stiffness(Inf, μs),);        # R = 1, incompressible grain
    interfaces = (SpringInterface(Inf, kt),), # rigid normally, kₜ tangentially
)
nothing # hide
```

The average strain of the grain is read on the outer lip of its interface
(their eq. 43): it includes the sliding of the contacts, which is what the
concentration tensor of a `LayeredSphere` counts by default. With ``R = 1``, the
behavior depends on the dimensionless contact stiffness ``\kappa = k_tR/\mu_{\mathrm s}``
and on ``\varphi``.

## 2. The self-consistent moduli in closed form

The self-consistent estimate ``\mathbb C^{\mathrm{hom}} = 3k\,\mathbb J + 2\mu\,\mathbb K`` solves
``\mathbb C^{\mathrm{hom}}:[(1-\varphi)\,\mathbb A_{\mathrm S} + \varphi\,\mathbb A_{\mathrm P}] = (1-\varphi)\,\mathbb B_{\mathrm S}``,
with the concentration tensors of the grain ``\mathrm S`` and of the pore
``\mathrm P`` computed in ``\mathbb C^{\mathrm{hom}}``. Kept general, with a compressible grain
and a compliant normal contact, the spherical equation is linear in ``k``:

```@example sandstone
using SymPy
@syms k::positive μ::positive ks::positive μs::positive kn::positive kt::positive
@syms R::positive φ::positive x::positive κ::positive ϵ::positive

C = iso_stiffness(k, μ)
S_sym = LayeredSphere((R,), (iso_stiffness(ks, μs),); interfaces = (SpringInterface(kn, kt),))
αS, βS = get_data(gradient_gradient_concentration(S_sym, C, C))
aS, bS = get_data(flux_gradient_concentration(S_sym, C, C))
αP, βP = get_data(gradient_gradient_concentration(Ellipsoid(1.0), iso_stiffness(0, 0), C))

k_sc = simplify(solve(3k * ((1 - φ) * αS + φ * αP) - (1 - φ) * aS, k)[1])
```

which is eq. (49) of the article. Substituted in the deviatoric equation, it
leaves a product of positive factors and of the cubic of eq. (50), the only
factor of degree three in ``\mu``:

```@example sandstone
shear = 2μ * ((1 - φ) * βS + φ * βP) - (1 - φ) * bS
num, _ = sympy.fraction(sympy.together(subs(shear, k => k_sc)))
[sympy.degree(f, μ) for (f, _) in sympy.factor_list(num)[2]]
```

The article then lets the grain and the normal contact become incompressible,
``k_{\mathrm s}, k_n \to \infty``. With ``k_{\mathrm s} = k_n = 1/\epsilon``, ``\mu = x\,\mu_{\mathrm s}`` and
``k_t = \kappa\mu_{\mathrm s}/R``, the lowest order in ``\epsilon`` of the deviatoric equation is
a positive multiple of the quadratic of eq. (52),
``16(5+\kappa)(3-\varphi)\,x^2 + [(9+77\varphi)\kappa + 114(3\varphi-1)]\,x + 57\kappa(2\varphi-1) = 0``:

```@example sandstone
function lowest_order(e)
    n, _ = sympy.fraction(sympy.together(e))
    c = sympy.Poly(n, ϵ).all_coeffs()
    return c[findlast(c -> c != 0, c)]
end
eq52 = lowest_order(subs(shear, k => k_sc, ks => 1 / ϵ, kn => 1 / ϵ))
eq52 = subs(eq52, μ => x * μs, kt => κ * μs / R)
q52 = 16 * (5 + κ) * (3 - φ) * x^2 + ((9 + 77φ) * κ + 114 * (3φ - 1)) * x + 57κ * (2φ - 1)
factor(eq52 / q52)
```

Its positive root is ``x = M(\kappa, \varphi)``, so that ``\mu = M\mu_{\mathrm s}`` and, from
the limit of (49), ``k = 4(1-\varphi)\mu/(3\varphi)`` (eqs. 51, 53). Perfect contacts
(``\kappa \to \infty``) give back the self-consistent porous polycrystal,
``M = 3(1-2\varphi)/(3-\varphi)`` (eq. 54): as ``\kappa`` grows, the quadratic is
dominated by its coefficient of ``\kappa``, which vanishes at that value.

```@example sandstone
simplify(subs(sympy.Poly(q52, κ).coeff_monomial(κ), x => 3 * (1 - 2φ) / (3 - φ)))
```

## 3. The numerical estimate

The self-consistent iteration keeps every modulus above
``\sqrt{\epsilon_{\mathrm{mach}}}`` times its starting value, a guard against the null fixed
point; above ``\varphi = 1/3`` the moduli fall to the order of ``k_tR`` as
``\kappa \to 0``, so the iteration starts from that scale when it is the smaller.

```@example sandstone
function moduli(μs, kt, φ)
    rve = RVE()
    add_phase!(rve, :S, brick(μs, kt), Dict(:C => iso_stiffness(1.0, 1.0)); fraction = 1 - φ)
    add_phase!(rve, :P, Ellipsoid(1.0), Dict(:C => iso_stiffness(0.0, 0.0)); fraction = φ)
    start = ForwardDiff.value(min(μs, kt))    # the scale of the moduli, also under a Dual
    sc = SelfConsistent(init = iso_stiffness(start, start), abstol = 1.0e-15, reltol = 1.0e-13, maxiters = 2000)
    return collect(k_mu(homogenize(rve, sc, :C)))
end

M52(κ, φ) = (a = 16 * (5 + κ) * (3 - φ); b = (9 + 77φ) * κ + 114 * (3φ - 1); c = 57κ * (2φ - 1);
    (-b + sqrt(b^2 - 4a * c)) / (2a))
for (κv, φv) in ((0.1, 0.35), (1.0, 0.25), (10.0, 0.30))
    kv, μv = moduli(1.0, κv, φv)
    @printf("κ = %4.1f, φ = %.2f: μ/μs = %.8f (eq. 52: %.8f), k/μ = %.6f (eq. 51: %.6f)\n",
        κv, φv, μv, M52(κv, φv), kv / μv, 4 * (1 - φv) / (3φv))
end
```

## 4. From the averages to the strength

Two averages drive the criteria, and both come from derivatives of the
effective moduli at fixed load (their eqs. 23 and 33):

- the quadratic mean ``\bar\varepsilon_d`` of the deviatoric strain in the grains,
  ``4(1-\varphi)\,\bar\varepsilon_d^2 = A\,\Sigma_m^2 + B\,\Sigma_d^2`` with
  ``A = k^{-2}\,\partial k/\partial\mu_{\mathrm s}`` and ``B = \mu^{-2}\,\partial\mu/\partial\mu_{\mathrm s}``;
- the quadratic mean ``\bar T_t`` of the tangential traction on the contacts,
  ``3(1-\varphi)^2(1-2\varphi)\,\bar T_t^2/R = C\,\Sigma_m^2 + D\,\Sigma_d^2`` with
  ``C = (k_t/k)^2\,\partial k/\partial k_t`` and ``D = (k_t/\mu)^2\,\partial\mu/\partial k_t``.

The mean normal traction on the contacts is
``\bar T_n = \Sigma_m/[(1-2\varphi)(1-\varphi)^2]`` (eq. 38), whatever the scheme. The four
derivatives are taken by automatic differentiation through the
self-consistent iteration, at ``\mu_{\mathrm s} = 1``:

```@example sandstone
function coefficients(κv, φv)
    k, μ = moduli(1.0, κv, φv)
    J = ForwardDiff.jacobian(p -> moduli(p[1], p[2], φv), [1.0, κv])
    return J[1, 1] / k^2, J[2, 1] / μ^2, κv^2 * J[1, 2] / k^2, κv^2 * J[2, 2] / μ^2
end
nothing # hide
```

**Grains alone.** When the grains yield, their secant shear modulus behaves as
``\mu_{\mathrm s} \sim K/(2\bar\varepsilon_d)``, and the first average gives
``(1-\varphi)K^2 = A\,\Sigma_m^2 + B\,\Sigma_d^2`` with ``A`` and ``B`` evaluated at
``\mu_{\mathrm s} = 1``: the moduli are proportional to ``\mu_{\mathrm s}``, so the criterion
does not depend on it. With perfect contacts this is the ellipse of the von
Mises porous solid (eq. 75),
``\frac{3\varphi}{4(1-\varphi)}\,\Sigma_m^2 + \Sigma_d^2 = M(\infty, \varphi)\,(1-\varphi)\,K^2``.

**Grains and contacts.** When the contacts slide too, the Mohr–Coulomb law
holds on the averages, ``\bar T_t + \alpha\,\bar T_n = 0``, and ``\kappa`` is no longer fixed:
it parameterizes the envelope (eq. 85). Eliminating ``\bar T_t`` gives the ratio
``H = \Sigma_d/|\Sigma_m|``, and the grain criterion then gives ``\Sigma_m``:

```math
H^2 = \frac{1}{D}\left[\frac{3\alpha^2}{(1-2\varphi)(1-\varphi)^2} - C\right],
\qquad
|\Sigma_m| = K\sqrt{\frac{1-\varphi}{A + B H^2}},
\qquad
\Sigma_d = H\,|\Sigma_m| ,
```

for ``\Sigma_m \le 0``, wherever ``H^2 \ge 0``. Two ends close the curve. As
``\kappa \to 0`` the contacts have no tangential stiffness left; above
``\varphi = 1/3`` the curve then tends to a finite point, and the contacts alone
fail along the Drucker–Prager line through the origin whose slope is that
limit of ``H`` (their eq. 81), the first part of the envelope. Below ``1/3``,
``H \to \infty`` and the curve starts on the deviatoric axis. At the other end,
either ``H^2`` vanishes at a finite ``\kappa`` and the curve comes down to the
mean-stress axis, or it stays positive and the curve meets the ellipse as
``\kappa \to \infty``. The first end is located by bisection between two values of
the grid.

```@example sandstone
κs = 10.0 .^ range(-8, 6; length = 141)
H²(c, φv, αv) = (3αv^2 / ((1 - 2φv) * (1 - φv)^2) - c[3]) / c[4]
function point(c, φv, αv)
    h² = max(H²(c, φv, αv), 0.0)
    Sm = sqrt((1 - φv) / (c[1] + c[2] * h²))
    return (Sm, Sm * sqrt(h²))
end
function envelope(φv, αv, coefs)
    pts = φv > 1 / 3 ? [(0.0, 0.0)] : Tuple{Float64, Float64}[]   # contacts alone
    for i in eachindex(κs)
        if H²(coefs[i], φv, αv) ≥ 0
            push!(pts, point(coefs[i], φv, αv))
        elseif i > 1                       # H² vanishes between κs[i-1] and κs[i]
            lo, hi = log10(κs[i - 1]), log10(κs[i])
            for _ in 1:40
                mid = (lo + hi) / 2
                H²(coefficients(10.0^mid, φv), φv, αv) ≥ 0 ? (lo = mid) : (hi = mid)
            end
            push!(pts, point(coefficients(10.0^lo, φv), φv, αv))
            break
        end
    end
    return pts
end
ellipse_radii(φv) = (c = coefficients(1.0e8, φv); (sqrt((1 - φv) / c[1]), sqrt((1 - φv) / c[2])))
plt = plot(; aspect_ratio = :equal, xlabel = "−Σm / K", ylabel = "Σd / K", legend = :topright,
    title = "Strength of a sandstone (K = 1)")
for (i, φv) in enumerate((0.35, 0.25))
    a, b = ellipse_radii(φv)
    t = range(0, π / 2; length = 100)
    plot!(plt, a .* cos.(t), b .* sin.(t); color = :red, lw = 2,
        label = i == 1 ? "grains alone (eq. 75)" : "")
    coefs = [coefficients(κv, φv) for κv in κs]
    for (j, αv) in enumerate((0.16, 0.20, 0.25, 0.30))
        pts = envelope(φv, αv, coefs)
        plot!(plt, first.(pts), last.(pts); color = :blue, lw = 1.5, marker = :cross, ms = 2,
            label = (i == 1 && j == 1) ? "grains and contacts (eq. 85)" : "")
    end
end
annotate!(plt, [(0.35, 0.05, text("φ = 0.35", 9)), (1.0, 0.05, text("φ = 0.25", 9))])
plt
```

For each porosity, the red ellipse is the strength of the porous solid when
only the grains yield; the blue curves, for ``\alpha = 0.16, 0.20, 0.25, 0.30`` from
the bottom up, are the strength when the contacts slide as well. The weaker
the friction, the lower the curve. Below ``\varphi = 1/3`` a polycrystal whose
contacts have lost all tangential stiffness still resists shear, ``M(0, \varphi) > 0``
(their eq. 56), so the curves leave the deviatoric axis at a finite stress;
above ``1/3`` it does not, and the envelope starts from the origin, with the
straight segment of the contacts failing alone, as in a sand. Frictional
contacts keep the curve up to the ellipse, which it meets as ``\kappa \to \infty``,
where the contacts stop sliding; weakly frictional ones bring it down to the
mean-stress axis first, so that a hydrostatic compression alone breaks the
rock before the grains reach their own limit.

The friction coefficient enters only through ``H``, so the whole family of
envelopes at ``\varphi = 0.25`` is one surface. Each envelope is built as in the
plane figure, its end included, then resampled at points evenly spaced along
its length, so that the edges of the surface are the envelopes and their ends
rather than the cells of the ``\kappa`` grid:

```@example sandstone
include(joinpath(pkgdir(MeanFieldHomogenization), "scripts", "common", "docviz.jl"))
φv = 0.25
coef = [coefficients(κv, φv) for κv in κs]
## n points evenly spaced along the polyline pts.
function resample(pts, n)
    s = vcat(0.0, cumsum([hypot((pts[k] .- pts[k - 1])...) for k in 2:length(pts)]))
    return map(range(0, s[end]; length = n)) do t
        k = clamp(searchsortedlast(s, t), 1, length(pts) - 1)
        w = s[k + 1] > s[k] ? (t - s[k]) / (s[k + 1] - s[k]) : 0.0
        (1 - w) .* pts[k] .+ w .* pts[k + 1]
    end
end
αs = range(0.12, 0.34; length = 23)
envs = [envelope(φv, αv, coef) for αv in αs]
pts = resample.(envs, 61)
grid = [pts[i][j] for i in eachindex(pts), j in 1:61]
Xs, Zs = first.(grid), last.(grid)
Ys = [αv for αv in αs, _ in 1:61]
surface = surface_trace(Xs, Ys, Zs; color = "#2c7fb8", opacity = 0.9)
rim = line_trace(
    hcat(permutedims(hcat(Xs[1, :], Ys[1, :], Zs[1, :])), permutedims(hcat(Xs[:, end], Ys[:, end], Zs[:, end])),
        permutedims(hcat(reverse(Xs[end, :]), reverse(Ys[end, :]), reverse(Zs[end, :]))),
        permutedims(hcat(reverse(Xs[:, 1]), reverse(Ys[:, 1]), reverse(Zs[:, 1]))));
    color = FIG_VECTOR, width = 4)
nothing # hide
```

Two families of planes cut it. A plane ``\alpha = \text{const}`` gives back the
envelope of that friction coefficient, as in the plane figure above. A plane
``\Sigma_m = \text{const}`` gives the strength in shear as a function of the
friction at a fixed confinement, read on each envelope where it crosses that
mean stress; it is empty where the rock breaks under that hydrostatic
compression alone, whatever the shear.

```@example sandstone
xm, zm = maximum(Xs), maximum(Zs)
## Plane α = const: the envelope itself.
αc = 0.12:0.02:0.34
envc = [envelope(φv, αv, coef) for αv in αc]
friction = CutFamily(
    "Plane α = const", "α", string.(αc),
    [plane_patch([xm / 2, αv, zm / 2], [1, 0, 0], [0, 0, 1], 0.55xm, 0.55zm) for αv in αc],
    [vcat(first.(e)', fill(αv, 1, length(e)), last.(e)') for (αv, e) in zip(αc, envc)],
    [vcat(first.(e)', last.(e)') for e in envc],
    ("−Σm / K", "Σd / K"); active = 5,
)
## Plane −Σm = c: Σd where each envelope crosses that mean stress.
function crossing(e, c)
    for k in 1:(length(e) - 1)
        (x0, z0), (x1, z1) = e[k], e[k + 1]
        (x0 - c) * (x1 - c) ≤ 0 && x1 != x0 && return z0 + (c - x0) / (x1 - x0) * (z1 - z0)
    end
    return NaN
end
cs = 0.1:0.1:floor(xm; digits = 1)
shear = [[crossing(e, c) for e in envs] for c in cs]
confinement = CutFamily(
    "Plane Σm = const", "−Σm / K", string.(cs),
    [plane_patch([c, (first(αs) + last(αs)) / 2, zm / 2], [0, 1, 0], [0, 0, 1], 0.55 * (last(αs) - first(αs)), 0.55zm) for c in cs],
    [vcat(fill(c, 1, length(αs)), collect(αs)', d') for (c, d) in zip(cs, shear)],
    [vcat(collect(αs)', d') for d in shear],
    ("friction α", "Σd / K"); active = 5, equal = false,
)
section_figure([surface, rim], [friction, confinement];
    uid = "sandstone-envelopes-3d", height = 620, aspectmode = "cube",
    xlabel = "−Σm / K", ylabel = "friction α", zlabel = "Σd / K",
    title = "Strength envelopes of a sandstone, φ = 0.25")
```

The buttons above the surface choose a family of planes, or none, and the slider
below moves the plane; on the right, the section is drawn in coordinates of its
plane.

## Where to go next

The rigid-grain limit of this model, with frictional contacts only, is
[Friction of a granular medium from its contacts](@ref app-granular-friction).
The secant reasoning on the energy derivatives is the same as in
[Quasi-brittle strength of cement paste and mortar](@ref app-strength), where it
is applied to a brittle solid. The next application,
[Viscoelastic complex modulus of a bituminous mixture](@ref app-bituminous),
turns to time-dependent behavior.
