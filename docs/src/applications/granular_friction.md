# [Friction of a granular medium from its contacts](@id app-granular-friction)

!!! info "Before this page"
    [Layered inclusions](@ref man-layered) for the spring interface of a
    composite sphere, [Layered sphere §7](@ref th-layered-sphere-jumps) for the
    strain its opening carries, and [Homogenization schemes](@ref th-homogenization)
    for the self-consistent estimate.

A sand resists shear in proportion to the pressure it is under: its strength is
frictional, and the macroscopic friction coefficient decreases as the sand gets
looser. This page reproduces the micromechanical explanation of
[maalej2009](@citet): the grains are rigid and only their contacts deform, and
each contact obeys a Coulomb friction law. The question is how the friction of
the contacts becomes the friction of the assembly, and why porosity matters so
much.

The sand is a porous polycrystal. Each grain is a rigid sphere of radius ``a``
surrounded by an interface of normal and tangential stiffnesses ``K_n`` and
``K_t``, and the pores are spherical; neither phase plays the role of a matrix,
which calls for the self-consistent scheme. The contacts fail by sliding,
``|\underline T_t| \le -\alpha\,T_n`` with ``\alpha`` the friction coefficient, and the
assembly is expected to follow a Drucker–Prager criterion
``\Sigma_d \le -T\,\Sigma_m``, with ``\Sigma_m = \mathrm{tr}\,\boldsymbol\Sigma/3`` and
``\Sigma_d = \sqrt{\boldsymbol\Sigma^{\mathrm d}:\boldsymbol\Sigma^{\mathrm d}/2}``. The macroscopic friction
coefficient ``T`` is what this page computes.

## 1. A rigid grain bonded by springs

```@example granular
using MeanFieldHomogenization, TensND, ForwardDiff, JSON, Printf, Plots
gr()  # headless backend; GKSwstype is set to "100" in make.jl

grain(ρ) = LayeredSphere(
    (1.0,), (iso_stiffness(Inf, 1.0e20),);       # a = 1, rigid core
    interfaces = (SpringInterface(1.0, ρ),),     # Kₙ = 1, Kₜ = ρ Kₙ
)
nothing # hide
```

The bulk modulus is infinite and the shear modulus ``10^{20}`` times the contact
stiffness, which makes the core rigid to twenty digits. The stiffnesses are
measured in units of ``a\,K_n``, so the moduli below depend on the porosity
``\varphi`` and on ``\rho = K_t/K_n`` only.

A rigid grain does not strain: all the strain of a grain is the opening of its
contacts, ``\frac{1}{|G|}\oint[\![\underline\xi]\!]\stackrel{s}{\otimes}\underline n\,\mathrm dS`` (their eq. 30). That
is exactly what the concentration tensor of a `LayeredSphere` counts by
default, the opening of the outer interface included.

## 2. The self-consistent moduli in closed form

The self-consistent estimate ``\mathbb C^{\mathrm{hom}} = 3k\,\mathbb J + 2\mu\,\mathbb K`` makes
the average stress and the average strain consistent when every phase is
embedded in ``\mathbb C^{\mathrm{hom}}`` itself:

```math
\mathbb C^{\mathrm{hom}}:\big[(1-\varphi)\,\mathbb A_{\mathrm G} + \varphi\,\mathbb A_{\mathrm P}\big]
 = (1-\varphi)\,\mathbb B_{\mathrm G},
```

``\mathbb A`` and ``\mathbb B`` being the strain and stress concentration tensors of the grain
``\mathrm G`` and of the pore ``\mathrm P``, whose stress vanishes. On ``\mathbb J`` and
``\mathbb K`` this is two scalar equations. Written with symbolic moduli, and with the
rigid grain obtained as ``k_{\mathrm s} = \mu_{\mathrm s} = 1/\epsilon``, ``\epsilon \to 0``, they
give back the two results of the article:

```@example granular
using SymPy
@syms k::positive μ::positive Kn::positive Kt::positive a::positive φ::positive
@syms ϵ::positive X::positive ρ::positive

C = iso_stiffness(k, μ)
G_sym = LayeredSphere((a,), (iso_stiffness(1 / ϵ, 1 / ϵ),); interfaces = (SpringInterface(Kn, Kt),))
αG, βG = get_data(strain_strain_loc(G_sym, C, C))
aG, bG = get_data(stress_strain_loc(G_sym, C, C))
αP, βP = get_data(strain_strain_loc(Ellipsoid(1.0), iso_stiffness(0, 0), C))

# The lowest order in ϵ of the numerator: the equation of the rigid grain.
function rigid(e)
    num, _ = sympy.fraction(sympy.together(e))
    c = sympy.Poly(num, ϵ).all_coeffs()
    return c[findlast(x -> x != 0, c)]
end

bulk = rigid(3k * ((1 - φ) * αG + φ * αP) - (1 - φ) * aG)
k_sc = simplify(solve(bulk, k)[1])
```

This is eq. (39) of the article, ``k = 4(1-\varphi)\,aK_n\,\mu/[3(4\mu + \varphi\,aK_n)]``.
Substituting it in the deviatoric equation and writing ``\mu = X\,aK_n`` and
``K_t = \rho K_n`` leaves a product of factors:

```@example granular
shear = rigid(subs(2μ * ((1 - φ) * βG + φ * βP) - (1 - φ) * bG, k => k_sc))
factors = [f for (f, _) in sympy.factor_list(subs(shear, μ => X * a * Kn, Kt => ρ * Kn))[2]]
```

All are positive but the last, the cubic (42) of the article, whose positive
root is the dimensionless shear modulus ``X = F(\rho, \varphi)``:

```math
128X^3 + 16\,[3\varphi + 2 + 2\rho(3\varphi-1)]\,X^2 + 2\,[3(3\varphi-1) + 2\rho(12\varphi-5)]\,X + 3\rho\,(2\varphi-1) = 0 .
```

## 3. The numerical estimate

The same scheme with numbers. The fixed point is approached slowly as
``\varphi`` comes close to ``1/3``, where the shear stiffness of a frictionless
assembly vanishes, hence a generous iteration count. The iteration starts from
moduli of the order of the contact stiffness ``K_t``: the solver keeps every
modulus above ``\sqrt{\epsilon_{\mathrm{mach}}}`` times its starting value, a guard against the
null fixed point, and the moduli of a loose assembly with soft contacts fall
far below that bound when the start is ``K_n``.

```@example granular
function moduli(φ, ρ)
    rve = RVE()
    add_phase!(rve, :G, grain(ρ), Dict(:C => iso_stiffness(1.0, 1.0)); fraction = 1 - φ)
    add_phase!(rve, :P, Ellipsoid(1.0), Dict(:C => iso_stiffness(0.0, 0.0)); fraction = φ)
    start = ForwardDiff.value(ρ)              # the scale of Kₜ, also under a Dual
    sc = SelfConsistent(init = iso_stiffness(start, start), abstol = 1.0e-16, reltol = 1.0e-13, maxiters = 2000)
    return k_mu(homogenize(rve, sc, :C))
end

cubic(X, φ, ρ) = 128X^3 + 16 * (3φ + 2 + 2ρ * (3φ - 1)) * X^2 +
    2 * (3 * (3φ - 1) + 2ρ * (12φ - 5)) * X + 3ρ * (2φ - 1)
for (φv, ρv) in ((0.34, 0.055), (0.40, 0.055), (0.45, 1.0e-3))
    kv, μv = moduli(φv, ρv)
    @printf("φ = %.2f, ρ = %.3g: k = %.6e (eq. 39: %.6e), cubic(μ) / (3ρ) = %.1e\n",
        φv, ρv, kv, 4 * (1 - φv) * μv / (3 * (4μv + φv)), cubic(μv, φv, ρv) / (3ρv))
end
```

## 4. From the contacts to the macroscopic friction

Two averages of the contact traction are needed, and both follow from the
macroscopic stress without solving anything else. The mean normal traction on
the contacts is ``\Sigma_m/[(1-\varphi)\chi]`` (their eq. 17), with
``\chi = (1-2\varphi)(1-\varphi)`` the fraction of the grain surface in contact
(eq. 27). The quadratic mean of the tangential traction follows from the
derivative of the elastic energy with respect to the contact compliance
``1/K_t`` (eq. 24), that is from ``\partial k/\partial\rho``. Requiring the Coulomb
law of the contacts on these averages (eq. 54) gives the Drucker–Prager slope
(eq. 57), with ``G = k/(aK_n)``:

```math
T = \sqrt{\frac{9\alpha^2\varphi}{4(1-\varphi)^3(1-2\varphi)}\,
  \frac{G^2}{\rho^2\,\partial G/\partial\rho} - \frac{3\varphi}{4(1-\varphi)}} .
```

The derivative is taken by automatic differentiation through the
self-consistent iteration, so it is exact to rounding:

```@example granular
function slope(φ, α, ρ)
    G = moduli(φ, ρ)[1]
    dG = ForwardDiff.derivative(r -> moduli(φ, r)[1], ρ)
    return sqrt(9α^2 * φ / (4 * (1 - φ)^3 * (1 - 2φ)) * G^2 / (ρ^2 * dG) - 3φ / (4 * (1 - φ)))
end
nothing # hide
```

Two failure modes are compared by the authors. In a **ductile** contact the
tangential stiffness vanishes as it slides, ``\rho \to 0``, and the limit is in
closed form (eq. 58):

```math
T = \sqrt{\frac{\alpha^2}{2(1-\varphi)^2(\varphi - 1/3)} - \frac{3\varphi}{4(1-\varphi)}}
\quad (\varphi > 1/3),
\qquad T = \infty \quad (\varphi < 1/3).
```

In a **brittle** contact the stiffness ratio keeps its elastic value until
failure. The friction coefficient ``\alpha = 0.27`` and the brittle ratio
``\rho = 0.055`` are the values the authors fit, and they compare both models with
the empirical rule of Caquot and Kérisel for uniform sands,
``\tan\phi_f = 0.55\,(1-\varphi)/\varphi``, turned into a Drucker–Prager slope for
triaxial compression by ``T = 2\sqrt 3\sin\phi_f/(3 - \sin\phi_f)`` (eqs. 64–65):

```@example granular
data = JSON.parsefile(joinpath(pkgdir(MeanFieldHomogenization), "data", "literature", "maalej2009.json"))["quantities"]
α = data["friction_coefficient"]["value"]
ρ_brittle = data["stiffness_ratio_brittle"]["value"]
c_ck = data["caquot_kerisel_coefficient"]["value"]

ductile_limit(φ) = sqrt(α^2 / (2 * (1 - φ)^2 * (φ - 1 / 3)) - 3φ / (4 * (1 - φ)))
function caquot_kerisel(φ)
    s = sin(atan(c_ck * (1 - φ) / φ))
    return 2 * sqrt(3) * s / (3 - s)
end

φs = range(0.335, 0.495; length = 33)   # eq. (57) is singular at φ = 1/2, where χ = 0
T_ductile = [slope(φ, α, 1.0e-6) for φ in φs]
T_brittle = [slope(φ, α, ρ_brittle) for φ in φs]
@printf("ductile, φ = 0.34: T = %.4f at ρ = 1e-6, %.4f in the limit (eq. 58)\n",
    slope(0.34, α, 1.0e-6), ductile_limit(0.34))

φf = range(0.335, 0.495; length = 200)
plot(φs, T_ductile; seriestype = :scatter, ms = 3, label = "ductile, ρ = 10⁻⁶",
    xlabel = "porosity φ", ylabel = "macroscopic friction T", ylims = (0, 6),
    title = "Drucker–Prager slope of a granular medium (α = $α)")
plot!(φf, ductile_limit.(φf); lw = 2, ls = :dash, color = 1, label = "ductile limit, eq. (58)")
plot!(φs, T_brittle; lw = 2, marker = :circle, ms = 2, color = 2, label = "brittle, ρ = $ρ_brittle")
plot!(φf, caquot_kerisel.(φf); lw = 2, color = :gray, label = "Caquot–Kérisel")
```

The ductile slope grows without bound as the porosity decreases to ``1/3``,
which the experimental trend does not do: the ductile model describes loose
sands only. The brittle model, with its finite ``\rho``, stays close to the
empirical rule over the whole range, which is the authors' conclusion that the
two mechanisms are complementary. Near ``\varphi = 1/3`` the self-consistent fixed
point is reached only after many iterations; an iteration capped too early
overestimates the ductile slope there.

The transition between the two modes is a matter of ``\rho`` alone, and the
whole family can be seen at once:

```@example granular
include(joinpath(pkgdir(MeanFieldHomogenization), "scripts", "common", "docviz.jl"))
φg = range(0.34, 0.5; length = 17)
lρ = range(-4, -0.5; length = 15)
Tg = [min(slope(φ, α, 10.0^l), 6.0) for l in lρ, φ in φg]
plotly_surface(φg, lρ, Tg; uid = "granular-slope-3d", title = "Macroscopic friction T(φ, ρ)",
    xlabel = "porosity φ", ylabel = "log₁₀ ρ", zlabel = "T", colorscale = "Viridis",
    reversescale = false, height = 480)
```

## Where to go next

Grains that are not rigid can fail too, and when they do the criterion is no
longer a cone: [The strength of a sandstone](@ref app-sandstone-strength)
combines the sliding contacts of this page with a von Mises criterion in the
grains, on the same self-consistent polycrystal.
