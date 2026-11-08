# # Permeability from Stokes flow: Brinkman inhomogeneities
#
# This script follows the method of [bignonnetIJES2022](@citet) for estimating
# the permeability of a porous medium from the Stokes flow in its pores. It has
# two parts.
#
# 1. **Symbolic** (§1–§4). TensND's curvilinear differential operators and SymPy
#    solve the Brinkman inhomogeneity problems in closed form: a sphere, an
#    infinite cylinder, a planar layer and a coated sphere. Each solution is
#    compared with the `Brinkman*` inclusions of the package. Then the
#    symbolic consequences: the dilute and Mori–Tanaka estimates, the
#    percolation threshold of the self-consistent scheme, its dilute expansion,
#    and Stokes' paradox.
# 2. **Numerical** (§5–§9). The schemes of the package (dilute, Mori–Tanaka,
#    self-consistent, differential, Maxwell) are applied to granular, fibrous
#    and tubular media and to a double-porosity medium, with sensitivities by
#    automatic differentiation.
#
# The idea in one paragraph. At the scale of the pores the flow obeys Stokes'
# equations and vanishes in the solid; at the scale of the material it obeys
# Darcy's law ``\underline{V} = -(\boldsymbol{k}/\mu)\cdot\nabla P``. Extending
# the Stokes problem to the solid by a body force ``\underline{f}`` (zero in the
# fluid, of mean ``\nabla P``) turns the permeability into a volume average of a
# concentration tensor, which splits over the phases as in elasticity. Each phase
# average is estimated from an *inhomogeneity problem*: one inclusion embedded in
# an unbounded medium where the flow obeys Brinkman's equation
# ``-\nabla p + \mu\Delta\underline{v} - \mu k_0^{-1}\underline{v} = \underline{0}``.
# Its solution depends on the size ``R`` of the inclusion through
# ``\alpha_0 = R/\sqrt{k_0}``, and there is no Hill tensor. The schemes themselves
# are those of conduction.
#
# **§0** Setup · **§1** Sphere · **§2** Cylinder and layer · **§3** Coated
# sphere · **§4** Symbolic consequences · **§5** Granular media · **§6** Fibrous
# media · **§7** Spanning pores · **§8** Double porosity · **§9** Sensitivities.

import Pkg                                                          #jl
Pkg.activate(joinpath(@__DIR__, "..", "docs"); io = devnull)        #jl

using MeanFieldHomogenization
using TensND
using SymPyPythonCall
using SpecialFunctions
using ForwardDiff
using LinearAlgebra
using Printf
using Plots

# ## §0 Setup
#
# Two helpers: `iso(x)` is the isotropic second-order tensor ``x\boldsymbol{1}``, and
# `val(t)` reads the scalar of an isotropic tensor. `bisect` finds the root of a
# scalar equation from a sign change; it serves as an independent reference for
# the closed-form estimates of the paper.

println("="^78)                                                     #jl
println("  § 0  SETUP")                                             #jl
println("="^78)                                                     #jl

iso(x) = TensISO{3}(x)
val(t) = first(TensND.get_data(t))
function bisect(f, a, b; n = 200)
    fa = f(a)
    for _ in 1:n
        m = (a + b) / 2
        fm = f(m)
        (fm > 0) == (fa > 0) ? (a = m; fa = fm) : (b = m)
    end
    return (a + b) / 2
end
## Numerical value of a SymPy expression at given values of its symbols.
num(x, vals...) = Float64(N(subs(x, vals...)))

# ## §1 The Brinkman sphere, solved symbolically
#
# A sphere of radius ``R`` and permeability ``k_1 = \ell_1^2`` sits in a Brinkman
# medium of permeability ``k_0 = \ell_0^2``, under a uniform remote velocity
# ``V\underline{e}_3``. The angular dependence is fixed by the load, and
# incompressibility then fixes the orthoradial velocity from the radial one:
#
# ```math
# \underline{v} = u(r)\cos\theta\,\underline{e}_r
#   - \Big(u + \frac{r u'}{2}\Big)\sin\theta\,\underline{e}_\theta ,
# \qquad p = P(r)\cos\theta .
# ```
#
# Everything below is produced by the operators of TensND (`symgrad`,
# `divergence`) in spherical coordinates. The route is that of TensND's tutorial
# on the elastic sphere; nothing is transcribed from a table.

println("="^78)                                                     #jl
println("  § 1  BRINKMAN SPHERE — SYMBOLIC")                       #jl
println("="^78)                                                     #jl

Spherical = coorsys_spherical()
θ, ϕ, r = get_coords(Spherical)
𝐞ᶿ, 𝐞ᵠ, 𝐞ʳ = unitvec(Spherical)
@set_coorsys Spherical
𝟏 = tens_Id2(Val(3), Val(Sym))
μ, ℓ, R = symbols("μ ℓ R", positive = true)
u = SymFunction("u", real = true)
P = SymFunction("P", real = true)

velocity(ue) = ue * cos(θ) * 𝐞ʳ - (ue + r * diff(ue, r) / 2) * sin(θ) * 𝐞ᶿ
𝐯 = velocity(u(r))
println("div v = ", simplify(divergence(𝐯)))

# The stress is that of the Newtonian fluid, and the Brinkman drag
# ``-\mu\ell^{-2}\underline{v}`` acts as a body force:

𝛔 = -P(r) * cos(θ) * 𝟏 + 2μ * symgrad(𝐯)
balance = divergence(𝛔) - μ / ℓ^2 * 𝐯
eʳ = simplify(balance ⋅ 𝐞ʳ / cos(θ))
eᶿ = simplify(balance ⋅ 𝐞ᶿ / sin(θ))

# The angle drops out of both components. The orthoradial one gives the pressure,
# and the radial one becomes a fourth-order equation for ``u``:

Psol = solve(eᶿ, P(r))[1]
ode = simplify(subs(eʳ, P(r) => Psol).doit() * r)
println("radial equation: ", ode)

# Four independent solutions: two are those of Stokes flow (``1``, ``r^{-3}``) and
# two carry the Brinkman screening length ``\ell``. Their Wronskian does not
# vanish, so they span all solutions:

basis = [Sym(1), r^-3, (ℓ - r) * exp(r / ℓ) / r^3, (ℓ + r) * exp(-r / ℓ) / r^3]
@assert all(iszero(simplify(subs(ode, u(r) => b).doit())) for b in basis)
W = simplify(det(Sym[diff(b, r, k) for k in 0:3, b in basis]))
println("Wronskian: ", W)

# Inside the sphere the solution must be regular at the origin. ``r^{-3}`` is
# excluded, and so is each exponential term alone: only their *difference*
# stays finite.

println("regular combination near 0: ", series(((ℓ - r) * exp(r / ℓ) - (ℓ + r) * exp(-r / ℓ)) / r^3, r, 0, 3))

# The velocity, its radial derivative and the two traction components
# ``\underline{e}_r\cdot\boldsymbol{\sigma}\cdot\underline{e}_r`` and
# ``\underline{e}_\theta\cdot\boldsymbol{\sigma}\cdot\underline{e}_r`` are continuous at ``r = R``.

function radial_fields(ue, Pe)
    t = (-Pe * cos(θ) * 𝟏 + 2μ * symgrad(velocity(ue))) ⋅ 𝐞ʳ
    return (ue, diff(ue, r), simplify((t ⋅ 𝐞ʳ) / cos(θ)), simplify(-(t ⋅ 𝐞ᶿ) / sin(θ)))
end
brinkman_fields(ue, ℓe) = radial_fields(ue, subs(subs(Psol, ℓ => ℓe), u(r) => ue).doit())

ℓ₀, ℓ₁ = symbols("ℓ₀ ℓ₁", positive = true)
a₁, a₃, b₂, b₄ = symbols("a₁ a₃ b₂ b₄")
u_in = a₁ + a₃ * ((ℓ₁ - r) * exp(r / ℓ₁) - (ℓ₁ + r) * exp(-r / ℓ₁)) / r^3
u_out = 1 + b₂ / r^3 + b₄ * (ℓ₀ + r) * exp(-r / ℓ₀) / r^3     # unit remote velocity
F_in, F_out = brinkman_fields(u_in, ℓ₁), brinkman_fields(u_out, ℓ₀)
sol = solve([subs(F_in[k] - F_out[k], r => R) for k in 1:4], [a₁, a₃, b₂, b₄])

# The average velocity over a ball equals ``u(R)`` (divergence theorem:
# ``\langle\underline{v}\rangle = |\omega|^{-1}\int_{\partial\omega}\underline{x}\,(\underline{v}\cdot\underline{n})\,dS``),
# so the velocity concentration of the porous sphere is:

α₀, α₁ = symbols("α₀ α₁", positive = true)
A_porous = simplify(subs(subs(subs(u_in, r => R), sol), ℓ₀ => R / α₀, ℓ₁ => R / α₁))

# The package evaluates the same closed form, rewritten so as to stay accurate
# when ``\alpha_1 \to 0``. Compared at a few points:

for (a, b) in ((0.4, 2.0), (3.0, 0.7), (12.0, 25.0))
    r₀, r₁ = (a / 1.3)^2, (b / 1.3)^2
    A_mfh = val(gradient_gradient_concentration(BrinkmanSphere(1.3), iso(r₁), iso(r₀)))
    @printf("  α₀ = %5.2f, α₁ = %5.2f :  symbolic %.15f   package %.15f\n", a, b, num(A_porous, α₀ => a, α₁ => b), A_mfh)
end

# **Darcy's limit.** For a sphere much larger than both screening lengths
# (``R \to \infty`` at fixed ``t = \alpha_1/\alpha_0 = \sqrt{k_0/k_1}``) the
# concentration is that of Darcy flow, ``3k_1/(k_1 + 2k_0)``:

t = symbols("t", positive = true)
println("Darcy limit: ", simplify(limit(subs(A_porous, α₁ => α₀ * t), α₀ => oo)))

# **Stokes sphere.** Inside a clear fluid the screening terms disappear, and the
# regular Stokes solution is ``u = a_1 + a_4 r^2``:

Psol_stokes = subs(expand(Psol), ℓ => oo)
a₄ = symbols("a₄")
u_S = a₁ + a₄ * r^2
F_S = radial_fields(u_S, subs(Psol_stokes, u(r) => u_S).doit())
sol_S = solve([subs(F_S[k] - F_out[k], r => R) for k in 1:4], [a₁, a₄, b₂, b₄])
A_stokes = factor(simplify(subs(subs(subs(u_S, r => R), sol_S), ℓ₀ => R / α₀)))
println("Stokes sphere: A = ", A_stokes)

# **Impervious sphere.** The velocity vanishes at ``r = R``, and the average drag
# density is the force on the sphere divided by its volume,
# ``\langle\boldsymbol{r}\cdot\boldsymbol{A}\rangle = (T_r + 2T_\theta)/(\mu R)``
# per unit remote velocity. This is Brinkman's drag:

sol_I = solve([subs(F_out[1], r => R), subs(F_out[2], r => R)], [b₂, b₄])
drag = factor(simplify(subs(subs((F_out[3] + 2F_out[4]) / (μ * R), r => R), sol_I)))
println("impervious sphere: ⟨r·A⟩ = ", simplify(subs(drag, ℓ₀ => R / α₀)))
@assert iszero(simplify(subs(A_stokes, α₀ => Sym(2)) - Sym(3) * (15 + 30 + 24 + 8) / (45 + 90 + 24 + 8)))

# ## §2 The cylinder and the layer
#
# An infinite cylinder of axis ``\underline{e}_z`` carries two problems. Under an
# axial body force the velocity is axial, ``w(r)\,\underline{e}_z``, and obeys a
# modified Bessel equation. Under a transverse remote velocity
# ``\underline{v} = u(r)\cos\theta\,\underline{e}_r - (r u)'\sin\theta\,\underline{e}_\theta``
# (divergence-free), it has the separable solutions ``1``, ``r^{-2}``,
# ``K_1(r/\ell)/r`` and ``I_1(r/\ell)/r``.

println("="^78)                                                     #jl
println("  § 2  CYLINDER AND LAYER — SYMBOLIC")                    #jl
println("="^78)                                                     #jl

Cyl = coorsys_cylindrical()
ρc, θc, z = get_coords(Cyl)
𝐞ᴿ, 𝐞ᶿᶜ, 𝐞ᶻ = unitvec(Cyl)
@set_coorsys Cyl
F = symbols("F", positive = true)
w = SymFunction("w", real = true)

σ_axial(we) = -F * z * 𝟏 + 2μ * symgrad(we * 𝐞ᶻ)
axial = simplify((divergence(σ_axial(w(ρc))) - μ / ℓ^2 * w(ρc) * 𝐞ᶻ) ⋅ 𝐞ᶻ)
println("axial equation: ", axial)

c₁, c₂ = symbols("c₁ c₂")
w_in = -F * ℓ₁^2 / μ + c₁ * besseli(0, ρc / ℓ₁)
w_out = -F * ℓ₀^2 / μ + c₂ * besselk(0, ρc / ℓ₀)
@assert iszero(simplify(subs(axial, ℓ => ℓ₁, w(ρc) => w_in).doit()))
@assert iszero(simplify(subs(axial, ℓ => ℓ₀, w(ρc) => w_out).doit()))
τ_axial(we) = (σ_axial(we) ⋅ 𝐞ᴿ) ⋅ 𝐞ᶻ
sol_ax = solve([subs(w_in - w_out, ρc => R), subs(τ_axial(w_in) - τ_axial(w_out), ρc => R)], [c₁, c₂])
V₀ = -F * ℓ₀^2 / μ
A_n = simplify(subs(simplify(integrate(subs(w_in, sol_ax) * 2ρc, (ρc, 0, R)) / R^2 / V₀), ℓ₀ => R / α₀, ℓ₁ => R / α₁))
println("axial concentration: A_n = ", A_n)

# The transverse problem, by the same steps as for the sphere:

ut = SymFunction("u", real = true)
Pt = SymFunction("P", real = true)
velocity_t(ue) = ue * cos(θc) * 𝐞ᴿ - diff(ρc * ue, ρc) * sin(θc) * 𝐞ᶿᶜ
σ_t(ue, Pe) = -Pe * cos(θc) * 𝟏 + 2μ * symgrad(velocity_t(ue))
bal_t = divergence(σ_t(ut(ρc), Pt(ρc))) - μ / ℓ^2 * velocity_t(ut(ρc))
Psol_t = solve(simplify(bal_t ⋅ 𝐞ᶿᶜ / sin(θc)), Pt(ρc))[1]
ode_t = simplify(subs(simplify(bal_t ⋅ 𝐞ᴿ / cos(θc)), Pt(ρc) => Psol_t).doit())
@assert all(iszero(simplify(subs(ode_t, ut(ρc) => b).doit())) for b in (Sym(1), ρc^-2, besselk(1, ρc / ℓ) / ρc, besseli(1, ρc / ℓ) / ρc))
function fields_t(ue, ℓe)
    Pe = subs(subs(Psol_t, ℓ => ℓe), ut(ρc) => ue).doit()
    tr_ = σ_t(ue, Pe) ⋅ 𝐞ᴿ
    return (ue, diff(ue, ρc), simplify((tr_ ⋅ 𝐞ᴿ) / cos(θc)), simplify(-(tr_ ⋅ 𝐞ᶿᶜ) / sin(θc)))
end
d₁, d₄, e₂, e₃ = symbols("d₁ d₄ e₂ e₃")
ut_in = d₁ + d₄ * besseli(1, ρc / ℓ₁) / ρc
ut_out = 1 + e₂ / ρc^2 + e₃ * besselk(1, ρc / ℓ₀) / ρc
G_in, G_out = fields_t(ut_in, ℓ₁), fields_t(ut_out, ℓ₀)
sol_t = solve([subs(G_in[k] - G_out[k], ρc => R) for k in 1:4], [d₁, d₄, e₂, e₃])
A_t = simplify(subs(subs(subs(ut_in, ρc => R), sol_t), ℓ₀ => R / α₀, ℓ₁ => R / α₁))

for (a, b) in ((0.5, 3.0), (4.0, 0.3))
    r₀, r₁ = a^2, b^2
    A = canonical_components(gradient_gradient_concentration(BrinkmanCylinder(1.0), iso(r₁), iso(r₀)))
    @printf(
        "  α₀ = %.1f, α₁ = %.1f :  Aₙ %.14f / %.14f   Aₜ %.14f / %.14f\n", a, b,
        num(A_n, α₀ => a, α₁ => b), A[3, 3], num(A_t, α₀ => a, α₁ => b), A[1, 1]
    )
end

# **The layer** ``|z| < h`` needs no curvilinear system. Under an in-plane body
# force the velocity is ``w(z)``, and ``\mu w'' - \mu w/\ell^2 = F`` holds in each
# medium. The flow normal to the layer is uniform, so ``A_n = 1``:

zc = symbols("z", real = true)
h = symbols("h", positive = true)
f₁, f₂ = symbols("f₁ f₂")
wl_in = ℓ₁^2 / ℓ₀^2 + f₁ * cosh(zc / ℓ₁)       # unit remote velocity, symmetric
wl_out = 1 + f₂ * exp(-zc / ℓ₀)
sol_l = solve([subs(wl_in - wl_out, zc => h), subs(diff(wl_in - wl_out, zc), zc => h)], [f₁, f₂])
A_layer = simplify(subs(integrate(subs(wl_in, sol_l), (zc, 0, h)) / h, ℓ₀ => h / α₀, ℓ₁ => h / α₁))
for (a, b) in ((0.5, 3.0), (4.0, 0.3))
    A = canonical_components(gradient_gradient_concentration(BrinkmanLayer(1.0), iso(b^2), iso(a^2)))
    @printf("  layer α₀ = %.1f, α₁ = %.1f :  Aₜ %.14f / %.14f\n", a, b, num(A_layer, α₀ => a, α₁ => b), A[1, 1])
end

# Its limits are those of the clear-fluid layer, ``1 + \alpha_0 + \alpha_0^2/3``,
# and of Darcy flow, ``k_1/k_0 = \alpha_0^2/\alpha_1^2``:

println("fluid layer: ", simplify(limit(A_layer, α₁ => 0)))
println("Darcy layer: ", simplify(limit(subs(A_layer, α₁ => α₀ * t), α₀ => oo)))

# ## §3 The coated sphere
#
# An impervious core of radius ``R`` wrapped in a Stokes shell up to ``R/\eta`` is
# the inclusion of the classical cell models, here embedded in a Brinkman
# medium. Six conditions (no slip on the core, continuity at ``R/\eta``) fix the
# six constants. The averages run over the whole coated sphere.

println("="^78)                                                     #jl
println("  § 3  COATED SPHERE")                                     #jl
println("="^78)                                                     #jl

@set_coorsys Spherical
η = symbols("η", positive = true)
s₁, s₂, s₃, s₄ = symbols("s₁ s₂ s₃ s₄")
u_sh = s₁ + s₂ / r^3 + s₃ / r + s₄ * r^2                  # Lamb's solution of Stokes flow
F_sh = radial_fields(u_sh, subs(Psol_stokes, u(r) => u_sh).doit())
R_out = R / η
conds = vcat([subs(F_sh[1], r => R), subs(F_sh[2], r => R)], [subs(F_sh[k] - F_out[k], r => R_out) for k in 1:4])
sol_C = solve(conds, [s₁, s₂, s₃, s₄, b₂, b₄])
A_coat = factor(simplify(subs(subs(subs(u_sh, r => R_out), sol_C), ℓ₀ => R / α₀)))
drag_coat = factor(simplify(subs(subs(subs((F_sh[3] + 2F_sh[4]) * R^2 / (μ * R_out^3), r => R), sol_C), ℓ₀ => R / α₀)))
for (a, e) in ((0.5, 1 / 3), (3.0, 2 / 3))
    c = BrinkmanCoatedSphere(1.0, 1 / e)
    A = val(gradient_gradient_concentration(c, iso(0.0), iso(a^2)))
    rA = val(flux_gradient_concentration(c, iso(0.0), iso(a^2)))
    @printf(
        "  α₀ = %.1f, η = %.3f :  A %.14f / %.14f   ⟨r·A⟩ %.14f / %.14f\n", a, e,
        num(A_coat, α₀ => a, η => e), A, num(drag_coat, α₀ => a, η => e, R => 1), rA
    )
end

# ## §4 Symbolic consequences
#
# **Dilute and Mori–Tanaka estimates.** With the clear fluid as matrix
# (``\boldsymbol{r}_0 = \boldsymbol{0}``, so ``\alpha_0 = 0``) and impervious spheres at volume fraction
# ``f = 1 - \varphi``, both schemes accept symbols and return the estimates in
# closed form:

println("="^78)                                                     #jl
println("  § 4  SYMBOLIC CONSEQUENCES")                             #jl
println("="^78)                                                     #jl

fs = symbols("f", positive = true)
rve_sym = RVE(T = Sym)
add_phase!(rve_sym, :p, BrinkmanSphere(R, BrinkmanFluid()), Dict(:r => iso(Sym(0))); fraction = :rest)
add_phase!(rve_sym, :s, BrinkmanSphere(R, BrinkmanSolid()), Dict(:r => iso(Sym(0))); fraction = fs)
println("dilute      r = ", simplify(val(homogenize(rve_sym, Dilute(:p), :r))))
println("Mori–Tanaka r = ", simplify(val(homogenize(rve_sym, MoriTanaka(:p), :r))))

# **Self-consistent equation.** The self-consistent scheme is the condition
# ``\sum_i \varphi_i \boldsymbol{R}_i = \boldsymbol{0}`` on the resistance contribution tensors
# ``\boldsymbol{R}_i = \langle\boldsymbol{r}\cdot\boldsymbol{A}\rangle_i - \boldsymbol{r}_0\cdot\boldsymbol{A}_i``, evaluated in the
# effective medium itself. Built here as a tensor from the symbolic concentrations
# of the package, with spherical pores and grains of the same radius and
# ``\alpha = R\sqrt{r}``:

α, φs = symbols("α φ", positive = true)
r_sc = α^2 / R^2
contribution(i, P₁) = flux_gradient_concentration(i, P₁, iso(r_sc)) - iso(r_sc) ⋅ gradient_gradient_concentration(i, P₁, iso(r_sc))
residual = (1 - φs) * contribution(BrinkmanSphere(R, BrinkmanSolid()), iso(Sym(0))) +
    φs * contribution(BrinkmanSphere(R, BrinkmanFluid()), iso(Sym(0)))
sc_equation = factor(simplify(val(residual) * R^2))
println("self-consistent equation: ", sc_equation, " = 0")

# The resistance diverges, ``\alpha \to \infty``, when the porosity falls to the
# percolation threshold of spherical pores:

φc = solve(limit(sc_equation / α^2, α => oo), φs)
println("percolation threshold: φ_c = ", φc)

# Near the dilute limit (``x = 1 - \varphi \to 0``) the resistance departs from the
# dilute one by a term in ``\sqrt{x}``. The coefficient below is the one that the
# multiple-scattering analyses of random beds of spheres predict:

xs, τs = symbols("x τ", positive = true)                  # x = τ²
c1, c2 = symbols("c1 c2")
eq_x = subs(sc_equation, φs => 1 - τs^2, α => c1 * τs + c2 * τs^2)
ser = expand(series(eq_x / τs^2, τs, 0, 2).removeO())
coef = solve([ser.coeff(τs, 0), ser.coeff(τs, 1)], [c1, c2])
coef = [cc for cc in coef if N(cc[1]) > 0][1]
ratio = series(2 * (coef[1] * τs + coef[2] * τs^2)^2 / (9τs^2), τs, 0, 2)
println("r_sc / r_dil = ", ratio, "   (τ = √(1 − φ))")

# **Stokes' paradox.** The drag on an impervious cylinder vanishes as the
# reference tends to a clear fluid, so no dilute, Mori–Tanaka or differential
# estimate exists for fibers in a fluid matrix: the package refuses it.

drag_cyl = α₀ * (2 + α₀ * besselk(0, α₀) / besselk(1, α₀)) / (besselk(0, α₀) / besselk(1, α₀))
println("impervious cylinder drag R²⟨r·A⟩ₙ as α₀ → 0: ", limit(drag_cyl, α₀ => 0))

# ## §5 Granular media
#
# Impervious spheres of radius ``R`` in a pore space of porosity ``\varphi``. The
# phases are described in the resistance form. The pore phase is the matrix of
# the dilute, Mori–Tanaka and differential schemes, and the self-consistent scheme
# needs a shape for the pores as well. Following [bignonnetIJES2022](@citet), the
# pore radius is set so that both inclusions show the same specific surface
# (``R_p = R`` for spheres, ``R_p = 2R/3`` for cylinders).

println("="^78)                                                     #jl
println("  § 5  GRANULAR MEDIA")                                    #jl
println("="^78)                                                     #jl

Rg = 1.0
sc = SelfConsistent(algorithm = NewtonDefault(), init = iso(1.0), reltol = 1.0e-12)
function granular(φ, pore)
    rve = RVE()
    add_phase!(rve, :p, pore, Dict(:r => iso(0.0)); fraction = :rest, symmetrize = :iso)
    add_phase!(rve, :s, BrinkmanSphere(Rg, BrinkmanSolid()), Dict(:r => iso(0.0)); fraction = 1 - φ)
    return rve
end
## Effective-medium approximation (eq. 65): r = ⟨r·A⟩ of the coated sphere in r.
function ema(φ)
    c = BrinkmanCoatedSphere(Rg, Rg / cbrt(1 - φ))
    return bisect(x -> val(flux_gradient_concentration(c, iso(0.0), iso(x))) - x, 1.0e-8, 1.0e8)
end
## Brinkman's original estimate (eq. 64): the pore phase does not enter.
brinkman1947(φ) = bisect(x -> (1 - φ) * 3 * (3 + 3Rg * sqrt(x) + Rg^2 * x) / (2Rg^2) - x, 1.0e-8, 1.0e8)
## Cell models (eq. 66), KUBC and SUBC (Happel), η = (1 − φ)^{1/3}.
k_kubc(φ) = (η = cbrt(1 - φ); (1 - η)^3 * (4 + 7η + 4η^2) / (18η^3 * (1 + η + η^2 + η^3 + η^4)) * Rg^2)
k_subc(φ) = (η = cbrt(1 - φ); (1 - η)^3 * (1 + η) * (2 + η + 2η^2) / (3η^3 * (3 + 2η^5)) * Rg^2)

φs_g = range(0.35, 0.95; length = 31)
estimates = Dict{String, Vector{Float64}}()
estimates["dilute"] = [1 / val(homogenize(granular(φ, BrinkmanSphere(Rg, BrinkmanFluid())), Dilute(:p), :r)) for φ in φs_g]
estimates["Mori–Tanaka"] = [1 / val(homogenize(granular(φ, BrinkmanSphere(Rg, BrinkmanFluid())), MoriTanaka(:p), :r)) for φ in φs_g]
estimates["differential"] = [1 / val(homogenize(granular(φ, BrinkmanSphere(Rg, BrinkmanFluid())), DifferentialScheme(:p), :r)) for φ in φs_g]
estimates["SC, spherical pores"] = [φ > 1 / 3 + 0.02 ? 1 / val(homogenize(granular(φ, BrinkmanSphere(Rg, BrinkmanFluid())), sc, :r)) : NaN for φ in φs_g]
estimates["SC, cylindrical pores"] = [1 / val(homogenize(granular(φ, BrinkmanCylinder(2Rg / 3, BrinkmanFluid())), sc, :r)) for φ in φs_g]
estimates["SC, coated sphere"] = map(φs_g) do φ
    rve = RVE()
    add_phase!(rve, :c, BrinkmanCoatedSphere(Rg, Rg / cbrt(1 - φ)), Dict(:r => iso(0.0)); fraction = 1.0)
    1 / val(homogenize(rve, sc, :r))
end
estimates["EMA, coated sphere"] = [1 / ema(φ) for φ in φs_g]
estimates["Brinkman (1947)"] = [φ > 1 / 3 + 0.02 ? 1 / brinkman1947(φ) : NaN for φ in φs_g]
estimates["cell, KUBC"] = k_kubc.(φs_g)
estimates["cell, SUBC"] = k_subc.(φs_g)

# The differential scheme of the package against Wilkinson's closed integration
# (eq. 58), and the self-consistent scheme against the scalar equation above:

for φ in (0.9, 0.6, 0.4)
    αdif = bisect(a -> -1.5log(φ) - log(1 + a + a^2 / 3) + 2sqrt(3) * (atan((2a + 3) / sqrt(3)) - π / 3), 0.0, 1.0e3)
    rdif = val(homogenize(granular(φ, BrinkmanSphere(Rg, BrinkmanFluid())), DifferentialScheme(:p; reltol = 1.0e-11, abstol = 1.0e-13), :r))
    eq61(a) = num(sc_equation, α => a, φs => φ, R => Rg)
    αsc = bisect(eq61, 1.0e-6, 1.0e3)
    rsc = val(homogenize(granular(φ, BrinkmanSphere(Rg, BrinkmanFluid())), sc, :r))
    @printf("  φ = %.2f :  differential %.10g / %.10g    SC %.10g / %.10g\n", φ, rdif, αdif^2, rsc, αsc^2)
end

order = [
    "dilute", "Mori–Tanaka", "differential", "SC, spherical pores", "SC, cylindrical pores",
    "SC, coated sphere", "EMA, coated sphere", "Brinkman (1947)", "cell, KUBC", "cell, SUBC",
]
## Line style by family: matrix-based schemes, self-consistent, effective medium, cells.
style(name) = startswith(name, "SC") ? :dash : startswith(name, "cell") ? :dot :
    (startswith(name, "EMA") || startswith(name, "Brinkman")) ? :dashdot : :solid
plt_g = plot(;
    xlabel = "porosity φ", ylabel = "k / R²", yscale = :log10, legend = :bottomright,
    title = "Impervious spheres", size = (760, 520), palette = :tab10
)
for name in order
    plot!(plt_g, φs_g, estimates[name]; label = name, lw = 2, ls = style(name))
end

const figdir = joinpath(@__DIR__, "figures")                        #jl
isdir(figdir) || mkdir(figdir)                                       #jl
savefig(plt_g, joinpath(figdir, "71_granular.png"))                  #jl

# ## §6 Fibrous media
#
# Impervious fibers (infinite cylinders) of radius ``R`` with isotropic
# orientations. A fluid matrix is out of reach (Stokes' paradox, §4), so only the
# self-consistent scheme applies, with spherical pores (``R_p = 3R/2``) or
# cylindrical ones (``R_p = R``). The cell models of
# Happel and Cunningham average their axial and transverse permeabilities as
# ``(k_n + 2k_t)/3``.

println("="^78)                                                     #jl
println("  § 6  FIBROUS MEDIA")                                     #jl
println("="^78)                                                     #jl

function fibrous(φ, pore)
    rve = RVE()
    add_phase!(rve, :p, pore, Dict(:r => iso(0.0)); fraction = φ, symmetrize = :iso)
    add_phase!(rve, :s, BrinkmanCylinder(Rg, BrinkmanSolid()), Dict(:r => iso(0.0)); fraction = 1 - φ, symmetrize = :iso)
    return rve
end
function cell_fiber(φ, bc)
    η = sqrt(1 - φ)
    kn, kt = bc === :kubc ?
        (-(log(η) + (1 - η^2) / (1 + η^2)) / 2, -(log(η) + (1 - η^2) / (1 + η^2)) / 4) :
        (-(log(η) + (1 - η^2) * (3 - η^2) / 4) / 2, -(log(η) + (1 - η^4) / (2 * (1 + η^4))) / 4)
    return Rg^2 / η^2 * (kn + 2kt) / 3
end
φs_f = range(0.5, 0.97; length = 25)
plt_f = plot(;
    xlabel = "porosity φ", ylabel = "k / R²", yscale = :log10, legend = :topleft,
    title = "Impervious fibers, isotropic orientations", size = (720, 480)
)
plot!(plt_f, φs_f, [φ > 0.37 ? 1 / val(homogenize(fibrous(φ, BrinkmanSphere(1.5Rg, BrinkmanFluid())), sc, :r)) : NaN for φ in φs_f]; label = "SC, spherical pores", lw = 2)
plot!(plt_f, φs_f, [1 / val(homogenize(fibrous(φ, BrinkmanCylinder(Rg, BrinkmanFluid())), sc, :r)) for φ in φs_f]; label = "SC, cylindrical pores", lw = 2)
plot!(plt_f, φs_f, cell_fiber.(φs_f, :kubc); label = "cell, KUBC", lw = 2, ls = :dash)
plot!(plt_f, φs_f, cell_fiber.(φs_f, :subc); label = "cell, SUBC", lw = 2, ls = :dash)
savefig(plt_f, joinpath(figdir, "71_fibrous.png"))                   #jl

# ## §7 Spanning pores in an impervious matrix
#
# Here the flow runs only in cylindrical tubes of radius ``R`` or planar slits of
# thickness ``2R``, with isotropic orientations, in an impervious matrix. The
# matrix has ``k_0 = 0``, so the phases are described in the permeability form:
# the force concentration replaces the velocity concentration, and a tube in an
# impervious matrix carries Poiseuille's flow,
# ``\boldsymbol{k}\cdot\boldsymbol{B} = (R^2/8)\,\underline{n}\otimes\underline{n}``.
# The Mori–Tanaka estimates are then ``k = R^2\varphi/(24(1-\varphi))`` for
# tubes and ``2R^2\varphi/(9(1-\varphi))`` for slits.

println("="^78)                                                     #jl
println("  § 7  SPANNING PORES (PERMEABILITY FORM)")                #jl
println("="^78)                                                     #jl

Pf = PermeabilityForm()
function spanning(φ, pore; Rs = Rg)
    rve = RVE()
    add_phase!(rve, :s, BrinkmanSphere(Rs, BrinkmanSolid(); form = Pf), Dict(:k => iso(0.0)); fraction = :rest)
    add_phase!(rve, :p, pore, Dict(:k => iso(0.0)); fraction = φ, symmetrize = :iso)
    return rve
end
tube, slit = BrinkmanCylinder(Rg, BrinkmanFluid(); form = Pf), BrinkmanLayer(Rg, BrinkmanFluid(); form = Pf)
for φ in (0.1, 0.3)
    @printf(
        "  φ = %.1f :  MT tubes %.8g (closed form %.8g)   MT slits %.8g (closed form %.8g)\n", φ,
        val(homogenize(spanning(φ, tube), MoriTanaka(:s), :k)), Rg^2 * φ / (24 * (1 - φ)),
        val(homogenize(spanning(φ, slit), MoriTanaka(:s), :k)), 2Rg^2 * φ / (9 * (1 - φ))
    )
end

# The self-consistent scheme represents the solid by spheres of the same specific
# surface as the pores (``R_s = 3R/2`` for tubes, ``3R`` for slits). It is
# insensitive to the formulation, since both forms describe the same flow:

scK = SelfConsistent(algorithm = NewtonDefault(), init = iso(0.01), reltol = 1.0e-12)
φs_t = range(0.02, 0.6; length = 30)
plt_t = plot(;
    xlabel = "porosity φ", ylabel = "k / R²", yscale = :log10, legend = :topleft,
    title = "Tubes and slits in an impervious matrix", size = (720, 480)
)
for (pore, name, Rs) in ((tube, "tubes", 1.5Rg), (slit, "slits", 3Rg))
    plot!(plt_t, φs_t, [val(homogenize(spanning(φ, pore), MoriTanaka(:s), :k)) for φ in φs_t]; label = "Mori–Tanaka, $name", lw = 2)
    plot!(plt_t, φs_t, [val(homogenize(spanning(φ, pore), DifferentialScheme(:s), :k)) for φ in φs_t]; label = "differential, $name", lw = 2, ls = :dash)
    plot!(plt_t, φs_t, [val(homogenize(spanning(φ, pore; Rs = Rs), scK, :k)) for φ in φs_t]; label = "SC, $name", lw = 2, ls = :dot)
end
savefig(plt_t, joinpath(figdir, "71_spanning.png"))                  #jl

# ## §8 Double porosity
#
# A microporous matrix of permeability ``k_\mu`` holds spherical mesopores of
# radius ``R`` at volume fraction ``\varphi``. This is a three-dimensional
# variant, with spherical mesopores, of the double-porosity media studied by
# [bignonnetIJES2022](@citet). The ratio ``\alpha = R/\sqrt{k_\mu}`` measures how
# much larger the mesopores are than the micropores.
#
# As ``\alpha \to \infty`` the Brinkman layer around each mesopore becomes thin
# and the estimates tend to those of Darcy–Darcy homogenization, where a pore is
# an inclusion of infinite permeability. For Mori–Tanaka this limit is
# ``k/k_\mu = (1 + 2\varphi)/(1 - \varphi)``.
#
# Maxwell's scheme needs the Hill tensor of the shape of the distribution
# (here a sphere) in the matrix. In the permeability form that is the Hill tensor
# of conduction, ``1/(3k_\mu)``. Because Brinkman inhomogeneities carry a size
# effect while the distribution shape does not, Maxwell's estimate can leave the
# admissible range. At large mesoporosity it falls below the Reuss-type bound
# ``k \ge k_\mu/(1-\varphi)``.

println("="^78)                                                     #jl
println("  § 8  DOUBLE POROSITY")                                   #jl
println("="^78)                                                     #jl

kμ = 1.0
function double_porosity(φ, αm; maxwell = false)
    Rm = αm * sqrt(kμ)
    rve = maxwell ? RVE(; distribution_shape = Ellipsoid(1.0)) : RVE()
    add_phase!(rve, :m, BrinkmanSphere(Rm; form = Pf), Dict(:k => iso(kμ)); fraction = :rest)
    add_phase!(rve, :p, BrinkmanSphere(Rm, BrinkmanFluid(); form = Pf), Dict(:k => iso(0.0)); fraction = φ)
    return rve
end
φs_d = range(0.0, 0.6; length = 31)
plt_d = plot(;
    xlabel = "mesoporosity φ", ylabel = "k / k_μ", legend = :topleft, yscale = :log10,
    title = "Spherical mesopores in a microporous matrix", size = (760, 520), palette = :tab10
)
for (αm, c) in ((3.0, 1), (30.0, 2))
    plot!(plt_d, φs_d, [val(homogenize(double_porosity(φ, αm), MoriTanaka(:m), :k)) / kμ for φ in φs_d]; label = "Mori–Tanaka, α = $αm", color = c, lw = 2)
    plot!(plt_d, φs_d, [val(homogenize(double_porosity(φ, αm), DifferentialScheme(:m), :k)) / kμ for φ in φs_d]; label = "differential, α = $αm", color = c, lw = 2, ls = :dash)
    plot!(plt_d, φs_d, [val(homogenize(double_porosity(φ, αm), SelfConsistent(algorithm = NewtonDefault(), init = iso(kμ)), :k)) / kμ for φ in φs_d]; label = "SC, α = $αm", color = c, lw = 2, ls = :dot)
end
plot!(plt_d, φs_d, [val(homogenize(double_porosity(φ, 3.0; maxwell = true), Maxwell(), :k)) / kμ for φ in φs_d]; label = "Maxwell, α = 3.0", color = 3, lw = 2, ls = :dashdot)
plot!(plt_d, φs_d, (1 .+ 2φs_d) ./ (1 .- φs_d); label = "Mori–Tanaka, Darcy limit", color = :black, lw = 1)
plot!(plt_d, φs_d, 1 ./ (1 .- φs_d); label = "bound k_μ/(1 − φ)", color = :gray, lw = 1, ls = :dot)
savefig(plt_d, joinpath(figdir, "71_double_porosity.png"))           #jl

for φ in (0.2, 0.5, 0.7)
    kmax = val(homogenize(double_porosity(φ, 3.0; maxwell = true), Maxwell(), :k))
    @printf("  φ = %.1f :  Maxwell k/kμ = %8.4f   bound 1/(1−φ) = %6.4f\n", φ, kmax / kμ, 1 / (1 - φ))
end
println("  Darcy limit, Mori–Tanaka at φ = 0.3: ", val(homogenize(double_porosity(0.3, 1.0e6), MoriTanaka(:m), :k)), " vs ", 1.6 / 0.7)

# ## §9 Sensitivities
#
# The inclusions are generic in their number type, so ForwardDiff reaches the
# radius of the grains through the self-consistent fixed point. At fixed porosity
# the permeability scales as ``R^2``, so ``\partial\ln k/\partial\ln R = 2``. The
# derivative with respect to the porosity has no such shortcut.

println("="^78)                                                     #jl
println("  § 9  SENSITIVITIES")                                     #jl
println("="^78)                                                     #jl

function k_sc(R, φ)
    rve = RVE()
    add_phase!(rve, :p, BrinkmanSphere(R, BrinkmanFluid()), Dict(:r => iso(0.0)); fraction = :rest)
    add_phase!(rve, :s, BrinkmanSphere(R, BrinkmanSolid()), Dict(:r => iso(0.0)); fraction = 1 - φ)
    return 1 / val(homogenize(rve, sc, :r))
end
@printf("  ∂ln k/∂ln R at φ = 0.6 : %.12f\n", ForwardDiff.derivative(R -> log(k_sc(R, 0.6)), 1.0))
@printf("  ∂k/∂φ at φ = 0.6       : %.8f R²\n", ForwardDiff.derivative(φ -> k_sc(1.0, φ), 0.6))
