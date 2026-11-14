```@meta
EditURL = "../../../../scripts/69_layered_sphere_symbolic.jl"
```

# [Layered sphere: the transfer matrices, symbolically](@id tut-layered-sphere-symbolic)

!!! tip "The whole page as one script"
    Every block of this page, in order, is [`scripts/69_layered_sphere_symbolic.jl`](https://github.com/MicroPoroChemoMechanics/MeanFieldHomogenization.jl/blob/main/scripts/69_layered_sphere_symbolic.jl).
    From a clone of the repository, `julia --project=docs -e 'using Pkg; Pkg.instantiate()'`
    once, then `julia scripts/69_layered_sphere_symbolic.jl`, runs it end to end.

!!! info "Before this page"
    The theory page
    [Layered sphere — bulk + shear recurrences and imperfect interfaces](@ref th-layered-sphere)
    states the recurrences of an `n`-layer sphere. This page derives them with
    TensND's symbolic differential operators: the transfer matrices of
    [herve1993](@citet), their extension to imperfect interfaces
    [herveLuanco2014](@cite), and the order of the products when the moduli
    are operators. The cylinder has its own page,
    [Layered cylinder: the transfer relations, symbolically](@ref tut-layered-cylinder-symbolic).

A [`LayeredSphere`](@ref) is a core and concentric isotropic shells in an
infinite isotropic matrix, under a remote uniform strain. That strain splits
into a spherical part and a deviatoric one, and the displacement of each
layer is, for each part, a combination of a few elementary fields, the
**modes**. The method is to find the modes, to write what an interface
transmits, and to chain the layers.

````@example layered_sphere_symbolic
using MeanFieldHomogenization
using TensND
using SymPyPythonCall
using LinearAlgebra
````

## 1. An isotropic layer in spherical coordinates

TensND orders the spherical coordinates ``(\theta, \varphi, r)``, ``\theta``
being the colatitude. The stiffness of a layer is
``\mathbb{C} = 3k\,\mathbb{J} + 2\mu\,\mathbb{K}``.

````@example layered_sphere_symbolic
Sph = coorsys_spherical()
θ, φ, r = get_coords(Sph)
𝐞ᶿ, 𝐞ᵠ, 𝐞ʳ = unitvec(Sph)
k, μ = symbols("k mu", positive = true)
ℂ = TensISO{3}(3k, 2μ)
stress(u) = ℂ ⊡ symgrad(u, Sph)
equilibrium(u) = divergence(stress(u), Sph)
nothing #hide
````

## 2. Spherical part: the matrices ``\boldsymbol J`` and ``\boldsymbol N``

Under a remote spherical strain the displacement is radial,
``\underline{u} = f(r)\,\underline{e}_r``, and equilibrium reduces to

````@example layered_sphere_symbolic
f = SymFunction("f", real = true)
simplify((equilibrium(f(r) * 𝐞ʳ) ⋅ 𝐞ʳ) / (3k + 4μ))
````

an Euler equation of solutions ``r`` and ``1/r^2``: ``f = A r + B/r^2``. The
quantities continuous across a perfect interface are the displacement and
the radial stress; they depend linearly on ``(A, B)``:

````@example layered_sphere_symbolic
A, B = symbols("A B", real = true)
u_bulk = A * r + B / r^2
σrr_bulk = simplify((stress(u_bulk * 𝐞ʳ) ⋅ 𝐞ʳ) ⋅ 𝐞ʳ)
J(ρ, kk, μμ) = [ρ 1 / ρ^2; 3kk -4μμ / ρ^3]
@assert iszero(simplify(σrr_bulk - (J(r, k, μ) * [A, B])[2]))
J(r, k, μ)
````

Continuity at the radius ``R`` between a layer ``(k_1, \mu_1)`` and the next one
``(k_2, \mu_2)`` is ``\boldsymbol J_2(R)\,(A_2, B_2) = \boldsymbol J_1(R)\,(A_1, B_1)``,
so the amplitudes are carried by the matrix

````@example layered_sphere_symbolic
k₁, μ₁, k₂, μ₂, R = symbols("k_1 mu_1 k_2 mu_2 R", positive = true)
N = simplify.(inv(J(R, k₂, μ₂)) * J(R, k₁, μ₁))
````

The product of these matrices from the core (where ``B = 0``) to the matrix
gives every amplitude per unit core amplitude; the remote strain fixes the
latter. For one sphere in a matrix, the strain of the sphere per unit remote
strain is the classical

````@example layered_sphere_symbolic
simplify(1 / N[1, 1])
````

The package propagates the state ``(u_r, \sigma_{rr})`` rather than the
amplitudes, which keeps an incompressible layer (``k = \infty``) regular; the
transfer across a layer is the same product in the other order:

````@example layered_sphere_symbolic
const LS = MeanFieldHomogenization.LayeredSpheres
r₁, r₂ = symbols("r_1 r_2", positive = true)
@assert all(iszero, simplify.(LS._bulk_layer_transfer(r₂, r₁, k, μ) .- J(r₂, k, μ) * inv(J(r₁, k, μ))))
````

## 3. Deviatoric part: the matrices ``\boldsymbol L`` and ``\boldsymbol M``

Every deviatoric strain is a sum of rotated copies of the axisymmetric one
``\underline{e}_z\otimes\underline{e}_z - \tfrac12(\underline{e}_x\otimes\underline{e}_x +
\underline{e}_y\otimes\underline{e}_y)``, whose displacement is carried by the
Legendre polynomial ``P_2(\cos\theta)``:
``\underline{u} = U(r)\,P_2\,\underline{e}_r + W(r)\,P_2'\,\underline{e}_\theta``,
the prime being the derivative in ``\theta``. With power laws ``U = a\,r^s``,
``W = b\,r^s``:

````@example layered_sphere_symbolic
s, a, b = symbols("s a b", real = true)
P₂ = (3cos(θ)^2 - 1) / 2
u_dev(U, W) = U * P₂ * 𝐞ʳ + W * diff(P₂, θ) * 𝐞ᶿ
eq = equilibrium(u_dev(a * r^s, b * r^s))
E₁ = simplify((eq ⋅ 𝐞ʳ) / (r^(s - 2) * P₂))
E₂ = simplify((eq ⋅ 𝐞ᶿ) / (r^(s - 2) * diff(P₂, θ)))
system = [diff(E₁, a) diff(E₁, b); diff(E₂, a) diff(E₂, b)]
factor(det(system))
````

Four modes, ``s \in \{1, 3, -4, -2\}``, of shapes

````@example layered_sphere_symbolic
shapes = [simplify.(subs.(system, s => q).nullspace()[1]') for q in (1, 3, -4, -2)]
````

The first one is the uniform strain. With the amplitudes ``(a, b, c, d)`` and
the normalization of [christensenLo1979](@citet),

````@example layered_sphere_symbolic
a, b, c, d = symbols("a b c d", real = true)
x = k / μ
U = 2r * a + 6(3x - 2) * r^3 * b + 3c / r^4 + 3(x + 1) * d / r^2
W = r * a + (15x + 11) * r^3 * b - c / r^4 + d / r^2
@assert all(iszero, simplify.(TensND.canonical_components(equilibrium(u_dev(U, W)))))
````

The traction has the same angular shapes, ``\sigma_{rr} \propto P_2`` and
``\sigma_{r\theta} \propto P_2'``; its radial functions are read at one angle:

````@example layered_sphere_symbolic
coefficient(e, g) = factor(simplify(subs(e, θ => PI / 3) / subs(g, θ => PI / 3)))
σ = stress(u_dev(U, W))
state = [U, W, coefficient((σ ⋅ 𝐞ʳ) ⋅ 𝐞ʳ, P₂), coefficient((σ ⋅ 𝐞ʳ) ⋅ 𝐞ᶿ, diff(P₂, θ))]
L = simplify.([diff(q, amp) for q in state, amp in (a, b, c, d)])
````

``\boldsymbol L(r)`` maps the amplitudes to the state
``(U, W, \sigma_{rr}, \sigma_{r\theta})`` (radial functions of the displacement and
of the traction), and
``\boldsymbol M = \boldsymbol L_2(R)^{-1}\boldsymbol L_1(R)`` carries the
amplitudes across a perfect interface, as ``\boldsymbol N`` does for the
spherical part. The package uses the same modes, the second and the fourth
scaled by ``s_\mu = \mu/(k + \mu)`` so that they stay finite for an
incompressible layer:

````@example layered_sphere_symbolic
sμ = μ / (k + μ)
L_pkg = LS._shear_M_matrix(r, k, μ)
@assert all(iszero, simplify.(L_pkg .- L * Diagonal([1, sμ, 1, sμ])))
````

## 4. Imperfect interfaces

Across an imperfect interface the state is not continuous: it is multiplied
by a jump matrix ``\boldsymbol T``, and the transfer of the amplitudes becomes
``\boldsymbol J_2(R)^{-1}\,\boldsymbol T\,\boldsymbol J_1(R)`` (resp.
``\boldsymbol L_2(R)^{-1}\,\boldsymbol T\,\boldsymbol L_1(R)``).

A **spring** opens in proportion to the traction, which stays continuous:
``[\![u_r]\!] = \sigma_{rr}/k_n``, ``[\![u_\theta]\!] = \sigma_{r\theta}/k_t``.
The jump matrices carry ``1/k_n`` and ``1/k_t`` above the diagonal:

````@example layered_sphere_symbolic
kn, kt = symbols("k_n k_t", positive = true)
LS._bulk_interface_T(SpringInterface(kn, kt), k, μ, R)
````

````@example layered_sphere_symbolic
LS._shear_interface_T(SpringInterface(kn, kt), k, μ, R)
````

A **membrane** keeps the displacement continuous and carries a surface stress
``\boldsymbol\sigma^{\mathrm s} = \lambda^{\mathrm s}(\operatorname{tr}\boldsymbol\varepsilon^{\mathrm s})
\boldsymbol 1^{\mathrm s} + 2\mu^{\mathrm s}\boldsymbol\varepsilon^{\mathrm s}`` of the strain
in the tangent plane; its equilibrium makes the traction jump,
``[\![\boldsymbol\sigma]\!]\cdot\underline{e}_r = -\operatorname{div}^{\mathrm s}
\boldsymbol\sigma^{\mathrm s}``. The surface stress, read on the sphere and
extended in the radial direction as a homogeneous function of degree zero, has
the surface divergence as its divergence on ``r = R``:

````@example layered_sphere_symbolic
λs, μs = symbols("lambda_s mu_s", positive = true)
𝟏 = TensISO{3}(Sym(1))
Pᵗ = 𝟏 - 𝐞ʳ ⊗ 𝐞ʳ
function traction_jump(u)
    εs = Pᵗ ⋅ symgrad(u, Sph) ⋅ Pᵗ
    σs = tsubs(λs * tr(εs) * Pᵗ + 2μs * εs, r => R)
    return -tsubs(divergence(σs, Sph), r => R)
end
u₀, U₀, W₀ = symbols("u_0 U_0 W_0", real = true)
jump_bulk = simplify(traction_jump(u₀ * 𝐞ʳ) ⋅ 𝐞ʳ)
````

````@example layered_sphere_symbolic
t = traction_jump(u_dev(U₀, W₀))
jump_dev = (coefficient(t ⋅ 𝐞ʳ, P₂), coefficient(t ⋅ 𝐞ᶿ, diff(P₂, θ)))
````

With ``\kappa^{\mathrm s} = \lambda^{\mathrm s} + \mu^{\mathrm s}``, these are the entries
of the jump matrices of [`MembraneInterface`](@ref):

````@example layered_sphere_symbolic
κs = λs + μs
Tb = LS._bulk_interface_T(MembraneInterface(κs, μs), k, μ, R)
Td = LS._shear_interface_T(MembraneInterface(κs, μs), k, μ, R)
@assert iszero(simplify(Tb[2, 1] * u₀ - jump_bulk))
@assert all(iszero, simplify.((Td * [U₀, W₀, 0, 0])[3:4] .- collect(jump_dev)))
Td
````

In conduction the temperature of the mode is ``T = (A r + B/r^2)\cos\theta``.
A **Kapitza resistance** makes it jump, ``[\![T]\!] = -\rho\,q_r``, the flux
being continuous; a **conducting surface** carries the flux
``-k^{\mathrm s}\nabla^{\mathrm s}T``, whose surface divergence the normal flux
balances, ``[\![q_r]\!] = -\operatorname{div}^{\mathrm s}\underline{q}^{\mathrm s}``:

````@example layered_sphere_symbolic
ks, T₀ = symbols("k_s T_0", positive = true)
simplify(-subs(divergence(-ks * (Pᵗ ⋅ grad(T₀ * cos(θ), Sph)), Sph), r => R) / cos(θ))
````

````@example layered_sphere_symbolic
LS._cond_interface_T(SurfaceConductiveInterface(ks), k, R)
````

## 5. Two closed forms

A grain ``k_1`` bonded to its matrix by a spring answers a pressure as a grain
of bulk modulus ``k_1/(1 + 3k_1/(k_n R))``:

````@example layered_sphere_symbolic
k₀, μ₀ = symbols("k_0 mu_0", positive = true)
grain = LayeredSphere((R,), (TensISO{3}(3k₁, 2μ₁),); interfaces = (SpringInterface(kn, kt),))
α = TensND.get_data(gradient_gradient_concentration(grain, TensISO{3}(3k₀, 2μ₀), TensISO{3}(3k₀, 2μ₀)))[1]
@assert iszero(simplify(α - (3k₀ + 4μ₀) / (3k₁ / (1 + 3k₁ / (kn * R)) + 4μ₀)))
factor(simplify(α))
````

A core ``k_1`` of radius ``a`` coated up to the radius 1 by ``(k_2, \mu_2)``
answers a pressure as a homogeneous sphere whose bulk modulus is that of the
composite-sphere model:

````@example layered_sphere_symbolic
ρ = symbols("a", positive = true)
coated = LayeredSphere((ρ, Sym(1)), (TensISO{3}(3k₁, 2μ₁), TensISO{3}(3k₂, 2μ₂)))
states, outer = LS._bulk_state_seq(coated, k₂, μ₂)
k_eff = factor(simplify(outer[2] / (3 * outer[1])))
````

````@example layered_sphere_symbolic
cf = ρ^3
@assert iszero(simplify(k_eff - (k₂ + cf * (k₁ - k₂) * (3k₂ + 4μ₂) / (3k₂ + 4μ₂ + 3 * (1 - cf) * (k₁ - k₂)))))
````

## 6. Moduli that do not commute

In aging linear viscoelasticity ``k`` and ``\mu`` are Volterra operators and the
amplitudes are histories. The matrices above keep their form, each product
read in the order the mechanics gives: a modulus acts on the left of the
strain it multiplies. Two places need care.

**The transfer across an interface.** With non-commuting symbols, the
continuity of the radial stress and the opening of a spring of compliance
``s_n`` at the radius ``R`` give the amplitude ``A_2`` of the outer layer:

````@example layered_sphere_symbolic
K₁, M₁, K₂, M₂, Sₙ, A₁, B₁ = symbols("K_1 M_1 K_2 M_2 S_n A_1 B_1", commutative = false)
σ⁻ = 3K₁ * A₁ - 4M₁ * B₁ / R^3
u⁺ = R * A₁ + B₁ / R^2 + Sₙ * σ⁻
expand(σ⁻ + 4M₂ * u⁺ / R)                    # = (3K₂ + 4M₂) ∘ A₂
````

The factor of ``A_1`` is ``3K_1 + 4M_2 + (12/R)\,M_2\circ S_n\circ K_1``: outer
shear modulus, then the compliance, then the inner bulk modulus, and the
division by ``3K_2 + 4M_2`` is on the left. No other order is admissible.

**The shapes of the deviatoric modes.** Equilibrium is linear in the moduli,
each one on the left of an amplitude: the two equations of a mode
``U = \alpha\,r^s``, ``W = \beta\,r^s`` are obtained from the ones of section 3
by putting the moduli back on the left.

````@example layered_sphere_symbolic
K, Mo, αo, βo, bo, do_ = symbols("K M alpha beta b d", commutative = false)
operator_form(E, q) = expand(
    sum(
        subs(diff(E, mod, amp), s => q) * Op * H
            for (mod, Op) in ((k, K), (μ, Mo)), (amp, H) in ((a, αo), (b, βo))
    )
)
mode_equations(q) = (operator_form(E₁, q), operator_form(E₂, q))
mode_equations(3)
````

The modes ``r^3`` and ``r^{-2}`` of section 3, with ``x`` read as
``M^{-1}\circ K`` — the inverse on the left —, satisfy them identically:

````@example layered_sphere_symbolic
X = Mo^-1 * K
check(q, αe, βe) = [simplify(expand(Mo^-1 * subs(e, αo => αe, βo => βe))) for e in mode_equations(q)]
(check(3, 6(3X - 2) * bo, (15X + 11) * bo), check(-2, 3(X + 1) * do_, do_))
````

and with the other order, ``K\circ M^{-1}``, they do not:

````@example layered_sphere_symbolic
Xʳ = K * Mo^-1
check(3, 6(3Xʳ - 2) * bo, (15Xʳ + 11) * bo)
````

This is the form the aging recurrences of the package use: the ratio
``\widetilde\mu^{-1}\circ\widetilde k`` applied on the left in the mode shapes,
and the products of section 6 in the transfers. Their use is described in
[Aging viscoelasticity](@ref man-viscoelasticity).

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

