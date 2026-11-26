```@meta
EditURL = "../../../../scripts/103_layered_sphere_first_order_symbolic.jl"
```

# [Layered sphere: the systems of the first order, symbolically](@id tut-layered-sphere-first-order-symbolic)

!!! tip "The whole page as one script"
    Every block of this page, in order, is [`scripts/103_layered_sphere_first_order_symbolic.jl`](https://github.com/MicroPoroChemoMechanics/MeanFieldHomogenization.jl/blob/main/scripts/103_layered_sphere_first_order_symbolic.jl).
    From a clone of the repository, `julia --project=docs -e 'using Pkg; Pkg.instantiate()'`
    once, then `julia scripts/103_layered_sphere_first_order_symbolic.jl`, runs it end to end.

!!! info "Before this page"
    [Layered sphere: modes and transfer matrices, symbolically](@ref tut-layered-sphere-symbolic),
    which solves the isotropic layers by their radial modes, and the theory
    pages [Layered spheres and cylinders: the radial problem](@ref th-radial-problems)
    and [Layered sphere](@ref th-layered-sphere), which state the results
    derived here.

The layers of a [`LayeredSphere`](@ref) are solved by two strategies. The
first one finds the radial modes of a layer, powers of ``r``, and chains
their amplitudes: it is the subject of the page above. The second one never
looks for a mode: it writes the equations of a loading mode as a system of
the first order, ``r\,\mathrm{d}Y/\mathrm{d}r = \boldsymbol{M}\,Y``, on the
state ``Y`` that an interface transmits, and crosses a layer by the power of
the matrix ``\boldsymbol{M}``. This page builds the systems of an **isotropic**
layer from the field equations, with TensND's differential operators in
spherical coordinates, for moduli that may depend on the radius. It then
crosses a uniform layer, and finds the transfer matrices of the first
strategy again.

````@example layered_sphere_first_order_symbolic
using MeanFieldHomogenization
using TensND
using SymPyPythonCall
using LinearAlgebra
````

## 1. An isotropic layer whose moduli depend on the radius

TensND orders the spherical coordinates ``(\theta, \varphi, r)``. The bulk
modulus ``K`` and the shear modulus ``G`` of the layer are taken as functions
of ``r``, to see whether their derivatives enter the systems; the stiffness
is ``\mathbb{C} = 3K\,\mathbb{J} + 2G\,\mathbb{K}`` at each radius.

````@example layered_sphere_first_order_symbolic
Sph = coorsys_spherical()
θ, φ, r = get_coords(Sph)
𝐞ᶿ, 𝐞ᵠ, 𝐞ʳ = unitvec(Sph)
K, G = SymFunction("K", positive = true)(r), SymFunction("G", positive = true)(r)
ℂ = TensISO{3}(3K, 2G)
stress(u) = ℂ ⊡ symgrad(u, Sph)
component(σ, a, b) = (σ ⋅ a) ⋅ b
nothing #hide
````

The **state** of a loading mode gathers the radial functions of the
displacement and those of the traction on the sphere of radius ``r``,
multiplied by ``r``. The function `first_order` builds the matrix
``\boldsymbol{M}`` of ``r\,Y' = \boldsymbol{M}\,Y`` from the expression of the
state and from the equilibrium equations: the constitutive law, solved for
the derivatives of the displacement, gives the first rows, and equilibrium
the last ones. `mode_states` gives the eigenvectors of a constant matrix.

````@example layered_sphere_first_order_symbolic
include(joinpath(pkgdir(MeanFieldHomogenization), "scripts", "common", "radial_first_order.jl"))
````

```@raw html
<details class="source-file"><summary>Show the code of <code>scripts/common/radial_first_order.jl</code>, which defines <code>first_order</code> and <code>mode_states</code></summary>
```

```@eval
Main.show_source("scripts/common/radial_first_order.jl")
```

```@raw html
</details>
```

## 2. Spherical part

Under a remote spherical strain the displacement is radial,
``\underline{u} = f(r)\,\underline{e}_r``. The state is
``Y = (f,\; r\,\sigma_{rr})``, and the only equation of equilibrium is the
radial one:

````@example layered_sphere_first_order_symbolic
f = SymFunction("f", real = true)(r)
σ = stress(f * 𝐞ʳ)
Y_bulk = [f, simplify(r * component(σ, 𝐞ʳ, 𝐞ʳ))]
````

````@example layered_sphere_first_order_symbolic
M_bulk = first_order(Y_bulk, [f], [simplify(divergence(σ, Sph) ⋅ 𝐞ʳ)], r)
````

The matrix holds the moduli at the radius ``r`` and no derivative of them:
it is that of a layer whose moduli vary with the radius as well as that of a
uniform one. The first row is the constitutive law
``\sigma_{rr} = (K + \tfrac43 G)\,f' + 2\,(K - \tfrac23 G)\,f/r`` solved for
``r f'``, the second one the equilibrium
``(r^2\sigma_{rr})' = r\,(\sigma_{\theta\theta} + \sigma_{\varphi\varphi})``.

From now on the layer is uniform, of constant moduli ``\kappa`` and ``\mu``.
The package holds this matrix for a layer transversely isotropic about the
radius, of which the isotropic one is a case; it is the same:

````@example layered_sphere_first_order_symbolic
const LS = MeanFieldHomogenization.LayeredSpheres
const RS = MeanFieldHomogenization.Core
κ, μ = symbols("kappa mu", positive = true)
uniform = (K => κ, G => μ)
h = hill_moduli(iso_stiffness(κ, μ))
M₁ = simplify.(subs.(M_bulk, uniform...))
@assert all(iszero, simplify.(M₁ .- LS._bulk_system(h)))
M₁
````

### The exponents are the eigenvalues

A solution ``Y = r^s\,\underline{y}`` of ``r\,Y' = \boldsymbol{M}\,Y`` has
``\boldsymbol{M}\,\underline{y} = s\,\underline{y}``: the exponents of the
radial modes are the eigenvalues of ``\boldsymbol{M}``, and the states of the
modes its eigenvectors. Nothing was assumed about them:

````@example layered_sphere_first_order_symbolic
s = symbols("s", real = true)
𝟙₂ = Sym[1 0; 0 1]
factor(det(M₁ - s * 𝟙₂))
````

The exponents ``1`` and ``-2`` of the modes ``r`` and ``r^{-2}`` of
[Layered sphere: modes and transfer matrices, symbolically](@ref tut-layered-sphere-symbolic).
The eigenvector of an exponent is in the kernel of
``\boldsymbol{M} - s\,\boldsymbol{1}`` (`mode_states`); normalized by its
displacement,

````@example layered_sphere_first_order_symbolic
V₁ = mode_states(M₁, (1, -2))
````

The columns are the states ``(u_r,\; r\,\sigma_{rr})`` of ``u_r = r`` and of
``u_r = r^{-2}`` at ``r = 1``: ``r\,\sigma_{rr} = 3\kappa\,r`` for the first
one, a uniform strain, and ``-4\mu\,r^{-2}`` for the second one.

### Crossing a layer

With ``\tau = \ln r`` the system is
``\mathrm{d}Y/\mathrm{d}\tau = \boldsymbol{M}\,Y``, of constant matrix: the
state is carried from the radius ``r_1`` to the radius ``r_2`` by
``\xi^{\boldsymbol{M}} = \exp(\boldsymbol{M}\ln\xi)``, ``\xi = r_2/r_1``.
On the eigenvectors the power is that of the exponents:

````@example layered_sphere_first_order_symbolic
ξ = symbols("xi", positive = true)
P₁ = simplify.(V₁ * Diagonal([ξ, ξ^(-2)]) * inv(V₁))
````

It is the solution of
``\xi\,\mathrm{d}\boldsymbol{P}/\mathrm{d}\xi = \boldsymbol{M}\,\boldsymbol{P}``
equal to the identity at ``\xi = 1``:

````@example layered_sphere_first_order_symbolic
@assert all(iszero, simplify.(ξ .* diff.(P₁, ξ) .- M₁ * P₁)) && all(iszero, simplify.(subs.(P₁, ξ => 1) .- 𝟙₂))
````

The package computes the same power in closed form, without the eigenvectors
(`radial_power`); here on numbers:

````@example layered_sphere_first_order_symbolic
numbers = (κ => 4.0, μ => 3.0, ξ => 1.7)
Float64.(N.(subs.(P₁, numbers...))) ≈ RS.radial_power(LS._bulk_system(hill_moduli(iso_stiffness(4.0, 3.0))), 1.7)
````

### The transfer matrix of the modes, found again

The first strategy carries the state ``(u_r, \sigma_{rr})``, without the
factor ``r``, by the transfer matrix
``\boldsymbol{T}(r_2, r_1) = \boldsymbol{\Phi}(r_2)\,\boldsymbol{\Phi}(r_1)^{-1}``
built on the modes. With ``\boldsymbol{D}(r) = \operatorname{diag}(1, r)``,
which turns one state into the other, the two strategies give the same
matrix:

````@example layered_sphere_first_order_symbolic
r₁, r₂ = symbols("r_1 r_2", positive = true)
D₂(x) = Diagonal([1, x])
T₁ = inv(D₂(r₂)) * subs.(P₁, ξ => r₂ / r₁) * D₂(r₁)
@assert all(iszero, simplify.(T₁ .- LS._bulk_layer_transfer(r₂, r₁, κ, μ)))
simplify.(T₁)
````

## 3. Deviatoric part

Under the remote deviatoric strain
``\underline{e}_z\otimes\underline{e}_z - \tfrac12(\underline{e}_x\otimes\underline{e}_x + \underline{e}_y\otimes\underline{e}_y)``
the displacement is carried by the Legendre polynomial ``P_2(\cos\theta)``,
``\underline{u} = U(r)\,P_2\,\underline{e}_r + W(r)\,P_2'\,\underline{e}_\theta``,
the prime being the derivative in ``\theta``. The traction has the same
angular shapes, ``\sigma_{rr}`` on ``P_2`` and ``\sigma_{r\theta}`` on
``P_2'``; the radial functions are read at one angle. The state is
``Y = (U,\; W,\; r\,\sigma_{rr},\; r\,\sigma_{r\theta})``:

````@example layered_sphere_first_order_symbolic
P₂ = (3cos(θ)^2 - 1) / 2
at(e, g) = simplify(subs(e, θ => PI / 3) / subs(g, θ => PI / 3))
U, W = SymFunction("U", real = true)(r), SymFunction("W", real = true)(r)
σ = stress(U * P₂ * 𝐞ʳ + W * diff(P₂, θ) * 𝐞ᶿ)
eq = divergence(σ, Sph)
Y_dev = [U, W, at(r * component(σ, 𝐞ʳ, 𝐞ʳ), P₂), at(r * component(σ, 𝐞ʳ, 𝐞ᶿ), diff(P₂, θ))]
````

````@example layered_sphere_first_order_symbolic
M_dev = first_order(Y_dev, [U, W], [at(eq ⋅ 𝐞ʳ, P₂), at(eq ⋅ 𝐞ᶿ, diff(P₂, θ))], r)
````

Again without any derivative of the moduli. The angular shapes are the right
ones: read at another angle, the two equations of equilibrium are the same
(the difference is evaluated at numerical values of the functions).

````@example layered_sphere_first_order_symbolic
trial = (U => r^2 + 1 / r, W => 2r - r^3, K => 3 + r, G => 1 + r^2)
other(e, g) = subs(simplify(subs(e, θ => PI / 5) / subs(g, θ => PI / 5)), θ => PI / 5)
gap = [other(eq ⋅ 𝐞ʳ, P₂) - at(eq ⋅ 𝐞ʳ, P₂), other(eq ⋅ 𝐞ᶿ, diff(P₂, θ)) - at(eq ⋅ 𝐞ᶿ, diff(P₂, θ))]
@assert all(e -> abs(N(subs(subs(e, trial...).doit(), r => Sym(3) / 2))) < 1.0e-12, gap)
````

For a uniform layer it is the matrix of the package, and its eigenvalues are
the four exponents of [christensenLo1979](@citet):

````@example layered_sphere_first_order_symbolic
M₂ = simplify.(subs.(M_dev, uniform...))
@assert all(iszero, simplify.(M₂ .- LS._shear_system(h)))
𝟙₄ = Sym.(Matrix(1I, 4, 4))
factor(det(M₂ - s * 𝟙₄))
````

The eigenvectors give the shape ``(U, W)`` of each mode and its traction:

````@example layered_sphere_first_order_symbolic
exponents = (1, 3, -4, -2)
V₂ = mode_states(M₂, exponents)
````

The first column is the uniform strain, ``(U, W) = (1, \tfrac12)`` up to a
factor; the ratios ``W/U`` of the others are those of the modes ``r^3``,
``r^{-4}`` and ``r^{-2}`` of the first strategy, here with ``x = \kappa/\mu``:

````@example layered_sphere_first_order_symbolic
x = κ / μ
@assert all(iszero, simplify.(V₂[2, :] .- [Sym(1) / 2, (15x + 11) / (6 * (3x - 2)), -Sym(1) / 3, 1 / (3 * (x + 1))]))
````

The power of the matrix follows, and the transfer matrix of the first
strategy on the state ``(U, W, \sigma_{rr}, \sigma_{r\theta})``, with
``\boldsymbol{D}(r) = \operatorname{diag}(1, 1, r, r)``:

````@example layered_sphere_first_order_symbolic
P₂ᵐ = simplify.(V₂ * Diagonal([ξ^Sym(q) for q in exponents]) * inv(V₂))
@assert all(iszero, simplify.(ξ .* diff.(P₂ᵐ, ξ) .- M₂ * P₂ᵐ)) && all(iszero, simplify.(subs.(P₂ᵐ, ξ => 1) .- 𝟙₄))
D₄(x) = Diagonal([1, 1, x, x])
T₂ = inv(D₄(r₂)) * subs.(P₂ᵐ, ξ => r₂ / r₁) * D₄(r₁)
@assert all(iszero, simplify.(T₂ .- LS._shear_layer_transfer(r₂, r₁, κ, μ)))
````

At size four the package does not use the eigenvectors: it computes the
exponential of ``\boldsymbol{M}\ln\xi``, which holds when two exponents are
equal or complex, as layers transversely isotropic about the radius allow.
On numbers the two agree:

````@example layered_sphere_first_order_symbolic
maximum(abs, Float64.(N.(subs.(P₂ᵐ, numbers...))) .- RS.radial_power(LS._shear_system(hill_moduli(iso_stiffness(4.0, 3.0))), 1.7))
````

## 4. Conduction

With a conductivity ``k(r)`` and a temperature
``T = \hat T(r)\cos\theta`` under a remote uniform gradient, the flux is
``\underline{q} = -k\,\underline{\operatorname{grad}}\,T`` and the state
``Y = (\hat T,\; r\,\hat q)``, ``\hat q`` being the radial function of the
outward flux on ``\cos\theta``. The balance is ``\operatorname{div}\underline{q} = 0``:

````@example layered_sphere_first_order_symbolic
kᶜ = SymFunction("k", positive = true)(r)
T̂ = SymFunction("T", real = true)(r)
T = T̂ * cos(θ)
q = -kᶜ * (diff(T, r) * 𝐞ʳ + diff(T, θ) / r * 𝐞ᶿ)
Y_cond = [T̂, simplify(r * (q ⋅ 𝐞ʳ) / cos(θ))]
M_cond = first_order(Y_cond, [T̂], [simplify(divergence(q, Sph) / cos(θ))], r)
````

The matrix of the package, and the exponents ``1`` and ``-2`` of the modes
``r`` and ``r^{-2}`` of the temperature:

````@example layered_sphere_first_order_symbolic
k₀ = symbols("k", positive = true)
M₃ = subs.(M_cond, kᶜ => k₀)
@assert all(iszero, simplify.(M₃ .- LS._cond_system(radial_ti(SphericalChart(), k₀, k₀))))
factor(det(M₃ - s * 𝟙₂))
````

The transfer matrix of the modes is found as for the spherical part, on the
state ``(\hat T, \hat q)``:

````@example layered_sphere_first_order_symbolic
V₃ = mode_states(M₃, (1, -2))
P₃ = simplify.(V₃ * Diagonal([ξ, ξ^(-2)]) * inv(V₃))
Φ(x) = [x 1/x^2; -k₀ 2k₀/x^3]                       # the states of the modes r and r⁻²
@assert all(iszero, simplify.(inv(D₂(r₂)) * subs.(P₃, ξ => r₂ / r₁) * D₂(r₁) .- Φ(r₂) * inv(Φ(r₁))))
P₃
````

## 5. The center, and an interface

At the center the solutions that stay bounded are those of the positive
exponents. Their states at the radius of the core are the corresponding
eigenvectors: one for the spherical part and for conduction, two for the
deviatoric part,

````@example layered_sphere_first_order_symbolic
V₁[:, 1], V₂[:, 1:2]
````

The package finds their span without the eigenvectors, as the range of a
projector (`regular_states`); at size two it is the same direction, here
``r\,\sigma_{rr}/u_r = 3\kappa`` for ``\kappa = 4``:

````@example layered_sphere_first_order_symbolic
y = RS.regular_states(LS._bulk_system(hill_moduli(iso_stiffness(4.0, 3.0))), Val(1))
y[2] / y[1]
````

An imperfect interface multiplies the state by its jump matrix. On ``Y``
the factor ``r`` of the traction enters it: a spring of normal compliance
``s_n`` at the radius ``R``, ``[\![u_r]\!] = s_n\,\sigma_{rr}``, and a membrane of
surface bulk modulus ``\kappa^{\mathrm{s}}``,
``[\![\sigma_{rr}]\!] = 4\kappa^{\mathrm{s}}\,u_r/R^2``, have on
``(u_r,\; r\,\sigma_{rr})`` the matrices

````@example layered_sphere_first_order_symbolic
sₙ, κˢ, R = symbols("s_n kappa_s R", positive = true)
jump(itf) = simplify.(D₂(R) * LS._bulk_interface_T(itf, κ, μ, R) * inv(D₂(R)))
jump(SpringInterface(; sn = sₙ, st = sₙ)), jump(MembraneInterface(κˢ, κˢ))
````

## 6. One sphere in a matrix

A sphere ``(\kappa_1, \mu_1)`` of radius ``R`` in a matrix
``(\kappa_0, \mu_0)``, under a remote spherical strain. In the sphere the
state is a multiple ``a`` of the regular one; in the matrix it is the uniform
strain, of unit amplitude, plus a multiple ``b`` of the mode that decays. The
continuity of the state at ``R`` fixes both:

````@example layered_sphere_first_order_symbolic
κ₁, μ₁, κ₀, μ₀, a, b = symbols("kappa_1 mu_1 kappa_0 mu_0 a b", positive = true)
states(κ, μ) = mode_states(Matrix(LS._bulk_system(hill_moduli(iso_stiffness(κ, μ)))), (1, -2))
inside = states(κ₁, μ₁)[:, 1]
uniform_strain, decaying = eachcol(states(κ₀, μ₀))
amplitudes = solve(a .* inside .- (uniform_strain .+ b .* decaying), [a, b])
factor(amplitudes[a])
````

The strain of the sphere per unit remote strain is the classical
``(3\kappa_0 + 4\mu_0)/(3\kappa_1 + 4\mu_0)``, which the first strategy finds
from the amplitudes of the modes, and the package from either:

````@example layered_sphere_first_order_symbolic
sphere = LayeredSphere((Sym(1),), (iso_stiffness(κ₁, μ₁),))
@assert iszero(simplify(amplitudes[a] - LS._bulk_concentration(sphere, κ₀, μ₀)[1]))
````

## Where to go next

- [Layered spheres and cylinders: the radial problem](@ref th-radial-problems):
  the two strategies, and the layers each one reaches.
- [Layered sphere: layers transversely isotropic about the radius, symbolically](@ref tut-layered-sphere-radial-symbolic):
  the same systems for a layer whose axis turns with the point, where the
  exponents are no longer integers.
- [Layered cylinder: the systems of the first order, symbolically](@ref tut-layered-cylinder-first-order-symbolic):
  the cylinder.
- [Graded isotropic layers with interfaces at given radii](@ref tut-graded-layers-interfaces):
  the systems integrated numerically when ``K`` and ``G`` vary with the radius.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

