```@meta
EditURL = "../../../../scripts/102_layered_cylinder_radial_symbolic.jl"
```

# [Layered cylinder: layers transversely isotropic about the radius, symbolically](@id tut-layered-cylinder-radial-symbolic)

!!! tip "The whole page as one script"
    Every block of this page, in order, is [`scripts/102_layered_cylinder_radial_symbolic.jl`](https://github.com/MicroPoroChemoMechanics/MeanFieldHomogenization.jl/blob/main/scripts/102_layered_cylinder_radial_symbolic.jl).
    From a clone of the repository, `julia --project=docs -e 'using Pkg; Pkg.instantiate()'`
    once, then `julia scripts/102_layered_cylinder_radial_symbolic.jl`, runs it end to end.

!!! info "Before this page"
    [Layered cylinder: the transfer relations, symbolically](@ref tut-layered-cylinder-symbolic),
    for the loading modes of the cylinder, and the theory pages
    [The radial problem](@ref th-radial-problems) and
    [Layered cylinder: layers transversely isotropic about the radius](@ref th-layered-cylinder-radial),
    which state the results derived here.

A layer of a [`LayeredCylinder`](@ref) may be transversely isotropic about
the radial direction ``\underline{e}_r`` of the section, with moduli that
depend on the radius. Each loading mode of the cylinder is then a system of
the first order ``r\,\mathrm{d}Y/\mathrm{d}r = \boldsymbol{M}(r)\,Y`` on the
state that an interface transmits. This page derives the matrices
``\boldsymbol{M}`` with TensND's differential operators in cylindrical
coordinates, compares them with those of the package, and solves a layer
graded as a power of the radius in closed form.

````@example layered_cylinder_radial_symbolic
using MeanFieldHomogenization
using TensND
using SymPyPythonCall
using LinearAlgebra
````

## 1. The layer in cylindrical coordinates

TensND orders the cylindrical coordinates ``(r, \theta, z)``: the radial
vector is the first of the local frame. The stiffness is written with Hill's
moduli about ``\underline{e}_r``, ``n = C_{rrrr}``,
``l = C_{rr\theta\theta} = C_{rrzz}``,
``k = \tfrac12(C_{\theta\theta\theta\theta} + C_{\theta\theta zz})``,
``m = \tfrac12(C_{\theta\theta\theta\theta} - C_{\theta\theta zz})`` and
``p = C_{r\theta r\theta} = C_{rzrz}``, taken as functions of ``r`` to see
whether their derivatives enter the systems. `radial_ti` writes the field
from these moduli and `tensor_at` gives it in the coordinate system:

````@example layered_cylinder_radial_symbolic
Cyl = coorsys_cylindrical()
r, θ, z = get_coords(Cyl)
𝐞ʳ, 𝐞ᶿ, 𝐞ᶻ = unitvec(Cyl)
fn(name) = SymFunction(name, positive = true)(r)
n, l, k, m, p = fn("n"), fn("l"), fn("k"), fn("m"), fn("p")
ℂ = tensor_at(radial_ti(CylindricalChart(); n, l, k, m, p), Cyl)
stress(u) = ℂ ⊡ symgrad(u, Cyl)
component(σ, a, b) = (σ ⋅ a) ⋅ b
nothing #hide
````

The package holds the same moduli as numbers or symbols, here `h`, in which
the matrices found below are compared with its own:

````@example layered_cylinder_radial_symbolic
const LC = MeanFieldHomogenization.LayeredCylinders
n₀, l₀, k₀, m₀, p₀ = symbols("n l k m p", positive = true)
uniform = (n => n₀, l => l₀, k => k₀, m => m₀, p => p₀)
h = LC.RadialModuli(n₀, l₀, k₀, m₀, p₀)
same(A, B) = all(iszero, simplify.(subs.(A, uniform...) .- B))
nothing #hide
````

The function `first_order` builds ``\boldsymbol{M}`` from the expression of a
state and from the equilibrium equations.

````@example layered_cylinder_radial_symbolic
include(joinpath(pkgdir(MeanFieldHomogenization), "scripts", "common", "radial_first_order.jl"))
````

```@raw html
<details class="source-file"><summary>Show the code of <code>scripts/common/radial_first_order.jl</code>, which defines <code>first_order</code> and <code>first_order_forcing</code></summary>
```

```@eval
Main.show_source("scripts/common/radial_first_order.jl")
```

```@raw html
</details>
```

## 2. Antiplane shear

With ``\underline{u} = \varphi(r)\cos\theta\,\underline{e}_z``, the state is
``Y = (\varphi,\; r\,\sigma_{rz})``, the stress being its radial function on
``\cos\theta``, and the only equation is the axial one:

````@example layered_cylinder_radial_symbolic
φ = SymFunction("varphi", real = true)(r)
σ = stress(φ * cos(θ) * 𝐞ᶻ)
Y_anti = [φ, simplify(r * component(σ, 𝐞ʳ, 𝐞ᶻ) / cos(θ))]
M_anti = first_order(Y_anti, [φ], [simplify(divergence(σ, Cyl) ⋅ 𝐞ᶻ / cos(θ))], r)
````

No derivative of a modulus appears, and the matrix is that of the package,
which solves the antiplane shear and the conduction across the axis as one
problem, with ``(p, m)`` or ``(k_r, k_t)``:

````@example layered_cylinder_radial_symbolic
@assert same(M_anti, LC._pot_system(LC._antiplane_modulus(h)))
````

## 3. In-plane dilatation and axial strain

With ``\underline{u} = f(r)\,\underline{e}_r + \varepsilon z\,\underline{e}_z``
and ``Y = (f,\; r\,\sigma_{rr})``:

````@example layered_cylinder_radial_symbolic
f = SymFunction("f", real = true)(r)
ε = symbols("varepsilon", real = true)
σ = stress(f * 𝐞ʳ + ε * z * 𝐞ᶻ)
Y_axi = [f, simplify(r * component(σ, 𝐞ʳ, 𝐞ʳ))]
equilibrium_axi = [simplify(divergence(σ, Cyl) ⋅ 𝐞ʳ)]
M_axi = first_order(Y_axi, [f], equilibrium_axi, r)
````

The state is not zero when ``Y`` is: the axial strain is a second member,
proportional to ``\varepsilon\,r``,

````@example layered_cylinder_radial_symbolic
F_axi = first_order_forcing(Y_axi, [f], equilibrium_axi, r)
````

The package appends ``\varepsilon\,r`` to the state, and
``J = \tfrac{1}{r}\int u_r\,\mathrm{d}r``, which the mean axial stress needs.
The first two rows of its matrix hold ``\boldsymbol{M}`` and the second
member:

````@example layered_cylinder_radial_symbolic
A = Matrix(LC._axi_system(h))
@assert same(M_axi, A[1:2, 1:2]) && same(F_axi ./ (ε * r), A[1:2, 3])
A
````

The third row is ``r\,(\varepsilon r)' = \varepsilon r`` and the last one
``r\,J' = u_r - J``. The axial stress, integrated across a layer of uniform
moduli, is read on the state at its two radii and on the integral of
``u_r``:

````@example layered_cylinder_radial_symbolic
σzz = component(stress(f * 𝐞ʳ + ε * z * 𝐞ᶻ), 𝐞ᶻ, 𝐞ᶻ)
expand(simplify(subs(r * σzz - diff(l * r * f, r), uniform...)))
````

that is ``r\,\sigma_{zz} = (l\,r\,u_r)' + (k - m - l)\,u_r + (k + m)\,\varepsilon\,r``.

## 4. In-plane shear

With
``\underline{u} = U(r)\cos 2\theta\,\underline{e}_r + V(r)\sin 2\theta\,\underline{e}_\theta``
and ``Y = (U,\; V,\; r\,\sigma_{rr},\; r\,\sigma_{r\theta})``, the stresses
being their radial functions on ``\cos 2\theta`` and ``\sin 2\theta``:

````@example layered_cylinder_radial_symbolic
U, V = SymFunction("U", real = true)(r), SymFunction("V", real = true)(r)
σ = stress(U * cos(2θ) * 𝐞ʳ + V * sin(2θ) * 𝐞ᶿ)
eq = divergence(σ, Cyl)
Y_shear = [U, V, simplify(r * component(σ, 𝐞ʳ, 𝐞ʳ) / cos(2θ)), simplify(r * component(σ, 𝐞ʳ, 𝐞ᶿ) / sin(2θ))]
M_shear = first_order(Y_shear, [U, V], [simplify(eq ⋅ 𝐞ʳ / cos(2θ)), simplify(eq ⋅ 𝐞ᶿ / sin(2θ))], r)
````

Again without any derivative of the moduli, and the matrix of the package:

````@example layered_cylinder_radial_symbolic
@assert same(M_shear, LC._shear_system(h))
````

Its exponents are symmetric about zero; those of an isotropic layer are
``\pm1`` and ``\pm3``:

````@example layered_cylinder_radial_symbolic
s = symbols("s", real = true)
K, G = symbols("K G", positive = true)
isotropic = (n₀ => K + 4G / 3, l₀ => K - 2G / 3, k₀ => K + G / 3, m₀ => G, p₀ => G)
factor(simplify(det(subs.(Matrix(LC._shear_system(h)), isotropic...) - s * Sym.(Matrix(1I, 4, 4)))))
````

## 5. A layer graded as a power of the radius

The matrices hold the moduli at the radius ``r``: they are those of a graded
layer. When the moduli follow a power of the radius, ``p = p_0\,r^\beta`` and
``m = m_0\,r^\beta``, the antiplane problem is an Euler equation, which is
integrated in closed form:

````@example layered_cylinder_radial_symbolic
β = symbols("beta", real = true)
power = (p => p₀ * r^β, m => m₀ * r^β)
antiplane_equation = simplify(subs(divergence(stress(φ * cos(θ) * 𝐞ᶻ), Cyl) ⋅ 𝐞ᶻ / cos(θ), power...).doit() * r^(2 - β) / p₀)
````

````@example layered_cylinder_radial_symbolic
solution = dsolve(antiplane_equation, φ)
````

The exponents are the roots of ``s\,(s + \beta) = m_0/p_0``. They are the
eigenvalues of the matrix of the reference moduli, ``\beta`` being subtracted
from the diagonal of its traction row
([The radial problem](@ref th-radial-problems), section 4):

````@example layered_cylinder_radial_symbolic
M₀ = Matrix(LC._pot_system(LC._antiplane_modulus(h)))
shifted = M₀ - β * [0 0; 0 1]
factor(det(shifted - s * [1 0; 0 1]))
````

The package crosses such a layer by the power of the shifted matrix,
`power_law_transfer`. The matrix it returns carries the state
``(\varphi,\; r\,\sigma_{rz})`` of the graded layer from the radius 1 to the
radius ``\rho``; it satisfies the system of the graded moduli and is the
identity at ``\rho = 1``:

````@example layered_cylinder_radial_symbolic
ρ = symbols("rho", positive = true)
P = Matrix(MeanFieldHomogenization.Core.power_law_transfer(LC._pot_system(LC._antiplane_modulus(h)), β, Sym(1), ρ))
M_graded = [0 1 / (p₀ * ρ^β); m₀ * ρ^β 0]
@assert all(iszero, simplify.(ρ .* diff.(P, ρ) .- M_graded * P)) && all(iszero, simplify.(subs.(P, ρ => 1) .- [1 0; 0 1]))
simplify.(subs.(P, β => 0, m₀ => p₀))
````

The last matrix is that of an isotropic uniform layer, of modes ``r`` and
``1/r``.

A profile without such a solution is integrated numerically. On the power
law, the integration is compared with the closed form: a fiber of
conductivity ``k_c`` in a coating of conductivity ``k_s\,r^\beta``, whose
temperature across the axis is ``B_1 r^{s_1} + B_2 r^{s_2}`` between ``A\,r``
in the fiber and ``r + g/r`` in the matrix.

````@example layered_cylinder_radial_symbolic
k_c, k_s, b, R, k_m = 8.0, 0.4, 2.5, 1.6, 1.3
function closed_form(b, R)
    e = ((-b + sqrt(b^2 + 4)) / 2, (-b - sqrt(b^2 + 4)) / 2)             # s (s + β) = 1
    flux(s, r) = k_s * r^b * s * r^(s - 1)
    S = [
        1 -1 -1 0
        k_c -flux(e[1], 1) -flux(e[2], 1) 0
        0 R^e[1] R^e[2] -1 / R
        0 flux(e[1], R) flux(e[2], R) k_m / R^2
    ]
    return -2k_m * (S \ [0, 0, R, k_m])[4] / R^2
end
coated = LayeredCylinder((1.0, R), (TensISO{3}(k_c), r -> TensISO{3}(k_s * r^b)))
first(TensND.get_data(property_contribution(coated, TensISO{3}(k_m)))), closed_form(b, R)
````

## 6. A wound coating

A fiber in a coating whose stiffness along the radius differs from that in
the tangent plane, in the Mori-Tanaka scheme. With isotropic moduli written
as a radial field, the isotropic coating is recovered:

````@example layered_cylinder_radial_symbolic
C₀, C₁, C₂ = iso_stiffness(10.0, 6.0), iso_stiffness(40.0, 20.0), iso_stiffness(4.0, 3.0)
function estimate(coating)
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C₀); fraction = :rest)
    add_phase!(rve, :I, LayeredCylinder((0.6, 1.0), (C₁, coating)), Dict(:C => C₁); fraction = 0.3)
    return ti_constants(best_fit_ti(homogenize(rve, MoriTanaka(), :C), (0.0, 0.0, 1.0)))
end
estimate(C₂)
````

````@example layered_cylinder_radial_symbolic
estimate(radial_ti(CylindricalChart(), C₂))
````

and a coating four times as stiff along the radius gives

````@example layered_cylinder_radial_symbolic
wound = radial_ti(CylindricalChart(), C₂)
estimate(set_coefficient(wound, :Crrrr, 4 * get_coefficient(wound, :Crrrr)))
````

## Where to go next

- [The radial problem](@ref th-radial-problems): the statement of the
  problem and its resolution across uniform, power-law and graded layers.
- [Layered cylinder: layers transversely isotropic about the radius](@ref th-layered-cylinder-radial):
  the matrices found here, the averages and the parameters of a layer.
- [Layered inclusions](@ref man-layered): the syntax.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

