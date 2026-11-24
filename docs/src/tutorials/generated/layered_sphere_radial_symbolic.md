```@meta
EditURL = "../../../../scripts/101_layered_sphere_radial_symbolic.jl"
```

# [Layered sphere: layers transversely isotropic about the radius, symbolically](@id tut-layered-sphere-radial-symbolic)

!!! tip "The whole page as one script"
    Every block of this page, in order, is [`scripts/101_layered_sphere_radial_symbolic.jl`](https://github.com/MicroPoroChemoMechanics/MeanFieldHomogenization.jl/blob/main/scripts/101_layered_sphere_radial_symbolic.jl).
    From a clone of the repository, `julia --project=docs -e 'using Pkg; Pkg.instantiate()'`
    once, then `julia scripts/101_layered_sphere_radial_symbolic.jl`, runs it end to end.

!!! info "Before this page"
    [Layered sphere: the transfer matrices, symbolically](@ref tut-layered-sphere-symbolic),
    whose method is followed step by step, and the theory page
    [Layered sphere: layers transversely isotropic about the radius](@ref th-layered-sphere-radial),
    which states the results derived here.

A layer of a [`LayeredSphere`](@ref) may be transversely isotropic about the
radial direction ``\underline{e}_r``, its symmetry axis turning with the
point (a field of TensND, `radial_ti(SphericalChart(), …)`). The sphere keeps its symmetry, so that the method
of the isotropic layers applies: find the radial modes of a layer, write the
state they carry to its boundaries, and chain the layers. What changes is the
modes: they are still powers of ``r``, with exponents that depend on the
moduli. This page derives them with TensND's differential operators in
spherical coordinates.

````@example layered_sphere_radial_symbolic
using MeanFieldHomogenization
using TensND
using SymPyPythonCall
using LinearAlgebra
````

## 1. The layer in spherical coordinates

TensND orders the spherical coordinates ``(\theta, \varphi, r)``. The
stiffness is written with Hill's moduli about ``\underline{e}_r``,
``n = C_{rrrr}``, ``l = C_{rr\theta\theta}``,
``k = \tfrac12(C_{\theta\theta\theta\theta} + C_{\theta\theta\varphi\varphi})``,
``m = \tfrac12(C_{\theta\theta\theta\theta} - C_{\theta\theta\varphi\varphi})``
and ``p = C_{r\theta r\theta}``; an isotropic layer is the case
``n = K + \tfrac43 G``, ``l = K - \tfrac23 G``, ``k = K + \tfrac13 G``,
``m = p = G``. As a tensor field, it has constant components in the frame of
the unit vectors of the spherical chart, which TensND writes from these
moduli (`radial_ti`) and gives in the coordinate system (`tensor_at`):

````@example layered_sphere_radial_symbolic
Sph = coorsys_spherical()
θ, φ, r = get_coords(Sph)
𝐞ᶿ, 𝐞ᵠ, 𝐞ʳ = unitvec(Sph)
n, l, k, m, p = symbols("n l k m p", positive = true)
ℂ = tensor_at(radial_ti(SphericalChart(); n, l, k, m, p), Sph)
stress(u) = ℂ ⊡ symgrad(u, Sph)
equilibrium(u) = divergence(stress(u), Sph)
K, G = symbols("K G", positive = true)
isotropic = (n => K + 4G / 3, l => K - 2G / 3, k => K + G / 3, m => G, p => G)
nothing #hide
````

## 2. Spherical part

Under a remote spherical strain the displacement is radial,
``\underline{u} = f(r)\,\underline{e}_r``, and equilibrium reduces to an Euler
equation:

````@example layered_sphere_radial_symbolic
f = SymFunction("f", real = true)
simplify(r^2 * (equilibrium(f(r) * 𝐞ʳ) ⋅ 𝐞ʳ))
````

Its solutions are powers ``r^s``, the exponent being a root of

````@example layered_sphere_radial_symbolic
s = symbols("s", real = true)
bulk_equation = expand(simplify((equilibrium(r^s * 𝐞ʳ) ⋅ 𝐞ʳ) / r^(s - 2)))
````

that is ``n\,s(s + 1) = 2(2k - l)``, or ``s = -\tfrac12 \pm t`` with
``t^2 = \tfrac14 + 2(2k - l)/n``: the exponents ``1`` and ``-2`` of an
isotropic layer,

````@example layered_sphere_radial_symbolic
factor(subs(bulk_equation, isotropic...))
````

are no longer integers. They are real for every positive definite stiffness:
``n k > l^2`` gives ``2(2k - l)/n \geq 4k/n - 2\sqrt{k/n}``, hence
``t^2 \geq (\tfrac12 - 2\sqrt{k/n})^2``. The state continuous across a perfect
interface is the displacement and the radial stress, here for the mode
``r^s``:

````@example layered_sphere_radial_symbolic
σrr = simplify((stress(r^s * 𝐞ʳ) ⋅ 𝐞ʳ) ⋅ 𝐞ʳ)
factor(σrr / r^(s - 1))
````

The same problem is a system of the first order on the state that an
interface transmits, the displacement and the traction. With
``Y = (f,\; r\,\sigma_{rr})``, the constitutive law gives ``r f'`` and
equilibrium gives ``r\,(r\sigma_{rr})'``, both linear in ``Y``:
``r\,Y' = \boldsymbol{M}\,Y``. The function below builds ``\boldsymbol{M}``
from the expression of the state and from the equilibrium equations, for any
of the three problems; the moduli are taken as functions of ``r``, to see
whether their derivatives enter.

````@example layered_sphere_radial_symbolic
fn(name) = SymFunction(name, positive = true)(r)
nʳ, lʳ, kʳ, mʳ, pʳ = fn("n"), fn("l"), fn("k"), fn("m"), fn("p")
ℂʳ = tensor_at(radial_ti(SphericalChart(); n = nʳ, l = lʳ, k = kʳ, m = mʳ, p = pʳ), Sph)
stressʳ(u) = ℂʳ ⊡ symgrad(u, Sph)
function first_order(Y, unknowns, equations)
    d₁ = [diff(g, r) for g in unknowns]
    d₂ = [diff(g, r, 2) for g in unknowns]
    y = [symbols("y$i", real = true) for i in eachindex(Y)]
    nu = length(unknowns)
    from_law = solve([Y[nu + i] - y[nu + i] for i in 1:nu], d₁)        # r g′ from the traction
    from_equilibrium = solve(equations, d₂)                             # g″ from equilibrium
    rhs = [r * diff(Yᵢ, r) for Yᵢ in Y]
    rhs = [subs(e, [d₂[i] => from_equilibrium[d₂[i]] for i in 1:nu]...) for e in rhs]
    rhs = [subs(e, [d₁[i] => from_law[d₁[i]] for i in 1:nu]...) for e in rhs]
    rhs = [subs(e, [unknowns[i] => y[i] for i in 1:nu]...) for e in rhs]
    return [simplify(diff(expand(simplify(e)), yⱼ)) for e in rhs, yⱼ in y]
end
g = SymFunction("g", real = true)(r)
σ = stressʳ(g * 𝐞ʳ)
M_bulk = first_order([g, simplify(r * ((σ ⋅ 𝐞ʳ) ⋅ 𝐞ʳ))], [g], [simplify(divergence(σ, Sph) ⋅ 𝐞ʳ)])
````

No derivative of a modulus appears: the matrix depends on the moduli at the
radius ``r`` alone, so that it holds for moduli that vary with the radius as
for uniform layers. It is the matrix of the package, and its eigenvalues are
the two exponents:

````@example layered_sphere_radial_symbolic
const LS = MeanFieldHomogenization.LayeredSpheres
uniform = (nʳ => n, lʳ => l, kʳ => k, mʳ => m, pʳ => p)
h = (n = n, l = l, k = k, m = m, p = p)
@assert all(iszero, simplify.(subs.(M_bulk, uniform...) .- LS._bulk_system(h)))
@assert iszero(expand(det(LS._bulk_system(h) - s * [1 0; 0 1]) * n - bulk_equation))
````

In a layer of uniform moduli the state is carried from the radius ``r_1`` to
the radius ``r_2`` by the power ``(r_2/r_1)^{\boldsymbol{M}}``, the solution of
``\rho\,\mathrm{d}\boldsymbol{P}/\mathrm{d}\rho = \boldsymbol{M}\,\boldsymbol{P}``
equal to the identity at ``\rho = 1``. For a system of size two,
``\boldsymbol{X} = \boldsymbol{M} + \tfrac12\boldsymbol{1}`` has the square
``t^2\,\boldsymbol{1}``, which gives the power in closed form:

````@example layered_sphere_radial_symbolic
ρ = symbols("rho", positive = true)
Mₕ = Matrix(LS._bulk_system(h))
X = Mₕ + [1 0; 0 1] / Sym(2)
@assert all(iszero, simplify.(X * X - (Sym(1) / 4 + 2 * (2k - l) / n) * [1 0; 0 1]))
P = Matrix(LS._radial_power(LS._bulk_system(h), ρ))
@assert all(iszero, simplify.(ρ .* diff.(P, ρ) .- Mₕ * P)) && all(iszero, simplify.(subs.(P, ρ => 1) .- [1 0; 0 1]))
simplify.(subs.(P, isotropic...))
````

The last matrix is that of an isotropic layer, where ``t = \tfrac32``.

## 3. Deviatoric part

The displacement under a remote deviatoric strain keeps the angular shapes of
the isotropic case,
``\underline{u} = U(r)\,P_2\,\underline{e}_r + W(r)\,P_2'\,\underline{e}_\theta``,
since the layer has the symmetry of the sphere. With power laws
``U = a\,r^s``, ``W = b\,r^s``, the two equilibrium equations are linear in
``(a, b)``; their radial functions are read at one angle:

````@example layered_sphere_radial_symbolic
a, b = symbols("a b", real = true)
P₂ = (3cos(θ)^2 - 1) / 2
u_dev(U, W) = U * P₂ * 𝐞ʳ + W * diff(P₂, θ) * 𝐞ᶿ
at(e, g) = simplify(subs(e, θ => PI / 3) / subs(g, θ => PI / 3))
eq = equilibrium(u_dev(a * r^s, b * r^s))
E₁ = expand(at(eq ⋅ 𝐞ʳ, r^(s - 2) * P₂))
E₂ = expand(at(eq ⋅ 𝐞ᶿ, r^(s - 2) * diff(P₂, θ)))
system = [diff(E₁, a) diff(E₁, b); diff(E₂, a) diff(E₂, b)]
````

The angular shapes are the right ones: at another angle the same two
equations are found (the difference is evaluated at numerical values of the
symbols).

````@example layered_sphere_radial_symbolic
values = (n => 12, l => 5, k => 19, m => 11, p => 4, s => Sym(7) / 3, a => 2, b => -3, r => Sym(3) / 2)
other = subs(simplify(subs(eq ⋅ 𝐞ʳ, θ => PI / 5) / subs(r^(s - 2) * P₂, θ => PI / 5)) - E₁, values...)
@assert abs(N(other)) < 1.0e-12
````

A mode exists when the determinant vanishes. It is a polynomial of degree two
in ``t^2``, with ``t = s + \tfrac12``:

````@example layered_sphere_radial_symbolic
t = symbols("t", real = true)
quartic = sympy.collect(expand(subs(det(system), s => t - Sym(1) / 2)), t)
````

``n p\,t^4 - B\,t^2 + C = 0``. For an isotropic layer its roots are
``t = \pm\tfrac32, \pm\tfrac72``, the exponents ``1, 3, -2, -4`` of
[christensenLo1979](@citet):

````@example layered_sphere_radial_symbolic
factor(subs(det(system), isotropic...))
````

The two values of ``t^2`` are not always real: for some positive definite
moduli the discriminant ``B^2 - 4npC`` is negative, the four exponents are
complex, ``-\tfrac12 \pm (\alpha \pm i\beta)``, and the modes oscillate as
``r^{-1/2 \pm \alpha}\cos(\beta \ln r)``. An example of such a layer, by the
eigenvalues of its system (below):

````@example layered_sphere_radial_symbolic
oscillating = radial_ti(SphericalChart(); n = 0.768, l = 0.622, k = 1.2855, m = 1.2335, p = 3.6)
eigvals(Matrix(LS._shear_system(LS._hill_moduli(oscillating))))
````

The shape ``(a, b)`` of a mode is in the kernel of the system. Taken from the
second equation, it satisfies the first one at a root of the determinant:

````@example layered_sphere_radial_symbolic
shape = [-system[2, 2], system[2, 1]]
@assert iszero(expand((system * shape)[1] + det(system)))
shape
````

The state is the pair ``(U, W)`` and the radial functions of the traction,
``\sigma_{rr} \propto P_2`` and ``\sigma_{r\theta} \propto P_2'``:

````@example layered_sphere_radial_symbolic
σ = stress(u_dev(a * r^s, b * r^s))
traction = [expand(at((σ ⋅ 𝐞ʳ) ⋅ 𝐞ʳ, r^(s - 1) * P₂)), expand(at((σ ⋅ 𝐞ʳ) ⋅ 𝐞ᶿ, r^(s - 1) * diff(P₂, θ)))]
````

As a system of the first order, on
``Y = (U,\; W,\; r\,\sigma_{rr},\; r\,\sigma_{r\theta})``:

````@example layered_sphere_radial_symbolic
Uʳ, Wʳ = SymFunction("U", real = true)(r), SymFunction("W", real = true)(r)
uʳ = u_dev(Uʳ, Wʳ)
σ = stressʳ(uʳ)
eqʳ = divergence(σ, Sph)
Y_dev = [Uʳ, Wʳ, at(r * ((σ ⋅ 𝐞ʳ) ⋅ 𝐞ʳ), P₂), at(r * ((σ ⋅ 𝐞ʳ) ⋅ 𝐞ᶿ), diff(P₂, θ))]
M_dev = first_order(Y_dev, [Uʳ, Wʳ], [at(eqʳ ⋅ 𝐞ʳ, P₂), at(eqʳ ⋅ 𝐞ᶿ, diff(P₂, θ))])
````

Again without any derivative of the moduli. It is the matrix of the package,
and its characteristic polynomial is the determinant found above with the
power laws, so that its eigenvalues are the four exponents:

````@example layered_sphere_radial_symbolic
@assert all(iszero, simplify.(subs.(M_dev, uniform...) .- LS._shear_system(h)))
𝟙₄ = Sym.(Matrix(1I, 4, 4))
@assert iszero(expand(det(Matrix(LS._shear_system(h)) - s * 𝟙₄) * n * p - det(system)))
````

The power of this matrix is computed numerically, by the exponential of
``\boldsymbol{M}\ln(r_2/r_1)``: real or complex exponents, distinct or equal,
are one case. The solutions that are regular at the center are those of the
exponents of real part above ``-\tfrac12``; their states span the range of
the projector ``\tfrac12(\boldsymbol{1} + \operatorname{sign}\boldsymbol{X})``,
``\boldsymbol{X} = \boldsymbol{M} + \tfrac12\boldsymbol{1}`` having its
eigenvalues in opposite pairs.

## 4. Conduction

With a conductivity ``k_r`` along the radius and ``k_t`` across it, the
temperature under a remote uniform gradient is ``T = r^s\cos\theta``:

````@example layered_sphere_radial_symbolic
kr, kt = symbols("k_r k_t", positive = true)
𝟏 = TensISO{3}(Sym(1))
𝐊 = kr * 𝐞ʳ ⊗ 𝐞ʳ + kt * (𝟏 - 𝐞ʳ ⊗ 𝐞ʳ)
grad_T = diff(r^s * cos(θ), r) * 𝐞ʳ + diff(r^s * cos(θ), θ) / r * 𝐞ᶿ
factor(simplify(divergence(𝐊 ⋅ grad_T, Sph) / (r^(s - 2) * cos(θ))))
````

``k_r\,s(s + 1) = 2k_t``. The state is the temperature and the outward flux,
``Y = (\hat T,\; r\,\hat q)`` with ``\hat q = -k_r\,\hat T'``, and the system of
the package is

````@example layered_sphere_radial_symbolic
M_cond = Matrix(LS._cond_system(radial_ti(SphericalChart(), kt, kr)))
````

whose characteristic polynomial is the equation of the exponents:

````@example layered_sphere_radial_symbolic
@assert iszero(expand(det(M_cond - s * [1 0; 0 1]) * kr - (kr * s * (s + 1) - 2kt)))
````

A homogeneous sphere of such a material, of regular
mode ``r^{s}`` with ``s = -\tfrac12 + \sqrt{\tfrac14 + 2k_t/k_r}``, has the
flux ``-k_r\,s\,r^{s-1}`` at its surface for the temperature ``r^s``: it
disturbs nothing in a medium of conductivity ``k_r\,s``, the result of
[schulgasser1983](@citet) (see also [milton2002](@cite), section 7.5). The
package finds a vanishing contribution there:

````@example layered_sphere_radial_symbolic
s₊ = -Sym(1) / 2 + sqrt(Sym(1) / 4 + 2kt / kr)
ball = LayeredSphere((1,), (radial_ti(SphericalChart(), kt, kr),))
simplify(only(TensND.get_data(property_contribution(ball, TensISO{3}(kr * s₊)))))
````

## 5. From the states to the means

The schemes need the mean strain and the mean stress of the sphere and of its
layers. In an isotropic layer they follow from the amplitudes of the modes.
Here they are read on the states at the boundaries, by the divergence
theorem: the mean strain of a ball of radius ``R`` is the integral of
``\underline{u}\stackrel{s}{\otimes}\underline{n}`` over its surface divided
by its volume, and its mean stress that of
``R\,\underline{t}\stackrel{s}{\otimes}\underline{n}``, with ``\underline{t}``
the traction. For the deviatoric state, whose remote strain is
``\underline{e}_z\otimes\underline{e}_z - \tfrac12(\underline{e}_x\otimes\underline{e}_x + \underline{e}_y\otimes\underline{e}_y)``,
the component ``zz`` of the mean strain is

````@example layered_sphere_radial_symbolic
U, W, R = symbols("U W R", positive = true)
u_z = U * P₂ * cos(θ) - W * diff(P₂, θ) * sin(θ)          # u·e_z, with e_r·e_z = cos θ and e_θ·e_z = −sin θ
mean_zz = simplify(integrate(u_z * cos(θ) * 2PI * R^2 * sin(θ), (θ, 0, PI)) / (4PI * R^3 / 3))
````

that is ``(U + 3W)/(5R)``: the moment ``R^2(U + 3W)/5`` divided by ``R^3``,
which is ``1`` for the uniform strain ``U = 2R``, ``W = R``. The same integral
on the traction gives ``R^3(\sigma_{rr} + 3\sigma_{r\theta})/5`` for the
stress. A layer is the difference of two balls, and nothing in these
expressions depends on its law.

## 6. A coated sphere

A core in a shell whose stiffness along the radius differs from that in the
tangent plane, in the Mori-Tanaka scheme. With isotropic moduli written as a
radial field, the isotropic sphere is recovered:

````@example layered_sphere_radial_symbolic
C₀, C₁, C₂ = iso_stiffness(10.0, 6.0), iso_stiffness(40.0, 20.0), iso_stiffness(4.0, 3.0)
function estimate(shell)
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C₀); fraction = :rest)
    add_phase!(rve, :I, LayeredSphere((0.6, 1.0), (C₁, shell)), Dict(:C => C₁); fraction = 0.3)
    return k_mu(homogenize(rve, MoriTanaka(), :C))
end
estimate(C₂), estimate(radial_ti(SphericalChart(), C₂))
````

and a shell four times as stiff along the radius as its isotropic
counterpart gives

````@example layered_sphere_radial_symbolic
estimate(set_coefficient(radial_ti(SphericalChart(), C₂), :Crrrr, 4 * get_coefficient(radial_ti(SphericalChart(), C₂), :Crrrr)))
````

## Where to go next

- [Layered sphere: layers transversely isotropic about the radius](@ref th-layered-sphere-radial):
  the statement of these results, the parameters of a layer and the second
  moments in a layer.
- [Layered inclusions](@ref man-layered): the syntax.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

