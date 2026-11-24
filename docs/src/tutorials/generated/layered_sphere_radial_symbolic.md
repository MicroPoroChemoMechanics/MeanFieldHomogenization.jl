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
point ([`RadialTI`](@ref)). The sphere keeps its symmetry, so that the method
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
``m = p = G``. As a tensor field, it is a transversely isotropic tensor whose
axis is the unit vector ``\underline{e}_r`` of the chart:

````@example layered_sphere_radial_symbolic
Sph = coorsys_spherical()
θ, φ, r = get_coords(Sph)
𝐞ᶿ, 𝐞ᵠ, 𝐞ʳ = unitvec(Sph)
n, l, k, m, p = symbols("n l k m p", positive = true)
ℂ = TensTI{4}(n, 2k, sqrt(Sym(2)) * l, 2m, 2p, 𝐞ʳ)
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

The package builds the matrix of the two modes from these two rows:

````@example layered_sphere_radial_symbolic
const LS = MeanFieldHomogenization.LayeredSpheres
h = (n = n, l = l, k = k, m = m, p = p)
t_bulk = sqrt(Sym(1) / 4 + 2 * (2k - l) / n)
state(q) = [r^q, (n * q + 2l) * r^(q - 1)]
@assert all(iszero, simplify.(LS._bulk_modes(h, r) .- hcat(state(-Sym(1) / 2 + t_bulk), state(-Sym(1) / 2 - t_bulk))))
````

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

The coefficients ``B`` and ``C`` are those of the package:

````@example layered_sphere_radial_symbolic
B, C = LS._shear_quadratic(h)
@assert iszero(expand(quartic - (n * p * t^4 - B * t^2 + C)))
````

The two values of ``t^2`` are not always real: for some positive definite
moduli the discriminant ``B^2 - 4npC`` is negative, the four exponents are
complex, ``-\tfrac12 \pm (\alpha \pm i\beta)``, and the modes oscillate as
``r^{-1/2 \pm \alpha}\cos(\beta \ln r)``. The solution stays real, the
computation being made on complex numbers. An example of such a layer:

````@example layered_sphere_radial_symbolic
oscillating = RadialTI{4}(; Crrrr = 0.768, Cθθθθ = 2.519, Cθθφφ = 0.052, Crrθθ = 0.622, Crθrθ = 3.6)
Bₒ, Cₒ = LS._shear_quadratic(LS._hill_moduli(oscillating))
Bₒ^2 - 4 * 0.768 * 3.6 * Cₒ
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

``k_r\,s(s + 1) = 2k_t``. A homogeneous sphere of such a material, of regular
mode ``r^{s}`` with ``s = -\tfrac12 + \sqrt{\tfrac14 + 2k_t/k_r}``, has the
flux ``-k_r\,s\,r^{s-1}`` at its surface for the temperature ``r^s``: it
disturbs nothing in a medium of conductivity ``k_r\,s``, the result of
[schulgasser1983](@citet) (see also [milton2002](@cite), section 7.5). The
package finds a vanishing contribution there:

````@example layered_sphere_radial_symbolic
s₊ = -Sym(1) / 2 + sqrt(Sym(1) / 4 + 2kt / kr)
ball = LayeredSphere((1,), (RadialTI{2}(kt, kr),))
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
`RadialTI`, the isotropic sphere is recovered:

````@example layered_sphere_radial_symbolic
C₀, C₁, C₂ = iso_stiffness(10.0, 6.0), iso_stiffness(40.0, 20.0), iso_stiffness(4.0, 3.0)
function estimate(shell)
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C₀); fraction = :rest)
    add_phase!(rve, :I, LayeredSphere((0.6, 1.0), (C₁, shell)), Dict(:C => C₁); fraction = 0.3)
    return k_mu(homogenize(rve, MoriTanaka(), :C))
end
estimate(C₂), estimate(RadialTI(C₂))
````

and a shell four times as stiff along the radius as its isotropic
counterpart gives

````@example layered_sphere_radial_symbolic
estimate(set_coefficient(RadialTI(C₂), :Crrrr, 4 * get_coefficient(RadialTI(C₂), :Crrrr)))
````

## Where to go next

- [Layered sphere: layers transversely isotropic about the radius](@ref th-layered-sphere-radial):
  the statement of these results, the parameters of a layer and the second
  moments in a layer.
- [Layered inclusions](@ref man-layered): the syntax.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

