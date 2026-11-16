# [The multipole expansion method in elasticity](@id th-mem-elasticity)

!!! info "Before this page"
    [The multipole expansion method](@ref th-mem), which sets the method out in
    conduction: solid harmonics, response of a sphere, translation, linear
    system, closure. This page follows the same steps with the displacement in
    place of the temperature.

In an isotropic matrix of shear modulus ``\mu_0`` and Poisson ratio ``\nu_0``,
the displacement solves the Lamé equation

```math
\frac{2(1-\nu)}{1-2\nu}\,\underline{\nabla}\,(\underline{\nabla}\cdot\underline{u})
  - \underline{\nabla}\times(\underline{\nabla}\times\underline{u}) = \underline{0} .
```

A harmonic function is replaced by three vector solutions per degree and
order, the response of a sphere by a matrix, and the translation formula by
one that mixes the three solutions [kushch2013](@cite). The scheme is the same
[`MultipoleExpansion`](@ref), on a finite [`ParticleAssembly`](@ref) of spheres.

## 1. Vector solid harmonics

With ``y_t^s`` and ``Y_t^s`` the [solid harmonics](@ref th-mem), three families of
harmonic vectors are built on each of them. The decaying ones are

```math
\underline{Y}^{(1)}_{ts} = \underline{\nabla}Y_t^s , \qquad
\underline{Y}^{(2)}_{ts} = \frac{1}{t}\,\underline{\nabla}Y_t^s\times\underline{x} , \qquad
\underline{\nabla}\times\underline{Y}^{(3)}_{ts} = \underline{Y}^{(2)}_{ts} ,\quad
\underline{\nabla}\cdot\underline{Y}^{(3)}_{ts} = -Y_t^s ,
```

a gradient, a rotation field, and a third family whose Cartesian components
are harmonics of degree ``t-1`` [kushch2013](@cite). The first two are
divergence-free. The regular families ``\underline{y}^{(i)}_{ts}`` obey the same
relations with ``y_t^s``. Their components are combinations of scalar solid
harmonics, which is how [`vector_regular_harmonic`](@ref) and
[`vector_irregular_harmonic`](@ref) evaluate them.

## 2. Solutions of the Lamé equation

The first two families are divergence-free and their curl is a gradient or
vanishes, so they solve the Lamé equation as they are. The third solution is
the field of Papkovich and Neuber built on the third family: for any harmonic
vector ``\underline{B}``, ``4(1-\nu)\,\underline{B} - \underline{\nabla}(\underline{x}\cdot\underline{B})``
solves the equation, and

```math
\underline{U}^{(1)}_{ts} = \underline{Y}^{(1)}_{ts} , \qquad
\underline{U}^{(2)}_{ts} = \underline{Y}^{(2)}_{ts} , \qquad
\underline{U}^{(3)}_{ts} = 4(1-\nu)\,\underline{Y}^{(3)}_{ts}
  - \underline{\nabla}\big(\underline{x}\cdot\underline{Y}^{(3)}_{ts}\big) ,
\qquad \underline{x}\cdot\underline{Y}^{(3)}_{ts} = \frac{r^2\,Y_t^s}{2t-1} ,
```

with the regular solutions ``\underline{u}^{(i)}_{ts}`` built in the same way on
``\underline{y}^{(i)}_{ts}``, for which
``\underline{x}\cdot\underline{y}^{(3)}_{ts} = -r^2\,y_t^s/(2t+3)``. Written in
this form, the third solution is that of [kushch2013](@citet) without its
biharmonic function, and the Poisson ratio enters through the factor
``4(1-\nu)`` alone. [`lame_regular`](@ref) and [`lame_irregular`](@ref) return
the solutions at a point.

The lowest degrees are the fields of the uniform strain and of the rigid
motions: ``\underline{u}^{(3)}_{00}`` is the dilatation,
``\underline{u}^{(1)}_{1s}`` the translations, ``\underline{u}^{(2)}_{1s}`` the
rotations and ``\underline{u}^{(1)}_{2s}`` the five uniform deviatoric strains.
Among the decaying ones, ``\underline{U}^{(1)}_{00}`` is the center of
dilatation, ``\underline{U}^{(3)}_{1s}`` the point forces and
``\underline{U}^{(2)}_{1s}`` the point torques.

```@example mem_el
using MeanFieldHomogenization, TensND, LinearAlgebra
x, ν = [0.3, -0.7, 0.45], 0.3
lame_regular(3, 0, 0, x, ν) ≈ (4ν - 2) / 3 * x, lame_regular(1, 2, 0, x, ν) ≈ [-x[1] / 2, -x[2] / 2, x[3]]
```

## 3. Response of a sphere

On a sphere of radius ``r`` the displacement and the traction of each solution
are combinations of the three vector surface harmonics of its degree and order,

```math
\underline{S}^{(1)}_{ts} = r\,\underline{\nabla}\chi_t^s , \qquad
\underline{S}^{(2)}_{ts} = r\,\underline{\nabla}\times(\underline{e}_r\,\chi_t^s) , \qquad
\underline{S}^{(3)}_{ts} = \underline{e}_r\,\chi_t^s ,
```

the tangential gradient, the tangential rotation and the normal, which are
orthogonal on the sphere. With ``n = t`` for the regular solutions and
``n = -(t+1)`` for the decaying ones, and up to the factorial of the
normalization [kushch2013](@cite),

| solution | displacement | traction ``/\,2\mu`` |
| :-- | :-- | :-- |
| first | ``r^{n-1}\,(\underline{S}^{(1)} + n\,\underline{S}^{(3)})`` | ``(n-1)/r`` times the displacement |
| second | ``-r^{n}\,\underline{S}^{(2)}/(n+1)`` | ``(n-1)/(2r)`` times the displacement |
| third | ``r^{n+1}\,(\beta_n\,\underline{S}^{(1)} + \gamma_n\,\underline{S}^{(3)})`` | ``r^{n}\,(b_n\,\underline{S}^{(1)} + g_n\,\underline{S}^{(3)})`` |

```math
\beta_n = \frac{n+5-4\nu}{(n+1)(2n+3)} , \quad \gamma_n = \frac{n-2+4\nu}{2n+3} , \quad
b_n = (n+1)\,\beta_n - \frac{2(1-\nu)}{n+1} , \quad g_n = (n+1)\,\gamma_n - 2\nu .
```

The state of a harmonic at a radius is the vector of its six components, three
of displacement and three of traction, on these surface harmonics. As for the
[layered sphere](@ref th-layered-sphere) under a uniform strain, it is carried
from the center to the outer radius, through each layer by the six solutions of
that layer and across each interface by the jump of its model. A spring
interface of compliances ``s_n`` and ``s_t`` adds ``s_t`` times the tangential
tractions and ``s_n`` times the normal one to the displacement. A membrane of
moduli ``\kappa^{\mathrm{s}}`` and ``\mu^{\mathrm{s}}`` adds to the traction the
surface divergence of its stress, which on the harmonic of degree ``t`` reads,
with ``(u_1, u_2, u_3)`` the components of the displacement
[kushch2013](@cite),

```math
\begin{aligned}
\Delta\tau_1 &= \big[-2\kappa^{\mathrm{s}}\,u_3 + \big(t(t+1)(\kappa^{\mathrm{s}}+\mu^{\mathrm{s}}) - 2\mu^{\mathrm{s}}\big)\,u_1\big]/r^2 ,\\
\Delta\tau_2 &= \big(t(t+1) - 2\big)\,\mu^{\mathrm{s}}\,u_2/r^2 ,\\
\Delta\tau_3 &= \big[4\kappa^{\mathrm{s}}\,u_3 - 2\,t(t+1)\,\kappa^{\mathrm{s}}\,u_1\big]/r^2 .
\end{aligned}
```

At ``t = 0`` and ``t = 2`` these are the jumps of the bulk and shear recurrences
of the layered sphere. Matching the state at the outer radius with the six
solutions of the matrix gives the response of the sphere, a matrix
``\mathbf{B}_t`` independent of the order since the sphere is isotropic:
receiving the regular solutions with the amplitudes ``c^{(j)}_{ts}``, it sends
the decaying ones with

```math
(t-s)!\,(t+s)!\;A^{(i)}_{ts} = \sum_j (\mathbf{B}_t)_{ij}\,c^{(j)}_{ts} .
```

[`harmonic_response`](@ref) returns ``\mathbf{B}_t``. A rigid motion sends
nothing back: the first two columns of ``\mathbf{B}_1`` vanish, so that a
sphere in equilibrium carries neither a net force nor a net torque.

```@example mem_el
C(μ, ν) = TensISO{3}(2μ * (1 + ν) / (1 - 2ν), 2μ)        # (3k, 2μ)
C₀, sphere = C(1.0, 0.3), LayeredSphere((0.8,), (C(7.0, 0.2),))
B₁ = harmonic_response(sphere, C₀, 1)
norm(B₁[:, 1:2]) < 1e-12 * norm(B₁)
```

## 4. From one center to another

The decaying vector harmonics expand about a second center, at
``\underline{R}`` from the first, on the regular ones of the same and of the
lower families [kushch2013](@cite):

```math
\underline{Y}^{(i)}_{ts}(\underline{r} + \underline{R}) =
  \sum_{j \le i}\,\sum_{k,l} (-1)^{k+l}\,\alpha^{(i)(j)}_{tksl}(\underline{R})\;\underline{y}^{(j)}_{kl}(\underline{r}) ,
```

```math
\begin{aligned}
\alpha^{(1)(1)} &= \alpha^{(2)(2)} = \alpha^{(3)(3)} = Y_{t+k}^{s-l} , \qquad
\alpha^{(2)(1)} = -\alpha^{(3)(2)} = \mathrm{i}\,\Big(\frac{l}{k} + \frac{s}{t}\Big)\,Y_{t+k-1}^{s-l} ,\\
\alpha^{(3)(1)} &= \Big[1 + \frac{l\,s}{k\,t} - \frac{t^2-s^2}{t(2t-1)} - \frac{k^2-l^2}{k(2k-1)}\Big]\,Y_{t+k-2}^{s-l} ,
\end{aligned}
```

the harmonics being taken at ``\underline{R}``. The first two Lamé solutions
translate as these vectors do. For the third, write the position from the first
center as ``\underline{r} + \underline{R}`` in its Papkovich-Neuber form: with
``\underline{B} = \underline{Y}^{(3)}_{ts}(\underline{r} + \underline{R})``,

```math
\underline{U}^{(3)}_{ts} = \big[4(1-\nu)\,\underline{B} - \underline{\nabla}(\underline{r}\cdot\underline{B})\big]
  - \underline{\nabla}(\underline{R}\cdot\underline{B}) .
```

``\underline{B}`` being expanded as above, the bracket is term by term a
regular Lamé solution about the second center:
``\underline{u}^{(3)}_{kl}`` for ``\underline{y}^{(3)}_{kl}``;
``4(1-\nu)\,\underline{u}^{(2)}_{kl}`` for ``\underline{y}^{(2)}_{kl}``, which is
tangent to the spheres; and ``(4(1-\nu) - k)\,\underline{u}^{(1)}_{kl}`` for
``\underline{y}^{(1)}_{kl} = \underline{\nabla}y_k^l``, since
``\underline{r}\cdot\underline{\nabla}y_k^l = k\,y_k^l``. The last term is the
gradient of the harmonic function ``\underline{R}\cdot\underline{B}``, whose
expansion follows from the scalar translation formula applied to the Cartesian
components of ``\underline{Y}^{(3)}_{ts}``: with ``H_{tksl}`` its coefficient on
``y_k^l``, it contributes ``-H_{tksl}\,\underline{u}^{(1)}_{kl}``. Hence

```math
\underline{U}^{(i)}_{ts}(\underline{r} + \underline{R}) =
  \sum_{j \le i}\,\sum_{k,l} (-1)^{k+l}\,\eta^{(i)(j)}_{tksl}(\underline{R})\;\underline{u}^{(j)}_{kl}(\underline{r}) ,
\qquad \|\underline{r}\| < \|\underline{R}\| ,
```

with ``\eta = \alpha`` for the first two solutions and

```math
\eta^{(3)(3)} = \alpha^{(3)(3)} , \qquad \eta^{(3)(2)} = 4(1-\nu)\,\alpha^{(3)(2)} , \qquad
\eta^{(3)(1)} = \big(4(1-\nu) - k\big)\,\alpha^{(3)(1)} - H ,
```

```math
H_{tksl} = \frac{(R_1 + \mathrm{i}R_2)\,(t+s-1)(t+s)\,Y_{t-1+k}^{s-1-l}
  - (R_1 - \mathrm{i}R_2)\,(t-s-1)(t-s)\,Y_{t-1+k}^{s+1-l}
  + 2R_3\,(t-s)(t+s)\,Y_{t-1+k}^{s-l}}{2\,t\,(2t-1)} .
```

Every coefficient is affine in ``4(1-\nu)``; [`lame_translation`](@ref) returns
its two parts. The formula is checked here against the direct evaluation of a
solution:

```@example mem_el
R, r, kmax = [1.1, -0.6, 0.9], [0.12, 0.2, -0.1], 24
Y = irregular_harmonics(R, kmax + 3)
series = zeros(ComplexF64, 3)
for k in 0:kmax, l in -k:k, j in 1:3
    a, b = lame_translation(3, j, 2, k, 1, l, Y, R)
    global series += (-1)^(k + l) * (a + 4 * (1 - ν) * b) * lame_regular(j, k, l, r, ν)
end
series ≈ lame_irregular(3, 2, 1, R + r, ν)
```

## 5. The linear system

Under the remote strain ``\boldsymbol{E}``, the displacement of the matrix is

```math
\underline{u}(\underline{x}) = \boldsymbol{E}\cdot\underline{x}
  + \sum_p\,\sum_{t,s}\,\sum_i A^{(i)(p)}_{ts}\,\underline{U}^{(i)}_{ts}(\underline{x} - \underline{X}_p) ,
```

and sphere ``q`` receives on ``\underline{u}^{(j)}_{kl}`` the amplitude

```math
c^{(j)(q)}_{kl} + \sum_{p \ne q}\,\sum_{i,t,s} (-1)^{k+l}\,
  \eta^{(i)(j)}_{tksl}(\underline{X}_q - \underline{X}_p)\,A^{(i)(p)}_{ts} ,
```

to which it answers by its response ``\mathbf{B}_k^{(q)}``. The remote field has
the degrees 0 and 2 only [kushch2013](@cite):

```math
c^{(3)}_{00} = \frac{\operatorname{tr}\boldsymbol{E}}{3\gamma_0(\nu_0)} , \quad
c^{(1)}_{20} = \frac{2E_{33} - E_{11} - E_{22}}{3} , \quad
c^{(1)}_{21} = E_{13} - \mathrm{i}E_{23} , \quad
c^{(1)}_{22} = E_{11} - E_{22} - 2\mathrm{i}E_{12} ,
```

with ``c^{(1)}_{2,-s} = (-1)^s\,\overline{c^{(1)}_{2s}}``. There are
``1 + 3\,t_{\max}(t_{\max}+2)`` unknowns per sphere, each scaled to a
displacement at the surface of its sphere.

## 6. Effective stiffness

Let ``\boldsymbol{\tau} = \boldsymbol{\sigma} - \mathbb{C}_0 : \boldsymbol{\varepsilon}``
be the polarization, which vanishes in the matrix. As for the flux in
conduction, its average over a volume ``V`` holding the spheres reduces to
integrals over their surfaces, taken on the side of the matrix, in which the
regular part of the field contributes nothing and the series contributes by its
dipoles alone: the center of dilatation and the five solutions of the third
family at the degree 2 [kushch2013](@cite),

```math
\begin{aligned}
\operatorname{tr}\langle\boldsymbol{\tau}\rangle &= \frac{24\pi\mu_0(1-\nu_0)}{(1-2\nu_0)\,V}\sum_q A^{(1)(q)}_{00} , &
2\langle\tau_{33}\rangle - \langle\tau_{11}\rangle - \langle\tau_{22}\rangle &= -\frac{32\pi\mu_0(1-\nu_0)}{V}\sum_q A^{(3)(q)}_{20} ,\\
\langle\tau_{13}\rangle - \mathrm{i}\langle\tau_{23}\rangle &= -\frac{16\pi\mu_0(1-\nu_0)}{V}\sum_q A^{(3)(q)}_{21} , &
\langle\tau_{11}\rangle - \langle\tau_{22}\rangle - 2\mathrm{i}\langle\tau_{12}\rangle &= -\frac{64\pi\mu_0(1-\nu_0)}{V}\sum_q A^{(3)(q)}_{22} .
\end{aligned}
```

Solving for the six unit strains gives the tensor ``\boldsymbol{\Sigma}`` such
that ``\langle\boldsymbol{\tau}\rangle = \boldsymbol{\Sigma} : \boldsymbol{E}``,
and a finite assembly in a domain ``\Omega`` is closed as in conduction,
``\mathbb{C}^{\mathrm{hom}} = \mathbb{C}_0 + \boldsymbol{\Sigma} :
(\mathbb{I} - \mathbb{P}_\Omega : \boldsymbol{\Sigma})^{-1}``.

For one sphere at the center of a ball the estimate is that of Maxwell, hence
of [Mori–Tanaka](@ref th-scheme-mori-tanaka), at any degree, the sphere
receiving the uniform strain alone:

```@example mem_el
asm = ParticleAssembly(; boundary = MixedBC(2.0))
add_matrix!(asm, Dict(:C => C₀))
add_particle!(asm, :p, (0.0, 0.0, 0.0), Ellipsoid(1.1), Dict(:C => C(100.0, 0.2)))
components(C) = (m = mandel(C, CanonicalBasis{3, Float64}()); round.((m[1, 1], m[1, 2], m[4, 4]); digits = 8))
components(homogenize(asm, MultipoleExpansion(; tmax = 4), :C)), components(homogenize(asm, MoriTanaka(), :C))
```

With several spheres the arrangement enters. For 27 nearly rigid spheres on a
cubic block of ``3\times3\times3`` sites, at the fraction 0.2 in the ball of the
same fraction, [kushch2013](@citet) reports ``k^{\mathrm{hom}}/k_0 = 1.4064`` and
two shear moduli, 1.6568 for ``(C_{1111} - C_{1122})/2`` and 1.4719 for
``C_{1212}``, in units of ``\mu_0``: the block is cubic, which no one-site
estimate sees.

```@example mem_el
function block(n, c, C₁)
    R = cbrt(3c / (4π))
    asm = ParticleAssembly(; boundary = MixedBC(R * cbrt(n^3 / c)))
    add_matrix!(asm, Dict(:C => C₀))
    for i in 1:n, j in 1:n, k in 1:n
        add_particle!(asm, Symbol(:p, i, j, k), (i, j, k) .- (n + 1) / 2, Ellipsoid(R), Dict(:C => C₁))
    end
    return asm
end
M = mandel(homogenize(block(3, 0.2, C(1e6, 0.3)), MultipoleExpansion(; tmax = 4), :C), CanonicalBasis{3, Float64}())
k₀ = 2 * 1.3 / (3 * 0.4)
round.(((M[1, 1] + 2M[1, 2]) / 3k₀, (M[1, 1] - M[1, 2]) / 2, M[4, 4] / 2); digits = 4)
```

The degree 4 gives the bulk modulus and misses the last digits of the two shear
moduli, which the degree 6 recovers.

The periodic assembly, whose sums over the images involve more than the lattice
harmonics of conduction, and the fields of each sphere are listed in the
[roadmap](@ref dev-roadmap).
