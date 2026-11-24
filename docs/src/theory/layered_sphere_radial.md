# [Layered sphere: layers transversely isotropic about the radius](@id th-layered-sphere-radial)

!!! info "Before this page"
    [Layered sphere](@ref th-layered-sphere), whose states, interface matrices
    and chaining of the layers are kept here;
    [Second moments of the local fields](@ref th-second-moments), for the last
    two sections.

The layers of a [`LayeredSphere`](@ref) need not be isotropic for the sphere to
keep its symmetry: it is enough that the material be, at each point,
transversely isotropic about the radial direction ``\underline{e}_r``. Its
symmetry axis then turns with the point, as in a coating deposited layer by
layer or in an oolite. Such a layer has five moduli in elasticity and two
conductivities, held by a field of TensND, `radial_ti(SphericalChart(), …)`. The remote strain still splits
into a spherical and a deviatoric part, each carried by a few radial modes, and
the layers are chained as isotropic ones are. Two things change: the modes are
powers of ``r`` whose exponents depend on the moduli, and the means over a
layer are read on its boundaries. The modes were given for the sphere by
[kushch2021](@citet) and [kushchMogilevskaya2022](@citet), in a setting
[norrisShuvalov2012](@citet) describe for every degree of harmonics; they are
derived again, symbolically, in
[the companion tutorial](@ref tut-layered-sphere-radial-symbolic).

## 1. The moduli of a layer

In the local frame ``(\underline{e}_r, \underline{e}_\theta, \underline{e}_\varphi)``,
the stiffness has the components ``C_{rrrr}``,
``C_{\theta\theta\theta\theta}``, ``C_{\theta\theta\varphi\varphi}``,
``C_{rr\theta\theta}`` and ``C_{r\theta r\theta}``. Hill's moduli about the
radius are used below:

```math
\begin{gathered}
n = C_{rrrr}, \qquad l = C_{rr\theta\theta}, \qquad p = C_{r\theta r\theta} , \\
k = \tfrac12\,(C_{\theta\theta\theta\theta} + C_{\theta\theta\varphi\varphi}), \qquad
m = \tfrac12\,(C_{\theta\theta\theta\theta} - C_{\theta\theta\varphi\varphi}) .
\end{gathered}
```

An isotropic layer of moduli ``(K, G)`` is the case ``n = K + \tfrac43 G``,
``l = K - \tfrac23 G``, ``k = K + \tfrac13 G``, ``m = p = G``. The conductivity
is ``k_r\,\underline{e}_r\otimes\underline{e}_r + k_t\,(\boldsymbol{1} - \underline{e}_r\otimes\underline{e}_r)``.

## 2. Spherical part

The displacement is radial, ``\underline{u} = f(r)\,\underline{e}_r``, and
equilibrium is an Euler equation,

```math
n\,r^2 f'' + 2n\,r f' - 2\,(2k - l)\,f = 0 ,
```

of solutions ``r^s`` with

```math
n\,s(s + 1) = 2\,(2k - l) ,
\qquad
s = -\tfrac12 \pm t , \quad t^2 = \tfrac14 + \frac{2\,(2k - l)}{n} .
```

A positive definite stiffness has ``n k > l^2``, hence
``t^2 \geq (\tfrac12 - 2\sqrt{k/n})^2``: the two exponents are real, one of each
sign. The isotropic layer has ``s = 1`` and ``s = -2``. The mode ``r^s`` carries
the state

```math
u_r = r^s , \qquad \sigma_{rr} = (n\,s + 2l)\,r^{s-1} .
```

## 3. Deviatoric part

With the angular shapes of the isotropic sphere,
``\underline{u} = U(r)\,P_2\,\underline{e}_r + W(r)\,P_2'\,\underline{e}_\theta``,
and ``U = a\,r^s``, ``W = b\,r^s``, the two equilibrium equations read

```math
\begin{aligned}
\bigl[n\,s(s+1) - 4k + 2l - 6p\bigr]\,a + \bigl[12k - 6l\,(s+1) - 6p\,(s-1)\bigr]\,b &= 0 , \\
\bigl[2k + (l + p)\,s + 2p\bigr]\,a + \bigl[p\,s(s+1) - 2p - 6k - 4m\bigr]\,b &= 0 .
\end{aligned}
```

Their determinant is a polynomial of degree two in ``t^2``, ``t = s + \tfrac12``,

```math
n p\,t^4 - B\,t^2 + C = 0 ,
```

```math
\begin{aligned}
B &= 6kn + 4kp - 6l^2 - 14lp + 4mn + \tfrac52\,np , \\
C &= 16km + \tfrac32\,kn + 9kp - \tfrac32\,l^2 - 8lm + \tfrac92\,lp + mn + 24mp + \tfrac{9}{16}\,np ,
\end{aligned}
```

whose roots are ``t = \pm\tfrac32, \pm\tfrac72`` for an isotropic layer: the
exponents ``1, 3, -2, -4`` of [christensenLo1979](@citet). Each exponent gives a
mode of shape ``(a, b)`` in the kernel of the system, and of state

```math
\begin{gathered}
U = a\,r^s , \qquad W = b\,r^s , \\
\sigma_{rr} = \bigl[(n\,s + 2l)\,a - 6l\,b\bigr]\,r^{s-1} , \qquad
\sigma_{r\theta} = p\,\bigl[a + (s - 1)\,b\bigr]\,r^{s-1} ,
\end{gathered}
```

``\sigma_{rr}`` and ``\sigma_{r\theta}`` being the radial functions of the
traction, on ``P_2`` and ``P_2'``.

The two values of ``t^2`` are not always real. When ``B^2 < 4npC``, which
positive definite moduli allow, the exponents are complex,
``s = -\tfrac12 \pm (\alpha \pm i\beta)``, and the modes oscillate as
``r^{-1/2 \pm \alpha}\cos(\beta\ln r)``. The layers are crossed without the
exponents (section 5), so that this case needs no particular treatment.

```@example radial
using MeanFieldHomogenization, TensND, LinearAlgebra
const LS = MeanFieldHomogenization.LayeredSpheres

shell = radial_ti(SphericalChart(); n = 12.0, l = 5.0, k = 19.0, m = 11.0, p = 4.0)
oscillating = radial_ti(SphericalChart(); n = 0.768, l = 0.622, k = 1.2855, m = 1.2335, p = 3.6)
exponents(c) = eigvals(Matrix(LS._shear_system(LS._hill_moduli(c))))     # of the system of section 5
(isotropic = exponents(radial_ti(SphericalChart(), iso_stiffness(4.0, 3.0))), shell = exponents(shell), oscillating = exponents(oscillating))
```

## 4. Conduction

Under a remote uniform gradient the temperature is ``T = \hat T(r)\cos\theta``,
and ``\hat T = r^s`` with

```math
k_r\,s(s + 1) = 2k_t , \qquad s = -\tfrac12 \pm \sqrt{\tfrac14 + \frac{2k_t}{k_r}} ,
```

of state ``(\hat T, \hat q) = (r^s, -k_r\,s\,r^{s-1})``, ``\hat q`` being the
outward flux. A homogeneous sphere, whose regular mode has the exponent
``s_+``, disturbs nothing in a medium of conductivity ``k_r\,s_+``
[schulgasser1983](@cite), [milton2002](@cite):

```@example radial
kᵣ, kₜ = 2.0, 5.0
s₊ = -1 / 2 + sqrt(1 / 4 + 2kₜ / kᵣ)
ball = LayeredSphere((1.0,), (radial_ti(SphericalChart(), kₜ, kᵣ),))
property_contribution(ball, TensISO{3}(kᵣ * s₊))
```

## 5. Chaining the layers, and the means

Each of the three problems is a system of the first order on the state that an
interface transmits. With ``Y`` the radial functions of the displacement and
``r`` times those of the traction (the temperature and ``r`` times the outward
flux),

```math
r\,\frac{\mathrm{d}Y}{\mathrm{d}r} = \boldsymbol{M}\,Y ,
```

where ``\boldsymbol{M}`` depends on the moduli at the radius ``r`` alone. For
the spherical part, ``Y = (u_r,\; r\,\sigma_{rr})``, and for conduction,
``Y = (\hat T,\; r\,\hat q)``:

```math
\boldsymbol{M} = \begin{pmatrix} -\dfrac{2l}{n} & \dfrac{1}{n} \\[2mm] 4k - \dfrac{4l^2}{n} & \dfrac{2l}{n} - 1 \end{pmatrix} ,
\qquad
\boldsymbol{M} = \begin{pmatrix} 0 & -\dfrac{1}{k_r} \\[2mm] -2k_t & -1 \end{pmatrix} .
```

For the deviatoric part,
``Y = (U,\; W,\; r\,\sigma_{rr},\; r\,\sigma_{r\theta})`` and, with
``c = k - l^2/n``,

```math
\boldsymbol{M} = \begin{pmatrix}
 -\dfrac{2l}{n} & \dfrac{6l}{n} & \dfrac{1}{n} & 0 \\[2mm]
 -1 & 1 & 0 & \dfrac{1}{p} \\[2mm]
 4c & -12c & \dfrac{2l}{n} - 1 & 6 \\[2mm]
 -2c & 6c + 4m & -\dfrac{l}{n} & -2
\end{pmatrix} .
```

The exponents of sections 2 to 4 are the eigenvalues of these matrices. In a
layer of uniform moduli ``\boldsymbol{M}`` is constant, and the system is
solved by a power of ``r``: the state is carried from the inner radius of the
layer to its outer one by

```math
Y(r_{\mathrm{out}}) = \Bigl(\frac{r_{\mathrm{out}}}{r_{\mathrm{in}}}\Bigr)^{\boldsymbol{M}}\,Y(r_{\mathrm{in}}) ,
\qquad
\rho^{\boldsymbol{M}} = \exp(\boldsymbol{M}\ln\rho) ,
```

whatever the exponents: real or complex, distinct or equal. The eigenvalues of
``\boldsymbol{X} = \boldsymbol{M} + \tfrac12\boldsymbol{1}`` come in opposite
pairs ``\pm t``. For the two systems of size two,
``\boldsymbol{X}^2 = t^2\,\boldsymbol{1}`` and the power is the closed form

```math
\rho^{\boldsymbol{M}} = \rho^{-1/2}\,\Bigl[\cosh(t\ln\rho)\,\boldsymbol{1} + \frac{\sinh(t\ln\rho)}{t}\,\boldsymbol{X}\Bigr] .
```

At the center, the solutions that are regular are those of the exponents of
real part above ``-\tfrac12``; their states span the range of the projector
``\tfrac12\,(\boldsymbol{1} + \operatorname{sign}\boldsymbol{X})``, which is
``(\boldsymbol{X} + t\,\boldsymbol{1})/(2t)`` at size two and is computed by
Newton's iteration ``\boldsymbol{X} \leftarrow \tfrac12(\boldsymbol{X} + \boldsymbol{X}^{-1})``
at size four. These states (one in the spherical part and in conduction, two in
the deviatoric part) are carried outward: across a layer by the power above,
across an interface by its matrix ([Layered sphere](@ref th-layered-sphere),
section 3). In the matrix, which is isotropic, the amplitude of the uniform
strain is read and set to one.

The means that the schemes need are then read on the states, without the
amplitudes of the layers. The mean strain of a ball of radius ``R`` and its mean
stress are integrals over its surface, of unit normal ``\underline{n}``,

```math
\langle\boldsymbol{\varepsilon}\rangle = \frac{1}{V}\oint \underline{u}\stackrel{s}{\otimes}\underline{n}\,\mathrm{d}S ,
\qquad
\langle\boldsymbol{\sigma}\rangle = \frac{1}{V}\oint R\,(\boldsymbol{\sigma}\cdot\underline{n})\stackrel{s}{\otimes}\underline{n}\,\mathrm{d}S ,
```

the second one because the stress is in equilibrium. On the states of the two
parts they are, per unit remote strain and on ``\mathbb{J}`` and ``\mathbb{K}``,

| | spherical part | deviatoric part |
| :--- | :--- | :--- |
| mean strain of the ball | ``u_r/R`` | ``(U + 3W)/(5R)`` |
| mean stress of the ball | ``\sigma_{rr}`` | ``(\sigma_{rr} + 3\sigma_{r\theta})/5`` |

A layer is the difference of two balls. These expressions hold for any law of
the layers; with isotropic ones they return the tensors of the isotropic
sphere, which is the first check of the implementation:

```@example radial
C₀, C₁, C₂ = iso_stiffness(10.0, 6.0), iso_stiffness(40.0, 20.0), iso_stiffness(4.0, 3.0)
isotropic = LayeredSphere((0.6, 1.0), (C₁, C₂))
as_radial = LayeredSphere((0.6, 1.0), (C₁, radial_ti(SphericalChart(), C₂)))
TensND.get_data(property_contribution(isotropic, C₀, C₀)), TensND.get_data(property_contribution(as_radial, C₀, C₀))
```

The mean stress of a layer is not the product of a stiffness by its mean
strain, since the stiffness turns with the point: it is
`flux_gradient_concentration(sphere, C₀; layer)`.

## 6. The parameters of a layer

The moduli of the layers belong to the sphere. A coefficient of one layer is
designated by `layer_param(phase, layer, name)`, the name being a component in
the frame that turns with the point. The derivative with respect to
``C_{rrrr}`` of the second layer is taken at fixed
``C_{\theta\theta\theta\theta}``, ``C_{\theta\theta\varphi\varphi}``,
``C_{rr\theta\theta}`` and ``C_{r\theta r\theta}``:

```@example radial
rve = RVE()
add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C₀); fraction = :rest)
add_phase!(rve, :I, LayeredSphere((0.6, 1.0), (C₁, shell)), Dict(:C => C₁); fraction = 0.3)
shear(C) = k_mu(C)[2]
[name => derivative(rve, MoriTanaka(), layer_param(:I, 2, name); indexer = shear) for name in (:Crrrr, :Cθθθθ, :Cθθφφ, :Crrθθ, :Crθrθ)]
```

## 7. Second moments in the frame of a layer

By [Kreher's theorem](@ref th-second-moments), the derivative of the effective
energy when the stiffness of a layer moves along a tensor field
``\mathbb{B}(\underline{x})`` is the mean over the layer of
``\boldsymbol{\varepsilon}:\mathbb{B}(\underline{x}):\boldsymbol{\varepsilon}``.
With a field that turns with the point, the moments of the components of the
strain in the local frame follow, which no uniform tensor gives: the direction
``\underline{e}_r\otimes\underline{e}_r\otimes\underline{e}_r\otimes\underline{e}_r``,
the first element of the Walpole basis about the radius, gives
``\langle\varepsilon_{rr}^2\rangle``. The layer need not be anisotropic for
that: an isotropic layer is moved along a radial field.

```@example radial
iso_rve = RVE()
add_phase!(iso_rve, :M, Ellipsoid(1.0), Dict(:C => C₀); fraction = :rest)
add_phase!(iso_rve, :I, isotropic, Dict(:C => C₁); fraction = 0.3)
E = Tens(Symmetric([1.0 0.2 0.1; 0.2 -0.5 0.3; 0.1 0.3 0.7]))
𝕎₁ = radial_ti(SphericalChart(), 1.0, 0.0, 0.0, 0.0, 0.0)
second_moment(iso_rve, Dilute(), (:I, 2), E, 𝕎₁)          # ⟨ε_rr²⟩ in the shell
```

In the dilute scheme the local field is that of one sphere in the matrix, which
[`LayeredSphereFields`](@ref) gives at every point of an isotropic sphere. The
mean of ``\varepsilon_{rr}^2`` over the shell, by a quadrature of that field,
is the same number:

```@example radial
using QuadGK
fields = LayeredSphereFields(isotropic, C₀)
function εrr²(r, c, φ)
    x = r .* [sqrt(1 - c^2) * cos(φ), sqrt(1 - c^2) * sin(φ), c]
    ε = local_gradient_gradient_concentration(fields, x) ⊡ E
    n = Tens(x / r)
    return (n ⋅ ε ⋅ n)^2
end
xs, ws = gauss(10)
shell_mean = sum(
    wr * wc * 0.2 * (0.8 + 0.2xr)^2 * εrr²(0.8 + 0.2xr, c, 2π * (j - 0.5) / 16) * 2π / 16
        for (xr, wr) in zip(xs, ws), (c, wc) in zip(xs, ws), j in 1:16
) / (4π / 3 * (1 - 0.6^3))
```

This identity, for the five directions of the Walpole basis and in each layer,
is the second check of the implementation: it compares the derivative of the
solution with radial layers, taken at isotropic ones, with the field of the
isotropic sphere, computed by other formulas.

## 8. Scope

- **Symmetry.** The axis of the layers is the radius. A layer of fixed axes
  breaks the symmetry of the sphere and is refused.
- **Matrix.** The reference medium is isotropic, as for the isotropic sphere.
- **Numbers.** Real and complex numbers (the complex moduli of a viscoelastic
  material at a frequency) and dual numbers, nested, in elasticity; symbols as
  well in conduction. Oscillating modes and equal exponents need no
  particular treatment.
- **Not available with radial layers:** the pointwise fields, the response to
  a harmonic of any degree (multipole expansion), and aging viscoelasticity.

## Where to go next

- [Layered sphere: layers transversely isotropic about the radius, symbolically](@ref tut-layered-sphere-radial-symbolic):
  the derivation of sections 2 to 5.
- [Layered inclusions](@ref man-layered): the syntax.
- The next page, [Layered cylinder](@ref th-layered-cylinder), treats the
  coaxial pattern, whose layers may be transversely isotropic about the axis of
  the fiber.
