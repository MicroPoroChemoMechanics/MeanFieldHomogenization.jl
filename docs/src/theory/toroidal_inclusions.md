# [Toroidal inclusions](@id th-toroidal-inclusions)

!!! info "Before this page"
    [Extension to conductivity](@ref th-conductivity) and
    [Concentration and contribution tensors](@ref th-concentration), for the
    tensors computed here, and, in the documentation of TensND,
    [Toroidal coordinates](@extref TensND th-toroidal) and
    [Toroidal functions](@extref TensND th-toroidal-harmonics), for the
    coordinates and the functions in which the problem is written.

A torus is not an ellipsoid: the field inside a toroidal heterogeneity is not
uniform and there is no Hill tensor. Its response to a remote loading is
nonetheless known as a series, the torus being a coordinate surface of the
toroidal system. This page solves the conduction problem of an insulating torus
in an isotropic matrix and derives its concentration tensor. The problem was
solved by [Radi and Sevostianov](@cite radiPRSA2016) with the same family of
functions; the derivation is written here in the notation of the package, and
the syntax is given in [Toroidal inclusions](@ref man-toroidal-inclusions).

## 1. The problem

A torus ``\Omega`` of axis ``\underline{n}``, of center-line radius ``R`` and
of tube radius ``r<R``, is embedded in an unbounded matrix of conductivity
``k_0``. No flux crosses its surface. Far from it the temperature gradient is
uniform, equal to ``\underline{G}``:

```math
\Delta T=0\ \text{outside }\Omega,
\qquad
\underline{\operatorname{grad}}\,T\cdot\underline{\nu}=0\ \text{on }\partial\Omega,
\qquad
T-\underline{G}\cdot\underline{x}\to0\ \text{at infinity},
```

with ``\underline{\nu}`` the unit normal of the torus, outward. The mean
gradient over the pore is defined from the values of the temperature on its
surface, and is linear in ``\underline{G}``:

```math
\frac{1}{|\Omega|}\oint_{\partial\Omega}T\,\underline{\nu}\,\mathrm{d}S=\boldsymbol{A}\cdot\underline{G} .
```

``\boldsymbol{A}`` is the gradient concentration tensor of the pore. Its flux
concentration vanishes, so that the two contributions per unit volume of the
torus are ``\boldsymbol{N}=-k_0\,\boldsymbol{A}`` and
``\boldsymbol{R}=\boldsymbol{A}/k_0``. By symmetry about the axis,

```math
\boldsymbol{A}=A_t\,\bigl(\boldsymbol{1}-\underline{n}\otimes\underline{n}\bigr)+A_n\,\underline{n}\otimes\underline{n} ,
```

and the two scalars depend on the ratio ``r/R`` only: they are obtained from a
gradient along the axis and from a gradient across it.

## 2. The torus in toroidal coordinates

In the toroidal system ``(\varphi,\alpha,\beta)`` of axis
``\underline{n}=\underline{e}_3`` and of focal radius ``a``, with
``\delta=\cosh\alpha-\cos\beta`` and ``\varrho`` the distance to the axis,

```math
\varrho=\frac{a\,\sinh\alpha}{\delta},\qquad x_3=\frac{a\,\sin\beta}{\delta},
\qquad
a=\sqrt{R^2-r^2},\qquad \cosh\alpha_0=\frac{R}{r} ,
```

the torus is the surface ``\alpha=\alpha_0``, its inside is
``\alpha>\alpha_0``, and its outward normal is
``\underline{\nu}=-\underline{e}^\alpha``: the condition on the torus is
``\partial T/\partial\alpha=0`` at ``\alpha_0``. A harmonic function is a
combination of the products

```math
\sqrt{\delta}\;F_{n-1/2}^m(\cosh\alpha)\,\bigl\{\cos n\beta,\ \sin n\beta\bigr\}\,\bigl\{\cos m\varphi,\ \sin m\varphi\bigr\},
\qquad F=P\ \text{or}\ Q ,
```

where ``P_{n-1/2}^m`` and ``Q_{n-1/2}^m`` are the toroidal functions. Outside
the torus, which contains the axis ``\alpha=0`` and the points at infinity
``\alpha=\beta=0``, a function that vanishes at infinity is a series of the
first kind, ``P``, bounded on the axis and multiplied by a factor
``\sqrt\delta`` that vanishes at infinity.

## 3. The remote field as a series

The remote field ``\underline{G}\cdot\underline{x}`` is harmonic everywhere.
Its series follows from that of the inverse distance,

```math
\frac{1}{\sqrt{2\,\delta}}=\frac{1}{\pi}\sum_{n\ge0}\epsilon_n\,Q_{n-1/2}(\cosh\alpha)\,\cos n\beta,
\qquad \epsilon_0=1,\quad\epsilon_{n\ge1}=2 ,
```

whose derivatives with respect to ``\beta`` and to ``\alpha`` are, since
``\partial\delta/\partial\beta=\sin\beta``,
``\partial\delta/\partial\alpha=\sinh\alpha`` and
``\mathrm{d}Q_{n-1/2}(\cosh\alpha)/\mathrm{d}\alpha=Q^1_{n-1/2}(\cosh\alpha)``,

```math
\frac{\sin\beta}{(2\delta)^{3/2}}=\frac{2}{\pi}\sum_{n\ge1}n\,Q_{n-1/2}\,\sin n\beta,
\qquad
\frac{\sinh\alpha}{(2\delta)^{3/2}}=-\frac{1}{\pi}\sum_{n\ge0}\epsilon_n\,Q^1_{n-1/2}\,\cos n\beta .
```

Multiplied by ``2^{3/2}a\,\sqrt\delta``, the left sides are ``x_3`` and
``\varrho``:

```math
x_3=a\,\sqrt{\delta}\,\sum_{n\ge1}\frac{4\sqrt2}{\pi}\,n\;Q_{n-1/2}(\cosh\alpha)\,\sin n\beta,
\qquad
x_1=\varrho\cos\varphi=a\,\sqrt{\delta}\,\cos\varphi\sum_{n\ge0}\Bigl(-\frac{2\sqrt2}{\pi}\,\epsilon_n\Bigr)\,Q^1_{n-1/2}(\cosh\alpha)\,\cos n\beta .
```

Both converge off the axis, their terms decaying as
``\mathrm{e}^{-n\alpha}``.

## 4. The condition on the torus

For a unit gradient, the temperature is the remote field plus a perturbation
that vanishes at infinity, of the same dependence on the angles:

```math
T=a\,\sqrt{\delta}\;\cos m\varphi\sum_{n}f_n(\alpha)\,w_n(\beta),
\qquad
f_n=s_n\,Q^m_{n-1/2}(\cosh\alpha)+b_n\,P^m_{n-1/2}(\cosh\alpha) ,
```

with ``m=0``, ``w_n=\sin n\beta`` and ``s_n=\tfrac{4\sqrt2}{\pi}n`` for the
gradient along the axis, and ``m=1``, ``w_n=\cos n\beta`` and
``s_n=-\tfrac{2\sqrt2}{\pi}\epsilon_n`` for the gradient across it. The
amplitudes ``b_n`` are the unknowns. The derivative with respect to ``\alpha``
is

```math
\frac{\partial}{\partial\alpha}\Bigl(\sqrt\delta\sum_nf_n\,w_n\Bigr)
=\frac{1}{2\sqrt\delta}\Bigl(\sinh\alpha\sum_nf_n\,w_n+2\cosh\alpha\sum_nf_n'\,w_n-\sum_nf_n'\,2\cos\beta\,w_n\Bigr) .
```

The last sum is rearranged with
``2\cos\beta\,\sin n\beta=\sin(n+1)\beta+\sin(n-1)\beta`` or
``2\cos\beta\,\cos n\beta=\cos(n+1)\beta+\cos(n-1)\beta``. In a series of
cosines the term ``n=0`` contributes ``2\cos\beta`` and is counted twice at the
index 1. The derivative vanishes for every ``\beta`` when, at
``\alpha=\alpha_0``,

```math
\sinh\alpha_0\,f_n+2\cosh\alpha_0\,f_n'-\omega_n\,f_{n-1}'-f_{n+1}'=0 ,
```

for ``n\ge1`` with ``\omega_1=0`` and ``\omega_{n\ge2}=1`` in the series of
sines, and for ``n\ge0`` with ``\omega_0=0``, ``\omega_1=2`` and
``\omega_{n\ge2}=1`` in the series of cosines. Each relation ties three
consecutive amplitudes.

## 5. Solving the relations

With the unknowns ``x_n=b_n\,P_n`` and the ratios
``\ell_n=P_n'/P_n`` at ``\alpha_0``, where ``P_n`` stands for
``P^m_{n-1/2}(\cosh\alpha_0)``, the relations are the tridiagonal system

```math
\bigl(\sinh\alpha_0+2\cosh\alpha_0\,\ell_n\bigr)\,x_n-\omega_n\,\ell_{n-1}\,x_{n-1}-\ell_{n+1}\,x_{n+1}
=-\Bigl[s_n\bigl(\sinh\alpha_0\,Q_n+2\cosh\alpha_0\,Q_n'\bigr)-\omega_n\,s_{n-1}\,Q_{n-1}'-s_{n+1}\,Q_{n+1}'\Bigr] .
```

Dividing the unknowns by ``P_n``, which grows as ``\mathrm{e}^{n\alpha_0}``,
keeps the coefficients of the system of the order of ``n``; it is solved by
elimination with pivoting. The right side decays as ``Q_n``, that is as
``\mathrm{e}^{-n\alpha_0}``, and ``P_n`` grows at the same rate, so that the
amplitudes ``b_n=x_n/P_n`` decay as ``\mathrm{e}^{-2n\alpha_0}``. The system is
cut at the index ``N`` where this factor reaches the rounding of the number
type, with ``b_{N+1}=0``; ``N`` grows as ``1/\alpha_0`` when the hole of the
torus closes.

## 6. From the far field to the tensor

The tensor is read on the perturbation ``\tilde T=T-\underline{G}\cdot\underline{x}``
far from the torus. Let ``\Phi(\underline{x}-\underline{y})=1/(4\pi|\underline{x}-\underline{y}|)``.
For a point ``\underline{x}`` outside the torus, the representation of the
harmonic function ``\tilde T``, which vanishes at infinity, and that of
``\underline{G}\cdot\underline{y}``, harmonic inside, are

```math
\tilde T(\underline{x})=\oint_{\partial\Omega}\Bigl(\tilde T\,\frac{\partial\Phi}{\partial\nu}-\Phi\,\frac{\partial\tilde T}{\partial\nu}\Bigr)\mathrm{d}S_y,
\qquad
0=\oint_{\partial\Omega}\Bigl(\underline{G}\cdot\underline{y}\,\frac{\partial\Phi}{\partial\nu}-\Phi\,\underline{G}\cdot\underline{\nu}\Bigr)\mathrm{d}S_y .
```

Their sum involves the temperature itself, whose normal derivative vanishes on
the torus:

```math
\tilde T(\underline{x})=\oint_{\partial\Omega}T(\underline{y})\,\frac{\partial\Phi}{\partial\nu_y}(\underline{x}-\underline{y})\,\mathrm{d}S_y
\ \underset{|\underline{x}|\to\infty}{\simeq}\ \frac{\underline{x}}{4\pi\,|\underline{x}|^3}\cdot\oint_{\partial\Omega}T\,\underline{\nu}\,\mathrm{d}S
=\frac{|\Omega|}{4\pi}\,\frac{\bigl(\boldsymbol{A}\cdot\underline{G}\bigr)\cdot\underline{x}}{|\underline{x}|^3} .
```

The perturbation is, far away, the field of a dipole whose moment is the mean
gradient over the pore times its volume ``|\Omega|=2\pi^2R\,r^2``.

Far from the torus ``\alpha`` and ``\beta`` tend to zero, with
``\varrho\simeq2a\alpha/(\alpha^2+\beta^2)``,
``x_3\simeq2a\beta/(\alpha^2+\beta^2)`` and
``|\underline{x}|\simeq2a/\sqrt{\alpha^2+\beta^2}``, so that

```math
\sqrt\delta\simeq\frac{\sqrt2\,a}{|\underline{x}|},\qquad
\beta\simeq\frac{2a\,x_3}{|\underline{x}|^2},\qquad
\alpha\simeq\frac{2a\,\varrho}{|\underline{x}|^2} .
```

There ``P_{n-1/2}\to1`` and ``\sin n\beta\simeq n\beta``; and
``P^1_{n-1/2}(\cosh\alpha)\simeq\tfrac12\bigl(n^2-\tfrac14\bigr)\alpha``, the
derivative of ``P_\nu`` at 1 being ``\nu(\nu+1)/2``. The two perturbations
become

```math
\tilde T\simeq2\sqrt2\,a^3\Bigl(\sum_nn\,b_n\Bigr)\frac{x_3}{|\underline{x}|^3},
\qquad
\tilde T\simeq\sqrt2\,a^3\Bigl(\sum_n\bigl(n^2-\tfrac14\bigr)b_n\Bigr)\frac{x_1}{|\underline{x}|^3} .
```

Identified with the dipole, with
``a^3/(R\,r^2)=\sinh^3\alpha_0/\cosh\alpha_0``, they give the two components of
the tensor:

```math
A_n=\frac{4\sqrt2}{\pi}\,\frac{\sinh^3\alpha_0}{\cosh\alpha_0}\sum_{n\ge1}n\,b_n,
\qquad
A_t=\frac{2\sqrt2}{\pi}\,\frac{\sinh^3\alpha_0}{\cosh\alpha_0}\sum_{n\ge0}\bigl(n^2-\tfrac14\bigr)\,b_n ,
```

the amplitudes being those of the series of sines for ``A_n`` and of the
series of cosines for ``A_t``.

## 7. Values

```@example thtorus
using MeanFieldHomogenization, TensND
using Plots

components(λ) = canonical_components(gradient_gradient_concentration(InsulatingTorus(1.0, λ), TensISO{3}(0.0), TensISO{3}(1.0)))[[1, 9]]
λs = range(0.01, 0.99; length = 99)
A = components.(λs)
plot(λs, [first.(A) last.(A)]; label = ["Aₜ (across the axis)" "Aₙ (along the axis)"],
    xlabel = "r / R", ylabel = "component of A", ylims = (1.2, 2.2),
    title = "Insulating torus: gradient concentration tensor", titlefontsize = 10, size = (640, 360))
```

A thin ring is, locally, a straight wire. A wire across the gradient has the
concentration 2 of an insulating circular cylinder, and a wire along it does
not disturb the field, with the concentration 1. A gradient along the axis is
across the wire all around the ring, and a gradient across the axis is across
the wire for half of it on average:

```math
A_n\to2,\qquad A_t\to\frac{1+2}{2}=\frac32
\qquad\text{when }r/R\to0 .
```

```@example thtorus
components(1.0e-4)
```

When the hole closes, ``r/R\to1``, both stay finite:

```@example thtorus
components(0.99)
```

For comparison, the insulating sphere has ``A=3/2`` in every direction.

## 8. Scope

The series solves the problem in an isotropic matrix, for a torus of circular
cross-section that conducts nothing. The number of terms is set by the value of
``\alpha_0``: the radii are numbers or dual numbers, and the derivatives with
respect to them go through the solve. [`InsulatingTorus`](@ref) is the
inclusion; [A torus in a matrix](@ref man-toroidal-inclusions) shows its use.
