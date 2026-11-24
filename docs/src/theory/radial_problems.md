# [The radial problem: uniform, power-law and graded layers](@id th-radial-problems)

!!! info "Before this page"
    [Layered sphere](@ref th-layered-sphere) and
    [Layered cylinder](@ref th-layered-cylinder), which solve the isotropic
    sphere and the cylinder of axial symmetry by radial modes and amplitudes,
    and whose interfaces and averages are kept here.

A sphere or a cylinder keeps its symmetry as long as its moduli depend on the
radius ``r`` alone, in the frame that turns with the point. This page states
the problem such an inclusion sets, in a form that does not ask what the layers
are made of, and solves it for three kinds of layers:

| Layer | Moduli | Crossed by |
| :--- | :--- | :--- |
| uniform | constant components in the local frame: isotropic, or transversely isotropic about ``\underline{e}_r`` | a power of a matrix |
| power law, [`PowerLawLayer`](@ref) | those of a reference, times ``(r/r_{\mathrm{ref}})^\beta`` | a power of a matrix |
| graded, [`GradedLayer`](@ref) | any function of ``r`` | a numerical integration |

The matrices of each loading mode are those of
[the sphere](@ref th-layered-sphere-radial) and of
[the cylinder](@ref th-layered-cylinder-radial), on the two pages that follow.

![The local frames of the sphere and of the cylinder, and the chain of the radial problem](../assets/theory/radial_problem.svg)

## 1. Loading modes and radial functions

In a sphere the moduli are given in the frame
``(\underline{e}_\theta, \underline{e}_\varphi, \underline{e}_r)`` of the
spherical chart, in a cylinder in the frame
``(\underline{e}_r, \underline{e}_\theta, \underline{e}_z)`` of the cylindrical
one; the radial vector is the third of the first frame and the first of the
second, and a field built by `radial_ti` knows which it is. A material that is
isotropic, or transversely isotropic about ``\underline{e}_r``, with
coefficients that depend on ``r`` alone, is unchanged by the rotations that
leave the inclusion unchanged.

A uniform remote strain ``\boldsymbol{E}`` (a remote gradient
``\underline{G}`` in conduction) then splits into **loading modes**, each of
which drives a field of fixed angular shape. Only radial functions remain
unknown:

| Inclusion | Loading mode | Field | Size of the state |
| :--- | :--- | :--- | :---: |
| sphere | spherical part of ``\boldsymbol{E}`` | ``\underline{u} = u_r(r)\,\underline{e}_r`` | 2 |
| sphere | deviatoric part of ``\boldsymbol{E}`` | ``\underline{u} = U(r)\,P_2\,\underline{e}_r + W(r)\,P_2'\,\underline{e}_\theta`` | 4 |
| sphere | ``\underline{G}`` | ``T = \hat T(r)\cos\theta`` | 2 |
| cylinder | in-plane dilatation and axial strain | ``\underline{u} = u_r(r)\,\underline{e}_r + \varepsilon z\,\underline{e}_z`` | 2 |
| cylinder | in-plane shear | ``\underline{u} = U(r)\cos 2\theta\,\underline{e}_r + V(r)\sin 2\theta\,\underline{e}_\theta`` | 4 |
| cylinder | antiplane shear | ``\underline{u} = u_z(r)\cos\theta\,\underline{e}_z`` | 2 |
| cylinder | ``\underline{G}`` across the axis | ``T = \hat T(r)\cos\theta`` | 2 |
| cylinder | ``\underline{G}`` along the axis | ``T = G z`` | none |

``P_2`` is the Legendre polynomial of ``\cos\theta`` and ``P_2'`` its derivative
with respect to ``\theta``. The modes do not interact: each one is solved
alone, and the tensors of the inclusion are assembled from their answers.

## 2. The state and its equation

The **state** of a mode at the radius ``r`` gathers what a perfect interface
transmits: the radial functions of the displacement, and those of the traction
on the surface of radius ``r``. The traction is multiplied by ``r``,

```math
Y(r) = \bigl(\text{displacement functions},\; r \times \text{traction functions}\bigr) ,
```

and, in conduction, ``Y = (\hat T,\; r \times \text{normal flux})``. The
state is continuous where the moduli jump, which neither the derivative of the
displacement nor the amplitudes of a modal solution are.

Take the simplest case, conduction in a sphere, with a radial conductivity
``k_r(r)`` and a tangential one ``k_t(r)``. With
``T = \hat T(r)\cos\theta``, the flux
``\underline{q} = -\boldsymbol{k}\cdot\underline{\operatorname{grad}}\,T`` has the
components

```math
q_r = \hat q(r)\cos\theta , \quad \hat q = -k_r\,\hat T' ,
\qquad
q_\theta = \frac{k_t\,\hat T}{r}\,\sin\theta ,
```

and its divergence vanishes when ``(r^2\hat q)' + 2k_t\,\hat T = 0``. The first
relation is the constitutive law along the radius, the second one the balance.
Written on ``Y = (\hat T,\; r\hat q)``, they are

```math
r\,\frac{\mathrm{d}Y}{\mathrm{d}r} = \boldsymbol{M}(r)\,Y ,
\qquad
\boldsymbol{M} = \begin{pmatrix} 0 & -\dfrac{1}{k_r} \\[2mm] -2k_t & -1 \end{pmatrix} .
```

Every mode of the table has this form. The first rows of ``\boldsymbol{M}`` are
the constitutive law solved for the radial derivatives of the displacement, the
last rows the equilibrium; ``\boldsymbol{M}`` holds the moduli at the radius
``r`` and no derivative of them. The factor ``r`` in front of the tractions is
what makes ``\boldsymbol{M}`` free of ``r`` when the moduli are.

## 3. A uniform layer: the power of a matrix

In a layer of uniform moduli ``\boldsymbol{M}`` is constant. With
``s = \ln r`` the equation is ``\mathrm{d}Y/\mathrm{d}s = \boldsymbol{M}\,Y``,
of solution ``Y(s) = \exp\bigl(\boldsymbol{M}(s - s_0)\bigr)\,Y(s_0)``: the state
is carried from the inner radius of the layer to its outer one by

```math
Y(r_{\mathrm{out}}) = \rho^{\boldsymbol{M}}\,Y(r_{\mathrm{in}}) ,
\qquad
\rho = \frac{r_{\mathrm{out}}}{r_{\mathrm{in}}} ,
\qquad
\rho^{\boldsymbol{M}} = \exp(\boldsymbol{M}\ln\rho) .
```

The eigenvalues of ``\boldsymbol{M}`` are the exponents of the radial modes: an
eigenvector ``\underline{y}`` of eigenvalue ``s`` gives the solution
``Y = r^s\,\underline{y}``. For the isotropic conducting sphere they are
``1`` and ``-2``, the modes ``r`` and ``r^{-2}`` of the temperature:

```@example radial_problems
using MeanFieldHomogenization, TensND, LinearAlgebra, StaticArrays
const RS = MeanFieldHomogenization.Core
const LS = MeanFieldHomogenization.LayeredSpheres

M = LS._cond_system(radial_ti(SphericalChart(), 3.0, 3.0))     # k_t = k_r = 3
eigvals(Matrix(M))
```

```@example radial_problems
ρ = 2.0
Y_in = SVector(1.0, -3.0)                    # T̂ = r at r = 1: r q̂ = -k r
RS.radial_power(M, ρ) * Y_in, SVector(ρ, -3ρ)
```

The power does not need the exponents, nor the amplitudes of the modes. It
holds when the exponents are complex, as the deviatoric part of a sphere with
radial layers allows, and when two of them are equal, where a modal solution
has to switch to ``r^s\ln r``.

Let ``N`` be the size of the state and
``\boldsymbol{X} = \boldsymbol{M} - \tfrac{1}{N}(\operatorname{tr}\boldsymbol{M})\,\boldsymbol{1}``.
The eigenvalues of ``\boldsymbol{X}`` come in opposite pairs: those of
``\boldsymbol{M}`` are symmetric about ``-\tfrac12`` in a sphere and about ``0``
in a cylinder. At size two, ``\boldsymbol{X}^2 = t^2\,\boldsymbol{1}`` with
``t^2 = -\det\boldsymbol{X}``, and the series of the exponential sums to

```math
\rho^{\boldsymbol{M}} = \rho^{\operatorname{tr}\boldsymbol{M}/2}\,\Bigl[\cosh(t\ln\rho)\,\boldsymbol{1} + \frac{\sinh(t\ln\rho)}{t}\,\boldsymbol{X}\Bigr] ,
```

which holds for symbols as well. At size four the exponential of the
``4\times4`` matrix is computed.

## 4. A layer that follows a power of the radius

Let the moduli of a layer be those of a reference, uniform, times
``(r/r_{\mathrm{ref}})^\beta``. A matrix ``\boldsymbol{M}`` has the blocks

```math
\boldsymbol{M}(r) = \begin{pmatrix} \boldsymbol{M}_{11} & \varrho^{-\beta}\,\boldsymbol{M}_{12} \\ \varrho^{\beta}\,\boldsymbol{M}_{21} & \boldsymbol{M}_{22} \end{pmatrix} ,
\qquad \varrho = \frac{r}{r_{\mathrm{ref}}} ,
```

``\boldsymbol{M}_{ij}`` being the blocks of the reference: ratios of moduli stay,
a compliance is divided by ``\varrho^\beta`` and a stiffness multiplied by it.
The tractions of the state, divided by ``\varrho^\beta``, give a state
``\tilde Y`` that obeys a system of constant matrix,

```math
r\,\frac{\mathrm{d}\tilde Y}{\mathrm{d}r} = \tilde{\boldsymbol{M}}\,\tilde Y ,
\qquad
\tilde{\boldsymbol{M}} = \begin{pmatrix} \boldsymbol{M}_{11} & \boldsymbol{M}_{12} \\ \boldsymbol{M}_{21} & \boldsymbol{M}_{22} - \beta\,\boldsymbol{1} \end{pmatrix} ,
```

so that the layer is crossed by ``\rho^{\tilde{\boldsymbol{M}}}``, exactly. In
conduction the exponents are the roots of ``s(s + 1 + \beta) = 2k_t/k_r``, the
classical result for a shell graded as a power.

```@example radial_problems
k_c, k_s, β, R, k₀ = 8.0, 0.4, 2.5, 1.6, TensISO{3}(1.3)
exact = LayeredSphere((1.0, R), (TensISO{3}(k_c), PowerLawLayer(TensISO{3}(k_s), β)))
property_contribution(exact, k₀)
```

## 5. A graded layer: numerical integration

For any other profile the system
``r\,\mathrm{d}Y/\mathrm{d}r = \boldsymbol{M}(r)\,Y`` is integrated across the
layer, in the variable
``s = \ln(r/r_{\mathrm{in}})/\ln(r_{\mathrm{out}}/r_{\mathrm{in}})`` from 0 to
1, by an explicit Runge-Kutta method with step control. A function given as the
modulus of a layer is such a profile:

```@example radial_problems
integrated = LayeredSphere((1.0, R), (TensISO{3}(k_c), r -> TensISO{3}(k_s * r^β)))
property_contribution(integrated, k₀)
```

The two answers agree to the tolerance of the integration (`reltol`, `abstol`
of [`GradedLayer`](@ref)). The integration carries real, complex and dual
numbers alike: the complex moduli of a viscoelastic material at a frequency,
and the derivatives with respect to a parameter of the profile or to a radius.

```@example radial_problems
using ForwardDiff
contribution(β) = TensND.get_data(property_contribution(LayeredSphere((1.0, R), (TensISO{3}(k_c), r -> TensISO{3}(k_s * r^β))), k₀))[1]
closed(β) = TensND.get_data(property_contribution(LayeredSphere((1.0, R), (TensISO{3}(k_c), PowerLawLayer(TensISO{3}(k_s), β))), k₀))[1]
ForwardDiff.derivative(contribution, β), ForwardDiff.derivative(closed, β)
```

Uniform layers that follow the profile, as [`graded_layers`](@ref) cuts them,
tend to the same answer as the square of their thickness:

```@example radial_problems
using Plots
profile(r) = TensISO{3}(k_s * r^β)
value(sphere) = TensND.get_data(property_contribution(sphere, k₀))[1]
ns = [2, 4, 8, 16, 32, 64, 128]
errors = map(ns) do n
    radii, moduli = graded_layers(profile, 1.0, R, n)
    abs(value(LayeredSphere((1.0, radii...), (TensISO{3}(k_c), moduli...))) / value(exact) - 1)
end
plot(ns, errors; xscale = :log10, yscale = :log10, marker = :circle, label = "n uniform layers",
    xlabel = "number of layers n", ylabel = "relative difference with the exact layer")
plot!(ns, errors[1] .* (ns[1] ./ ns) .^ 2; linestyle = :dash, label = "n⁻²")
hline!([abs(value(integrated) / value(exact) - 1)]; linestyle = :dot, label = "integrated layer")
```

## 6. The center

At the center the solutions that stay bounded are kept. In a **uniform core**
they are the modes whose exponent has a real part above that of the pairs,
``\tfrac{1}{N}\operatorname{tr}\boldsymbol{M}``: the eigenvectors of
``\boldsymbol{X}`` of eigenvalues of positive real part. Their span is the
range of the projector

```math
\boldsymbol{P} = \tfrac12\,\bigl(\boldsymbol{1} + \operatorname{sign}\boldsymbol{X}\bigr) ,
```

``\operatorname{sign}\boldsymbol{X}`` being the matrix that has the
eigenvectors of ``\boldsymbol{X}`` and the signs of the real parts of its
eigenvalues. At size two it is ``\boldsymbol{X}/t``; at size four it is the
limit of Newton's iteration
``\boldsymbol{X} \leftarrow \tfrac12\,(\boldsymbol{X} + \boldsymbol{X}^{-1})``.
A mode has one regular solution at size two and two at size four; independent
columns of ``\boldsymbol{P}`` are their states at the radius of the core.

In a **graded core** the moduli vary down to the center, where the profile must
have a value. Carrying two regular solutions from a small radius does not work:
both turn toward the one of largest exponent and the pair loses its second
direction. Their span is carried instead, by its **impedance**
``\boldsymbol{Z}``, the matrix that gives the tractions of a regular state from
its displacements, ``Y = (\underline{d},\; \boldsymbol{Z}\,\underline{d})``.
Inserted in the system, with the blocks of ``\boldsymbol{M}``,

```math
r\,\frac{\mathrm{d}\boldsymbol{Z}}{\mathrm{d}r} = \boldsymbol{M}_{21} + \boldsymbol{M}_{22}\,\boldsymbol{Z} - \boldsymbol{Z}\,\boldsymbol{M}_{11} - \boldsymbol{Z}\,\boldsymbol{M}_{12}\,\boldsymbol{Z} ,
```

a Riccati equation, integrated from the impedance of the uniform medium found at
the center. The regular solutions being the growing ones, their impedance is
the solution this equation is attracted to when ``r`` increases, and the
integration is stable.

## 7. Interfaces, the matrix, and the amplitude

The regular states of the core are carried outward: across a layer by sections
3 to 5, across an interface by its jump matrix, which acts on the state
(sections 3 of [Layered sphere](@ref th-layered-sphere) and of
[Layered cylinder](@ref th-layered-cylinder)). A spring or a Kapitza
resistance adds to the displacement a multiple of the traction, a membrane or a
conducting surface adds to the traction a multiple of the displacement; a layer
of any kind may have any of them on its outer radius.

At the outer radius ``R`` the states found are those of the inclusion up to
``n`` amplitudes, ``n`` being half the size of the state. The matrix is
uniform, its solutions are its radial modes, and only those that decay,
plus the uniform field of the loading, are present: ``n`` more amplitudes. The
continuity of the state at ``R``, across the last interface, is a linear system
of ``2n`` equations, which fixes all of them per unit remote loading.

A loading that acts inside the layers as well enters the system as a second
member. The axial strain ``\varepsilon`` of a cylinder is the only one; it is
appended to the state, as [the page of the cylinder](@ref th-layered-cylinder-radial)
shows, so that the system keeps its form.

## 8. Averages

The schemes need the mean strain and the mean stress of each layer per unit
remote strain. No amplitude of a layer being known, they are read on the
states. Over a ball or a disc ``\Omega`` of radius ``r``, of outward normal
``\underline{n}``,

```math
\langle\boldsymbol{\varepsilon}\rangle = \frac{1}{|\Omega|}\oint_{\partial\Omega} \underline{u}\stackrel{s}{\otimes}\underline{n}\,\mathrm{d}S ,
\qquad
\langle\boldsymbol{\sigma}\rangle = \frac{1}{|\Omega|}\oint_{\partial\Omega} \underline{x}\stackrel{s}{\otimes}(\boldsymbol{\sigma}\cdot\underline{n})\,\mathrm{d}S ,
```

the second one because the stress is in equilibrium. Both are linear in the
state at ``r``, and a layer is the difference of two balls or two discs. The
mean stress of a layer is not the product of a stiffness by its mean strain,
since the stiffness varies in the layer: it is read on the tractions.

The mean of the moduli over a layer, which the bounds and some schemes ask for,
is taken over the directions at each radius and, for a graded layer, by a
quadrature along the radius:

```@example radial_problems
LS.layer_property_average(integrated, k₀)        # volume-weighted mean of the conductivity
```

## 9. Scope

- **Matrix.** The reference medium is isotropic for the sphere, transversely
  isotropic about the axis for the cylinder.
- **Numbers.** Real, complex and dual numbers, nested, for every kind of layer.
  Symbols for uniform layers in conduction; the sign of a ``4\times4`` matrix
  and the step control of an integration compare values, which symbols do not
  have.
- **A power-law layer** cannot be the core, where its moduli vanish or are
  infinite. In a cylinder it is integrated as a graded layer.
- **Not available** with layers of these kinds: the pointwise fields, the
  multipole expansion, and aging viscoelasticity.

## Where to go next

- [Layered sphere: layers transversely isotropic about the radius](@ref th-layered-sphere-radial):
  the matrices of the three modes of the sphere, and the parameters of a layer.
- [Layered cylinder: layers transversely isotropic about the radius](@ref th-layered-cylinder-radial):
  the matrices of the modes of the cylinder.
- [Layered cylinder: layers transversely isotropic about the radius, symbolically](@ref tut-layered-cylinder-radial-symbolic):
  the systems derived with symbols, and a power-law layer integrated in closed form.
- [Layered inclusions](@ref man-layered): the syntax.
- [A graded interphase around a spherical particle](@ref app-graded-interphase):
  the zone of [lutzIJSS2005](@citet) around an aggregate.
