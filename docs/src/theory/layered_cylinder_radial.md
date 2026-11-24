# [Layered cylinder: layers transversely isotropic about the radius](@id th-layered-cylinder-radial)

!!! info "Before this page"
    [Layered cylinder](@ref th-layered-cylinder), whose loading modes,
    interfaces and averages are kept here, and
    [The radial problem](@ref th-radial-problems), which solves a mode once its
    matrix is known.

A layer of a [`LayeredCylinder`](@ref) may be transversely isotropic about the
axis of the fiber, as [the previous pages](@ref th-layered-cylinder) assume, or
about the radial direction ``\underline{e}_r`` of the section: a wound sheet, a
coating deposited on a fiber, the rings of a trunk. Its plane of isotropy
``(\underline{e}_\theta, \underline{e}_z)`` then turns around the axis. This
page gives the matrix ``\boldsymbol{M}`` of each loading mode for such a layer;
with moduli that depend on ``r`` it is the matrix of a graded one. The matrices
are derived symbolically in
[the companion tutorial](@ref tut-layered-cylinder-radial-symbolic).

## 1. The moduli of a layer

In the frame ``(\underline{e}_r, \underline{e}_\theta, \underline{e}_z)`` of the
cylindrical chart, whose first vector is the radial one, Hill's moduli about
the radius are

```math
\begin{gathered}
n = C_{rrrr}, \qquad l = C_{rr\theta\theta} = C_{rrzz}, \qquad p = C_{r\theta r\theta} = C_{rzrz} , \\
k = \tfrac12\,(C_{\theta\theta\theta\theta} + C_{\theta\theta zz}), \qquad
m = \tfrac12\,(C_{\theta\theta\theta\theta} - C_{\theta\theta zz}) = C_{\theta z\theta z} ,
\end{gathered}
```

with ``C_{zzzz} = C_{\theta\theta\theta\theta}``. An isotropic layer
``(\kappa, \mu)`` has ``n = \kappa + \tfrac43\mu``, ``l = \kappa - \tfrac23\mu``,
``k = \kappa + \tfrac13\mu`` and ``m = p = \mu``. The conductivity is
``k_r\,\underline{e}_r\otimes\underline{e}_r + k_t\,(\boldsymbol{1} - \underline{e}_r\otimes\underline{e}_r)``.
The chart is that of the cylinder, whatever the direction of its axis in space.

```@example cylinder_radial
using MeanFieldHomogenization, TensND, LinearAlgebra
chart = CylindricalChart()
wound = radial_ti(chart; n = 12.0, l = 5.0, k = 19.0, m = 11.0, p = 4.0)
tensor_at(wound, [0.0, 1.0, 0.0])        # at a point of the axis e₂: the stiff plane is (e₁, e₃)
```

## 2. Antiplane shear, and conduction across the axis

With ``u_z = \varphi(r)\cos\theta``, the stresses are
``\sigma_{rz} = p\,\varphi'\cos\theta`` and
``\sigma_{\theta z} = -m\,(\varphi/r)\sin\theta``, and the axial equilibrium
reads ``(r\,p\,\varphi')' = m\,\varphi/r``. On the state
``Y = (\varphi,\; r\,p\,\varphi')``,

```math
\boldsymbol{M} = \begin{pmatrix} 0 & \dfrac{1}{p} \\[2mm] m & 0 \end{pmatrix} ,
\qquad s = \pm\sqrt{m/p} .
```

Conduction across the axis is the same problem, ``T = \hat T(r)\cos\theta`` and
``Y = (\hat T,\; r\,k_r\,\hat T')``, with ``k_r`` in place of ``p`` and ``k_t``
in place of ``m``. A homogeneous cylinder, whose regular mode is
``r^{\sqrt{k_t/k_r}}``, presents to the matrix the impedance
``k_r\sqrt{k_t/k_r}``: it disturbs nothing in a medium of conductivity
``\sqrt{k_r k_t}``.

```@example cylinder_radial
kₜ, kᵣ = 5.0, 2.0
rod = LayeredCylinder((1.0,), (radial_ti(chart, kₜ, kᵣ),))
TensND.get_data(property_contribution(rod, TensISO{3}(sqrt(kᵣ * kₜ))))     # across the axis, along it
```

Along the axis the gradient is uniform and the flux is that of the tangential
conductivity: the second coefficient is ``k_t - \sqrt{k_r k_t}``.

## 3. In-plane dilatation and axial strain

With ``\underline{u} = u_r(r)\,\underline{e}_r + \varepsilon z\,\underline{e}_z``,

```math
\begin{aligned}
\sigma_{rr} &= n\,u_r' + l\,\frac{u_r}{r} + l\,\varepsilon , \\
\sigma_{\theta\theta} &= l\,u_r' + (k + m)\,\frac{u_r}{r} + (k - m)\,\varepsilon , \\
\sigma_{zz} &= l\,u_r' + (k - m)\,\frac{u_r}{r} + (k + m)\,\varepsilon ,
\end{aligned}
```

and the radial equilibrium is ``(r\,\sigma_{rr})' = \sigma_{\theta\theta}``. On
``Y = (u_r,\; r\,\sigma_{rr})``, with ``c = k + m - l^2/n``,

```math
r\,\frac{\mathrm{d}Y}{\mathrm{d}r} = \begin{pmatrix} -\dfrac{l}{n} & \dfrac{1}{n} \\[2mm] c & \dfrac{l}{n} \end{pmatrix} Y
+ \varepsilon\,r \begin{pmatrix} -\dfrac{l}{n} \\[2mm] c - 2m \end{pmatrix} ,
\qquad s = \pm\sqrt{\frac{k + m}{n}} .
```

The axial strain is a second member, which vanishes when ``l = 0`` and
``k = m`` only. Since ``g = \varepsilon\,r`` obeys ``r\,g' = g``, it is appended
to the state and the system is homogeneous again, of constant matrix in a
uniform layer.

The mean axial stress is not a traction on the lateral surface. Over a layer
``a < r < b``,

```math
\int_a^b r\,\sigma_{zz}\,\mathrm{d}r = l\,\bigl[r\,u_r\bigr]_a^b + (k - m - l)\int_a^b u_r\,\mathrm{d}r + (k + m)\,\varepsilon\,\frac{b^2 - a^2}{2} ,
```

which needs the integral of ``u_r`` across the layer. The quantity
``J = \tfrac{1}{r}\int_a^r u_r\,\mathrm{d}r`` obeys ``r\,J' = u_r - J``: it is
appended as well, and the state ``(u_r,\; r\,\sigma_{rr},\; \varepsilon r,\; J)``
of a uniform layer is carried by the power of a ``4\times4`` matrix. In a graded
layer the integral of ``r\,\sigma_{zz}`` itself is carried along the
integration.

A homogeneous cylinder under an in-plane dilatation has the regular mode
``r^{s_+}`` and the impedance ``r\sigma_{rr}/u_r = n\,s_+ + l``, that of an
isotropic disc of plane-strain bulk modulus
``\tfrac12\bigl(\sqrt{n\,(k + m)} + l\bigr)``.

## 4. In-plane shear

With ``\underline{u} = U(r)\cos 2\theta\,\underline{e}_r + V(r)\sin 2\theta\,\underline{e}_\theta``,
the strains are ``\varepsilon_{rr} = U'\cos 2\theta``,
``\varepsilon_{\theta\theta} = (U + 2V)/r\,\cos 2\theta`` and
``2\varepsilon_{r\theta} = (V' - V/r - 2U/r)\sin 2\theta``. The stresses
``\sigma_{rr} = n\,\varepsilon_{rr} + l\,\varepsilon_{\theta\theta}`` and
``\sigma_{r\theta} = 2p\,\varepsilon_{r\theta}``, solved for ``U'`` and ``V'``,
give the first two rows; the two equations of equilibrium in the plane, with
``\sigma_{\theta\theta} = c\,(U + 2V)/r + (l/n)\,\sigma_{rr}``, give the last
two. On ``Y = (U,\; V,\; r\,\sigma_{rr},\; r\,\sigma_{r\theta})``, the stresses
being the radial functions of ``\sigma_{rr}`` on ``\cos 2\theta`` and of
``\sigma_{r\theta}`` on ``\sin 2\theta``,

```math
\boldsymbol{M} = \begin{pmatrix}
 -\dfrac{l}{n} & -\dfrac{2l}{n} & \dfrac{1}{n} & 0 \\[2mm]
 2 & 1 & 0 & \dfrac{1}{p} \\[2mm]
 c & 2c & \dfrac{l}{n} & -2 \\[2mm]
 2c & 4c & \dfrac{2l}{n} & -1
\end{pmatrix} .
```

Its eigenvalues are ``\pm1`` and ``\pm3`` for an isotropic layer, the modes of
the [isotropic cylinder](@ref th-layered-cylinder); the axial moduli do not
enter, ``k`` and ``m`` appearing through ``c`` alone.

```@example cylinder_radial
const LC = MeanFieldHomogenization.LayeredCylinders
exponents(c) = eigvals(Matrix(LC._shear_system(LC._hill_moduli(c, (0.0, 0.0, 1.0)))))
(isotropic = exponents(radial_ti(chart, iso_stiffness(4.0, 3.0))), wound = exponents(wound))
```

## 5. Chaining and averages

The modes are solved as [the radial problem](@ref th-radial-problems) says: the
regular states on the axis, the layers crossed one after the other, the
interfaces of the [layered cylinder](@ref th-layered-cylinder) between them,
and a matrix that is transversely isotropic about the axis. A cylinder may mix
layers of both kinds, and graded ones:

```@example cylinder_radial
ez = (0.0, 0.0, 1.0)
hill(k, l, m, n, p) = ti_from_constants(k + m, k - m, l, n, p, ez)
fiber, C₀ = hill(30.0, 12.0, 9.0, 150.0, 14.0), hill(5.0, 1.0, 2.0, 9.0, 3.0)
coated = LayeredCylinder((0.6, 1.0), (fiber, wound); interfaces = (SpringInterface(20.0, 12.5), PerfectInterface()))
property_contribution(coated, C₀)
```

```@example cylinder_radial
stiffening(r) = radial_ti(chart; n = 12.0 + 3r, l = 5.0, k = 19.0 - 2r, m = 11.0, p = 4.0 + r)
property_contribution(LayeredCylinder((0.6, 1.0), (fiber, stiffening)), C₀)
```

The averages are read on the states as for the cylinder of axial symmetry
([Layered cylinder](@ref th-layered-cylinder), section 5): ``u_r/r``,
``(U - V)/(2r)`` and ``\varphi/r`` for the mean strain of a disc,
``\sigma_{rr}``, ``(\sigma_{rr} - \sigma_{r\theta})/2`` and ``\sigma_{rz}`` for
its mean stress, the axial stress by the integral of section 3. With isotropic
layers written as radial ones, the tensors of the isotropic cylinder are
recovered to rounding, which is the first check of the implementation:

```@example cylinder_radial
C₁, C₂ = iso_stiffness(40.0, 20.0), iso_stiffness(4.0, 3.0)
axial = LayeredCylinder((0.6, 1.0), (C₁, C₂))
radial = LayeredCylinder((0.6, 1.0), (C₁, radial_ti(chart, C₂)))
norm(property_contribution(axial, C₀) - property_contribution(radial, C₀)) / norm(property_contribution(axial, C₀))
```

## 6. The parameters of a layer, and second moments

A coefficient of a layer is designated as for the sphere
([Layered sphere: layers transversely isotropic about the radius](@ref th-layered-sphere-radial),
sections 6 and 7), by its name in the frame of the cylindrical chart:

```@example cylinder_radial
rve = RVE()
add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C₀); fraction = :rest)
add_phase!(rve, :I, LayeredCylinder((0.6, 1.0), (C₁, wound)), Dict(:C => C₁); fraction = 0.3)
axial_modulus(C) = canonical_components(C)[3, 3, 3, 3]
[name => derivative(rve, MoriTanaka(), layer_param(:I, 2, name); indexer = axial_modulus) for name in (:Crrrr, :Cθθθθ, :Cθθzz, :Crrθθ, :Crθrθ)]
```

and the mean of ``\varepsilon_{rr}^2`` over a layer is the
[second moment](@ref th-second-moments) along the first element of the Walpole
basis about the radius:

```@example cylinder_radial
E = Tens(Symmetric([1.0 0.2 0.1; 0.2 -0.5 0.3; 0.1 0.3 0.7]))
second_moment(rve, Dilute(), (:I, 2), E, radial_ti(chart, 1.0, 0.0, 0.0, 0.0, 0.0))
```

## 7. Scope

- **Symmetry.** A layer is transversely isotropic about the axis of the
  cylinder or about its radial direction; any other class is refused.
- **Numbers.** Real, complex and dual numbers, nested.
- **Not available with radial or graded layers:** the pointwise fields and
  aging viscoelasticity.

## Where to go next

- [Layered cylinder: layers transversely isotropic about the radius, symbolically](@ref tut-layered-cylinder-radial-symbolic):
  the derivation of sections 2 to 4, and a power-law layer in closed form.
- [Layered inclusions](@ref man-layered): the syntax.
- [Layered spheroid](@ref th-layered-spheroid): the next pattern, confocal
  spheroidal layers.
