# [Layered cylinder](@id th-layered-cylinder)

!!! info "Before this page"
    [Layered sphere — bulk + shear recurrences and imperfect interfaces](@ref th-layered-sphere),
    whose method is followed here, and
    [Concentration and contribution tensors](@ref th-concentration), which
    defines the tensors a pattern without a Hill tensor supplies to the
    [Homogenization schemes](@ref th-homogenization).

[`LayeredCylinder`](@ref) is an infinite fiber of circular section made of `n`
coaxial layers, in an infinite matrix, under a remote uniform strain or
temperature gradient. It is the cylindrical counterpart of the layered sphere,
with one difference in the constituents: each layer, and the matrix, may be
**transversely isotropic about the fiber axis**. This is what a
unidirectional composite needs, since its effective medium is itself
transversely isotropic; a self-consistent or a differential scheme on aligned
fibers therefore stays within the model.

The solution of the coated fiber goes back to the composite-cylinder model of
[hashinRosen1964](@citet) and, for the transverse shear, to the three-phase
model of [christensenLo1979](@citet); [herve1995](@citet) gave it for any
number of coatings by transfer matrices, [hashin1979](@citet) treated
anisotropic constituents and [herve2002](@citet) conduction. The derivation
below is made with symbolic computations on
[Layered cylinder: the transfer relations, symbolically](@ref tut-layered-cylinder-symbolic).

## 1. Geometry and constituents

The radii are ascending, ``0 = r_0 < r_1 < \cdots < r_N``; layer ``k`` lies
between ``r_{k-1}`` and ``r_k`` (layer 1 is the core) and the matrix occupies
``r > r_N``. The axis is ``\underline{e}_3`` by default.

A stiffness that is transversely isotropic about ``\underline{e}_3`` has five
constants, written here with Hill's moduli:

```math
k = \tfrac12(C_{1111} + C_{1122}),\quad
m = \tfrac12(C_{1111} - C_{1122}),\quad
l = C_{1133},\quad n = C_{3333},\quad p = C_{2323},
```

``k`` being the plane-strain bulk modulus, ``m`` and ``p`` the transverse and
axial shear moduli. An isotropic layer ``(\kappa, \mu)`` has
``k = \kappa + \mu/3``, ``l = \kappa - 2\mu/3``, ``n = \kappa + 4\mu/3`` and
``m = p = \mu``. A conductivity has a transverse value ``k_T`` and an axial one
``k_A``.

```@example layered_cylinder
using MeanFieldHomogenization, TensND
ez = (0.0, 0.0, 1.0)
hill(k, l, m, n, p) = ti_from_constants(k + m, k - m, l, n, p, ez)   # (C₁₁₁₁, C₁₁₂₂, C₁₁₃₃, C₃₃₃₃, C₂₃₂₃)
fiber = hill(30.0, 12.0, 9.0, 150.0, 14.0)
coating = TensISO{3}(3 * 4.0, 2 * 1.5)                                # isotropic: (3κ, 2μ)
C₀ = hill(5.0, 1.0, 2.0, 9.0, 3.0)
cyl = LayeredCylinder((0.6, 1.0), (fiber, coating))
```

## 2. Four radial problems

A remote strain ``\boldsymbol{E}`` has, about the axis, an axial part
``\varepsilon = E_{33}``, an in-plane equibiaxial part
``e = \tfrac12(E_{11} + E_{22})``, an in-plane shear and an antiplane shear.
Each one drives a displacement of fixed angular shape, and the problem reduces
to radial functions. In each layer these are combinations of a few **modes**;
the **state** gathers the quantities that an interface transmits.

| Problem | Displacement | Modes | State |
| :--- | :--- | :--- | :--- |
| Axisymmetric | ``u_r = A r + B/r``, ``u_z = \varepsilon z`` | ``r,\ r^{-1}`` | ``(u_r, \sigma_{rr})`` |
| Transverse shear | ``u_r = U\cos 2\theta``, ``u_\theta = V\sin 2\theta`` | ``r,\ r^{3},\ r^{-3},\ r^{-1}`` | ``(U, V, \sigma_{rr}, \sigma_{r\theta})`` |
| Antiplane shear | ``u_z = (A r + B/r)\cos\theta`` | ``r,\ r^{-1}`` | ``(u_z, \sigma_{rz})`` |
| Transverse conduction | ``T = (A r + B/r)\cos\theta`` | ``r,\ r^{-1}`` | ``(T, -q_r)`` |

The stresses of the states are

```math
\begin{aligned}
\text{axisymmetric:}\quad & \sigma_{rr} = 2k\,A - 2m\,\frac{B}{r^2} + l\,\varepsilon,\\
\text{antiplane:}\quad & \sigma_{rz} = p\,\Big(A - \frac{B}{r^2}\Big),
\qquad\text{conduction:}\quad -q_r = k_T\,\Big(A - \frac{B}{r^2}\Big).
\end{aligned}
```

The axial strain enters the radial stress through ``l\,\varepsilon``: the axial
and the in-plane loadings are coupled, and the fiber answers an axial strain by
an in-plane one.

For the transverse shear, with ``\chi = m/k`` and the amplitudes
``(a, b, c, d)`` of the four modes,

```math
\begin{aligned}
U &= a\,r + (\chi - 1)\,b\,r^3 + \frac{c}{r^3} + (1 + \chi)\,\frac{d}{r},
&\sigma_{rr} &= m\,\Big(2a - \frac{6c}{r^4} - \frac{4d}{r^2}\Big),\\
V &= -a\,r + (\chi + 2)\,b\,r^3 + \frac{c}{r^3} - \chi\,\frac{d}{r},
&\sigma_{r\theta} &= m\,\Big(-2a + 6b\,r^2 - \frac{6c}{r^4} - \frac{2d}{r^2}\Big).
\end{aligned}
```

Only ``k`` and ``m`` enter. The mode ``a`` is the uniform strain; the modes
``c`` and ``d`` are singular on the axis and absent from the core, the modes
``b`` (and ``B`` above) grow at infinity and are absent from the matrix.

Along the axis the temperature gradient is uniform, the same in every layer.

## 3. Interfaces

A perfect interface transmits the state unchanged. The imperfect ones come in
pairs, as for the sphere: one lets the field jump, the other its flux.

| | Field jumps | Flux jumps |
| :--- | :--- | :--- |
| Elasticity | [`SpringInterface`](@ref) | [`MembraneInterface`](@ref) |
| Conduction | [`KapitzaInterface`](@ref) | [`SurfaceConductiveInterface`](@ref) |

A spring of normal and tangential compliances ``s_n = 1/k_n``, ``s_t = 1/k_t``
opens in proportion to the traction [hashin1990](@cite):

```math
[\![u_r]\!] = s_n\,\sigma_{rr},\qquad
[\![u_\theta]\!] = s_t\,\sigma_{r\theta},\qquad
[\![u_z]\!] = s_t\,\sigma_{rz}.
```

A membrane carries the surface stress
``\boldsymbol\sigma^{\mathrm s} = \lambda^{\mathrm s}(\operatorname{tr}\boldsymbol\varepsilon^{\mathrm s})\boldsymbol 1^{\mathrm s} + 2\mu^{\mathrm s}\boldsymbol\varepsilon^{\mathrm s}``
of the strain in its tangent plane [chenDvorakYu2006](@cite), and its
equilibrium, ``[\![\boldsymbol\sigma]\!]\cdot\underline{e}_r = -\operatorname{div}^{\mathrm s}\boldsymbol\sigma^{\mathrm s}``,
gives at the radius ``R``, with ``\kappa^{\mathrm s} = \lambda^{\mathrm s} + \mu^{\mathrm s}``:

```math
\begin{aligned}
\text{axisymmetric:}\quad & [\![\sigma_{rr}]\!] = \frac{1}{R}\Big((\kappa^{\mathrm s} + \mu^{\mathrm s})\,\frac{u_r}{R} + (\kappa^{\mathrm s} - \mu^{\mathrm s})\,\varepsilon\Big),\\
\text{transverse shear:}\quad & [\![\sigma_{rr}]\!] = (\kappa^{\mathrm s} + \mu^{\mathrm s})\,\frac{U + 2V}{R^2},\qquad
[\![\sigma_{r\theta}]\!] = 2\,[\![\sigma_{rr}]\!],\\
\text{antiplane:}\quad & [\![\sigma_{rz}]\!] = \mu^{\mathrm s}\,\frac{u_z}{R^2}.
\end{aligned}
```

In conduction, a Kapitza resistance ``\rho`` gives ``[\![T]\!] = -\rho\,q_r`` and
a conducting surface ``k^{\mathrm s}`` gives ``[\![q_r]\!] = -k^{\mathrm s}\,T/R^2``.

Each of these is the limit of a thin layer [benvenisteMiloh2001](@cite): a
spring is a soft layer of thickness ``h`` with ``s_n = h/(k + m)`` and
``s_t = h/m = h/p``, a membrane a stiff isotropic one with
``\mu^{\mathrm s} = h\mu`` and ``\lambda^{\mathrm s} = 2h\lambda\mu/(\lambda + 2\mu)``.
The test suite checks every jump against that limit, computed with perfect
interfaces.

## 4. Chaining the layers

In a layer, the amplitudes give the state at any radius, and a state gives the
amplitudes back. The recurrence starts in the core, where only the regular
modes exist, with a unit amplitude; at each radius it reads the state, applies
the jump of the interface, and extracts the amplitudes of the next layer. In
the matrix, the remote loading fixes the amplitude of the growing mode, which
normalizes the whole sequence. This is the transfer-matrix method of
[herve1995](@citet), written on states.

Two problems need a second sequence. The axisymmetric one is solved for
``e`` and for ``\varepsilon`` separately. The transverse shear has two regular
modes in the core, ``a`` and ``b``; two sequences are run and combined so that
the matrix carries ``(a, b) = (1, 0)``.

## 5. Averages

Over a disc of radius ``r`` of the section, the average strain is a contour
integral of the displacement, and the average of an equilibrated stress a
contour integral of the traction:

```math
\langle\boldsymbol\varepsilon\rangle = \frac{1}{\pi r^2}\oint \underline{u}\otimes^{\mathrm s}\underline{n}\,\mathrm{d}s,
\qquad
\langle\boldsymbol\sigma\rangle = \frac{1}{\pi r^2}\oint \underline{x}\otimes^{\mathrm s}(\boldsymbol\sigma\cdot\underline{n})\,\mathrm{d}s .
```

They are therefore read on the state:

| Problem | Average strain (gradient) | Average stress (flux) |
| :--- | :--- | :--- |
| Axisymmetric | ``e = u_r/r`` | ``\tfrac12(\sigma_{11} + \sigma_{22}) = \sigma_{rr}`` |
| Transverse shear | ``(U - V)/(2r)`` | ``(\sigma_{rr} - \sigma_{r\theta})/2`` |
| Antiplane, conduction | ``u_z/r``, ``T/r`` | ``\sigma_{rz}``, ``-q_r`` |

A layer is the difference of two discs. Read on the outer side of an
interface, an average includes what the interface carries — the opening of a
spring, which is a strain without stress, or the surface stress of a
membrane — and on its inner side it does not. The keywords `external` and
`internal` choose the side, with the meaning they have for the sphere
([Averages and interface jumps](@ref th-layered-sphere-jumps)).

The axial stress is not given by a traction on the lateral surface. It is
summed over the layers, ``\sigma_{zz} = 2l\,A + n\,\varepsilon`` in each, plus
the axial line force ``\sigma^{\mathrm s}_{zz}`` of each membrane; likewise a
conducting surface adds ``2k^{\mathrm s}r_k/r_N^2`` to the axial flux.

The strain concentration ``\mathbb{A}`` and the average stress ``\mathbb{B}``
per unit remote strain are transversely isotropic about the axis. They are
not symmetric: ``\mathbb A`` maps an axial strain to an in-plane one and not the
reverse. The contribution ``\mathbb N = \mathbb B - \mathbb C_0:\mathbb A`` is
symmetric.

```@example layered_cylinder
A = gradient_gradient_concentration(cyl, C₀, C₀)      # ⟨ε⟩ = A : E
B = flux_gradient_concentration(cyl, C₀, C₀)          # ⟨σ⟩ = B : E
N = property_contribution(cyl, C₀)
N ≈ B - C₀ ⊡ A
```

## 6. The composite-cylinder model recovered

A fiber coated by a matrix, read from outside, answers three of the four
loadings as a homogeneous fiber, whatever the medium around it: its moduli
``k``, ``l``, ``n`` and ``p`` are those of the composite-cylinder model
[hashinRosen1964](@cite), and satisfy Hill's universal relations
[hill1964](@cite). Its effective stiffness is ``\mathbb B:\mathbb A^{-1}``:

```@example layered_cylinder
c = 0.36                                        # fiber fraction
k_f, p_f = 30.0, 14.0
k_m, m_m, p_m = 4.0 + 1.5 / 3, 1.5, 1.5           # coating: k = κ + μ/3, m = p = μ
C_eq = B ⊡ inv(A)
C₁₁₁₁, C₁₁₂₂, _, _, C₂₃₂₃ = ti_constants(best_fit_ti(C_eq, ez))
k_HR = k_m + c / (1 / (k_f - k_m) + (1 - c) / (k_m + m_m))
p_HR = p_m * (p_f * (1 + c) + p_m * (1 - c)) / (p_f * (1 - c) + p_m * (1 + c))
((C₁₁₁₁ + C₁₁₂₂) / 2, k_HR), (C₂₃₂₃, p_HR)
```

Under transverse shear this equivalence does not hold: the answer of the
coated fiber depends on the surrounding medium, and the model only bounds the
modulus ``m``. Taking the surrounding medium as the unknown effective one is
the three-phase model [christensenLo1979](@cite), a fixed point of the
schemes rather than a property of the pattern.

## 7. In the schemes

A layered cylinder has no Hill tensor: it enters every scheme by its
tensors ``\mathbb A``, ``\mathbb B`` and ``\mathbb N``, evaluated in the reference of
the scheme. That reference must be isotropic or transversely isotropic about
the fiber axis. In a self-consistent or a differential scheme it is the
running estimate, which aligned fibers keep in that class; a matrix phase is
then given the shape of a one-layer cylinder.

```@example layered_cylinder
rve = RVE()
add_phase!(rve, :matrix, LayeredCylinder((1.0,), (coating,)), Dict(:C => coating); fraction = :rest)
add_phase!(rve, :fiber, cyl, Dict(:C => fiber); fraction = 0.4)
C_sc = homogenize(rve, SelfConsistent(), :C)
ti_constants(best_fit_ti(C_sc, ez))             # (C₁₁₁₁, C₁₁₂₂, C₁₁₃₃, C₃₃₃₃, C₂₃₂₃)
```

With an orientation distribution on the phase (`symmetrize = :iso`), the
tensors are averaged over the orientations, as for any inclusion.

## 8. Conduction, and impermeable layers

The transverse problem is that of the antiplane shear, with ``k_T`` in place of
``p``. A layer of zero conductivity is admissible at any rank. No flux crosses
it: the layers it encloses are shielded and carry no transverse gradient, and
its own field, the limit of a vanishing conductivity, vanishes on its inner
face, ``T = A\,(r - r_{k-1}^2/r)\cos\theta``. Two adjacent impermeable layers
leave the inner one undefined and are refused.

```@example layered_cylinder
K(t, a) = TensTI{2}(t, a, ez)                   # (transverse, axial)
pipe = LayeredCylinder((0.4, 0.7, 1.0), (K(5.0, 7.0), K(0.0, 0.0), K(1.0, 2.0)))
[TensND.get_data(gradient_gradient_concentration(pipe, K(2.0, 3.0); layer = j)) for j in 1:3]
```

## 9. Aging viscoelasticity

In aging linear viscoelasticity each modulus is a Volterra operator and each
amplitude a history ([Aging viscoelasticity (ALV)](@ref man-viscoelasticity)).
The relations of sections 2 to 5 hold with the products read in the order the
mechanics gives them: a modulus acts on the left of the strain it multiplies,
a division by a modulus is a left division, and a normalization by the remote
amplitude acts on the right. In particular ``\chi`` is ``k^{-1}\circ m``; the
other order does not satisfy equilibrium
([the symbolic page](@ref tut-layered-cylinder-symbolic) shows both). The
package writes the radial problems once, for numbers and for operators, so
the aging cylinder is the elastic one evaluated on Volterra matrices:
[`gradient_gradient_concentration_alv`](@ref) and
[`property_contribution_alv`](@ref) take a `LayeredCylinder` whose layers are
elastic tensors or `ViscoLaw`s, with the interfaces of section 3, which do
not age.

The reference of the Hill kernels of the aging schemes is isotropic. A
layered cylinder needs no Hill kernel, so a cell made only of coaxial layered
cylinders is solved by the self-consistent and differential schemes with a
transversely isotropic running estimate; among other shapes, a cylinder phase
takes an isotropic orientation average.

## 10. Pointwise fields

In the cylindrical frame of a point the local strain depends on the radius
only: with ``D = \tfrac12(E_{rr} - E_{\theta\theta})``,

```math
\begin{aligned}
\varepsilon_{rr} &= \Big(A_e - \frac{B_e}{r^2}\Big)e + \Big(A_\varepsilon - \frac{B_\varepsilon}{r^2}\Big)\varepsilon + U'\,D,
& \varepsilon_{r\theta} &= \tfrac12\Big(\frac{2U}{r} - V' + \frac{V}{r}\Big)E_{r\theta},\\
\varepsilon_{\theta\theta} &= \Big(A_e + \frac{B_e}{r^2}\Big)e + \Big(A_\varepsilon + \frac{B_\varepsilon}{r^2}\Big)\varepsilon + \frac{U + 2V}{r}\,D,
& \varepsilon_{rz} &= \varphi'\,E_{rz},\quad \varepsilon_{\theta z} = \frac{\varphi}{r}\,E_{\theta z},
\end{aligned}
```

``(A_e, B_e)`` and ``(A_\varepsilon, B_\varepsilon)`` being the axisymmetric
amplitudes under ``e`` and ``\varepsilon``, and ``\varphi = A r + B/r`` the
antiplane function. [`LayeredCylinderFields`](@ref) caches the amplitudes;
[`local_gradient`](@ref), [`local_flux`](@ref) and
[`local_displacement`](@ref) evaluate the fields at a point given as a vector
or by its cylindrical coordinates, in a layer or in the matrix, and `side`
chooses the limit on an interface.

```@example layered_cylinder
bonded = LayeredCylinder((0.6, 1.0), (fiber, coating); interfaces = (SpringInterface(20.0, 12.5), PerfectInterface()))
fields = LayeredCylinderFields(bonded, C₀)
E = Tens([0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 1.0])     # axial strain
jump = local_displacement(fields, 0.6, 0.0, 0.0, E; side = :outer) -
    local_displacement(fields, 0.6, 0.0, 0.0, E; side = :inner)
```

The radial opening of the spring under an axial strain is the Poisson effect
of the fiber against its coating.

## Where to go next

- [Layered cylinder: the transfer relations, symbolically](@ref tut-layered-cylinder-symbolic):
  the modes, the jumps and the operator order, derived.
- [Layered inclusions](@ref man-layered): the syntax shared by the layered
  patterns.
- [Cylindrical inclusions](@ref man-cylindrical-inclusions): the homogeneous
  cylinder, of any section, through its Hill tensor.
- [API — LayeredCylinder](@ref api-layered-cylinder).
