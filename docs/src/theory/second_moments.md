# [Second moments of the local fields](@id th-second-moments)

!!! info "Before this page"
    [Homogenization schemes](@ref th-homogenization), for the effective
    stiffness and the concentration tensors of an estimate;
    [Levin's theorem](@ref th-levin), which draws the mean of a field over a
    phase from them;
    [Derivatives by automatic differentiation](@ref th-autodiff), for the
    derivative along a tensor used throughout.

The concentration tensor of a phase gives the mean of the strain over that
phase, ``\langle\boldsymbol{\varepsilon}\rangle_r = \mathbb{A}_r:\boldsymbol{E}``.
A criterion written on the local strain (a yield stress, a strength) is not
linear in it: it involves the mean of a quadratic form,
``\langle\boldsymbol{\varepsilon}:\mathbb{B}:\boldsymbol{\varepsilon}\rangle_r``,
which the mean of the strain does not give, since the strain fluctuates within
the phase. These **second moments** follow from the effective stiffness itself:
they are its derivatives with respect to the stiffness of the phase. The result
is due to [kreher1990](@citet); it is the basis of the modified secant method of
nonlinear homogenization [suquet1997](@cite), whose estimates are those of the
variational procedure of [ponteCastaneda1991](@citet). This page states it,
proves it, and writes it in every symmetry class. The function is
[`second_moment`](@ref); its syntax is in
[Sensitivities](@ref man-sensitivities-moments).

## 1. Statement

The representative volume element ``\Omega`` is made of phases of stiffness
``\mathbb{C}_r`` and volume fraction ``f_r``, and is subjected to the
macroscopic strain ``\boldsymbol{E}``. Its effective stiffness
``\mathbb{C}^{\mathrm{hom}}`` is a function of the stiffnesses of the phases.
For any tensor ``\mathbb{B}`` having the symmetries of a stiffness,

```math
f_r\,\langle\boldsymbol{\varepsilon}:\mathbb{B}:\boldsymbol{\varepsilon}\rangle_r
  = \frac{\mathrm{d}}{\mathrm{d}t}\,
    \boldsymbol{E}:\mathbb{C}^{\mathrm{hom}}\bigl(\mathbb{C}_r + t\,\mathbb{B}\bigr):\boldsymbol{E}\,\Big|_{t=0} ,
```

the stiffnesses of the other phases being fixed: the mean over a phase of a
quadratic form of the strain is the derivative of twice the effective energy
when the stiffness of that phase moves along the tensor of the form.

## 2. Proof

The local strain ``\boldsymbol{\varepsilon}`` derives from a displacement equal
to ``\boldsymbol{E}\cdot\underline{x}`` on the boundary of ``\Omega``, and the
stress ``\boldsymbol{\sigma} = \mathbb{C}:\boldsymbol{\varepsilon}`` is in
equilibrium. Hill's lemma [hill1963](@cite) states that the mean of the work of
a stress field in equilibrium in a compatible strain field is the work of
their means; applied to the solution, it gives the effective energy as the
mean of the local one:

```math
\boldsymbol{E}:\mathbb{C}^{\mathrm{hom}}:\boldsymbol{E}
  = \langle\boldsymbol{\sigma}\rangle:\langle\boldsymbol{\varepsilon}\rangle
  = \langle\boldsymbol{\varepsilon}:\mathbb{C}:\boldsymbol{\varepsilon}\rangle .
```

Let the stiffness field change by ``\mathrm{d}\mathbb{C}``, the macroscopic
strain being kept. The strain field changes by
``\mathrm{d}\boldsymbol{\varepsilon}``, which derives from a displacement that
vanishes on the boundary, so that
``\langle\mathrm{d}\boldsymbol{\varepsilon}\rangle = \boldsymbol{0}``. The
stiffness having the major symmetry,

```math
\mathrm{d}\,\langle\boldsymbol{\varepsilon}:\mathbb{C}:\boldsymbol{\varepsilon}\rangle
  = \langle\boldsymbol{\varepsilon}:\mathrm{d}\mathbb{C}:\boldsymbol{\varepsilon}\rangle
  + 2\,\langle\boldsymbol{\sigma}:\mathrm{d}\boldsymbol{\varepsilon}\rangle ,
\qquad
\langle\boldsymbol{\sigma}:\mathrm{d}\boldsymbol{\varepsilon}\rangle
  = \langle\boldsymbol{\sigma}\rangle:\langle\mathrm{d}\boldsymbol{\varepsilon}\rangle = 0 ,
```

the second equality being Hill's lemma again. The change of the strain field
does not contribute at the first order: the energy is stationary with respect
to the field. What remains is

```math
\boldsymbol{E}:\mathrm{d}\mathbb{C}^{\mathrm{hom}}:\boldsymbol{E}
  = \langle\boldsymbol{\varepsilon}:\mathrm{d}\mathbb{C}:\boldsymbol{\varepsilon}\rangle ,
```

and the statement is the case of a change ``\mathrm{d}t\,\mathbb{B}`` in phase
``r`` and none elsewhere. Nothing is assumed of ``\mathbb{B}`` beyond the
symmetries of a stiffness: it need not belong to the symmetry class of
``\mathbb{C}_r``.

## 3. By symmetry class

Written in a symmetry class, the stiffness of the phase is a combination of
fixed tensors, ``\mathbb{C}_r = \sum_a c_a\,\mathbb{B}_a``, and the derivative
along ``\mathbb{B}_a`` is the partial derivative with respect to ``c_a``
([Derivatives by automatic differentiation](@ref th-autodiff), section 6).
Each coefficient thus gives the second moment of the quadratic form it
multiplies in the energy:

```math
f_r\,\langle\boldsymbol{\varepsilon}:\mathbb{B}_a:\boldsymbol{\varepsilon}\rangle_r
  = \frac{\partial}{\partial c_a}\bigl(\boldsymbol{E}:\mathbb{C}^{\mathrm{hom}}:\boldsymbol{E}\bigr) .
```

**Isotropic phase**, ``\mathbb{C}_r = 3k_r\,\mathbb{J} + 2\mu_r\,\mathbb{K}``.
With ``\boldsymbol{\varepsilon}^d = \mathbb{K}:\boldsymbol{\varepsilon}`` the
deviator of the strain,
``\boldsymbol{\varepsilon}:3\mathbb{J}:\boldsymbol{\varepsilon} = (\mathrm{tr}\,\boldsymbol{\varepsilon})^2``
and
``\boldsymbol{\varepsilon}:\mathbb{K}:\boldsymbol{\varepsilon} = \boldsymbol{\varepsilon}^d:\boldsymbol{\varepsilon}^d``,
so that

```math
f_r\,\langle(\mathrm{tr}\,\boldsymbol{\varepsilon})^2\rangle_r
  = \frac{\partial}{\partial k_r}\bigl(\boldsymbol{E}:\mathbb{C}^{\mathrm{hom}}:\boldsymbol{E}\bigr) ,
\qquad
2 f_r\,\langle\boldsymbol{\varepsilon}^d:\boldsymbol{\varepsilon}^d\rangle_r
  = \frac{\partial}{\partial\mu_r}\bigl(\boldsymbol{E}:\mathbb{C}^{\mathrm{hom}}:\boldsymbol{E}\bigr) .
```

When the effective stiffness is isotropic as well,
``\boldsymbol{E}:\mathbb{C}^{\mathrm{hom}}:\boldsymbol{E} = k^{\mathrm{hom}}\,(\mathrm{tr}\,\boldsymbol{E})^2 + 2\mu^{\mathrm{hom}}\,\boldsymbol{E}^d:\boldsymbol{E}^d``
and the two moments are

```math
\begin{aligned}
f_r\,\langle(\mathrm{tr}\,\boldsymbol{\varepsilon})^2\rangle_r
  &= \frac{\partial k^{\mathrm{hom}}}{\partial k_r}\,(\mathrm{tr}\,\boldsymbol{E})^2
   + 2\,\frac{\partial\mu^{\mathrm{hom}}}{\partial k_r}\,\boldsymbol{E}^d:\boldsymbol{E}^d , \\
f_r\,\langle\boldsymbol{\varepsilon}^d:\boldsymbol{\varepsilon}^d\rangle_r
  &= \frac12\,\frac{\partial k^{\mathrm{hom}}}{\partial\mu_r}\,(\mathrm{tr}\,\boldsymbol{E})^2
   + \frac{\partial\mu^{\mathrm{hom}}}{\partial\mu_r}\,\boldsymbol{E}^d:\boldsymbol{E}^d .
\end{aligned}
```

A hydrostatic macroscopic strain produces deviatoric strains in the phases as
soon as ``k^{\mathrm{hom}}`` depends on ``\mu_r``, although their mean has no
deviator.

**Transversely isotropic phase** about the unit vector ``\underline{n}``. With
``\boldsymbol{1}_T = \boldsymbol{1} - \underline{n}\otimes\underline{n}``, the
strain splits into its axial part
``\varepsilon_n = \underline{n}\cdot\boldsymbol{\varepsilon}\cdot\underline{n}``,
its part in the transverse plane
``\boldsymbol{\varepsilon}_T = \boldsymbol{1}_T\cdot\boldsymbol{\varepsilon}\cdot\boldsymbol{1}_T``
and the shear
``\underline{\gamma} = \boldsymbol{1}_T\cdot\boldsymbol{\varepsilon}\cdot\underline{n}``
between them. The five coefficients of the stiffness on the Walpole basis
([Conventions](@ref th-notation-bases)) multiply the forms

| coefficient | quadratic form of the strain |
| :--- | :--- |
| ``\ell_1`` | ``\varepsilon_n^2`` |
| ``\ell_2`` | ``\tfrac12\,(\mathrm{tr}\,\boldsymbol{\varepsilon}_T)^2`` |
| ``\ell_3`` (on ``\mathbb{W}_3 + \mathbb{W}_4``) | ``\sqrt2\,\varepsilon_n\,\mathrm{tr}\,\boldsymbol{\varepsilon}_T`` |
| ``\ell_5`` | ``\boldsymbol{\varepsilon}_T:\boldsymbol{\varepsilon}_T - \tfrac12\,(\mathrm{tr}\,\boldsymbol{\varepsilon}_T)^2`` |
| ``\ell_6`` | ``2\,\underline{\gamma}\cdot\underline{\gamma}`` |

whose five moments are the five partial derivatives of the energy.

**Orthotropic phase** in the orthonormal frame
``(\underline{e}_1, \underline{e}_2, \underline{e}_3)``, with
``\varepsilon_{ij} = \underline{e}_i\cdot\boldsymbol{\varepsilon}\cdot\underline{e}_j``.
The nine components of the stiffness multiply ``\varepsilon_{ii}^2``
(``C_{iiii}``), ``2\,\varepsilon_{ii}\,\varepsilon_{jj}`` (``C_{iijj}``,
``i \neq j``) and ``4\,\varepsilon_{ij}^2`` (``C_{ijij}``, ``i \neq j``).

A class gives as many moments as it has coefficients, and no more: the
derivatives with respect to ``k_r`` and ``\mu_r`` do not separate
``\langle\varepsilon_n^2\rangle_r`` from the other terms of
``\langle\boldsymbol{\varepsilon}^d:\boldsymbol{\varepsilon}^d\rangle_r``. But the
class is that of the *direction*, not of the phase: the moment of any form is
obtained by moving an isotropic stiffness along a tensor that is not isotropic.

## 4. The whole tensor

The tensors having the symmetries of a stiffness form a space of dimension 21
(6 in the plane), with the scalar product
``\mathbb{A}::\mathbb{B} = A_{ijkl}\,B_{ijkl}``. The tensor
``\boldsymbol{\varepsilon}\otimes\boldsymbol{\varepsilon}`` belongs to it, and
``\boldsymbol{\varepsilon}:\mathbb{B}:\boldsymbol{\varepsilon} = (\boldsymbol{\varepsilon}\otimes\boldsymbol{\varepsilon})::\mathbb{B}``.
Defining the derivative of a scalar ``\varphi`` with respect to a stiffness as
the tensor of that space such that
``\mathrm{d}\varphi = (\partial\varphi/\partial\mathbb{C}_r)::\mathrm{d}\mathbb{C}_r``
for every change having these symmetries, the statement becomes

```math
f_r\,\langle\boldsymbol{\varepsilon}\otimes\boldsymbol{\varepsilon}\rangle_r
  = \frac{\partial}{\partial\mathbb{C}_r}\bigl(\boldsymbol{E}:\mathbb{C}^{\mathrm{hom}}:\boldsymbol{E}\bigr) ,
```

and, on an orthonormal basis ``(\mathbb{B}_a)`` of the space,
``\langle\boldsymbol{\varepsilon}\otimes\boldsymbol{\varepsilon}\rangle_r = \sum_a \langle\boldsymbol{\varepsilon}:\mathbb{B}_a:\boldsymbol{\varepsilon}\rangle_r\,\mathbb{B}_a``:
21 derivatives, taken in one evaluation of the estimate. The Kelvin-Mandel
matrix ([Conventions](@ref th-notation-storage)) provides such a basis, since
it turns ``::`` into the scalar product of matrices.

The cell below is a matrix with aligned oblate spheroids, under a strain that
has no particular direction:

```@example sm
using MeanFieldHomogenization, TensND, LinearAlgebra

function cell(C₀, C₁, f; shape = Ellipsoid(1.0), key = :C)
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(key => C₀); fraction = :rest)
    add_phase!(rve, :I, shape, Dict(key => C₁); fraction = f)
    return rve
end
k₀, μ₀, k₁, μ₁, f = 10.0, 6.0, 40.0, 20.0, 0.3
rve = cell(iso_stiffness(k₀, μ₀), iso_stiffness(k₁, μ₁), f; shape = Ellipsoid(1.0, 1.0, 0.4))
E = Tens(Symmetric([1.0 0.2 0.1; 0.2 -0.5 0.3; 0.1 0.3 0.7]))
𝕁, 𝕂 = TensISO{3}(1.0, 0.0), TensISO{3}(0.0, 1.0)

M = second_moment(rve, MoriTanaka(), :M, E)           # ⟨ε ⊗ ε⟩ over the matrix
(M ⊙ 𝕂, second_moment(rve, MoriTanaka(), :M, E, 𝕂))   # ⟨εᵈ:εᵈ⟩, from the tensor and alone
```

The form may be any one: here the square of the strain along the axis of the
spheroids, ``\langle\varepsilon_n^2\rangle``, in each phase.

```@example sm
n = [0.0, 0.0, 1.0]
𝕎₁ = Tens(n) ⊗ Tens(n) ⊗ Tens(n) ⊗ Tens(n)
[second_moment(rve, MoriTanaka(), φ, E, 𝕎₁) for φ in (:M, :I)]
```

## 5. Fluctuations, and the moments of the stress

The second moment exceeds the square of the mean by the covariance of the
strain in the phase,

```math
\langle\boldsymbol{\varepsilon}\otimes\boldsymbol{\varepsilon}\rangle_r
  - \langle\boldsymbol{\varepsilon}\rangle_r\otimes\langle\boldsymbol{\varepsilon}\rangle_r
  = \bigl\langle(\boldsymbol{\varepsilon} - \langle\boldsymbol{\varepsilon}\rangle_r)
    \otimes(\boldsymbol{\varepsilon} - \langle\boldsymbol{\varepsilon}\rangle_r)\bigr\rangle_r ,
```

a positive tensor (its quadratic form on symmetric tensors of order 2 is
positive), which vanishes when the strain is uniform in the phase and only
then. The mean ``\langle\boldsymbol{\varepsilon}\rangle_r = \mathbb{A}_r:\boldsymbol{E}``
comes from the concentration tensor of the estimate. In the cell above, the
eigenvalues of the covariance, in each phase:

```@example sm
function covariance(scheme, φ)
    ε̄ = concentration_tensors(rve, scheme, :C).strain[φ] ⊡ E
    return second_moment(rve, scheme, φ, E) - ε̄ ⊗ ε̄
end
spectrum(T) = extrema(eigvals(Symmetric(collect(mandel(T, CanonicalBasis{3, Float64}())))))
(matrix = spectrum(covariance(MoriTanaka(), :M)), inclusions = spectrum(covariance(MoriTanaka(), :I)))
```

The Mori-Tanaka estimate gives the ellipsoidal inclusions a uniform strain,
zero covariance up to rounding, and lets the matrix fluctuate. In a
self-consistent estimate both phases fluctuate:

```@example sm
sc = SelfConsistent(; abstol = 0.0, reltol = 1.0e-12)
(matrix = spectrum(covariance(sc, :M)), inclusions = spectrum(covariance(sc, :I)))
```

The stiffness being uniform in the phase,
``\boldsymbol{\sigma} = \mathbb{C}_r:\boldsymbol{\varepsilon}`` there and the
moments of the stress follow from those of the strain:

```math
\langle\boldsymbol{\sigma}\otimes\boldsymbol{\sigma}\rangle_r
  = \mathbb{C}_r:\langle\boldsymbol{\varepsilon}\otimes\boldsymbol{\varepsilon}\rangle_r:\mathbb{C}_r .
```

Taking ``\mathbb{B} = \mathbb{C}_r`` and summing over the phases returns the
energy, ``\sum_r f_r\,\langle\boldsymbol{\varepsilon}:\mathbb{C}_r:\boldsymbol{\varepsilon}\rangle_r = \boldsymbol{E}:\mathbb{C}^{\mathrm{hom}}:\boldsymbol{E}``.
For an estimate, this is Euler's identity for a function homogeneous of degree
one in the stiffnesses, which every scheme is:

```@example sm
sum(volume_fraction(rve, φ) * second_moment(rve, sc, φ, E, C) for (φ, C) in ((:M, iso_stiffness(k₀, μ₀)), (:I, iso_stiffness(k₁, μ₁)))),
    E ⊡ (homogenize(rve, sc, :C) ⊡ E)
```

## 6. With prestresses

The phases may carry prestresses, as in
[Levin's theorem](@ref th-levin):
``\boldsymbol{\sigma} = \mathbb{C}_r:\boldsymbol{\varepsilon} + \boldsymbol{\sigma}^\ast_r``.
The potential of the cell under the macroscopic strain ``\boldsymbol{E}`` is
then

```math
W = \Bigl\langle \tfrac12\,\boldsymbol{\varepsilon}:\mathbb{C}:\boldsymbol{\varepsilon}
  + \boldsymbol{\sigma}^\ast:\boldsymbol{\varepsilon} \Bigr\rangle ,
\qquad
\mathrm{d}W = \tfrac12\,\langle\boldsymbol{\varepsilon}:\mathrm{d}\mathbb{C}:\boldsymbol{\varepsilon}\rangle
  + \langle\mathrm{d}\boldsymbol{\sigma}^\ast:\boldsymbol{\varepsilon}\rangle
  + \langle\boldsymbol{\sigma}:\mathrm{d}\boldsymbol{\varepsilon}\rangle ,
```

and the last term vanishes for the same reason as in section 2. The two
moments of the strain in a phase are the two derivatives of the potential,

```math
f_r\,\langle\boldsymbol{\varepsilon}\rangle_r = \frac{\partial W}{\partial\boldsymbol{\sigma}^\ast_r} ,
\qquad
\frac{f_r}{2}\,\langle\boldsymbol{\varepsilon}\otimes\boldsymbol{\varepsilon}\rangle_r
  = \frac{\partial W}{\partial\mathbb{C}_r}\Big|_{\boldsymbol{\sigma}^\ast} ,
```

the stiffness being moved at fixed prestresses. This is the setting of
[kreher1990](@citet), and that of the strength of a frictional matrix, whose
comparison composite is prestressed
([Strength of a frictional matrix](@ref ml-theory-frictional-strength)).

## 7. Conduction

The proof uses the symmetry of the local law and Hill's lemma, which hold for
the conduction problem with the dictionary of
[Extension to conductivity](@ref th-conductivity): under the macroscopic
gradient ``\underline{G}``, with ``\underline{g}`` the local gradient,
``\boldsymbol{K}_r`` the conductivity of the phase and ``\boldsymbol{B}`` a
symmetric tensor of order 2,

```math
\begin{aligned}
f_r\,\langle\underline{g}\cdot\boldsymbol{B}\cdot\underline{g}\rangle_r
  &= \frac{\mathrm{d}}{\mathrm{d}t}\,
    \underline{G}\cdot\boldsymbol{K}^{\mathrm{hom}}\bigl(\boldsymbol{K}_r + t\,\boldsymbol{B}\bigr)\cdot\underline{G}\,\Big|_{t=0} , \\
f_r\,\langle\underline{g}\otimes\underline{g}\rangle_r
  &= \frac{\partial}{\partial\boldsymbol{K}_r}\bigl(\underline{G}\cdot\boldsymbol{K}^{\mathrm{hom}}\cdot\underline{G}\bigr) .
\end{aligned}
```

```@example sm
cond = cell(TensISO{3}(1.0), TensISO{3}(2.5), 0.2; key = :K)
G = Tens([1.0, 0.0, 0.0])
second_moment(cond, MoriTanaka(), :M, G, TensISO{3}(1.0); property = :K)   # ⟨g·g⟩ over the matrix
```

## 8. What an estimate gives

For the exact effective stiffness of a microstructure, the theorem is exact.
A scheme provides an estimate of ``\mathbb{C}^{\mathrm{hom}}`` as a function of
the stiffnesses of the phases; its derivative is the second moment in the
microstructures of which the scheme is the exact solution, and it *defines* the
second moment associated with the estimate otherwise. Two cases where the
estimate is an exact solution serve as checks.

**A laminate.** The fields are uniform in each layer of the exact solution
([Laminate](@ref th-laminate)): the second moment is the square of the mean.

```@example sm
lam = Laminate(; normal = (0, 0, 1))
add_layer!(lam, :A, Dict(:C => iso_stiffness(k₁, μ₁)); thickness = 0.3)
add_layer!(lam, :B, Dict(:C => iso_stiffness(k₀, μ₀)); thickness = 0.7)
ε_A = concentration_tensors(lam, Laminated(), :C).strain[:A] ⊡ E
norm(second_moment(lam, Laminated(), :A, E) - ε_A ⊗ ε_A) / norm(ε_A ⊗ ε_A)
```

**The composite sphere.** Under a hydrostatic strain
``\boldsymbol{E} = e\,\boldsymbol{1}``, a sphere of radius ``a`` and moduli
``(k_1, \mu_1)`` coated by a shell of outer radius ``b`` and moduli
``(k_0, \mu_0)``, with ``f = a^3/b^3``, deforms as the homogeneous medium that
surrounds it when the latter has the bulk modulus of
[hashin1962spheres](@citet), which is also the Mori-Tanaka estimate of spheres.
The displacement is radial, ``A_1\,r`` in the core and ``A\,r + B/r^2`` in the
shell. Its continuity and that of the radial stress at ``r = a``, with the
displacement ``e\,b`` at ``r = b``, give

```math
A = \frac{(3k_1 + 4\mu_0)\,e}{3k_1 + 4\mu_0 - 3f\,(k_1 - k_0)} ,
\qquad
B = -\frac{3\,(k_1 - k_0)\,a^3}{3k_1 + 4\mu_0}\,A ,
\qquad
A_1 = A + \frac{B}{a^3} .
```

In the shell, ``\mathrm{tr}\,\boldsymbol{\varepsilon} = 3A`` is uniform and the
deviator has the radial component ``-2B/r^3`` and the two others ``B/r^3``, so
that ``\boldsymbol{\varepsilon}^d:\boldsymbol{\varepsilon}^d = 6B^2/r^6``, whose
mean over the shell is

```math
\langle\boldsymbol{\varepsilon}^d:\boldsymbol{\varepsilon}^d\rangle_0
  = \frac{3}{b^3 - a^3}\int_a^b \frac{6B^2}{r^6}\,r^2\,\mathrm{d}r
  = \frac{6B^2}{a^3\,b^3} .
```

The shell shears although neither the macroscopic strain nor the mean strain
of the shell has a deviator. The derivative of the Mori-Tanaka estimate returns
these moments:

```@example sm
spheres = cell(iso_stiffness(k₀, μ₀), iso_stiffness(k₁, μ₁), f)
e = 1.0
A = (3k₁ + 4μ₀) * e / (3k₁ + 4μ₀ - 3f * (k₁ - k₀))
B = -3(k₁ - k₀) * f / (3k₁ + 4μ₀) * A                  # with b = 1, a³ = f
(
    shear = (second_moment(spheres, MoriTanaka(), :M, TensISO{3}(e), 𝕂), 6B^2 / f),
    volume = (second_moment(spheres, MoriTanaka(), :M, TensISO{3}(e), 3𝕁), 9A^2),
)
```

and so does it with symbols, the difference with the closed form simplifying
to zero:

```@example sm
using Symbolics
@variables K₀ M₀ K₁ M₁ F
sym = cell(iso_stiffness(K₀, M₀), iso_stiffness(K₁, M₁), F)
Bˢ = -3(K₁ - K₀) * F / (3K₁ + 4M₀ - 3F * (K₁ - K₀))
simplify(second_moment(sym, MoriTanaka(), :M, TensISO{3}(1), 𝕂) - 6Bˢ^2 / F)
```

## Where to go next

The next chapter keeps every scheme and changes the physics:
[Extension to conductivity](@ref th-conductivity) states the dictionary through
which the transport problem of order 2 reuses the whole chain, the second
moments included.

- [Sensitivities](@ref man-sensitivities-moments): the syntax of
  `second_moment` and of the derivative along a tensor.
- [Secant methods](@ref ml-theory-secant) and
  [Nonlinear homogenization by the secant method](@ref tut-secant-elastoplasticity):
  the second moment of the equivalent strain as the effective strain of a
  nonlinear phase.
- [From derivatives to a strength criterion](@ref tut-strength-criteria) and
  [Quasi-brittle strength of cement paste and mortar](@ref app-strength): the
  second moment of the stress in a strength criterion.
