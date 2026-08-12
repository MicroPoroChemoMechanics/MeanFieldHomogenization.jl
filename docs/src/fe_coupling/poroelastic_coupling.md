# [The coupled poroelastic problem](@id fe-poro-coupling)

What an FE code has to solve once the material returns two fluxes instead of
one, and what is integrated in time. The homogenized coefficients themselves —
``\boldsymbol{B}``, ``M`` — are a property of the microstructure, not of the
coupling: see [Poromechanics](@ref manual-poromechanics).

## The two balances

Unknowns are the skeleton displacement ``\underline{u}`` and the pore pressure
``p``. Momentum balance, fluid mass balance and Darcy's law:

```math
\operatorname{div}\boldsymbol{\Sigma} = \underline{0},
\qquad
\dot{\varphi} + \operatorname{div}\underline{q} = 0,
\qquad
\underline{q} = -\frac{\boldsymbol{K}}{\mu}\cdot\underline{\nabla} p ,
```

closed by the material, which returns **both** fluxes and the permeability:

```math
\dot{\boldsymbol{\Sigma}} = \mathbb{C}^{\rm hom} : \dot{\boldsymbol{E}}
                          - \dot{p}\,\boldsymbol{B},
\qquad
\dot{\varphi} = \boldsymbol{B} : \dot{\boldsymbol{E}} + \frac{\dot{p}}{M},
\qquad
\boldsymbol{K} = \boldsymbol{K}(\omega_i) .
```

## What drives the microstructure

Not ``\boldsymbol{\Sigma}``, and not the Biot effective stress, but the
**Terzaghi** one. The loading ``(\boldsymbol{\Sigma}, p)`` splits into a dry
problem under ``\boldsymbol{\Sigma}' = \boldsymbol{\Sigma} + p\,\boldsymbol{1}``,
during which fractures open and close, plus a uniform field carrying no strain
singularity — which cannot move a flat crack ([barthelemyARMA2011](@cite) § 1.1).
In terms of what the FE code hands over,

```math
\boldsymbol{\Sigma}' = \mathbb{C}^{\rm hom} : \boldsymbol{E}
                     + p\,(\boldsymbol{1} - \boldsymbol{B}) ,
```

so the pressure term disappears for an incompressible solid
(``\boldsymbol{B} = \boldsymbol{1}``), as it must. Each family then follows

```math
\Delta\omega_i = \underline{n}_i \cdot
  (\mathbb{S}_i : \Delta\boldsymbol{\Sigma}') \cdot \underline{n}_i ,
\qquad
C_i = C_i^0 \left(\frac{\omega_i}{\omega_i^0}\right)^{\!3} ,
```

with ``\mathbb{S}_i`` the family's own contribution to the macroscopic
compliance ([`crack_family_compliances`](@ref MeanFieldHomogenization.Schemes.crack_family_compliances))
and the second relation the cubic (Poiseuille) law carrying the aperture into
the [fracture conductivity](@ref fe-permeability).

## Discretized in time and space

Backward Euler on ``[t_n, t_{n+1}]``, unknowns at ``t_{n+1}``, everything
dualized by a test pair ``(\delta\underline{u}, \delta p)``:

```math
\begin{aligned}
\int_\Omega \boldsymbol{\Sigma}_{n+1} : \nabla^{\rm s}\delta\underline{u}
   \,{\rm d}\Omega
&= \int_{\Gamma_T} \underline{T}^{\rm g}\cdot\delta\underline{u}\,{\rm d}S ,
\\[2pt]
\int_\Omega (\varphi_{n+1} - \varphi_n)\,\delta p \,{\rm d}\Omega
 + \Delta t \int_\Omega \underline{\nabla}\delta p \cdot
   \left(\frac{\boldsymbol{K}}{\mu}\cdot\underline{\nabla} p_{n+1}\right){\rm d}\Omega
&= -\,\Delta t \int_{\Gamma_Q} q^{\rm g}\,\delta p \,{\rm d}S .
\end{aligned}
```

Multiplying the mass balance by ``\Delta t`` is what keeps the two equations of
comparable magnitude, and makes the steady limit ``\Delta t \to \infty`` the
plain Darcy problem.

Newton on that pair uses exactly the four tangent blocks the material declares,
plus the Darcy term:

```math
\begin{bmatrix}
\displaystyle\int \nabla^{\rm s}\delta\underline{u} : \mathbb{C}^{\rm hom}
   : \nabla^{\rm s}\underline{u}
&
-\displaystyle\int (\nabla^{\rm s}\delta\underline{u} : \boldsymbol{B})\, p
\\[6pt]
\displaystyle\int \delta p\,(\boldsymbol{B} : \nabla^{\rm s}\underline{u})
&
\displaystyle\int \frac{\delta p\, p}{M}
 + \Delta t \int \underline{\nabla}\delta p \cdot
   \left(\frac{\boldsymbol{K}}{\mu}\cdot\underline{\nabla} p\right)
\end{bmatrix}
```

which is [`mfh_poro_element!`](@ref fe-backends), line for line.

## The four time-integration choices

| | choice | why |
|:--|:--|:--|
| **Implicit in** ``(\underline{u}, p)`` | backward Euler | unconditionally stable; a well test spans four decades in time and an explicit scheme would be unusable at the small end |
| **Explicit in** ``\boldsymbol{K}`` | the mobility is evaluated once per step from the converged state | ``\boldsymbol{K}`` follows the apertures through a self-consistent solve; differentiating it would couple the flow block to the mechanics through that solve, for a term of order ``\Delta\omega/\omega`` |
| **Sub-stepped at events** | the step is split at every closure and reopening | between two events the law is exactly linear, so the tangent is exact and the answer does not depend on how the loading was subdivided |
| **Logarithmic steps** | ``t_k`` geometric in each phase | the pressure diffuses as ``\sqrt{t}``; uniform steps resolve nothing early and waste everything late |

!!! note "Equal-order interpolation"
    ``\mathbb{Q}_1/\mathbb{Q}_1`` is used for ``(\underline{u}, p)``. The
    inf-sup condition would bite for an incompressible undrained limit; here the
    storage term ``1/M > 0`` — the fractures are compressible even though the
    fluid is not — keeps the pressure block regular. A vanishing ``1/M`` would
    need a Taylor–Hood pair instead.

The worked model is the [ARMA 2011 well test](@ref fe-arma2011).
