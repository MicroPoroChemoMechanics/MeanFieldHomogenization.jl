# [Scale transition](@id fe-scale-transition)

## What the FE code needs

At each quadrature point a finite-element code solving a nonlinear problem needs
two things from the material, per step:

```math
\boldsymbol{\sigma}_{n+1}
  = \mathcal{F}\!\left(\boldsymbol{\varepsilon}_{n+1},\, \alpha_n\right),
\qquad
\mathbb{C}^{\text{tg}}
  = \frac{\partial \boldsymbol{\sigma}_{n+1}}{\partial \boldsymbol{\varepsilon}_{n+1}} ,
```

the stress and the **consistent tangent**, together with the updated internal
state ``\alpha_{n+1}``. The residual uses the first; the Newton Jacobian uses the
second. A tangent that is merely close costs quadratic convergence — nothing
else, which is why a wrong one is easy to ship unnoticed.

## Where homogenization enters

The material response *is* a homogenization: the strain drives an RVE, the
scheme returns its effective stiffness, and

```math
\boldsymbol{\sigma} = \mathbb{C}^{\rm hom}(\alpha) : \boldsymbol{\varepsilon} .
```

All the nonlinearity sits in ``\alpha`` — the crack apertures, the open/closed
set, a damage variable. Between two events that change ``\alpha``, the law is
**linear**, so

```math
\mathbb{C}^{\text{tg}} = \mathbb{C}^{\rm hom}(\alpha)
```

is exact, not an approximation. No numerical differentiation is needed, and no
algorithmic tangent has to be derived: the scheme already returns it.

!!! note "This is what makes the coupling affordable"
    Because ``\mathbb{C}^{\rm hom}`` depends on the state only through a
    *discrete* configuration (which families are open), the expensive scheme
    solve is shared by every quadrature point in the same configuration — see
    [`MaterialCache`](@ref).

## Several gradients, several fluxes

A poroelastic material takes a strain **and** a pore pressure, and returns a
stress **and** a variation of fluid content:

```math
\begin{aligned}
\dot{\boldsymbol{\Sigma}} &= \mathbb{C}^{\rm hom} : \dot{\boldsymbol{E}} - \dot{p}\,\boldsymbol{B}, \\
\dot{\varphi}           &= \boldsymbol{B} : \dot{\boldsymbol{E}} + \frac{\dot{p}}{M} .
\end{aligned}
```

The contract is therefore written with **named gradients and fluxes**, and the
tangent as a set of blocks keyed *flux then gradient*:

| block | value | key |
|:--|:--|:--|
| ``\partial\boldsymbol{\Sigma}/\partial\boldsymbol{E}`` | ``\mathbb{C}^{\rm hom}`` | `:σε` |
| ``\partial\boldsymbol{\Sigma}/\partial p``         | ``-\boldsymbol{B}``          | `:σp` |
| ``\partial\varphi/\partial\boldsymbol{E}``           | ``\boldsymbol{B}``           | `:φε` |
| ``\partial\varphi/\partial p``                   | ``1/M``                  | `:φp` |

A purely mechanical law declares only `:σε` and never sees the rest. This is the
same shape MGIS uses for MFront's generic behaviors.

## The one rule that bites

A `Tensors.jl` tensor — what Ferrite and every Julia FE code speaks — has **no
basis**: its components are always global.

A TensND tensor returns its components in **its own** basis, and a homogenized
stiffness whose RVE carries tilted crack families comes back in a *rotated*
basis. Handing its raw array to an FE assembler rotates the material silently.
Cross the boundary with [`to_tensors`](@ref) and [`from_tensors`](@ref), never
with `get_array`.

```@example frames
using MeanFieldHomogenization, TensND

C₀ = TensISO{3}(3 * 30.0, 2 * 18.0)
rve = RVE(:M)
add_matrix!(rve, Ellipsoid(1.0), Dict(:C => C₀))
add_phase!(rve, :F, PennyCrack(1.0; euler_angles = (π / 4, 0.0)), Dict(:C => C₀);
           density = 0.08)

C_hom = homogenize(rve, MoriTanaka())
raw    = get_array(C_hom)[1, 1, 1, 1]          # components in ITS OWN basis
global_ = to_tensors(C_hom)[1, 1, 1, 1]        # components an FE code expects
(raw, global_)
```

The two differ: the first is expressed in the crack frame, the second in the
global one.
