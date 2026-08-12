# [Fractured permeability](@id fe-permeability)

The hydraulic half of a fractured-rock model. A fracture is not an obstacle to
flow but the **preferential path** through an almost impermeable matrix, which
is the opposite of what an ordinary [crack](@ref man-cracks) does in transport.

## The flowing crack

Flatten an oblate spheroid ``(a, a, c)`` while its conductivity ``k_f``
diverges, keeping the **fracture conductivity** finite:

```math
C = 2\,c\,k_f \qquad (\text{conductivity} \times \text{aperture}).
```

With ``\gamma = C/2a``, the contribution ``\boldsymbol{k}`` per unit
``(4\pi/3)\,d`` in an isotropic matrix ``k_0`` is purely in-plane, ``\underline{n}``
being the unit normal to the fracture plane:

```math
\boldsymbol{k} = \frac{\gamma}{1 + \dfrac{\pi\gamma}{4k_0}}
            \left(\boldsymbol{1} - \underline{n}\otimes\underline{n}\right),
\qquad
\boldsymbol{k} \xrightarrow[C\to\infty]{} \frac{4k_0}{\pi}
            \left(\boldsymbol{1} - \underline{n}\otimes\underline{n}\right).
```

```@example perm
using MeanFieldHomogenization, TensND

cr = ConductiveCrack(1.0; conductivity = 2.0e-2)
K  = conductivity_contribution(cr, TensISO{3}(1.0e-3))
(in_plane = K[1, 1], normal = K[3, 3])      # the normal component vanishes
```

Mechanically a flowing crack *is* an open crack, so the whole elastic branch is
inherited unchanged — one object serves both physics.

!!! note "Anisotropic reference media"
    No closed form exists there. Rather than derive one, the same ``\omega\to0``
    limit is taken on the package's own spheroid contribution and Richardson
    extrapolated — two Hill solves, no new theory to get wrong. It reproduces
    the closed form above to seven digits.

## The self-consistent estimate

```math
\boldsymbol{K}^{\rm hom} = k_s\,\boldsymbol{1}
  + \sum_i \frac{4\pi}{3}\,d_i\,\boldsymbol{k}_i(\boldsymbol{K}^{\rm hom})
```

Each family is read **in the effective medium**, which is what lets fractures
see one another and produces a percolation threshold.

```@example perm
fams = (ConductiveCrack(1.0; conductivity = 1.0e-3, euler_angles = (π/2, 0.0)),
        ConductiveCrack(1.0; conductivity = 1.0e-3, euler_angles = (π/2, π/2)))

K = fracture_permeability(1.0e-6, fams, (0.05, 0.05))
(K₁₁ = K[1, 1], K₃₃ = K[3, 3])
```

Two vertical families with normals ``\underline{e}_1`` and ``\underline{e}_2`` leave
``\underline{e}_3`` lying in *both* fracture planes, so the vertical direction
conducts most — visible in the numbers above.

!!! warning "The matrix must not be exactly impermeable"
    [`fracture_permeability`](@ref) is written out rather than routed through
    [`SelfConsistent`](@ref) on `:K`, whose crack branch is built for
    *insulating* cracks: its volumetric accumulator
    ``\sum_\alpha f_\alpha\,\boldsymbol{K}_\alpha\cdot\boldsymbol{A}_\alpha``
    drops the ``0\times\infty`` product a flowing crack is, so with ``k_s = 0``
    it collapses to ``\boldsymbol{K}^{\rm hom} = 0`` for any input. Use a small
    but non-zero matrix conductivity.

    A dense, strongly conducting network can also pass the percolation
    threshold, where the estimate diverges; the solver warns rather than
    returning a converged-looking number.
