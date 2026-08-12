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

With ``\gamma = C/2a``, the contribution per unit ``(4\pi/3)\,d`` in an
isotropic matrix ``k_0`` is purely in-plane:

```math
\mathbb K = \frac{\gamma}{1 + \dfrac{\pi\gamma}{4k_0}}
            \left(\boldsymbol\delta - \hat{\mathbf n}\otimes\hat{\mathbf n}\right),
\qquad
\mathbb K \xrightarrow[C\to\infty]{} \frac{4k_0}{\pi}
            \left(\boldsymbol\delta - \hat{\mathbf n}\otimes\hat{\mathbf n}\right).
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
\mathbf K = k_s\,\boldsymbol\delta
  + \sum_i \frac{4\pi}{3}\,d_i\,\mathbb K_i(\mathbf K)
```

Each family is read **in the effective medium**, which is what lets fractures
see one another and produces a percolation threshold.

```@example perm
fams = (ConductiveCrack(1.0; conductivity = 1.0e-3, euler_angles = (π/2, 0.0)),
        ConductiveCrack(1.0; conductivity = 1.0e-3, euler_angles = (π/2, π/2)))

K = fracture_permeability(1.0e-6, fams, (0.05, 0.05))
(K₁₁ = K[1, 1], K₃₃ = K[3, 3])
```

Two vertical families with normals ``\mathbf e_1`` and ``\mathbf e_2`` leave
``\mathbf e_3`` lying in *both* fracture planes, so the vertical direction
conducts most — visible in the numbers above.

!!! warning "The matrix must not be exactly impermeable"
    [`fracture_permeability`](@ref) is written out rather than routed through
    [`SelfConsistent`](@ref) on `:K`, whose crack branch is built for
    *insulating* cracks: its volumetric accumulator ``\sum f_\alpha K_\alpha
    A_\alpha`` drops the ``0\times\infty`` product a flowing crack is, so with
    ``k_s = 0`` it collapses to ``\mathbf K = 0`` for any input. Use a small but
    non-zero matrix conductivity.

    A dense, strongly conducting network can also pass the percolation
    threshold, where the estimate diverges; the solver warns rather than
    returning a converged-looking number.
