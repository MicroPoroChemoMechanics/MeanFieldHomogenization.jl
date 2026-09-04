# [Derivatives and sensitivities](@id tut-sensitivities)

Every scheme in `MeanFieldHomogenization` is ordinary, generic Julia code — no
finite differences, no symbolic engine, no hand-written Jacobians. That
means [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) can
differentiate **any** homogenization result with respect to **any**
scalar input — a volume fraction, a modulus, a geometric parameter — at
machine precision, simply by running the same code on `Dual` numbers
instead of `Float64`.

## Parameter lenses

`MeanFieldHomogenization` exposes `derivative`, `gradient`, and `jacobian`, each
taking a *lens* describing which scalar input to differentiate against,
plus an `indexer` selecting a scalar output from the resulting tensor:

```@example tutsens
using MeanFieldHomogenization
using ForwardDiff
using TensND
using LinearAlgebra

rve = RVE()
add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => iso_stiffness(30.0, 10.0)); fraction = :rest)
add_phase!(rve, :I, Ellipsoid(1.0), Dict(:C => iso_stiffness(60.0, 20.0)); fraction = 0.2)

idxC = C -> get_array(C)[1, 1, 1, 1]

∂f = derivative(rve, MoriTanaka(), amount(:I); indexer = idxC)          # ∂C[1111]/∂f_I
∂K = derivative(rve, MoriTanaka(), property(:I, :C, :bulk); indexer = idxC)  # ∂C[1111]/∂K_I
(∂f, ∂K)
```

`amount(:I)` is the phase's volume fraction (or crack density), and
`property(:I, :C, :bulk)` its bulk-modulus coefficient — see
[the manual](../manual/sensitivities.md#Parameter-lenses) for the full
list, including `geometry` (a semi-axis, say) and `shape_param` (a
distribution-shape field). Several lenses combine into a `gradient`:

```@example tutsens
∇ = gradient(rve, MoriTanaka(), [amount(:I), property(:I, :C, :bulk)]; indexer = idxC)
```

!!! note "A symmetric shape has no first-order geometric sensitivity"
    Differentiating w.r.t. a semi-axis of a *sphere* returns exactly
    zero — the derivative respects the shape's own symmetry. A
    non-trivial geometric sensitivity needs an already non-degenerate
    (triaxial, prolate, or oblate) starting shape.

## The generic closure fallback

Whatever a lens cannot express — a composite parameter, a custom
inclusion field — can still be differentiated by writing an ordinary
closure and passing it to [`sensitivity`](@ref):

```@example tutsens
f_eval = K_inc -> begin
    r = RVE()
    add_phase!(r, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(3 * 30.0, 2 * 10.0)); fraction = :rest)
    add_phase!(r, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(K_inc, 2 * 20.0)); fraction = 0.2)
    return idxC(homogenize(r, MoriTanaka()))
end
sensitivity(f_eval, 3 * 60.0)
```

`sensitivity` auto-detects derivative vs. gradient vs. Jacobian from the
shape of its input and output — this is exactly what the lens-based
`derivative`/`gradient`/`jacobian` do internally, exposed for cases the
lenses do not cover.

## Validation against a closed form

Mori–Tanaka has a known closed-form sensitivity of the effective bulk
modulus to volume fraction [christensen1990](@cite):

```math
\frac{\partial k_{\text{MT}}}{\partial f} =
\Delta k\;\frac{\zeta_m(\zeta_m+\Delta k)}{D^2},
\qquad
\zeta_m = k_m+\tfrac{4}{3}\mu_m,
\qquad
D = \zeta_m+(1-f)\,\Delta k.
```

```@example tutsens
k_m, μ_m = 10.0, 5.0
k_i, μ_i = 40.0, 20.0
ζm = k_m + 4μ_m / 3
Δk = k_i - k_m
bulk = C -> sum(get_array(C)[i, i, j, j] for i in 1:3, j in 1:3) / 9

for f in (0.05, 0.1, 0.2, 0.3, 0.4)
    r = RVE()
    add_phase!(r, :M, Ellipsoid(1.0), Dict(:C => iso_stiffness(k_m, μ_m)); fraction = :rest)
    add_phase!(r, :I, Ellipsoid(1.0), Dict(:C => iso_stiffness(k_i, μ_i)); fraction = f)
    ∂_ad = derivative(r, MoriTanaka(), amount(:I); indexer = bulk)
    D = ζm + (1 - f) * Δk
    ∂_cf = Δk * ζm * (ζm + Δk) / D^2
    println("f=", f, "  AD=", round(∂_ad, digits = 6), "  closed form=", round(∂_cf, digits = 6))
end
```

The two match to machine precision. Because this holds for *every*
scheme, not just Mori–Tanaka, `ForwardDiff` derivatives of
`homogenize` are a reliable building block for anything downstream that
needs a sensitivity — including, as the next tutorial shows, a
macroscopic strength criterion built entirely from such derivatives.
