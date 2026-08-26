# [The rheological model library](@id man-rheological-models)

A linear viscoelastic material is described by four equivalent functions — the
relaxation modulus `R(t)`, the creep compliance `J(t)`, and their
Laplace-Carson transforms `R*(p)` and `J*(p)`. Which of them is *elementary*
depends entirely on the model, so this library is built so that you never have
to care: **every model answers all four**, whichever it was defined by.

```@example rheo
using MeanFieldHomogenization

m = zener_maxwell(2.0, 5.0, 1.0)      # E_∞ = 2, one branch (E₁ = 5, τ₁ = 1)

(relaxation(m, 0.7), creep(m, 0.7), carson_relaxation(m, 0.7), carson_creep(m, 0.7))
```

## 1. The interface

Five generics cover every model, scalar or tensorial:

| function | meaning |
|:---|:---|
| [`carson_relaxation`](@ref)`(m, p)` | ``R^{*}(p)`` — **the only method a model must define** |
| [`carson_creep`](@ref)`(m, p)` | ``J^{*}(p)``; defaults to `1/carson_relaxation` |
| [`relaxation`](@ref)`(m, t)` | ``R(t)``; defaults to inverting ``R^{*}`` |
| [`creep`](@ref)`(m, t)` | ``J(t)``; defaults to inverting ``J^{*}`` |
| [`complex_modulus`](@ref)`(m, ω)` | ``E^{*}(\omega) = R^{*}(i\omega)`` |

plus the limits [`glassy_modulus`](@ref) (``t \to 0``) and
[`equilibrium_modulus`](@ref) (``t \to \infty``), the predicate
[`is_fluid`](@ref), and the dynamic quantities
[`storage_modulus`](@ref), [`loss_modulus`](@ref), [`loss_factor`](@ref).

That fallback lattice *is* the design. Defining a new model means writing one
method:

```@example rheo
struct MyModel <: MeanFieldHomogenization.Viscoelasticity.AbstractRheology
    E::Float64
    τ::Float64
end

MeanFieldHomogenization.Viscoelasticity.carson_relaxation(m::MyModel, p) =
    m.E * (p * m.τ)^0.5 / (1 + (p * m.τ)^0.5)

creep(MyModel(10.0, 1.0), 2.0)        # obtained by numerical inversion
```

Values obtained by inversion are accurate to about `1e-12` with the default
method and are differentiable — see
[the inversion manual](@ref man-laplace-inversion).

## 2. The catalog

### Elementary elements

| | ``J(t)`` | ``R(t)`` | ``R^{*}(p)`` |
|:---|:---|:---|:---|
| [`Spring`](@ref)`(E)` | ``1/E`` | ``E`` | ``E`` |
| [`Dashpot`](@ref)`(η)` | ``t/\eta`` | ``\eta\,\delta(t)`` | ``p\eta`` |
| [`MaxwellUnit`](@ref)`(E, η)` | ``1/E + t/\eta`` | ``E e^{-t/\tau}`` | ``E\,\dfrac{p\tau}{1+p\tau}`` |
| [`KelvinUnit`](@ref)`(E, η)` | ``(1-e^{-t/\tau})/E`` | Dirac | ``E(1+p\tau)`` |

with ``\tau = \eta/E`` in both cases. [`Dashpot`](@ref) and
[`KelvinUnit`](@ref) have a Dirac impulse for a relaxation function, so
`relaxation` and `glassy_modulus` **throw** on them rather than return a wrong
number — assemble them with a spring first.

### Discrete spectra

[`PronyRelaxation`](@ref) and [`PronyCreep`](@ref) are the workhorses:

```math
R(t) = E_\infty + \sum_i E_i\,e^{-t/\tau_i},
\qquad
J(t) = J_0 + \sum_i J_i\bigl(1 - e^{-t/\tau_i}\bigr) + \varphi\,t .
```

`E_inf == 0` means a fluid; so does `phi > 0` on the creep side. The two are
converted into one another **exactly** by [`maxwell_to_kelvin`](@ref) and
[`kelvin_to_maxwell`](@ref) — see [the theory](@ref th-interlacing) for why that
is robust and [the tutorial](@ref tut-kelvin-maxwell) for what it looks like.

Named special cases: [`zener_maxwell`](@ref) and [`zener_kelvin`](@ref) (the
standard linear solid, both ways round) and [`burgers`](@ref).

```@example rheo
b = burgers(1.0, 3.0, 2.0, 6.0)       # k_s, η_s, k_p, η_p
r = kelvin_to_maxwell(b)
(is_fluid(b), length(r), equilibrium_modulus(r))   # a fluid gains one branch
```

### Fractional elements

A single exponential spans one decade; real polymers and bitumen relax over six
or more. The fractional elements broaden the spectrum without adding branches.

| | ``R^{*}(p)`` | notes |
|:---|:---|:---|
| [`ScottBlair`](@ref)`(V, α)` | ``V p^{\alpha}`` | the springpot; `α = 0` a spring, `α = 1` a dashpot. All four functions closed form |
| [`FractionalMaxwell`](@ref)`(V_a, α, V_b, β)` | two springpots in series | a fluid |
| [`FractionalKelvin`](@ref)`(V_a, α, V_b, β)` | ``V_a p^{\alpha} + V_b p^{\beta}`` | a solid when `β = 0` |
| [`FractionalZener`](@ref)`(E_∞, E_0, τ, α)` | ``E_\infty + (E_0-E_\infty)\dfrac{(p\tau)^{\alpha}}{1+(p\tau)^{\alpha}}`` | the Cole-Cole model; ``R(t)`` is a Mittag-Leffler function |
| [`Rabotnov`](@ref)`(μ₀, λ₀, α, β)` | ``\mu_0\left(1 + \dfrac{\lambda_0}{p^{\alpha+1}+\beta}\right)`` | the ECHOES benchmark kernel |

!!! tip "Rabotnov's transform is elementary"
    The kernel is usually written with a Mittag-Leffler function,
    ``\mathfrak{I}(t) = \bigl(1 - E_{\alpha+1,1}(-\beta t^{\alpha+1})\bigr)/\beta``.
    Its Carson transform is simply ``1/(p^{\alpha+1}+\beta)``, which is what the
    table above shows — no special function is involved on this route at all.
    A passive material has ``\lambda_0 < 0``; the benchmark of
    [barthelemyIJES2019](@cite) §5 uses `μ₀ = 1.7`, `λ₀ = -0.495`, `α = -0.46`,
    `β = 0.98`.

The two models whose *time-domain* form is a Mittag-Leffler function use it
when `MittagLeffler.jl` is loaded, and fall back on numerical inversion of the
closed-form transform otherwise. Both routes agree to about `1e-10`:

```@example rheo
fz = FractionalZener(2.0, 10.0, 1.0, 0.6)
(relaxation(fz, 1.0), inverse_carson(p -> carson_relaxation(fz, p), 1.0))
```

!!! warning "An upstream defect worth knowing about"
    `MittagLeffler.jl` v1.0.0 returns `1.0` for `1 < a < 2` and small `|z|`,
    silently — `mittleff(1.3, 1.0, -0.5)` gives `1.0` where the series gives
    `0.633`. The extension therefore answers only for `0 < a ≤ 1`, the domain
    checked against the series in the test suite; anything else falls back on
    the inversion, which is correct.

### Bituminous binders and mixtures

[`HuetSayegh`](@ref) and [`Model2S2P1D`](@ref) are the reference models of the
field:

```math
E^{*}(p) = E_{00} + \frac{E_0 - E_{00}}{\varphi^{*}(p)},
\qquad
\varphi^{*}(p) = 1 + \delta\,(p\tau_E)^{-k} + (p\tau_E)^{-h}
   \;\bigl[\;+\;\tfrac{1}{\beta p \tau_E}\;\bigr]
```

with the bracketed series-dashpot term present in 2S2P1D and absent in
Huet-Sayegh — which is exactly the difference between a fluid and a solid.
`E00` is the **static** modulus and `E0` the **glassy** one, following
Di Benedetto and Olard and the ECHOES sources; `0 < k < h < 1`.

!!! note "2S2P1D is exact in both domains"
    Because ``\mathcal{LC}\{t^{a}\} = \Gamma(a+1)p^{-a}``, the denominator
    ``\varphi^{*}`` is the transform of an ordinary function of time, term by
    term:

    ```math
    \varphi(t) = 1 + \delta\,\frac{(t/\tau_E)^{k}}{\Gamma(k+1)}
                   + \frac{(t/\tau_E)^{h}}{\Gamma(h+1)}
                   + \frac{t}{\beta\,\tau_E}.
    ```

    So the same object is an exact Laplace-Carson model *and* an exact Volterra
    model. [`creep_kernel`](@ref) gives ``\varphi(t)``,
    [`carson_creep_kernel`](@ref) gives ``\varphi^{*}(p)``, and
    [`creep_kernel_law`](@ref) packages the former as a [`ViscoLaw`](@ref) so
    that the *ageing* pipeline can consume it:

    ```julia
    Φ = trapezoidal_matrix(creep_kernel_law(m), times)
    R = m.E00 * I + (m.E0 - m.E00) * volterra_inverse(Φ; block_size = 1)
    ```

    That discrete `R` must agree with `relaxation(m, t)` obtained by inverting
    the transform. It does, and that is the sharpest available check that the
    two routes of the package are consistent.

```@example rheo
binder = Model2S2P1D(1.0e-7, 1000.0, 2.2, 1.94507827e-3, 0.22, 0.63, 50.0)
E = complex_modulus(binder, 2π * 10)          # 10 Hz
(abs(E), rad2deg(angle(E)))                   # norm and phase angle
```

### Long-term creep of concrete

[`LogarithmicCreep`](@ref)`(E, C, τ)` is ``J(t) = 1/E + \ln(1+t/\tau)/C``, whose
transform involves the exponential integral,
``J^{*}(p) = 1/E + e^{p\tau}E_1(p\tau)/C``. It is evaluated through
`SpecialFunctions.expintx`, the scaled form, so it stays finite where
``e^{p\tau}`` alone would overflow.

The *ageing* version — where `E`, `C` and `τ` depend on the loading age — is a
different object and belongs to the [time-domain route](@ref man-viscoelasticity).

## 3. Fitting a chain to something that is not one

When the material is given as an arbitrary transform — measured data, or a
homogenized `C*(p)` — [`prony_fit_relaxation`](@ref) fits a discrete spectrum by
collocation, and everything above becomes available:

```@example rheo
fz2 = FractionalZener(2.0, 10.0, 1.0, 0.6)
fit = prony_fit_relaxation(p -> carson_relaxation(fz2, p), exp10.(range(-2, 2; length = 14)))
(count(>(0), fit.E), relaxation(fit, 1.0), relaxation(fz2, 1.0))
```

The fit is non-negative by default. That is what makes the result completely
monotone and hence passive — and it also fits better: the unconstrained
least-squares solution puts half its moduli below zero and is three times worse
on the master curve.

!!! warning "A fit is a calibration step, not a differentiable one"
    With `nonneg = true` the solve is an active-set method, so the coefficients
    are a piecewise-smooth function of the input with a combinatorial switch in
    the middle. Differentiate the resulting *model*, not the fit. An
    unconstrained fit (`nonneg = false`) is a plain linear solve and is
    differentiable.

## 4. Lifting to a fourth-order tensor

[`iso_rheology`](@ref) pairs two scalar models into an isotropic tensor model,
one per channel:

```@example rheo
iso = iso_rheology(Spring(2500.0), binder)    # elastic bulk, 2S2P1D shear
carson_relaxation(iso, im * 2π * 10)
```

[`iso_rheology_E_nu`](@ref) takes a Young's-modulus channel and a Poisson
ratio, which may be a constant or a model of its own.

!!! warning "A relaxing Poisson ratio only makes sense in the transform domain"
    ``3k^{*} = E^{*}/(1-2\nu^{*})`` is a pointwise identity in `p`. In the time
    domain the same relation is a **Volterra** quotient: `ν(t)` does not divide,
    it deconvolves. A constant `nu` is unambiguous and is handled as a special
    case, including for the closed-form time values.

## 5. One object, both routes

This is what the library is for. From a single model:

```@example rheo
using TensND

Zm = iso_rheology(zener_maxwell(1.0, 2.0, 1.0), zener_maxwell(0.6, 1.2, 0.7))

Cstar = p -> carson_relaxation(Zm, p)   # → homogenize / homogenize_lc
law   = ViscoLaw(Zm)                    # → homogenize_alv

(TensND.get_data(law(1.0, 0.0)), TensND.get_data(relaxation(Zm, 1.0)))
```

`ViscoLaw(model)` builds a genuinely non-ageing kernel `(t, t') ↦ R(t - t')`,
which the existing ageing pipeline consumes unchanged. The two routes then
compute the same thing by disjoint means, which is what
[the three-route comparison](@ref tut-freq-vs-time) uses as a cross-check.

!!! note "Cost of the bridge"
    If the model has no closed-form time value, every entry of the trapezoidal
    matrix costs one numerical inversion. For an `n`-point grid that is
    `O(n²N)` transform evaluations. When the model is a Prony series — or has
    been fitted to one — the time value is closed form and no inversion runs.

## See also

* [the model gallery](@ref tut-rheological-models) — every model plotted, with
  master curves, Cole-Cole and Black diagrams;
* [Kelvin ⇄ Maxwell](@ref tut-kelvin-maxwell) — the exact conversion;
* [choosing an inversion](@ref man-laplace-inversion);
* [the theory](@ref th-laplace-carson).
