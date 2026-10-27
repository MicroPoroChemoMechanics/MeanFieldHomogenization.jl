# [Strength of a frictional matrix with a non-associated flow rule](@id mp-ex-frictional-strength)

This example computes the macroscopic limit states of composites with a
Drucker–Prager matrix whose flow is less dilatant than the normality rule, by
the [method of Maghous, Dormieux and Barthélémy](@ref mp-theory-frictional-strength),
and compares them with the closed forms of [maghous2009](@citet).

## A porous frictional solid

A solid with a tensile strength ``h = 1`` and a friction coefficient
``T = 0.3`` holds 15 % of spherical pores. Three dilatancy coefficients are
compared, from the normality rule ``t = T`` to a nearly isochoric flow:

```@example mp_ex_frictional
using MeanFieldHomogenization, TensND, Plots

φ, h, T = 0.15, 1.0, 0.3
rve = RVE()
add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(3.0, 2.0)); fraction = :rest)
add_phase!(rve, :P, Ellipsoid(1.0), Dict(:C => TensISO{3}(3.0, 2.0)); fraction = φ)
composite(t; a = 1e-2) = FrictionalComposite(rve, DruckerPrager(; h, friction = T, dilatancy = t);
    heterogeneities = Pores(), regularization = a)
nothing # hide
```

| code | theory |
|:--|:--|
| `DruckerPrager(; h, friction, dilatancy)` | the criterion ``\sigma_d + T(\sigma_m - h) \le 0`` and the flow potential ``\sigma_d + t\,\sigma_m`` |
| `Pores()` | the phases other than the matrix carry no stress |
| `regularization` | the parameter ``a`` of the viscous matrix, relative to ``t\lvert\boldsymbol{D}\rvert`` |
| `limit_stress(composite, D)` | the limit state reached under the macroscopic strain rate ``\boldsymbol{D}`` |

Each direction of the strain rate, from compaction to expansion, gives a point
of the boundary. Two regularizations and a Richardson extrapolation remove the
first-order error in ``a``:

```@example mp_ex_frictional
𝟏 = Mandel{Float64}(1, 1, 1, 0, 0, 0)
shear = Mandel{Float64}(1, -0.5, -0.5, 0, 0, 0)
mean_dev(σ) = (m = sum(σ[1:3]) / 3; (m, sqrt(sum(abs2, σ - m * 𝟏))))
limit(t, D) = 2 * limit_stress(composite(t; a = 5e-3), D).σ - limit_stress(composite(t; a = 1e-2), D).σ
rates = [sin(θ) * 𝟏 / 3 + cos(θ) * shear for θ in range(-1.4, 1.4; length = 25)]

# The closed-form domain, eq. (50): A Σm² + B Σm + C + D Σd² = 0.
A, B, C, Dd = 3φ / (2T^2) - 1, 2(1 - φ) * h, -(1 - φ)^2 * h^2, (1 + 2φ / 3) / T^2
Σm_line = range(-(B + sqrt(B^2 - 4A * C)) / (2A), (-B + sqrt(B^2 - 4A * C)) / (2A); length = 300)
p = plot(Σm_line, [sqrt(max(-(A * x^2 + B * x + C) / Dd, 0)) for x in Σm_line]; lw = 2, color = :black,
    label = "closed form, eq. (50)", xlabel = "Σ_m / h", ylabel = "Σ_d / h", legend = :topleft)
for (t, marker) in ((0.3, :circle), (0.15, :diamond), (0.05, :utriangle))
    pts = mean_dev.(limit.(t, rates))
    scatter!(p, first.(pts), last.(pts); marker, ms = 4, label = "t = $t")
end
p
```

The three dilatancies give the same domain, as the paper predicts, and the
points fall on the closed form. The domain is closed in compression: a porous
solid ends up crushing its pores.

## The flow rule depends on the dilatancy

The domain does not depend on ``t``, but the direction of the plastic flow at a
given stress does. With the normality rule, the strain rate is normal to the
domain; with ``t < T``, it is normal to the plastic potential of eqs. (53) and
(55), the same ellipse with ``T^2`` replaced by ``T t``:

```@example mp_ex_frictional
θs = range(-1.2, 1.2; length = 9)
q = plot(Σm_line, [sqrt(max(-(A * x^2 + B * x + C) / Dd, 0)) for x in Σm_line]; lw = 2, color = :black,
    label = "", xlabel = "Σ_m / h", ylabel = "Σ_d / h", aspect_ratio = :equal)
for (t, color) in ((0.3, :steelblue), (0.05, :firebrick))
    for θ in θs
        D = sin(θ) * 𝟏 / 3 + cos(θ) * shear
        Σm, Σd = mean_dev(limit(t, D))
        Dv, Ddev = sum(D[1:3]), sqrt(sum(abs2, D - sum(D[1:3]) / 3 * 𝟏))
        n = 0.15 / hypot(Dv, Ddev)
        plot!(q, [Σm, Σm + n * Dv], [Σd, Σd + n * Ddev]; color, lw = 2, label = θ == first(θs) ? "flow direction, t = $t" : "")
    end
end
q
```

The arrows give the macroscopic strain rate ``(D_v, D_d)`` at each limit state.
For ``t = T`` they are normal to the domain. For a small dilatancy they lean
toward the deviatoric axis: the porous solid flows with less volume change than
the normality rule would give.

## Rigid inclusions: a frictional cone

With rigid inclusions instead of pores, the limit states form a
Drucker–Prager cone, eq. (40), whose friction coefficient ``T^{\mathrm{hom}}`` grows
with the fraction of inclusions and depends on the dilatancy. The strain rates
that reach it follow the macroscopic flow rule ``D_v = t^{\mathrm{hom}} D_d``,
eq. (39); the point on the cone depends on their distance to it, of the order
of the regularization:

```@example mp_ex_frictional
t_hom(φ, t) = t * sqrt((1 + 1.5φ) / (1 - 2φ * t^2 / 3))
T_hom(φ, t) = T * sqrt((1 + 1.5φ) * (1 - 2φ * t^2 / 3)) / (1 - 2φ * t * T / 3)
function friction(φ, t; offset = 0.4)
    cell = RVE()
    add_phase!(cell, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(3.0, 2.0)); fraction = :rest)
    add_phase!(cell, :R, Ellipsoid(1.0), Dict(:C => TensISO{3}(3.0, 2.0)); fraction = φ)
    D_on = t_hom(φ, t) * sqrt(1.5) * 𝟏 / 3 + shear
    T_of(a) = begin
        fc = FrictionalComposite(cell, DruckerPrager(; h, friction = T, dilatancy = t); heterogeneities = RigidInclusions(), regularization = a)
        Σm, Σd = mean_dev(limit_stress(fc, D_on - offset * a * t * sqrt(sum(abs2, D_on)) * 𝟏 / 3).σ)
        Σd / (h - Σm)
    end
    return 2T_of(5e-3) - T_of(1e-2)
end
fractions = range(0.05, 0.5; length = 10)
r = plot(; xlabel = "fraction of rigid inclusions", ylabel = "macroscopic friction T^hom", legend = :topleft)
for (t, color) in ((0.3, :steelblue), (0.05, :firebrick))
    plot!(r, fractions, T_hom.(fractions, t); color, lw = 2, label = "closed form, t = $t")
    scatter!(r, fractions, friction.(fractions, t); color, ms = 4, label = "computed, t = $t")
end
r
```

The inclusions raise the friction coefficient, more with the normality rule
than with a nearly isochoric flow, and the computed values meet the closed form
of eq. (40).

## Going further

- Any scheme of the package replaces Mori–Tanaka (`scheme = SelfConsistent()`),
  and any shape of heterogeneity, with no closed form to compare with.
- The [rigid–plastic secant composite](@ref mp-ex-strength) is the von Mises
  limit of this one, ``h \to \infty`` and ``T \to 0`` with ``T h`` fixed.
