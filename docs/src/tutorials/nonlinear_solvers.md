# [Nonlinear solvers for the self-consistent fixed point](@id tut-nonlinear-solvers)

The [self-consistent scheme](bounds_and_schemes.md) is a fixed point
``\mathbb C = \mathrm{step}(\mathbb C)``. Three solver families are available
(see [Homogenization schemes](../manual/schemes.md)): the default damped Picard
[`AndersonDefault`](@ref), the dependency-free [`NewtonDefault`](@ref), and —
the subject of this page — the weak extension
`MeanFieldHomogenizationNonlinearSolveExt`, which hands the same fixed point to any
[NonlinearSolve.jl](https://github.com/SciML/NonlinearSolve.jl) algorithm.

Two things need checking: that they agree, and that `ForwardDiff`
differentiates *through* an external solve correctly.

## The three solver families

```@example tutnls
using MeanFieldHomogenization
using TensND
using ForwardDiff
using NonlinearSolve
using LinearAlgebra
using Printf

const k_m, μ_m = 30.0, 10.0
const k_i, μ_i = 60.0, 20.0

rve = RVE()
add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(k_m, μ_m)); fraction = :rest)
add_phase!(rve, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(k_i, μ_i)); fraction = 0.3)

C_picard = homogenize(rve, SelfConsistent())                                  # built-in damped Picard (default)
C_newton = homogenize(rve, SelfConsistent(; algorithm = NewtonDefault()))     # built-in Newton, dependency-free
C_nr     = homogenize(rve, SelfConsistent(; algorithm = NewtonRaphson()))     # NonlinearSolve.jl
C_tr     = homogenize(rve, SelfConsistent(; algorithm = TrustRegion()))       # NonlinearSolve.jl
C_auto   = homogenize(rve, SelfConsistent(; algorithm = AutoNonlinear()))     # auto-resolving

k_mu.((C_picard, C_newton, C_nr, C_tr, C_auto))
```

All five agree on the same fixed point. [`AutoNonlinear`](@ref) resolves at
runtime to `TrustRegion` when the extension is active and to
[`NewtonDefault`](@ref) otherwise. `AsymmetricSelfConsistent` accepts
`algorithm` the same way, on both its stiffness- and compliance-form
branches.

SciML algorithms are opt-in rather than the default (see
[Homogenization schemes](../manual/schemes.md)): reach for them on
well-conditioned, high-contrast problems away from a bifurcation, where they can
be markedly faster.

## Benchmark: time and memory

The numbers below use `@elapsed`/`@allocated` (minimum of a few samples after
warm-up) and are indicative of *relative* cost on this problem only; the
repeatable benchmark is `scripts/bench/bench_sc_solvers.jl`.

```@example tutnls
function bench(f, label; warmups = 3, samples = 5)
    for _ in 1:warmups
        f()
    end
    t_min, bytes = Inf, 0
    for _ in 1:samples
        b = @allocated f()
        t = @elapsed f()
        if t < t_min
            t_min, bytes = t, b
        end
    end
    return (; label, t_ms = t_min * 1.0e3, mib = bytes / 2^20)
end

results = [
    bench(() -> homogenize(rve, SelfConsistent()), "Picard"),
    bench(() -> homogenize(rve, SelfConsistent(; algorithm = NewtonDefault())), "Newton (built-in)"),
    bench(() -> homogenize(rve, SelfConsistent(; algorithm = NewtonRaphson())), "NewtonRaphson (NLS)"),
    bench(() -> homogenize(rve, SelfConsistent(; algorithm = TrustRegion())), "TrustRegion (NLS)"),
]
for r in results
    @printf "%-22s  %8.3f ms   %8.4f MiB\n" r.label r.t_ms r.mib
end
```

All four converge in a handful of iterations; the differences reflect
iteration count against per-iterate cost (Picard: cheap iterate, more of them;
Newton-type: quadratic, one Jacobian per step). Which one wins depends on the
contrast and the proximity to a bifurcation — hence the keyword.

## Differentiating through a SciML solve

`ForwardDiff` computes `derivative(rve, scheme, p)` by seeding a `Dual` into
the RVE, so the nonlinear solver sees `Dual`-valued inputs and would seed *its
own* `Dual` on top — **nested** duals, fragile in tag ordering and wasteful.

`MeanFieldHomogenizationNonlinearSolveExt` avoids this with an implicit-function-theorem
(IFT) **lift**: it solves the *primal* problem (inputs stripped to `Float64`
via `ForwardDiff.value`, explicit finite-difference Jacobian, so
`NonlinearSolve` never seeds a `Dual`) and recovers the caller's partials with
one linear-algebra correction.

Write the self-consistent scheme as a root-finding problem. Let ``p`` collect the
unknown effective moduli, ``\theta`` the parameter being differentiated, and

```math
F(p;\theta) = p - \mathcal{H}(p;\theta)
```

the **residual** of the self-consistent fixed point, where
``\mathcal{H}(p;\theta)`` is one application of the scheme — the map that takes a
trial effective medium ``p``, computes each phase's localization in it, and
returns the resulting effective moduli. A self-consistent solution is exactly a
root ``F(p^\star;\theta) = 0``. The IFT correction is then

```math
p^\star_{\text{dual}} = p^\star - \Big(\frac{\partial F}{\partial p}\Big)^{-1} F(p^\star; \theta),
```

evaluated with ``p^\star`` the primal root and the residual re-evaluated on the
caller's `Dual` inputs. Only the *caller's* `Dual` tag is ever present; nothing
is nested.

### Why that one line is the implicit function theorem

The formula looks like a Newton step, and that is exactly what it is — but
taken in *dual* arithmetic from a point where the primal residual already
vanishes. Write ``\theta = \theta_0 + \varepsilon`` for the caller's seeded
`Dual` (``\varepsilon`` the infinitesimal unit, ``\varepsilon^2 = 0``).

**Step 1 — the primal root carries no partials.** The extension solves the
primal problem with every input stripped through `ForwardDiff.value`, so
``p^\star`` is an ordinary real number and

```math
F(p^\star;\theta_0) = 0 .
```

**Step 2 — re-evaluating the residual at the *same* ``p^\star``, with the dual
``\theta``, isolates ``\partial F/\partial\theta``.** Since ``p^\star`` has zero
partials, a first-order expansion in ``\varepsilon`` gives

```math
F(p^\star;\theta) = \underbrace{F(p^\star;\theta_0)}_{=\;0}
                  + \varepsilon\,\frac{\partial F}{\partial\theta}
                  = \varepsilon\,\frac{\partial F}{\partial\theta} .
```

This is the crux: **the real part cancels because ``p^\star`` is a root**, so
the residual is purely infinitesimal.

**Step 3 — the correction is therefore purely infinitesimal too.** ``\partial
F/\partial p`` is the primal Jacobian, an ordinary real matrix, so

```math
p^\star - \Big(\frac{\partial F}{\partial p}\Big)^{-1} F(p^\star;\theta)
  = p^\star - \varepsilon
      \Big(\frac{\partial F}{\partial p}\Big)^{-1}\frac{\partial F}{\partial\theta} .
```

**Step 4 — recognize the IFT.** Differentiating ``F(p^\star(\theta);\theta) = 0``
with respect to ``\theta`` gives
``\frac{\partial F}{\partial p}\frac{dp^\star}{d\theta} + \frac{\partial F}{\partial\theta} = 0``,
i.e.

```math
\frac{dp^\star}{d\theta}
  = -\Big(\frac{\partial F}{\partial p}\Big)^{-1}\frac{\partial F}{\partial\theta} .
```

Substituting into step 3,

```math
p^\star_{\text{dual}} = p^\star + \varepsilon\,\frac{dp^\star}{d\theta} ,
```

which is precisely the `Dual` whose value is the primal root and whose partial
is the sensitivity. Nothing is iterated and no second-order information is
needed — one linear solve with the Jacobian already available at the root.

!!! note "The residual is not exactly zero in practice"
    The solver returns ``p^\star`` with ``\|F\| \lesssim`` `abstol`, not exactly
    ``0``. The real part of the formula is then
    ``p^\star - (\partial F/\partial p)^{-1}F(p^\star;\theta_0)`` — one extra
    *primal* Newton step. So the returned value can differ from the solver's
    ``p^\star`` by ``O(\texttt{abstol})``, in the direction of a **better**
    root. This is benign, but it explains why the value returned through the
    IFT path is not always bit-identical to the one returned by a plain solve.
The built-in [`NewtonDefault`](@ref) instead differentiates straight
through its own iterations (safe because it is entirely
hand-written), so both paths give the same answer by construction —
this section checks that they do:

```@example tutnls
idxC = C -> get_array(C)[1, 1, 1, 1]

d_picard = derivative(rve, SelfConsistent(), property(:I, :C, :bulk); indexer = idxC)
d_newton = derivative(rve, SelfConsistent(; algorithm = NewtonDefault()), property(:I, :C, :bulk); indexer = idxC)
d_nr     = derivative(rve, SelfConsistent(; algorithm = NewtonRaphson()), property(:I, :C, :bulk); indexer = idxC)
d_tr     = derivative(rve, SelfConsistent(; algorithm = TrustRegion()), property(:I, :C, :bulk); indexer = idxC)

(d_picard, d_newton, d_nr, d_tr)
```

```@example tutnls
# Central finite difference — a solver-independent ground truth.
function f_modulus(K_I)
    r = RVE()
    add_phase!(r, :SOLID, Ellipsoid(1.0), Dict(:C => TensISO{3}(k_m, μ_m)); fraction = :rest)
    add_phase!(r, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(K_I, μ_i)); fraction = 0.3)
    return idxC(homogenize(r, SelfConsistent()))
end
h = 1.0e-5
d_fd = (f_modulus(k_i + h) - f_modulus(k_i - h)) / (2h)
@printf "central FD = %.6f   |TrustRegion − FD| / |FD| = %.3e\n" d_fd abs(d_tr - d_fd) / abs(d_fd)
```

All four agree to within the solver tolerances: the IFT lift is exact to
first order and costs one linear solve at the root instead of propagating
partials at every step. The same holds for the *inclusion* fraction or the
*matrix* shear modulus — parameters living on a phase other than the one the
initial estimate is built from, which a naive type-from-the-initial-guess
dispatch can miss:

```@example tutnls
d_f_picard = derivative(rve, SelfConsistent(), amount(:I); indexer = idxC)
d_f_tr     = derivative(rve, SelfConsistent(; algorithm = TrustRegion()), amount(:I); indexer = idxC)
d_m_picard = derivative(rve, SelfConsistent(), property(:M, :C, :shear); indexer = idxC)
d_m_tr     = derivative(rve, SelfConsistent(; algorithm = TrustRegion()), property(:M, :C, :shear); indexer = idxC)

(; d_f_picard, d_f_tr, d_m_picard, d_m_tr)
```

## Strength criterion, revisited

The [capstone tutorial](strength_criteria.md) builds a macroscopic strength
ellipse from `ForwardDiff` derivatives of
``(k_{\mathrm{hom}}, \mu_{\mathrm{hom}})`` with respect to the solid's shear
modulus. Swapping the SC solve for a SciML algorithm changes nothing, since
`derivative` only sees `homogenize`'s public interface:

```@example tutnls
const k_s, μs_value = 1.0e6, 1.0
const TINY = 1.0e-12
const ω_aspect = 0.1
const φ_value = 0.15

function C_hom_iso_2vec(μs::T, scheme) where {T}
    r = RVE(; T = T)
    add_phase!(r, :SOLID, Spheroid(ω_aspect), Dict(:C => iso_stiffness(convert(T, k_s), μs)); fraction = :rest, symmetrize = IsoSymmetrize())
    add_phase!(r, :PORE, Spheroid(ω_aspect), Dict(:C => iso_stiffness(convert(T, TINY), convert(T, TINY)));
               fraction = convert(T, φ_value), symmetrize = IsoSymmetrize())
    C = homogenize(r, scheme, :C)
    return [k_mu(best_fit_iso(C))...]
end

function ellipse_AB(scheme)
    k_hom, μ_hom = C_hom_iso_2vec(μs_value, scheme)
    dk_dμs, dμ_dμs = ForwardDiff.derivative(μ -> C_hom_iso_2vec(μ, scheme), μs_value)
    A = (μs_value / k_hom)^2 * dk_dμs
    B = (μs_value / μ_hom)^2 * dμ_dμs
    return A, B
end

for (scheme, label) in (
        (SelfConsistent(; abstol = 1.0e-10, maxiters = 300, select_best = true), "Picard"),
        (SelfConsistent(; algorithm = TrustRegion()), "TrustRegion"),
    )
    A, B = ellipse_AB(scheme)
    @printf "%-12s  A = %.6g   B = %.6g\n" label A B
end
```

The two solver families produce the same ellipse — the strength criterion is a
property of the *scheme*, not of how its fixed point happens to be solved.
