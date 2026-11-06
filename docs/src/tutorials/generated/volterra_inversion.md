```@meta
EditURL = "../../../../scripts/67_volterra_interpolation_inversion.jl"
```

# [Volterra operators: interpolation and inversion](@id tut-volterra-inversion)

!!! tip "The whole page as one script"
    Every block of this page, in order, is [`scripts/67_volterra_interpolation_inversion.jl`](https://github.com/MicroPoroChemoMechanics/MeanFieldHomogenization.jl/blob/main/scripts/67_volterra_interpolation_inversion.jl).
    From a clone of the repository, `julia --project=docs -e 'using Pkg; Pkg.instantiate()'`
    once, then `julia scripts/67_volterra_interpolation_inversion.jl`, runs it end to end.

!!! info "Before this page"
    The theory page [Aging linear Volterra behavior (ALV)](@ref th-viscoelasticity),
    for the Volterra product, the trapezoidal rule and
    [the kernel between the instants](@ref th-visco-kernel-between), and the
    [Volterra operators](@ref man-volterra-operators) section of the manual, for the
    objects used below.

Every aging computation of `MeanFieldHomogenization` rests on three objects: a
**history** known at the instants of a time grid, a **kernel** ``R(t, t')`` turned into a
lower-triangular matrix on that grid, and the **inverse** of that matrix, which turns a
relaxation kernel into a creep kernel. This page opens each of them on scalar examples,
small enough to print, before the homogenization schemes of the next pages use them on
tensors. It answers four questions:

1. what a [`TimeHistory`](@ref) is between two instants, and at an instant where the
   load jumps;
2. what a [`VolterraOperator`](@ref) returns between two instants, and why the entries
   of its matrix are not kernel values;
3. how good the creep function obtained by inverting a relaxation operator is, checked
   on two materials whose creep **and** relaxation functions are both known in closed
   form: a non-aging one, then an aging one;
4. how the error depends on the grid.

````@example volterra_inversion
using MeanFieldHomogenization
using LinearAlgebra
using Printf
using Plots
gr()  # headless backend; GKSwstype is set to "100" before Literate runs
````

## §1 A history between the instants

A scalar strain history ``\varepsilon(t)`` is given either as a function, evaluated at
the instants of the grid, or as its values at those instants. The grid is a sorted
vector of instants; here five of them, deliberately coarse.

````@example volterra_inversion
grid = [0.0, 1.0, 2.0, 4.0, 6.0]
ε_fun(t) = 1.0e-3 * (1 - exp(-t / 2))

h_fun = TimeHistory(ε_fun, grid)
h_val = TimeHistory([ε_fun(t) for t in grid], grid)
````

Both hold the same values at the instants, `h[i]` (one component for a scalar):

````@example volterra_inversion
[h_fun[i][1] for i in eachindex(grid)] == [h_val[i][1] for i in eachindex(grid)]
````

Called at an arbitrary time, they differ. The history built from a function returns the
function itself; the one built from values interpolates them **linearly** between two
instants. The second is the history that the trapezoidal rule actually integrates, as
the theory page shows: whatever the function was, the operator only sees the broken line
through its values.

````@example volterra_inversion
ts = range(0.0, 6.0; length = 241)
plt_history = plot(;
    xlabel = "t", ylabel = "ε(t)", legend = :bottomright, framestyle = :box,
    title = "A history on a five-instant grid", size = (720, 420),
)
plot!(plt_history, ts, [h_fun(t)[1] for t in ts]; label = "h_fun(t): the function", lw = 2)
plot!(plt_history, ts, [h_val(t)[1] for t in ts]; label = "h_val(t): linear between the instants", lw = 2, ls = :dash)
scatter!(plt_history, grid, [h_val[i][1] for i in eachindex(grid)]; label = "values at the instants", color = :black)
plt_history
````

Outside the grid a history built from values has nothing to interpolate, and refuses:

````@example volterra_inversion
try
    h_val(7.0)
catch err
    err isa DomainError || rethrow()
    println(sprint(showerror, err))
end
````

### A load applied at an instant: give the instant twice

A strain imposed at ``t = 2`` and held, ``\varepsilon(t) = \varepsilon_0\,H(t - 2)`` with
``H`` the Heaviside step function, jumps at ``t = 2``. On a grid where ``2`` appears
once, the broken line through the values climbs over the whole interval ``[1, 2]``: the
load is spread over the interval before the jump. Given **twice**, the instant carries
the two limits of the jump, ``\varepsilon(2^-) = 0`` on the first node and
``\varepsilon(2^+) = \varepsilon_0`` on the second, and the history is a true step.
The function is written right-continuous, `t ≥ 2`, and is evaluated just before the
instant on the first of the two nodes.

````@example volterra_inversion
const ε₀ = 1.0e-3
step_at(a) = t -> t ≥ a ? ε₀ : 0.0

once = [0.0, 1.0, 2.0, 4.0, 6.0]
twice = [0.0, 1.0, 2.0, 2.0, 4.0, 6.0]
h_once = TimeHistory([step_at(2.0)(t) for t in once], once)
h_twice = TimeHistory(step_at(2.0), twice)
[h_twice[i][1] for i in eachindex(twice)]

plt_step = plot(;
    xlabel = "t", ylabel = "ε(t)", legend = :bottomright, framestyle = :box,
    title = "A strain step at t = 2", size = (720, 420),
)
plot!(plt_step, ts, [h_once(t)[1] for t in ts]; label = "instant 2 given once", lw = 2, ls = :dash)
plot!(plt_step, ts, [h_twice(t)[1] for t in ts]; label = "instant 2 given twice", lw = 2)
plt_step
````

At the doubled instant itself the history takes the right limit, the value after the
jump:

````@example volterra_inversion
h_twice(2.0)[1], h_twice(prevfloat(2.0))[1]
````

## §2 A kernel between the instants

The first material is the **aging Maxwell model**: a spring of modulus ``E`` in series
with a dashpot whose viscosity ``\eta`` grows linearly with time, as a binder stiffens
when it sets,

```math
\eta(t) = \eta_0\,\Big(1 + \frac{t}{t_0}\Big),
\qquad
\dot\varepsilon(t) = \frac{\dot\sigma(t)}{E} + \frac{\sigma(t)}{\eta(t)} ,
```

where ``\sigma`` is the stress, ``\eta_0`` the viscosity at ``t = 0`` and ``t_0`` the
time over which it doubles. Both of its kernels have closed forms; they are derived in
§4 and used here.

````@example volterra_inversion
const E, η₀, t₀ = 1.0, 2.0, 1.0
const m = E * t₀ / η₀                    # the exponent of the relaxation function

R_aging(t, t′) = E * ((t₀ + t′) / (t₀ + t))^m
J_aging(t, t′) = 1 / E + (t₀ / η₀) * log((t₀ + t) / (t₀ + t′))
````

A function of ``(t, t')`` becomes an operator on a grid with [`VolterraOperator`](@ref).
The grid below is coarse on purpose: six instants on ``[0, 5]``.

````@example volterra_inversion
coarse = [0.0, 0.5, 1.0, 2.0, 3.5, 5.0]
R_op = VolterraOperator(R_aging, coarse)
````

### The matrix holds weights, not kernel values

The operator is a lower-triangular ``n \times n`` matrix ([`trapezoidal_matrix`](@ref)
describes its entries): the stress at ``t_i`` is the row ``i`` applied to the strain
values. An entry is a **weight** of the trapezoidal rule, a combination of kernel values
at neighboring instants, not ``R(t_i, t_j)``. Compare the row of the last instant with
the kernel sampled at the same pairs:

````@example volterra_inversion
i = length(coarse)
hcat(Matrix(R_op)[i, 1:i], [R_aging(coarse[i], coarse[j]) for j in 1:i])
````

The two columns have nothing in common. The kernel is in the matrix all the same: the
sum of row ``i`` from column ``j`` on is the weight of the interval that ends at
``t_j``, and the kernel values follow from these sums by a recurrence
([the kernel between the instants](@ref th-visco-kernel-between)).
[`operator_law`](@ref) runs it and returns all the samples:

````@example volterra_inversion
law_back = operator_law(R_op)
law_back.samples[i, 1:i] ≈ [R_aging(coarse[i], coarse[j]) for j in 1:i]
````

### Calling the operator

An operator is also a function of two times, `R_op(t, t′)`. What it returns depends on
where it comes from:

| operator | `R(t, t′)` between the instants |
| :-- | :-- |
| built from a law (a function, a `ViscoLaw`) | the law itself, exact everywhere, zero for ``t' > t`` |
| computed (a sum, a product, an inverse, a homogenization result) | the samples, interpolated bilinearly, inside the grid only |

The second kind is obtained here by rebuilding the operator from its bare matrix, which
forgets the law:

````@example volterra_inversion
R_computed = VolterraOperator(Matrix(R_op), coarse)

plt_kernel = plot(;
    xlabel = "t", ylabel = "R(t, t′) / E", legend = :topright, framestyle = :box,
    title = "Relaxation kernel read between the instants", size = (720, 420),
)
for (t′, col) in ((0.0, :blue), (1.0, :red))
    tt = range(t′, 5.0; length = 200)
    plot!(plt_kernel, tt, [R_op(t, t′) for t in tt]; color = col, lw = 2, label = "law, t′ = $t′")
    plot!(plt_kernel, tt, [R_computed(t, t′) for t in tt]; color = col, lw = 2, ls = :dash, label = "computed, t′ = $t′")
end
scatter!(plt_kernel, coarse, [R_aging(t, 0.0) for t in coarse]; color = :black, label = "samples at t′ = 0")
plt_kernel
````

Between the instants, the computed kernel is the broken line through the samples: it is
exactly the kernel the trapezoidal rule integrates. A computed operator called outside
its grid has no samples to read, and refuses, as the history did.

## §3 A first check: a non-aging pair

The **standard linear solid** (Zener model) is a spring ``E_\infty`` in parallel with a
Maxwell branch of modulus ``E_1`` and relaxation time ``\tau``. Its relaxation and creep
functions depend on ``t - t'`` only:

```math
R(t, t') = E_\infty + E_1\,e^{-(t-t')/\tau},
\qquad
J(t, t') = \frac{1}{E_\infty} - \Big(\frac{1}{E_\infty} - \frac{1}{E_\infty + E_1}\Big)
  e^{-(t-t')/\tau_r},
\qquad
\tau_r = \tau\,\frac{E_\infty + E_1}{E_\infty} .
```

The creep function starts at the instantaneous compliance ``1/(E_\infty + E_1)``, ends
at the relaxed one ``1/E_\infty``, and its retardation time ``\tau_r`` is longer than
``\tau``. The pair is the one the conversion of
[Generalized Kelvin ⇄ generalized Maxwell](@ref tut-kelvin-maxwell) reproduces exactly.

````@example volterra_inversion
const E∞, E₁, τ = 2.0, 5.0, 1.0
const τr = τ * (E∞ + E₁) / E∞

R_zener(t, t′) = E∞ + E₁ * exp(-(t - t′) / τ)
J_zener(t, t′) = 1 / E∞ - (1 / E∞ - 1 / (E∞ + E₁)) * exp(-(t - t′) / τr)
````

Creep and relaxation are **Volterra inverses** of one another:
``J \circ R = R \circ J = H``, the Volterra product ``\circ`` playing the role of the
ordinary product and the step ``H`` that of ``1``. On the grid the Volterra product is
the matrix product and the Volterra inverse the matrix inverse, so the creep operator
is one call, `inv`:

````@example volterra_inversion
T_zener = collect(range(0.0, 8.0; length = 81))
Rz = VolterraOperator(R_zener, T_zener)
Jz = inv(Rz)
````

The creep function under a stress applied at ``t = 0`` is the response of ``J`` to a
unit stress step. The grid starts at ``0``, where a history is zero before the first
instant, so a constant unit history *is* that step:

````@example volterra_inversion
unit = TimeHistory(t -> 1.0, T_zener)
J_num = [(Jz * unit)[i][1] for i in eachindex(T_zener)]
J_exact = [J_zener(t, 0.0) for t in T_zener]
@printf "largest relative error on 81 instants: %.2e\n" maximum(abs.(J_num .- J_exact) ./ J_exact)

plt_zener = plot(;
    xlabel = "t / τ", ylabel = "J(t, 0) E∞", legend = :bottomright, framestyle = :box,
    title = "Zener model: creep from the inverted relaxation", size = (720, 420),
)
plot!(plt_zener, T_zener, E∞ .* J_exact; label = "closed form", lw = 2)
scatter!(plt_zener, T_zener[1:4:end], E∞ .* J_num[1:4:end]; label = "inv(R) applied to a unit step", ms = 3)
plt_zener
````

## §4 The aging pair, and its derivation

The aging Maxwell model of §2 has the same structure as the Zener model, but its
kernels depend on ``t`` and ``t'`` separately. Both follow from the differential law.

**Creep.** Under ``\sigma = H(t - t')``, the stress is ``1`` and ``\dot\sigma = 0`` after
the jump, which the spring follows instantly, ``\varepsilon(t'^+) = 1/E``. Integrating
``\dot\varepsilon = 1/\eta`` from ``t'`` to ``t`` gives

```math
J(t, t') = \frac{1}{E} + \int_{t'}^{t}\frac{\mathrm{d}s}{\eta(s)}
         = \frac{1}{E} + \frac{t_0}{\eta_0}\,\ln\frac{t_0 + t}{t_0 + t'} .
```

**Relaxation.** Under ``\varepsilon = H(t - t')``, the strain rate vanishes after the
jump, so ``\dot\sigma = -E\,\sigma/\eta``, with ``\sigma(t'^+) = E``. This is a linear
equation whose solution is

```math
R(t, t') = E\,\exp\Big(-E\int_{t'}^{t}\frac{\mathrm{d}s}{\eta(s)}\Big)
         = E\,\Big(\frac{t_0 + t'}{t_0 + t}\Big)^{m},
\qquad m = \frac{E\,t_0}{\eta_0} .
```

The ratio ``(t_0 + t')/(t_0 + t)`` is not a function of ``t - t'``: a specimen loaded
late creeps less than one loaded early, the signature of aging. Two checks of the pair
are immediate: ``R(t'^+, t') = 1/J(t'^+, t') = E`` (the spring alone at the jump), and
``\eta \to \infty`` gives ``m \to 0``, ``R = E``, ``J = 1/E``.

The grid covers ``[0, 100]`` on a logarithmic scale, with the loading ages ``1`` and
``10`` given twice so that loads applied there are true steps (§1).

````@example volterra_inversion
const AGES = (0.0, 1.0, 10.0)
log_grid(n) = sort(vcat(0.0, 10 .^ range(-2, 2; length = n - 1)))
with_ages(g) = sort(vcat(g, [a for a in AGES if a > 0], [a for a in AGES if a > 0 && !(a in g)]))
T_aging = with_ages(log_grid(120))

Ra = VolterraOperator(R_aging, T_aging)
Ja = inv(Ra)
````

The creep function for a load applied at the age ``t'`` is the response of ``J`` to
a unit stress step at ``t'``, the history `step_at(t′)` of §1 divided by ``\varepsilon_0``.
The same values are also the kernel of the inverse read at ``(t, t')``, since a step
applied on a doubled instant meets the kernel at its right limit.

````@example volterra_inversion
creep_at(J, grid, t′) = [(J * TimeHistory(t -> t ≥ t′ ? 1.0 : 0.0, grid))[i][1] for i in eachindex(grid)]

plt_aging = plot(;
    xlabel = "t", ylabel = "J(t, t′) E", xscale = :log10, legend = :topleft, framestyle = :box,
    title = "Aging Maxwell model: creep from the inverted relaxation", size = (760, 460),
)
for (t′, col) in zip(AGES, (:blue, :red, :green))
    sel = findall(t -> t > max(t′, 1.0e-2), T_aging)
    tt = T_aging[sel]
    plot!(plt_aging, tt, [E * J_aging(t, t′) for t in tt]; color = col, lw = 2, label = "closed form, t′ = $t′")
    scatter!(plt_aging, tt[1:6:end], E .* creep_at(Ja, T_aging, t′)[sel][1:6:end]; color = col, ms = 3, label = "inv(R), t′ = $t′")
end
plt_aging
````

The three curves are the same law loaded at three ages, and they separate: the later the
loading, the slower the creep, the dashpot being stiffer by then. The kernel read
directly on the computed inverse gives the same numbers:

````@example volterra_inversion
for t′ in AGES[2:end]
    @printf "t′ = %4.1f   J(100, t′) E: inverse %.6f   kernel read %.6f   exact %.6f\n" t′ creep_at(Ja, T_aging, t′)[end] * E Ja(100.0, t′) * E J_aging(100.0, t′) * E
end
````

Two properties of the Volterra algebra can be checked on the same objects. The inverse
acts on both sides, ``J \circ R = R \circ J = H``, and undoes the operator on any
history:

````@example volterra_inversion
ε_hist = TimeHistory(t -> 1.0e-3 * sqrt(t), T_aging)
@printf "‖J ∘ R - I‖ = %.1e   ‖R ∘ J - I‖ = %.1e   ‖J(R ε) - ε‖/‖ε‖ = %.1e\n" opnorm(Matrix(Ja * Ra) - I) opnorm(Matrix(Ra * Ja) - I) norm(Matrix(Ja * (Ra * ε_hist)) - Matrix(ε_hist)) / norm(Matrix(ε_hist))
````

Unlike the product of two numbers, the Volterra product does not commute once a kernel
ages. Two non-aging kernels commute, since both are convolutions, and so do their
matrices on a **uniform** grid; the aging kernel composed with the Zener one gives two
different operators in the two orders. This is why every formula of the ALV pages keeps
its factors in order.

````@example volterra_inversion
commutator(A, B) = opnorm(Matrix(A * B - B * A)) / opnorm(Matrix(A * B))
R_zener2(t, t′) = 1.0 + 3.0 * exp(-(t - t′) / 0.3)
U = collect(range(0.0, 8.0; length = 81))
@printf "Zener ∘ Zener′, both non-aging:   relative commutator %.1e\n" commutator(VolterraOperator(R_zener, U), VolterraOperator(R_zener2, U))
@printf "aging Maxwell ∘ Zener:            relative commutator %.1e\n" commutator(VolterraOperator(R_aging, U), VolterraOperator(R_zener, U))
````

On a non-uniform grid even the two non-aging matrices stop commuting, by an amount of
the order of the discretization error: the commutation of convolutions holds for the
kernels, and for their matrices only when every row sees the same intervals.

## §5 How the error depends on the grid

The inverse is computed exactly on the matrices, so the only error is that of the
trapezoidal rule. It is also the reason why `inv(R)` is **not** the operator built from
the closed-form ``J``: the two are two different second-order approximations of the
same kernel, which agree as the grid is refined, not on a given grid.

````@example volterra_inversion
J_built = VolterraOperator(J_aging, T_aging)
@printf "‖inv(R) - J built from the law‖ / ‖J‖ = %.2e\n" opnorm(Matrix(Ja) - Matrix(J_built)) / opnorm(Matrix(J_built))
````

The error on the creep function under a load at ``t' = 0`` is measured below against the
closed form, as the number ``n`` of instants grows, on two grids of ``[0, 100]``: uniform,
and logarithmic from ``10^{-2}``.

````@example volterra_inversion
function creep_error(grid)
    J = inv(VolterraOperator(R_aging, grid))
    num = creep_at(J, grid, 0.0)
    return maximum(abs(num[i] - J_aging(t, 0.0)) / J_aging(t, 0.0) for (i, t) in enumerate(grid))
end

ns = [20, 40, 80, 160, 320]
err_uniform = [creep_error(collect(range(0.0, 100.0; length = n))) for n in ns]
err_log = [creep_error(log_grid(n)) for n in ns]
for (n, eu, el) in zip(ns, err_uniform, err_log)
    @printf "n = %3d   uniform %.2e   logarithmic %.2e\n" n eu el
end

plt_conv = plot(;
    xlabel = "number of instants n", ylabel = "largest relative error on J(t, 0)",
    xscale = :log10, yscale = :log10, legend = :bottomleft, framestyle = :box,
    title = "Convergence of the inverted relaxation", size = (720, 440),
)
plot!(plt_conv, ns, err_uniform; marker = :circle, lw = 2, label = "uniform grid")
plot!(plt_conv, ns, err_log; marker = :square, lw = 2, label = "logarithmic grid")
plot!(plt_conv, ns, err_log[1] .* (ns[1] ./ ns) .^ 2; ls = :dot, color = :black, label = "slope −2")
plt_conv
````

On the logarithmic grid the error falls as ``n^{-2}`` from the start, the order of the
trapezoidal rule. On the uniform grid it reaches that rate only once the first interval
is short enough to follow the fast start of the creep, and it stays more than ten times
larger at equal cost: the creep function varies fast just after loading and slowly
afterwards, and the logarithmic grid puts its instants where the kernel varies.

A load applied at an age that is not doubled, ``t' = 10`` given once, is the last
check. The broken line through the history values then spreads the jump over the
interval before ``10``, and the creep function is wrong by much more than the
discretization error just after loading:

````@example volterra_inversion
let g1 = sort(vcat(log_grid(120), 10.0)), g2 = with_ages(log_grid(120))
    for (lbl, g) in (("10 given once ", g1), ("10 given twice", g2))
        num = creep_at(inv(VolterraOperator(R_aging, g)), g, 10.0)
        k = findlast(==(10.0), g) + 1
        @printf "%s: J(%.2f, 10) E = %.5f   exact %.5f\n" lbl g[k] num[k] * E J_aging(g[k], 10.0) * E
    end
end
````

!!! tip "What to remember"
    - A history and a computed kernel are linear between the instants: that is what the
      trapezoidal rule integrates, so it is what they return when called.
    - An entry of the operator's matrix is a weight; [`operator_law`](@ref) or calling
      the operator gives the kernel.
    - `inv(R)` is the exact inverse of the discretized operator and a second-order
      approximation of the creep kernel; put the instants where the kernel varies,
      logarithmically after each loading.
    - A load or a phase that appears at an instant needs that instant twice.

## Where to go next

The same operators carry tensors: [Aging viscoelastic schemes side by side](@ref tut-alv-schemes)
homogenizes an aging matrix with four schemes and reads the creep of the composite
off the inverse, as §4 did here, and
[Aging creep: loading age against inclusion shape](@ref tut-aging-ages-aspect) loads it
at several ages. The full list of operations on operators and histories is in
[Volterra operators](@ref man-volterra-operators), and an application to a real
concrete in [Aging basic creep of concrete](@ref app-aging-basic-creep).

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

