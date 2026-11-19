# [Approximation by Gaussians](@id th-gaussian-approximation)

The closed forms of the previous pages hold for ellipsoids and for a few cracks.
A crack of any contour, or a heterogeneity of any shape, is the solution of an
integral equation that has to be solved numerically. `MeanFieldHomogenization`
does it by expanding the unknown on Gaussians centered at the nodes of a regular
grid, an approximation introduced by [Maz'ya and Schmidt](@cite mazya1996) and
developed in [their monograph](@cite mazya2007). This page presents the
approximation alone, on functions of one variable; the
[next one](@ref th-gaussian-cracks) applies it to cracks.

## 1. A function rebuilt from its samples

Let ``u`` be a function of ``\underline{x}\in\mathbb{R}^d``, known at the nodes
``\underline{x}_m = h\,\underline{m}`` (``\underline{m}\in\mathbb{Z}^d``) of a
regular grid of step ``h``. The approximation of ``u`` is the sum of one
Gaussian per node, weighted by the value of ``u`` there:

```math
u_h(\underline{x}) = \sum_{\underline{m}} u(\underline{x}_m)\,
    \varphi(\underline{x}-\underline{x}_m),
\qquad
\varphi(\underline{x}) = \frac{1}{(\pi H)^{d/2}}
    \exp\!\left(-\frac{|\underline{x}|^{2}}{H h^{2}}\right).
\tag{1}
```

``\varphi`` is the *generating function* and ``H`` a dimensionless *shape
parameter*: the Gaussians have the width ``h\sqrt{H}``, so that neighbors
overlap. The prefactor makes the integral of ``\varphi`` equal to the volume
``h^{d}`` of a cell.

Nothing is solved to obtain ``u_h``: its coefficients are the samples of ``u``.
It is not an interpolant either, since ``u_h(\underline{x}_m)`` is a weighted
mean of ``u`` around ``\underline{x}_m`` rather than ``u(\underline{x}_m)``.
[`quasi_interpolate`](@ref) builds it on a [`GaussianGrid`](@ref):

```@example gaussapprox
using MeanFieldHomogenization
using Plots

u(x) = sinpi(2x) + x^2
grid = GaussianGrid((0.0,), (1.0,), 10; H = 2.0)       # 10 nodes on [0, 1]
uh = quasi_interpolate(x -> u(x[1]), grid)

G = MeanFieldHomogenization.GaussianApproximation
xs = range(0, 1; length = 400)
plt = plot(xs, u.(xs); label = "u", lw = 2, xlabel = "x", legend = :topleft)
plot!(plt, xs, [uh((x,)) for x in xs]; label = "u_h, 10 nodes", lw = 2, ls = :dash)
for xm in G.nodes(grid)                                   # the ten weighted Gaussians
    plot!(plt, xs, [u(xm[1]) * gaussian_generator((x - xm[1])^2, grid.h, grid.H, Val(1)) for x in xs];
        label = "", color = :gray, lw = 0.5)
end
plt
```

The two curves separate near the ends of the interval, where the Gaussians of
the nodes that would lie outside are missing: (1) is a statement about the
interior of the grid. Inside, ten nodes are too few for this function: ``u_h``
smooths its extrema.

Both defects go with a grid that extends beyond the interval and a smaller
step. Below, the nodes cover ``[-1, 2]`` and the function is rebuilt on
``[0, 1]`` with the steps ``h = 1/10``, ``1/20`` and ``1/40``; the lower panel
is the gap ``u_h - u``.

```@example gaussapprox
extended(n) = quasi_interpolate(x -> u(x[1]), GaussianGrid((-1.0,), (2.0,), 3n; H = 2.0))   # h = 1/n

xs = range(0, 1; length = 400)
top = plot(xs, u.(xs); label = "u", lw = 4, color = :lightgray, legend = :bottomleft)
bottom = plot(; xlabel = "x", ylabel = "u_h - u", legend = false)
for (color, (n, style)) in enumerate(((10, :dot), (20, :dash), (40, :solid)))
    uh = extended(n)
    plot!(top, xs, [uh((x,)) for x in xs]; label = "u_h, h = 1/$n", lw = 1.5, ls = style, color)
    plot!(bottom, xs, [uh((x,)) - u(x) for x in xs]; lw = 1.5, ls = style, color)
end
plot(top, bottom; layout = @layout([a{0.62h}; b]), size = (700, 560))
```

With forty nodes per unit length the two curves are superposed. The largest gap
on the interval, for each step:

```@example gaussapprox
for n in (10, 20, 40, 80)
    uh = extended(n)
    println("h = 1/", n, "   max |u_h - u| = ", round(maximum(abs(uh((x,)) - u(x)) for x in xs); sigdigits = 3))
end
```

Halving the step divides the gap by four. Section 3 gives the law, and the
level below which no refinement goes.

## 2. The sum of the Gaussians is not one

Take ``u = 1`` in one dimension. The sum
``s(x) = \sum_m \varphi(x - mh)`` is periodic of period ``h``, hence a Fourier
series in ``x/h``, whose coefficient of order ``\nu`` is the Fourier transform
of ``\varphi`` taken at the frequency ``\nu/h`` and divided by ``h`` (Poisson's
summation formula). The transform of a Gaussian is a Gaussian,
``\int\varphi(x)\,\mathrm{e}^{-2\pi i\nu x/h}\,\mathrm{d}x =
h\,\mathrm{e}^{-\pi^{2}H\nu^{2}}``, so that

```math
\sum_{m\in\mathbb{Z}} \varphi(x-mh)
= 1 + 2\sum_{\nu\ge 1} \mathrm{e}^{-\pi^{2}H\nu^{2}}\cos\frac{2\pi\nu x}{h}.
\tag{2}
```

A constant is therefore not reproduced exactly: the sum oscillates around one,
with the wavelength of the grid and the amplitude

```math
\epsilon_{\mathrm{sat}}(H) = 2\sum_{\nu\ge 1}\mathrm{e}^{-\pi^{2}H\nu^{2}}
\simeq 2\,\mathrm{e}^{-\pi^{2}H},
```

the *saturation error*, returned by [`saturation_error`](@ref). In dimension
``d`` the sum is the product of ``d`` such series and the amplitude is
``(1+\epsilon_{\mathrm{sat}})^{d}-1``.

```@example gaussapprox
for H in (1.0, 2.0, 3.0)
    println("H = ", H, "   saturation error ", saturation_error(H))
end
```

The oscillation is visible for ``H = 1`` and the amplitude is that of (2):

```@example gaussapprox
g1 = GaussianGrid((0.0,), (1.0,), 40; H = 1.0)
one_h = quasi_interpolate(x -> 1.0, g1)
xs = range(0.45, 0.55; length = 400)                       # four cells, far from the ends
plot(xs ./ g1.h, [one_h((x,)) - 1 for x in xs]; label = "sum of the Gaussians minus 1",
    xlabel = "x / h", lw = 2)
hline!([saturation_error(1.0), -saturation_error(1.0)]; label = "± saturation error", ls = :dash)
```

## 3. The error has two parts

For a smooth ``u``, expand ``u(x_m)`` around ``x`` in (1). The term of order
zero is ``u(x)\,s(x)``, with ``s`` the sum (2). The term of order one vanishes
with the first moment of ``\varphi``, up to an oscillation of the amplitude of
the saturation error. The term of order two brings the second moment of
``\varphi``, ``\int t^{2}\varphi(t)\,\mathrm{d}t = H h^{3}/2``. Altogether, in
dimension ``d`` and away from the ends of the grid,

```math
u_h(\underline{x}) - u(\underline{x})
= \frac{H h^{2}}{4}\,\Delta u(\underline{x}) + O(h^{4})
\;+\; \epsilon_{\mathrm{sat}}\text{-terms},
\tag{3}
```

where ``\Delta`` is the Laplacian and the last terms are ``u`` and its
derivatives times oscillations of amplitude ``\epsilon_{\mathrm{sat}}``
[mazya1996](@cite). The first part is of second order in the step, as for a
linear interpolation. The second does not decrease with the step: refining the
grid brings the error down to the saturation level and no further. The
approximation does not converge in the usual sense, hence the name
*approximate approximation*; but the level is set by ``H`` and falls below any
tolerance of practical interest as soon as ``H \ge 2``.

```@example gaussapprox
steps = [1 / n for n in (10, 20, 40, 80, 160, 320)]
function error_at_center(n, H)
    uh = quasi_interpolate(x -> sinpi(x[1]), GaussianGrid((0.0,), (1.0,), n; H))
    return abs(uh((0.5,)) - 1)
end
plt = plot(; xscale = :log10, yscale = :log10, xlabel = "step h", ylabel = "error at x = 1/2",
    legend = :bottomright)
for H in (1.0, 2.0, 3.0)
    plot!(plt, steps, [error_at_center(round(Int, 1 / h), H) for h in steps]; marker = :circle, label = "H = $H")
end
plot!(plt, steps, [2.0 * h^2 * π^2 / 4 for h in steps]; ls = :dash, color = :black,
    label = "H h² |u''| / 4, H = 2")
plt
```

With ``H = 1`` the error stops near ``10^{-4}``; with ``H = 2`` it follows
``H h^{2}|u''|/4`` over the whole range. The price of a larger ``H`` is on the
same figure: the smoothing error is proportional to ``H``.

## 4. Why Gaussians, for an integral equation

The unknown of an integral equation, for instance the opening of a crack, is
expanded as in (1), with unknown coefficients ``u_m`` in place of the samples.
Three properties then make the method simple.

- **One closed form.** A linear operator ``\mathcal{T}`` that commutes with
  translations acts on (1) as
  ``\mathcal{T}u_h = \sum_m u_m\,(\mathcal{T}\varphi)(\underline{x}-\underline{x}_m)``:
  only ``\mathcal{T}\varphi`` is needed. When ``\mathcal{T}`` is known by its
  Fourier symbol, ``\mathcal{T}\varphi`` is the inverse transform of the symbol
  times a Gaussian, which the [next page](@ref th-gaussian-cracks) obtains in
  closed form for the operator of a crack.
- **A convolution.** Writing the equation at the nodes (collocation), entry
  ``(m, m')`` of the matrix is
  ``(\mathcal{T}\varphi)(\underline{x}_m-\underline{x}_{m'})``: it depends on
  ``m - m'`` only. The matrix is never stored; its product by a vector is a
  discrete convolution, computed by fast Fourier transform.
- **No mesh.** A shape enters through the list of the nodes it contains.

One caution follows from (3). An operator that differentiates amplifies the
oscillation of the saturation error, of wavelength ``h``: the operator of a
crack, whose symbol grows like the wave number, multiplies it by
``2\pi\ell/h`` relative to a field that varies over the size ``\ell`` of the
crack. With 64 nodes across the crack this is a relative error of
``4\times10^{-2}`` on the traction for ``H = 1``, and of ``2\times10^{-6}`` for
``H = 2``. `MeanFieldHomogenization` takes ``H = 2`` by default.

## Where to go next

- [Planar cracks of any contour](@ref th-gaussian-cracks): the integral equation
  of a crack solved with this approximation.
- [Planar cracks of any contour](@ref man-planar-cracks), in the manual:
  describing a contour and computing with it.

## References

```@bibliography
Pages = ["gaussian_approximation.md"]
Canonical = false
```
