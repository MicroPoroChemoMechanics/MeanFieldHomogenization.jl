# =============================================================================
#  quasi_interpolation.jl — Maz'ya's quasi-interpolation by Gaussians
#  (mazya1996): a function sampled on a regular grid of step h is rebuilt as
#
#      u_h(x) = (πH)^(-d/2) Σ_m u(x_m) exp(-|x - x_m|² / (H h²)) .
#
#  The sum of the generating functions is not exactly one: it oscillates around
#  one with the amplitude `saturation_error(H, d)`, which no refinement of the
#  grid removes and which the parameter H makes as small as wanted.
# =============================================================================

"""
    gaussian_generator(r², h, H, Val(d))

Value of the generating function of the Gaussian approximation in dimension `d`,

```math
\\varphi(\\underline{x}) = (\\pi H)^{-d/2}\\,
    \\exp\\!\\left(-\\frac{|\\underline{x}|^2}{H h^2}\\right),
```

at squared distance `r²` from its node, for a grid of step `h` and a shape
parameter `H` [mazya1996](@cite). Its integral is ``h^d``.
"""
gaussian_generator(r², h, H, ::Val{d}) where {d} = exp(-r² / (H * h^2)) / sqrt(oftype(H / 1, π) * H)^d

"""
    saturation_error(H, d=1)

Amplitude of the oscillation of ``\\sum_m \\varphi(\\underline{x} - \\underline{x}_m)``
around one, for the Gaussian generating function of shape parameter `H` on a regular
grid of dimension `d`:

```math
\\epsilon_{\\mathrm{sat}} = \\Bigl(\\sum_{m \\in \\mathbb{Z}}
    \\mathrm{e}^{-\\pi^2 H m^2}\\Bigr)^{d} - 1 .
```

It is the part of the error of [`quasi_interpolate`](@ref) that does not vanish
with the grid step [mazya1996](@cite): about ``2d\\,\\mathrm{e}^{-\\pi^2 H}``.
"""
function saturation_error(H, d::Integer = 1)
    q = exp(-oftype(H / 1, π)^2 * H)
    # θ₃(0, q) - 1 = 2 Σ_{m ≥ 1} q^(m²); beyond m = 8 the terms are below any
    # floating-point precision as soon as H ≥ 0.1.
    s = 2 * sum(q^(m^2) for m in 1:8)
    # (1 + s)^d - 1 without the cancellation: s may be far below the machine precision
    return expm1(d * log1p(s))
end

"""
    QuasiInterpolant{d, T}

Gaussian quasi-interpolant of nodal values on a regular grid, returned by
[`quasi_interpolate`](@ref) and evaluated as a function of the point.
"""
struct QuasiInterpolant{d, T <: Number, V <: AbstractArray{T, d}, G <: GaussianGrid{d}}
    grid::G
    values::V
end

"""
    quasi_interpolate(u, grid::GaussianGrid) -> QuasiInterpolant
    quasi_interpolate(values::AbstractArray, grid::GaussianGrid) -> QuasiInterpolant

Quasi-interpolant of Maz'ya [mazya1996](@cite) of the function `u` (or of its
nodal `values`) on `grid`:

```math
u_h(\\underline{x}) = \\sum_m u(\\underline{x}_m)\\,\\varphi(\\underline{x} - \\underline{x}_m),
```

with ``\\varphi`` the [`gaussian_generator`](@ref). The result is called as
`uh(x)`. Away from the edges of the grid, ``u_h - u`` is of second order in the
grid step, down to the [`saturation_error`](@ref).

```julia
grid = GaussianGrid((0.0,), (1.0,), 50; H = 2.0)
uh = quasi_interpolate(x -> sinpi(x[1]), grid)
uh((0.5,))
```
"""
quasi_interpolate(u, grid::GaussianGrid) = quasi_interpolate(map(u, nodes(grid)), grid)
function quasi_interpolate(values::AbstractArray{T, d}, grid::GaussianGrid{d}) where {T, d}
    size(values) == grid.dims || throw(
        DimensionMismatch("the nodal values have size $(size(values)), the grid $(grid.dims)")
    )
    return QuasiInterpolant{d, T, typeof(values), typeof(grid)}(grid, values)
end

function (q::QuasiInterpolant{d})(x) where {d}
    g = q.grid
    ξ = SVector{d}(x)
    return sum(
        v * gaussian_generator(sum(abs2, ξ - y), g.h, g.H, Val(d))
            for (v, y) in zip(q.values, nodes(g))
    )
end
