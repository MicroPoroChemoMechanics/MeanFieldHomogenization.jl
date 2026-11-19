# =============================================================================
#  edge_rule.jl — where the edge of a crack sits between two nodes.
#
#  Keeping or dropping a node moves the edge of the discrete crack by a whole
#  grid step, and the Gaussians spread the opening beyond the last node kept:
#  left alone, both effects cost an error of first order in the step. They are
#  removed by two rules, calibrated once on a straight edge aligned with the
#  grid (a line crack, a one-dimensional problem):
#
#  - a node whose cell is covered by the crack over a fraction θ only is held by
#    a spring of stiffness γ (1-θ)/θ times the self-term of the kernel: none for
#    a cell inside (θ = 1), rigid for a cell outside (θ → 0). With the right γ,
#    the edge of the discrete crack follows the edge of the region continuously;
#  - it then sits a constant c₀ h beyond it: the coverage θ is computed on the
#    region eroded by c₀ h.
#
#  γ and c₀ depend on the shape parameter H only.
# =============================================================================

# Kernel of the line crack, symbol |ξ|, for the Gaussian exp(-x²/(H h²))/√(πH):
#     (h²/π) ∫₀^∞ ξ exp(-ξ² H h²/4) cos(ξ ρ h) dξ = (2/(πH)) [1 - 2z D(z)],
# z = ρ/√H, D the Dawson function.
_line_kernel(ρ, H) = (z = ρ / sqrt(H); 2 / (oftype(z, π) * H) * (1 - 2 * z * dawson(z)))

# Half-length of the line crack that the discrete solution represents, in grid
# steps, for a crack [-a, a] (a in steps) on nodes at the half-integers: the
# exact opening √(a² - x²) of the unit load has the integral π a²/2.
function _line_crack_half_length(a, H, γ)
    M = ceil(Int, a) + 1
    x = [i - 0.5 for i in (1 - M):M]
    θ = [clamp(min(xᵢ + 0.5, a) - max(xᵢ - 0.5, -a), 0.0, 1.0) for xᵢ in x]
    kept = findall(>(_MIN_COVERAGE), θ)
    A = [_line_kernel(x[i] - x[j], H) for i in kept, j in kept]
    w = (A + Diagonal(γ .* (1 .- θ[kept]) ./ θ[kept] .* _line_kernel(0.0, H))) \ ones(length(kept))
    return sqrt(2 * sum(w) / π)
end

# A cell covered over less than this fraction is outside: its spring would be
# rigid to within the tolerance of the solve.
const _MIN_COVERAGE = 1.0e-3

# Offsets of the discrete edge, in grid steps, as the edge of a crack of about
# 50 steps is swept across one cell.
_edge_offsets(H, γ) = [_line_crack_half_length(a, H, γ) - a for a in 50.05:0.1:50.95]

"""
    edge_rule(H) -> (γ, c₀)

Constants of the edge rule of a Gaussian crack of shape parameter `H`: the
spring coefficient `γ` for which the edge of the discrete crack follows the edge
of the region as it is swept across a cell, and the constant distance `c₀`, in
grid steps, by which it then lies beyond it. Calibrated on a line crack, the
local problem of any smooth edge, and memoized.
"""
edge_rule(H::Real) = lock(_EDGE_RULE_LOCK) do
    get!(_EDGE_RULES, Float64(H)) do
        # golden-section search of the γ that minimizes the sweep of the offset
        spread(γ) = (o = _edge_offsets(Float64(H), γ); maximum(o) - minimum(o))
        lo, hi, φ = 0.2, 1.5, (sqrt(5) - 1) / 2
        x₁, x₂ = hi - φ * (hi - lo), lo + φ * (hi - lo)
        f₁, f₂ = spread(x₁), spread(x₂)
        while hi - lo > 1.0e-4
            if f₁ < f₂
                hi, x₂, f₂ = x₂, x₁, f₁
                x₁ = hi - φ * (hi - lo)
                f₁ = spread(x₁)
            else
                lo, x₁, f₁ = x₁, x₂, f₂
                x₂ = lo + φ * (hi - lo)
                f₂ = spread(x₂)
            end
        end
        γ = (lo + hi) / 2
        o = _edge_offsets(Float64(H), γ)
        (γ, sum(o) / length(o))
    end
end

const _EDGE_RULES = Dict{Float64, Tuple{Float64, Float64}}()
const _EDGE_RULE_LOCK = ReentrantLock()

# Unit vectors that probe the erosion of a region: twelve around the circle in the
# plane, the axes and the diagonals of the cube in space.
_ring(::Val{2}) = [SVector(reverse(sincos(2π * k / 12))) for k in 0:11]
_ring(::Val{3}) = [normalize(SVector{3, Float64}(Tuple(I))) for I in CartesianIndices((-1:1, -1:1, -1:1)) if !iszero(I)]

# Is `y` in the region eroded by δ: in the region with the probes at the distance δ around it?
_in_eroded(y, region, δ, ring) = y ∈ region && (iszero(δ) || all(u -> (y + δ * u) ∈ region, ring))

"""
    coverage(region, grid; erosion=0, samples=6) -> Array

Fraction of each cell of `grid` covered by `region` eroded by the distance
`erosion`, counted on `samples` points per direction and per cell. A cell and
its neighbors that all agree at their centers are not subdivided.
"""
function coverage(region::AbstractRegion{d}, grid::GaussianGrid{d}; erosion::Real = 0, samples::Integer = 6) where {d}
    R = CartesianIndices(grid.dims)
    ring = _ring(Val(d))
    center = [_in_eroded(node(grid, I), region, erosion, ring) for I in R]
    offsets = [(SVector{d, Int}(Tuple(J)) .- 0.5) / samples .- 0.5 for J in CartesianIndices(ntuple(_ -> samples, Val(d)))]
    near = CartesianIndices(ntuple(_ -> -1:1, Val(d)))
    return map(R) do I
        around = (center[I + Δ] for Δ in near if checkbounds(Bool, center, I + Δ))
        # a cell on the side of the box may be cut by the contour without a neighbor to tell
        uniform = allequal(around) && all(k -> 1 < I[k] < grid.dims[k], 1:d)
        uniform && return Float64(center[I])
        x = node(grid, I)
        count(o -> _in_eroded(x + grid.h * o, region, erosion, ring), offsets) / samples^d
    end
end
