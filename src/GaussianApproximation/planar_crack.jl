# =============================================================================
#  planar_crack.jl — a flat crack of any contour.
#
#  The opening [[u]] of the crack is expanded on the Gaussians of the nodes that
#  fall inside its contour; the traction is collocated at the same nodes. The
#  crack then enters MFH as any other: only its COD tensor is computed here.
# =============================================================================

"""
    ContourShape

Shape tag of a flat crack of arbitrary contour, normalized by a reference length
``\\ell`` exactly as a penny-shaped crack is by its radius: density
``\\varepsilon = N \\ell^3``, ``\\pi \\ell^3\\,\\boldsymbol{B}\\cdot\\boldsymbol{\\Sigma}
\\cdot\\underline{n} = \\int [\\![\\underline{u}]\\!]\\,\\mathrm{d}S``,
``\\mathbb H = \\tfrac34\\,\\underline{n}\\stackrel{s}{\\otimes}\\boldsymbol{B}
\\stackrel{s}{\\otimes}\\underline{n}`` and ``\\Delta\\mathbb S = \\tfrac{4\\pi}{3}
\\varepsilon\\,\\mathbb H``. Carried by [`PlanarCrack`](@ref).
"""
struct ContourShape <: Cracks.CrackShape end

# One discretization of the crack: its grid, the nodes kept, and the springs that
# hold those of the edge (`edge_rule.jl`).
struct CrackLevel
    grid::GaussianGrid{2, Float64}
    inside::Vector{CartesianIndex{2}}
    spring::Vector{Float64}
end

function CrackLevel(region::AbstractRegion{2}, n::Integer, H::Float64, memory_limit::Integer)
    grid = GaussianGrid(map(SVector{2, Float64}, bounding_box(region))..., n; H)
    # the three opening components and their six kernel components
    _check_memory(grid.dims, 6, 3, memory_limit)
    γ, c₀ = edge_rule(H)
    θ = coverage(region, grid; erosion = c₀ * grid.h)
    inside = findall(>(_MIN_COVERAGE), θ)
    isempty(inside) && throw(
        ArgumentError(
            "no cell of the grid ($(join(grid.dims, "×")), h = $(grid.h)) is covered by the region " *
                "once its edge is located. Increase the number of nodes `n`."
        )
    )
    return CrackLevel(grid, inside, [γ * (1 - θ[I]) / θ[I] for I in inside])
end

"""
    PlanarCrack{T, R<:AbstractRegion{2}, B<:AbstractBasis}

Flat crack occupying the region `region` of its plane, of normal
``\\underline{n}`` (third vector of `basis`; the coordinates of the region are
taken along the first two). Its opening is computed by the Gaussian
approximation of the integral equation of the crack
(see [Planar cracks of any contour](@ref th-gaussian-cracks)).
"""
struct PlanarCrack{T <: Number, R <: AbstractRegion{2}, B <: TensND.AbstractBasis} <: MFH_Core.AbstractCrack{T}
    region::R
    size::T
    basis::B
    levels::Vector{CrackLevel}
    area::Float64
    rtol::Float64
    cache::FECache
end

"""
    PlanarCrack(region; size=nothing, n=64, H=2.0, extrapolate=true, euler_angles=(),
                rtol=1e-10, memory_limit=2^31)

Flat crack of contour `region` (any [`AbstractRegion`](@ref) of the plane: a
catalog shape, a set operation, a function, an image), oriented by ZYZ Euler
angles.

- `size`: the reference length ``\\ell`` that normalizes the COD tensor and
  defines the crack density ``\\varepsilon = N\\ell^3``. By default the radius of
  the disk of the same area, ``\\sqrt{S/\\pi}``, so that a disk is normalized as a
  [`PennyCrack`](@ref);
- `n`: number of nodes along the longest side of the bounding box of the region;
- `H`: shape parameter of the Gaussians;
- `extrapolate`: the mean openings are computed on the grids of `n` and `n ÷ 2`
  nodes and extrapolated to a vanishing step (their error is of first order in
  the step); `false` keeps the grid of `n` nodes alone;
- `rtol`: relative tolerance of the iterative solve;
- `memory_limit`: the crack is refused, before any allocation, when its grid
  needs more bytes than this.

A crack whose faces touch on islands is the region without the islands:

```julia
PlanarCrack(setdiff(Disk(1.0), Disk(0.2; center = (0.4, 0.0))); size = 1.0)
```
"""
function PlanarCrack(
        region::AbstractRegion{2}; size = nothing, n::Integer = 64, H::Real = 2.0,
        extrapolate::Bool = true, euler_angles::Tuple{Vararg{Real}} = (),
        rtol::Real = 1.0e-10, memory_limit::Integer = 2^31
    )
    extrapolate && n < 4 && throw(ArgumentError("extrapolating needs n ≥ 4 nodes, got n = $n"))
    _check_plain_contour(region)
    levels = [CrackLevel(region, k, Float64(H), memory_limit) for k in (extrapolate ? (n, n ÷ 2) : (n,))]
    fine = first(levels).grid
    area = sum(coverage(region, fine)) * fine.h^2
    ℓ = _reference_size(size, measure(region), area)
    T = TensND.float_like(typeof(ℓ))
    basis = TensND.euler_basis(euler_angles; T = T)
    return PlanarCrack{T, typeof(region), typeof(basis)}(
        region, T(ℓ), basis, levels, area, Float64(rtol), FECache()
    )
end

# The contour is seen through the nodes it contains, a discontinuous function of
# its parameters: a derivative with respect to them would silently be zero.
const _NO_CONTOUR_DERIVATIVE = "the contour of a `PlanarCrack` enters through the nodes it " *
    "contains, which do not vary continuously with it: derivatives with respect to the " *
    "geometry are not available. Derivatives with respect to the properties of the matrix are."

_check_plain_contour(region) = all(x -> x isa AbstractFloat, first(bounding_box(region))) ||
    throw(ArgumentError(_NO_CONTOUR_DERIVATIVE * " Got a contour of number type $(eltype(first(bounding_box(region))))."))

Schemes._replace_geom_field(::PlanarCrack, ::Val{name}, ::Nothing, value) where {name} =
    throw(ArgumentError(_NO_CONTOUR_DERIVATIVE))
Schemes._replace_geom_field(::PlanarCrack, ::Val{name}, ::Int, value) where {name} =
    throw(ArgumentError(_NO_CONTOUR_DERIVATIVE))

# Reference length: given, or the radius of the disk of the same area — the exact
# area when the region knows it, the area of its cells otherwise.
_reference_size(size, _, _) = (_check_positive(size, "reference length `size`"); size)
_reference_size(::Nothing, S, _) = sqrt(S / π)
_reference_size(::Nothing, ::Nothing, Sₕ) = sqrt(Sₕ / π)

MFH_Core.shape_trait(::PlanarCrack) = ContourShape
Cracks.semi_major(c::PlanarCrack) = c.size
Cracks.semi_minor(c::PlanarCrack) = c.size
_fe_cache(c::PlanarCrack) = c.cache

function MFH_Core.shape_tensor(c::PlanarCrack{T}) where {T}
    z = zero(T)
    return TensND.Tens(TensND.Tensors.SymmetricTensor{2, 3, T}((c.size, z, z, c.size, z, z)), c.basis)
end

Base.show(io::IO, c::PlanarCrack{T}) where {T} = print(
    io, "PlanarCrack{", T, "} (", nameof(typeof(c.region)), ", size ℓ=", c.size, ", ",
    length(first(c.levels).inside), " nodes, h=", first(c.levels).grid.h, ")"
)

"""
    crack_area(crack::PlanarCrack)

Area of the crack as its grid sees it: the sum of the covered fractions of its
cells.
"""
crack_area(c::PlanarCrack) = c.area

# ─── Mean openings ───────────────────────────────────────────────────────────

# Value at a vanishing step of a quantity whose error is of first order in the
# step, from its values on the levels of the crack (one level: the value itself).
_extrapolate(v, levels) = length(levels) == 1 ? first(v) :
    (h₁ = levels[1].grid.h; h₂ = levels[2].grid.h; (h₂ * v[1] - h₁ * v[2]) / (h₂ - h₁))

# Nodal openings of one level under the m unit loads, one column per load.
function _level_openings(l::CrackLevel, hs::SymbolHarmonics{m}, onsite, rtol) where {m}
    N = length(l.inside)
    R = kron(Matrix(I, m, m), ones(N))
    samples = _harmonic_samples(l.grid, hs, Val(m))
    # on each node: the spring of the edge rule, and the interface stiffness
    self = self_block(samples, Val(m))
    return gaussian_solve(SampledKernel(samples, [s * self + onsite for s in l.spring]), l.inside, R; rtol)
end

# ∫ [[u]] dS / (π ℓ³) under the m unit loads: the m × m matrix of the mean openings.
function _level_mean_openings(l::CrackLevel, crack::PlanarCrack, hs::SymbolHarmonics{m}, onsite) where {m}
    W = _level_openings(l, hs, onsite, crack.rtol)
    N = length(l.inside)
    s = l.grid.h^2 / (π * crack.size^3)
    G = SMatrix{m, m}([s * sum(@view W[((p - 1) * N + 1):(p * N), q]) for p in 1:m, q in 1:m])
    # symmetric up to the tolerance of the solve
    return (G + G') / 2
end

_solve_mean_openings(crack::PlanarCrack, hs, onsite) =
    _extrapolate([_level_mean_openings(l, crack, hs, onsite) for l in crack.levels], crack.levels)

# Memoized for floating-point properties, keyed by the harmonics of the symbol
# (which hold the reference medium as the crack sees it) and the on-site block;
# computed afresh for duals, whose derivatives a cache would lose.
_mean_openings(crack::PlanarCrack, hs::SymbolHarmonics{m, Float64}, onsite::SMatrix{m, m, Float64}) where {m} =
    _fe_memo(crack.cache, map(x -> round(x; sigdigits = 12), (reduce(vcat, vec.(hs.cosines))..., reduce(vcat, vec.(hs.sines))..., onsite...))) do
    crack.cache.assemblies += 1
    _solve_mean_openings(crack, hs, onsite)
end::typeof(onsite)
_mean_openings(crack::PlanarCrack, hs::SymbolHarmonics, onsite) = _solve_mean_openings(crack, hs, onsite)

# Components of an interface stiffness in the frame of the crack; none by default.
_onsite(::Nothing, _, ::Val{m}) where {m} = zero(SMatrix{m, m, Float64})
_onsite(K::TensND.AbstractTens{2, 3}, basis, ::Val{3}) = SMatrix{3, 3}(TensND.components(K, basis))
_onsite(α::Real, _, ::Val{1}) = SMatrix{1, 1}(α)

# ─── COD tensor ──────────────────────────────────────────────────────────────

"""
    cod_tensor(crack::PlanarCrack, C₀; K_interface=nothing, angles=32)
    cod_tensor(crack::PlanarCrack, K₀; α_interface=nothing, angles=32)

COD tensor ``\\boldsymbol{B}`` (elasticity) or thermal COD scalar ``b``
(conduction) of a flat crack of any contour in a matrix of any anisotropy,
normalized by its reference length ``\\ell``:

```math
\\pi\\,\\ell^3\\,\\boldsymbol{B}\\cdot\\boldsymbol{\\Sigma}\\cdot\\underline{n}
    = \\int_{\\mathcal{S}} [\\![\\underline{u}]\\!]\\,\\mathrm{d}S .
```

The opening is the solution of the Gaussian system of the crack; the solve is
memoized by the crack for each reference medium. An interface stiffness
(`K_interface`, traction ``\\boldsymbol{K}\\cdot[\\![\\underline{u}]\\!]`` between
the faces; `α_interface` in conduction) enters the integral equation itself, as
a block on every node. `angles` is the number of directions at which the symbol
of the crack operator is sampled.
"""
function Cracks.cod_tensor(
        crack::PlanarCrack, C₀::TensND.AbstractTens{4, 3};
        K_interface::Union{Nothing, TensND.AbstractTens{2, 3}} = nothing, angles::Integer = 32, kw...
    )
    MFH_Core._bump!(MFH_Core.COD_CALLS)
    hs = symbol_harmonics(_crack_symbol(C₀, crack.basis), Val(3); angles)
    G = _mean_openings(crack, hs, _onsite(K_interface, crack.basis, Val(3)))
    return TensND.Tens(TensND.Tensors.SymmetricTensor{2, 3}((G[1, 1], G[2, 1], G[3, 1], G[2, 2], G[3, 2], G[3, 3])), crack.basis)
end

function Cracks.cod_tensor(
        crack::PlanarCrack, K₀::TensND.AbstractTens{2, 3};
        α_interface::Union{Nothing, Real} = nothing, angles::Integer = 32, kw...
    )
    hs = symbol_harmonics(_crack_symbol(K₀, crack.basis), Val(1); angles)
    return only(_mean_openings(crack, hs, _onsite(α_interface, crack.basis, Val(1))))
end

# ─── Opening field ───────────────────────────────────────────────────────────

"""
    crack_opening(crack::PlanarCrack, C₀, Σ; K_interface=nothing) -> (points, openings)
    crack_opening(crack::PlanarCrack, K₀, σ; α_interface=nothing) -> (points, jumps)

Opening of the crack under the remote stress `Σ` (or jump of the potential under
the remote ``\\boldsymbol{\\sigma} \\equiv -\\underline{q}``), on its finest grid:
the positions of the nodes in the plane of the crack, and at each node the
displacement jump as its three components in the frame of the crack (a scalar
in conduction).
"""
function crack_opening(
        crack::PlanarCrack, C₀::TensND.AbstractTens{4, 3}, Σ::TensND.AbstractTens{2, 3};
        K_interface::Union{Nothing, TensND.AbstractTens{2, 3}} = nothing, angles::Integer = 32
    )
    l = first(crack.levels)
    hs = symbol_harmonics(_crack_symbol(C₀, crack.basis), Val(3); angles)
    W = _level_openings(l, hs, _onsite(K_interface, crack.basis, Val(3)), crack.rtol)
    t = SMatrix{3, 3}(TensND.components(Σ, crack.basis))[:, 3]
    N = length(l.inside)
    u = W * t
    return [node(l.grid, I) for I in l.inside], [SVector(u[j], u[N + j], u[2N + j]) for j in 1:N]
end

function crack_opening(
        crack::PlanarCrack, K₀::TensND.AbstractTens{2, 3}, σ::TensND.AbstractTens{1, 3};
        α_interface::Union{Nothing, Real} = nothing, angles::Integer = 32
    )
    l = first(crack.levels)
    hs = symbol_harmonics(_crack_symbol(K₀, crack.basis), Val(1); angles)
    W = _level_openings(l, hs, _onsite(α_interface, crack.basis, Val(1)), crack.rtol)
    return [node(l.grid, I) for I in l.inside], vec(W) * TensND.components(σ, crack.basis)[3]
end

"""
    opening_surface(crack::PlanarCrack, values) -> (x, y, Z)

The values that [`crack_opening`](@ref) returns at the nodes of the crack, laid
on its grid: the coordinates `x` and `y` of the lines of nodes in the plane of
the crack and the matrix `Z[i, j]` of the value at `(x[i], y[j])`, zero where
the faces do not open. For a surface plot.
"""
function opening_surface(crack::PlanarCrack, values::AbstractVector{<:Number})
    l = first(crack.levels)
    length(values) == length(l.inside) || throw(
        DimensionMismatch("$(length(values)) values for the $(length(l.inside)) nodes of the crack")
    )
    Z = zeros(eltype(values), l.grid.dims)
    Z[l.inside] .= values
    x, y = (l.grid.lo[k] .+ ((1:l.grid.dims[k]) .- 0.5) .* l.grid.h for k in 1:2)
    return x, y, Z
end

# ─── Intensity factors ───────────────────────────────────────────────────────

# Value at the point `x` of the plane of the field carried by the nodes of a level.
_field_at(l::CrackLevel, values, x) = sum(
    v * gaussian_generator(sum(abs2, x - node(l.grid, I)), l.grid.h, l.grid.H, Val(2))
        for (v, I) in zip(values, l.inside)
)

# Coefficient N′ of √s in the opening u(s) ≃ √(s + s₀) (N′ + M (s + s₀)) sampled
# at the distances `s` from the front. The position s₀ of the edge of the
# discrete crack is fitted with it: |u|² is a polynomial in s that vanishes there.
function _sqrt_coefficient(s::AbstractVector, u::AbstractVector)
    V = [one.(s) s s .^ 2]
    c = V \ sum.(abs2, u)
    # the root of c₁ + c₂ s + c₃ s² next to the front
    s₀ = 2 * c[1] / (c[2] + sqrt(c[2]^2 - 4 * c[1] * c[3]))
    r = sqrt.(s .+ s₀)
    F = [r r .^ 3]
    return [first(F \ getindex.(u, i)) for i in eachindex(first(u))]
end

"""
    sif(crack::PlanarCrack, C₀, Σ; point, normal, window=(3, 10), K_interface=nothing)
        -> (𝐊, (Kᴵ, Kᴵᴵ, Kᴵᴵᴵ))
    sif(crack::PlanarCrack, K₀, σ; point, normal, window=(3, 10), α_interface=nothing) -> K_T

Stress (or flux) intensity factors at the point `point` of the front of a crack
of any contour, where the unit outward normal to the front is `normal`; both are
given by their two coordinates in the plane of the crack.

The opening computed on the finest grid is sampled along the inward normal at
the distances `window[1] h` to `window[2] h` of the front (`h` the grid step)
and fitted by ``\\sqrt{s}\\,(\\underline{N}' + \\underline{M}\\,s)``; the local
displacement intensity factor ``\\underline{N} = \\underline{N}'/(4\\sqrt{2/\\pi})``
then gives ``\\underline{K}`` by the tangent ribbon ([`sif_from_dif`](@ref)).
The relation holds at a regular point of the front: not at a corner.
"""
function Cracks.sif(
        crack::PlanarCrack, C₀::TensND.AbstractTens{4, 3}, Σ::TensND.AbstractTens{2, 3};
        point, normal, window = (3, 10), K_interface = nothing, kw...
    )
    _, u = crack_opening(crack, C₀, Σ; K_interface)
    N′ = _front_coefficient(crack, u, point, normal, window)
    ν̂, n̂ = _front_vectors(crack, normal)
    N = TensND.Tens(N′ / (4 * sqrt(2 / π)), crack.basis)
    return Cracks.sif_from_dif(N, ν̂, n̂, C₀; kw...)
end

function Cracks.sif(
        crack::PlanarCrack, K₀::TensND.AbstractTens{2, 3}, σ::TensND.AbstractTens{1, 3};
        point, normal, window = (3, 10), α_interface = nothing, kw...
    )
    _, u = crack_opening(crack, K₀, σ; α_interface)
    N′ = only(_front_coefficient(crack, SVector.(u), point, normal, window))
    ν̂, n̂ = _front_vectors(crack, normal)
    return Cracks.sif_from_dif(N′ / (4 * sqrt(2 / π)), ν̂, n̂, K₀; kw...)
end

function _front_coefficient(crack::PlanarCrack, u, point, normal, window)
    l = first(crack.levels)
    x₀, ν = SVector{2, Float64}(point), normalize(SVector{2, Float64}(normal))
    s = range(window[1] * l.grid.h, window[2] * l.grid.h; length = 15)
    return _sqrt_coefficient(collect(s), [_field_at(l, u, x₀ - sᵢ * ν) for sᵢ in s])
end

# The outward normal to the front and the normal to the crack, as vectors of its basis.
_front_vectors(crack::PlanarCrack{T}, normal) where {T} = (
    TensND.Tens(SVector{3, T}(normalize(SVector{2, T}(normal))..., zero(T)), crack.basis),
    TensND.Tens(SVector{3, T}(zero(T), zero(T), one(T)), crack.basis),
)
