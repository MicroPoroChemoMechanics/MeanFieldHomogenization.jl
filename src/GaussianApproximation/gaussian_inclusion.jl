# =============================================================================
#  gaussian_inclusion.jl — a heterogeneity of any shape.
#
#  The strain in an infinite medium that holds a heterogeneity is
#  ε = E - 𝔾⁰ * τ, with τ = (ℂ₁ - ℂ₀) : ε the polarization, which vanishes
#  outside the heterogeneity. Written for τ, with 𝕄 = (ℂ₁ - ℂ₀)⁻¹,
#
#      𝕄 : τ + 𝔾⁰ * τ = E     in the heterogeneity,
#
#  one symmetric equation that covers every contrast: 𝕄 = 0 for a rigid
#  inclusion, -𝕊₀ for a pore. τ is expanded on the Gaussians of the nodes of a
#  grid and the equation is written at the same nodes; conduction is the same
#  with tensors of order 2.
#
#  The mean of τ over the heterogeneity, for the unit loads E, is all the
#  schemes need: ⟨ε⟩ = 𝕄 : ⟨τ⟩ and ⟨σ⟩ = ℂ₀ : ⟨ε⟩ + ⟨τ⟩.
# =============================================================================

"Shape tag of a heterogeneity described by a region and solved on a grid."
struct GaussianShape end

# One discretization: the grid, the cells the region touches, and the fraction
# of each that it covers.
struct VolumeLevel
    grid::GaussianGrid{3, Float64}
    inside::Vector{CartesianIndex{3}}
    θ::Vector{Float64}
end

function VolumeLevel(region::AbstractRegion{3}, n::Integer, H::Float64, memory_limit::Integer, samples::Int)
    grid = GaussianGrid(map(SVector{3, Float64}, bounding_box(region))..., n; H)
    # the six strain components and the 21 components of the kernel
    _check_memory(grid.dims, 21, 6, memory_limit)
    θ = coverage(region, grid; samples)
    inside = findall(>(_MIN_COVERAGE), θ)
    isempty(inside) && throw(
        ArgumentError(
            "no cell of the grid ($(join(grid.dims, "×")), h = $(grid.h)) is covered by the region. " *
                "Increase the number of nodes `n`."
        )
    )
    return VolumeLevel(grid, inside, θ[inside])
end

"""
    GaussianInclusion{T, R<:AbstractRegion{3}, B<:AbstractBasis}

Heterogeneity occupying the region `region` of space, described in the frame
`basis`. Its concentration tensors are computed by the Gaussian approximation of
the integral equation of the heterogeneity
(see [Heterogeneities of any shape](@ref th-gaussian-volume)); it enters the
schemes through the custom-inclusion interface.
"""
struct GaussianInclusion{T <: Number, R <: AbstractRegion{3}, B <: TensND.AbstractBasis} <: MFH_Core.AbstractCustomInclusion{T}
    region::R
    basis::B
    levels::Vector{VolumeLevel}
    rtol::Float64
    cache::FECache
end

"""
    GaussianInclusion(region; n=24, H=2.0, extrapolate=true, euler_angles=(), samples=4,
                      rtol=1e-8, memory_limit=2^31)
    GaussianInclusion(shape; kw...)

Heterogeneity of shape `region` (any [`AbstractRegion`](@ref) of space: a
function, set operations, voxels) or `shape` (an [`Ellipsoid`](@ref), a
[`Supersphere`](@ref), a [`Superspheroid`](@ref)), oriented by ZYZ Euler angles.

- `n`: number of nodes along the longest side of the bounding box of the region;
- `H`: shape parameter of the Gaussians;
- `extrapolate`: the concentration tensors are computed on the grids of `n` and
  `n ÷ 2` nodes and extrapolated to a vanishing step; `false` keeps the grid of
  `n` nodes alone;
- `samples`: points per direction on which the covered fraction of a cell cut by
  the surface is counted;
- `rtol`: relative tolerance of the iterative solve;
- `memory_limit`: the inclusion is refused, before any allocation, when its
  grid needs more bytes than this.

```julia
pore = GaussianInclusion(Torus(1.0, 0.3))
rve = RVE()
add_phase!(rve, :matrix, Ellipsoid(1.0), Dict(:C => C₀); fraction = :rest)
add_phase!(rve, :pores, pore, Dict(:C => 1.0e-9 * C₀); fraction = 0.1, symmetrize = IsoSymmetrize())
```
"""
function GaussianInclusion(
        region::AbstractRegion{3}; n::Integer = 24, H::Real = 2.0, extrapolate::Bool = true,
        euler_angles::Tuple{Vararg{Real}} = (), rtol::Real = 1.0e-8, memory_limit::Integer = 2^31,
        samples::Integer = 4
    )
    extrapolate && n < 4 && throw(ArgumentError("extrapolating needs n ≥ 4 nodes, got n = $n"))
    all(x -> x isa AbstractFloat, first(bounding_box(region))) || throw(ArgumentError(_NO_SHAPE_DERIVATIVE))
    levels = [VolumeLevel(region, k, Float64(H), memory_limit, Int(samples)) for k in (extrapolate ? (n, n ÷ 2) : (n,))]
    basis = TensND.euler_basis(euler_angles; T = Float64)
    return GaussianInclusion{Float64, typeof(region), typeof(basis)}(region, basis, levels, Float64(rtol), FECache())
end
GaussianInclusion(shape::Union{Elasticity.Ellipsoid{3}, Superspheres.AbstractSuperShape}; kw...) =
    GaussianInclusion(Region(shape); kw...)

const _NO_SHAPE_DERIVATIVE = "the shape of a `GaussianInclusion` enters through the cells it " *
    "covers, counted on a grid: derivatives with respect to the geometry are not available. " *
    "Derivatives with respect to the properties of the phases are."

Schemes._replace_geom_field(::GaussianInclusion, ::Val{name}, ::Nothing, value) where {name} =
    throw(ArgumentError(_NO_SHAPE_DERIVATIVE))
Schemes._replace_geom_field(::GaussianInclusion, ::Val{name}, ::Int, value) where {name} =
    throw(ArgumentError(_NO_SHAPE_DERIVATIVE))

MFH_Core.dimension(::GaussianInclusion) = 3
MFH_Core.inclusion_basis(g::GaussianInclusion) = g.basis
MFH_Core.shape_trait(::GaussianInclusion) = GaussianShape()
# Both concentration tensors are supplied: the contributions follow from them
# without the inverse of the property of the phase, infinite for a rigid inclusion.
MFH_Core.is_homogeneous_inclusion(::GaussianInclusion) = false
_fe_cache(g::GaussianInclusion) = g.cache

Base.show(io::IO, g::GaussianInclusion) = print(
    io, "GaussianInclusion (", nameof(typeof(g.region)), ", ", length(first(g.levels).inside),
    " nodes, h=", first(g.levels).grid.h, ")"
)

"""
    inclusion_volume(g::GaussianInclusion)

Volume of the heterogeneity as its finest grid sees it: the sum of the covered
fractions of its cells.
"""
inclusion_volume(g::GaussianInclusion) = (l = first(g.levels); sum(l.θ) * l.grid.h^3)

# ─── The solve ───────────────────────────────────────────────────────────────

# Matrix of a property in the frame of the inclusion: 3×3 in conduction,
# Kelvin–Mandel 6×6 in elasticity.
_local_matrix(P::TensND.AbstractTens{2, 3}, basis) = TensND.operator_matrix(P, basis)
_local_matrix(P::TensND.AbstractTens{4, 3}, basis) = SMatrix{6, 6}(TensND.mandel(P, basis))

# The contrast is stiffer (+1), softer (-1) or neither (0), read on the values.
function _contrast_sign(δ::SMatrix)
    λ = eigvals(Symmetric(TensND.primal.(Matrix(δ))))
    return all(>(0), λ) ? 1 : all(<(0), λ) ? -1 : 0
end

# Compliance of the contrast seen by a cell that the heterogeneity covers over
# the fraction θ only. The cell is given the effective property of a laminate:
# in series (uniform flux) for a stiffer heterogeneity, 𝕄 = (𝕄₁ + 𝕊₀)/θ - 𝕊₀,
# which stays finite when it is rigid; in parallel (uniform gradient)
# otherwise, 𝕄 = 𝕄₁/θ, which stays finite when it is a pore.
_cell_block(M, S₀, θ, sign) = sign > 0 ? (M + S₀) / θ - S₀ : M / θ

# The fields inside a rigid inclusion or a void are not unique, and the system
# degenerates with the contrast: the solve is stopped after this many iterations
# rather than left to run, and says so.
const _ITMAX = 2000

# Mean polarization over the heterogeneity under the m unit loads, on one level.
function _level_polarization(l::VolumeLevel, Γ::F, M::SMatrix{m, m}, S₀, sign, rtol) where {F, m}
    samples = sample_kernel(Γ, l.grid.dims, Val(m))
    N = length(l.inside)
    R = kron(Matrix(I, m, m), ones(N))
    blocks = [_cell_block(M, S₀, θ, sign) for θ in l.θ]
    # 𝕄 + 𝔾⁰ₕ is positive for a stiffer heterogeneity, negative for a softer one
    # (then solved with the opposite sign), of no definite sign otherwise
    W = sign > 0 ? gaussian_solve(SampledKernel(samples, blocks), l.inside, R; rtol, itmax = _ITMAX, method = :cg) :
        sign < 0 ? -gaussian_solve(SampledKernel(map(-, samples), map(-, blocks)), l.inside, R; rtol, itmax = _ITMAX, method = :cg) :
        gaussian_solve(SampledKernel(samples, blocks), l.inside, R; rtol, itmax = _ITMAX, method = :gmres)
    return SMatrix{m, m}([sum(@view W[((p - 1) * N + 1):(p * N), q]) for p in 1:m, q in 1:m]) / sum(l.θ)
end

# (⟨ε⟩, ⟨σ⟩) over the heterogeneity for the unit remote gradients, as matrices
# in the frame of the inclusion.
function _solve_concentrations(g::GaussianInclusion, P₁, P₀)
    C₁, C₀ = _local_matrix(P₁, g.basis), _local_matrix(P₀, g.basis)
    δ = C₁ - C₀
    # no contrast: the heterogeneity is the matrix
    iszero(δ) && return one(C₀), C₀
    M, S₀, sign = inv(δ), inv(C₀), _contrast_sign(δ)
    Γ = _smoothed_kernel(P₀, g.basis, first(g.levels).grid.H)
    T = _extrapolate([_level_polarization(l, Γ, M, S₀, sign, g.rtol) for l in g.levels], g.levels)
    A = M * T
    return A, C₀ * A + T
end

# The kernel as a function of the difference of nodes, in the frame of the
# inclusion. An isotropic medium is the same in every frame.
_smoothed_kernel(P₀::TensND.TensISO, _, H) = Δ -> smoothed_green_operator(P₀, Δ, H)
_smoothed_kernel(P₀, _, _) = throw(
    ArgumentError(
        "a `GaussianInclusion` needs an isotropic reference medium, got a property of type " *
            "$(nameof(typeof(P₀))). Register the phase with `symmetrize = IsoSymmetrize()`, " *
            "or give an isotropic reference."
    )
)

# Memoized for floating-point properties; computed afresh for duals.
_concentrations(g::GaussianInclusion, P₁::TensND.AbstractTens{o, 3, Float64}, P₀::TensND.AbstractTens{o, 3, Float64}) where {o} =
    _fe_memo(g.cache, map(x -> round(x; sigdigits = 12), (_local_matrix(P₁, g.basis)..., _local_matrix(P₀, g.basis)...))) do
    g.cache.assemblies += 1
    _solve_concentrations(g, P₁, P₀)
end
_concentrations(g::GaussianInclusion, P₁, P₀) = _solve_concentrations(g, P₁, P₀)

_from_local(A::SMatrix{3, 3}, basis) = TensND.Tens(A, basis)
_from_local(A::SMatrix{6, 6}, basis) = TensND.from_mandel(A, basis)

# ─── Level B of the custom-inclusion interface ────────────────────────────────

"""
    gradient_gradient_concentration(g::GaussianInclusion, P₁, P₀)
    flux_gradient_concentration(g::GaussianInclusion, P₁, P₀)

Mean strain (or potential gradient) and mean stress (or flux) in a heterogeneity
of any shape, of property `P₁`, per unit remote strain in an infinite medium of
property `P₀`: the two concentration tensors of the custom-inclusion interface,
from which every contribution tensor and every scheme follow. One solve gives
both, memoized by the inclusion for each pair of properties.
"""
MFH_Core.gradient_gradient_concentration(
    g::GaussianInclusion, P₁::TensND.AbstractTens{4, 3}, P₀::TensND.AbstractTens{4, 3}; kw...
) = _from_local(first(_concentrations(g, P₁, P₀)), g.basis)
MFH_Core.gradient_gradient_concentration(
    g::GaussianInclusion, P₁::TensND.AbstractTens{2, 3}, P₀::TensND.AbstractTens{2, 3}; kw...
) = _from_local(first(_concentrations(g, P₁, P₀)), g.basis)
MFH_Core.flux_gradient_concentration(
    g::GaussianInclusion, P₁::TensND.AbstractTens{4, 3}, P₀::TensND.AbstractTens{4, 3}; kw...
) = _from_local(last(_concentrations(g, P₁, P₀)), g.basis)
MFH_Core.flux_gradient_concentration(
    g::GaussianInclusion, P₁::TensND.AbstractTens{2, 3}, P₀::TensND.AbstractTens{2, 3}; kw...
) = _from_local(last(_concentrations(g, P₁, P₀)), g.basis)
