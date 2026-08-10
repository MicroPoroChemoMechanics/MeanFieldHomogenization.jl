# =============================================================================
#  generators.jl — ready-made microstructures for a `ParticleAssembly`.
#
#  Two families, matching the two literatures this module serves:
#
#    * `cubic_lattice` — the SC / BCC / FCC arrays of Molinari & El Mouden
#      (1996), whose figures are the reference results of the cluster model;
#    * `random_assembly` — the hard-particle Metropolis microstructures of
#      Brisard et al. (2014, 2023), whose statistics are what the equivalent
#      inclusion method is exercised on.
#
#  Both are deterministic given their inputs (the random one takes an explicit
#  `rng`), so a figure or a test built on them is reproducible.
# =============================================================================

# Site coordinates of the three cubic Bravais lattices, in units of the period.
# All sites of one lattice are related by a lattice translation, hence
# symmetrically equivalent — they form a single family and the N-body systems
# collapse to one unknown, which is why a lattice estimate is cheap.
const _CUBIC_SITES = Dict(
    :sc => (((0.0, 0.0, 0.0),)),
    :bcc => ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5)),
    :fcc => ((0.0, 0.0, 0.0), (0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5)),
)

"""
    max_packing_fraction(kind::Symbol) -> Float64

Largest volume fraction of equal, non-overlapping spheres on the cubic lattice
`kind` (`:sc`, `:bcc` or `:fcc`) — the point at which neighboring spheres
touch: `π/6 ≈ 0.5236`, `√3π/8 ≈ 0.6802` and `√2π/6 ≈ 0.7405`.

Worth knowing when reading Molinari & El Mouden's figures: their simple-cubic
curves stop near `f = 0.52` for exactly this reason.
"""
function max_packing_fraction(kind::Symbol)
    kind === :sc && return π / 6
    kind === :bcc && return sqrt(3) * π / 8
    kind === :fcc && return sqrt(2) * π / 6
    return throw(ArgumentError("max_packing_fraction: unknown lattice :$(kind)"))
end

"""
    cubic_lattice(kind, matrix_properties, particle_properties;
                  fraction = nothing, radius = nothing, period = 1.0,
                  cutoff = nothing, name_prefix = :p, T = Float64)
        -> ParticleAssembly

Build a [`ParticleAssembly`](@ref) on the simple-cubic (`:sc`), body-centered
(`:bcc`) or face-centered (`:fcc`) cubic lattice of equal spheres, with
[`PeriodicBox`](@ref) boundary conditions.

Give exactly one of `fraction` (total volume fraction of the spheres) or
`radius`. `cutoff` is forwarded to `PeriodicBox`; the default of three periods
sits inside the convergence plateau reported by
[Molinari & El Mouden 1996](@cite molinari1996).

`particle_properties` may be a single property dictionary — applied to every
site — or a vector of dictionaries, one per site of the motif, which is how a
multi-phase array such as their BCC arrangement of alternating voids and rigid
spheres is described. Sites carrying different properties are given different
families.

```julia
asm = cubic_lattice(:sc, Dict(:C => C_m), Dict(:C => C_i); fraction = 0.3)
homogenize(asm, ClusterModel(), :C)
```
"""
function cubic_lattice(
        kind::Symbol, matrix_properties::AbstractDict, particle_properties;
        fraction = nothing, radius = nothing, period = 1.0,
        cutoff = nothing, name_prefix::Symbol = :p, T::Type{<:Number} = Float64
    )
    haskey(_CUBIC_SITES, kind) ||
        throw(ArgumentError("cubic_lattice: unknown lattice :$(kind) (use :sc, :bcc or :fcc)"))
    sites = _CUBIC_SITES[kind]
    nsites = length(sites)
    count(!isnothing, (fraction, radius)) == 1 || throw(
        ArgumentError("cubic_lattice: give exactly one of `fraction` or `radius`")
    )
    a = if radius === nothing
        fmax = max_packing_fraction(kind)
        fraction < fmax || throw(
            ArgumentError(
                "cubic_lattice: a fraction of $(fraction) is not reachable on a " *
                    ":$(kind) lattice of equal spheres (maximum $(round(fmax; digits = 4)))"
            )
        )
        period * cbrt(3 * fraction / (4π * nsites))
    else
        radius
    end
    props = particle_properties isa AbstractDict ?
        fill(particle_properties, nsites) : collect(particle_properties)
    length(props) == nsites || throw(
        ArgumentError(
            "cubic_lattice: the :$(kind) motif has $(nsites) sites but " *
                "$(length(props)) property dictionaries were given"
        )
    )
    asm = ParticleAssembly(; boundary = PeriodicBox(period; cutoff = cutoff), T = T)
    add_matrix!(asm, matrix_properties)
    # Sites carrying the same properties are symmetrically equivalent and share
    # a family; distinct materials get distinct families.
    fam = Dict{UInt64, Int}()
    for (k, s) in enumerate(sites)
        h = hash(props[k])
        label = get!(fam, h, length(fam) + 1)
        add_particle!(
            asm, Symbol(name_prefix, k),
            (s[1] * period, s[2] * period, s[3] * period),
            Elasticity.Ellipsoid(a), props[k]; family = label
        )
    end
    return asm
end

"""
    random_assembly(n, matrix_properties, particle_properties;
                    fraction = nothing, radius = nothing, period = 1.0,
                    rng = Random.default_rng(), cycles = 20_000,
                    boundary = nothing, dim = 3, T = Float64)
        -> ParticleAssembly

Build a `ParticleAssembly` of `n` equal, non-overlapping spheres (`dim = 3`) or
disks (`dim = 2`) by the hard-particle Metropolis algorithm used by
[Brisard et al. 2014](@cite brisard2014), §5.

The particles start on a regular lattice and are then shuffled by `cycles`
sweeps of trial moves, the move amplitude being re-tuned every 50 sweeps to
hold the acceptance ratio near 0.3 — the recipe of the paper. Overlap is
tested in the *periodic* metric, so the resulting microstructure tiles space
whatever boundary treatment is then applied.

Pass `rng` explicitly for reproducibility; the default boundary is a
[`PeriodicBox`](@ref) of the given period, and `boundary = MixedBC(...)`
selects Brisard's ellipsoidal SVE instead.
"""
function random_assembly(
        n::Integer, matrix_properties::AbstractDict, particle_properties::AbstractDict;
        fraction = nothing, radius = nothing, period = 1.0,
        rng::Random.AbstractRNG = Random.default_rng(), cycles::Integer = 20_000,
        boundary::Union{Nothing, AbstractAssemblyBoundary} = nothing,
        dim::Integer = 3, name_prefix::Symbol = :p, T::Type{<:Number} = Float64
    )
    dim in (2, 3) || throw(ArgumentError("random_assembly: `dim` must be 2 or 3"))
    count(!isnothing, (fraction, radius)) == 1 || throw(
        ArgumentError("random_assembly: give exactly one of `fraction` or `radius`")
    )
    b = boundary === nothing ? PeriodicBox(period) : boundary
    a = radius === nothing ? _radius_for_fraction(b, fraction, n, dim, period) : radius
    centers = _metropolis_hard_particles(b, n, a, period, dim, rng, cycles)
    asm = ParticleAssembly(; boundary = b, T = T)
    add_matrix!(asm, matrix_properties)
    geom = dim == 3 ? Elasticity.Ellipsoid(a) : Elasticity.Ellipsoid(a, a)
    for k in 1:n
        add_particle!(asm, Symbol(name_prefix, k), centers[k], geom, particle_properties)
    end
    return asm
end

# Radius achieving a prescribed volume fraction, given the measure of the cell
# the boundary defines.
function _radius_for_fraction(b::PeriodicBox, fraction, n, dim, period)
    return dim == 3 ? period * cbrt(3 * fraction / (4π * n)) : period * sqrt(fraction / (π * n))
end

function _radius_for_fraction(b::MixedBC, fraction, n, dim, period)
    V = Interactions._inclusion_volume(b.shape)
    return dim == 3 ? cbrt(3 * fraction * V / (4π * n)) : sqrt(fraction * V / (π * n))
end

# Hard-particle Metropolis sampler.  The initial state is a regular grid inside
# the container (always overlap-free when the target fraction is attainable),
# so the chain starts admissible and every accepted move keeps it so.  Two
# containers are supported and differ only in their metric and their notion of
# "inside": a periodic box, and an ellipsoidal statistical volume element.
function _metropolis_hard_particles(
        b::AbstractAssemblyBoundary, n::Integer, a::Real, L::Real, dim::Integer,
        rng::Random.AbstractRNG, cycles::Integer
    )
    centers = _initial_state(b, n, a, L, dim)
    _min_pair_distance(b, centers, L) > 2a || throw(
        ArgumentError(
            "random_assembly: $(n) particles of radius $(a) cannot be placed " *
                "without overlap in this cell; reduce the fraction or the count"
        )
    )
    amp = _move_scale(b, L, n, dim)
    accepted = 0
    for cycle in 1:cycles
        for i in 1:n
            trial = _wrap(b, centers[i] .+ amp .* (2 .* rand(rng, dim) .- 1), L)
            _inside(b, trial, a) || continue
            ok = true
            for j in 1:n
                j == i && continue
                _distance(b, trial, centers[j], L) > 2a || (ok = false; break)
            end
            if ok
                centers[i] = trial
                accepted += 1
            end
        end
        if cycle % 50 == 0
            # Re-tune the move amplitude toward a 0.3 acceptance ratio, the
            # recipe of Brisard et al. (2014), §5.
            ratio = accepted / (50 * n)
            amp *= ratio > 0.3 ? 1.05 : 0.95
            amp = min(amp, L / 2)
            accepted = 0
        end
    end
    return centers
end

# ─── Container primitives ────────────────────────────────────────────────────

_move_scale(::PeriodicBox, L, n, dim) = L / (4 * n^(1 / dim))
_move_scale(b::MixedBC, L, n, dim) = maximum(b.shape.semi_axes) / (2 * n^(1 / dim))

_wrap(::PeriodicBox, x, L) = mod.(x, L)
_wrap(::MixedBC, x, L) = x

_distance(::PeriodicBox, x, y, L) = _periodic_distance(x, y, L)
_distance(::MixedBC, x, y, L) = sqrt(sum((x[k] - y[k])^2 for k in eachindex(x)))

# A particle is inside a periodic cell by construction; inside an SVE it must
# fit entirely, hence the shrunk ellipsoid.
_inside(::PeriodicBox, x, a) = true

function _inside(b::MixedBC, x, a)
    s = b.shape.semi_axes
    return sum((x[k] / (s[k] - a))^2 for k in eachindex(x)) ≤ 1
end

function _initial_state(::PeriodicBox, n::Integer, a::Real, L::Real, dim::Integer)
    2a < L || throw(
        ArgumentError(
            "random_assembly: a particle of radius $(a) does not fit in a " *
                "periodic cell of period $(L)"
        )
    )
    m = ceil(Int, n^(1 / dim))
    step = L / m
    pts = Vector{Vector{Float64}}()
    for idx in Iterators.product(ntuple(_ -> 0:(m - 1), dim)...)
        length(pts) == n && break
        push!(pts, collect(Float64.((idx .+ 0.5) .* step)))
    end
    return pts
end

# Inside an SVE the grid has to be over-sampled and filtered, since only the
# points falling in the (shrunk) ellipsoid are admissible.
function _initial_state(b::MixedBC, n::Integer, a::Real, L::Real, dim::Integer)
    s = b.shape.semi_axes
    all(s .> a) || throw(
        ArgumentError(
            "random_assembly: a particle of radius $(a) does not fit in an SVE " *
                "of semi-axes $(s)"
        )
    )
    for m in ceil(Int, n^(1 / dim)):(8 * ceil(Int, n^(1 / dim)))
        pts = Vector{Vector{Float64}}()
        for idx in Iterators.product(ntuple(k -> 0:(m - 1), dim)...)
            x = [(-s[k] + (idx[k] + 0.5) * 2 * s[k] / m) for k in 1:dim]
            _inside(b, x, a) && push!(pts, x)
        end
        length(pts) ≥ n && return pts[1:n]
    end
    return throw(
        ArgumentError(
            "random_assembly: could not seed $(n) particles of radius $(a) " *
                "inside the SVE; reduce the fraction or the count"
        )
    )
end


function _periodic_distance(x, y, L::Real)
    s = zero(float(L))
    for k in eachindex(x)
        d = abs(x[k] - y[k])
        d = min(d, L - d)
        s += d^2
    end
    return sqrt(s)
end

function _min_pair_distance(b::AbstractAssemblyBoundary, centers, L::Real)
    m = Inf
    for i in eachindex(centers), j in (i + 1):lastindex(centers)
        m = min(m, _distance(b, centers[i], centers[j], L))
    end
    return m
end
