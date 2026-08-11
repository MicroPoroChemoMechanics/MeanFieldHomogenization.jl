# =============================================================================
#  cache.jl — memoization of the expensive part of a Gauss-point law.
#
#  WHY THIS IS NOT OPTIONAL.  A self-consistent solve on a cracked RVE with a
#  genuinely anisotropic running estimate costs on the order of 10 ms per
#  iteration (each non-aligned family needs a numerical COD, and the Picard
#  iteration takes tens of steps). Called once per quadrature point, per Newton
#  iteration, per time step, that is several orders of magnitude too slow.
#
#  WHAT MAKES IT CHEAP.  For a *flat* crack the compliance contribution ℍ is the
#  ω → 0 limit: it does not depend on the aperture at all (`EllipticCrack` has no
#  thickness field). In the ARMA model the densities, radii and normals are fixed
#  data of the fracture network, and the only thing the Gauss-point state changes
#  is which families are open. So the homogenized quantities depend on the state
#  **only through the open/closed set** — a handful of discrete configurations,
#  at most `2^n` for `n` families, shared by every quadrature point in a region.
#
#  This is the same trick, and deliberately the same shape, as the finite-element
#  inclusion cache in `src/FiniteElements/common.jl`.
# =============================================================================

"""
    MaterialCache()

Memoization store shared by the quadrature points of a
[`AbstractMFHMaterial`](@ref).

Pass one to [`material_response`](@ref) via its `cache` keyword and reuse it
across the whole element loop — the entries are keyed on the *microstructural
configuration*, not on the point, so every quadrature point sharing a
configuration pays for it once.

```julia
cache = MaterialCache()
for cell in CellIterator(dh), qp in 1:getnquadpoints(cv)
    r = material_response(mat, ε[qp], states[qp], Δt; cache = cache)
end
@show cache_stats(cache)     # (; hits, misses, entries)
```

!!! warning "Not thread-safe"
    A `MaterialCache` is a plain `Dict` with no lock. Use one cache per thread
    in a threaded element loop, or none at all. Sharing one across threads is a
    data race, and the failure mode — a torn read of a partially built entry —
    would surface as a wrong stiffness rather than as an error.

See also [`cached!`](@ref), [`cache_stats`](@ref).
"""
mutable struct MaterialCache
    entries::Dict{Any, Any}
    hits::Int
    misses::Int
end

MaterialCache() = MaterialCache(Dict{Any, Any}(), 0, 0)

"""
    cached!(f, cache, key)
    cached!(f, ::Nothing, key)

Return `cache.entries[key]`, computing it with `f()` on a miss. With `nothing`
in place of a cache, `f()` is always called — so a material can be written once
and used with or without memoization.
"""
function cached!(f, cache::MaterialCache, key)
    hit = get(cache.entries, key, nothing)
    if hit !== nothing
        cache.hits += 1
        return hit
    end
    cache.misses += 1
    value = f()
    cache.entries[key] = value
    return value
end

cached!(f, ::Nothing, key) = f()

"""
    cache_stats(cache) -> NamedTuple

`(; hits, misses, entries)` for a [`MaterialCache`](@ref), or a zeroed tuple for
`nothing`.

Worth looking at once on a real mesh: for a model whose fracture network is
uniform over a region, `entries` should settle at the number of distinct
open/closed configurations actually visited (a handful), not grow with the
number of quadrature points. If it grows with the mesh, the cache key is
carrying continuous data and the memoization is not doing its job.
"""
cache_stats(c::MaterialCache) =
    (hits = c.hits, misses = c.misses, entries = length(c.entries))
cache_stats(::Nothing) = (hits = 0, misses = 0, entries = 0)

"""
    reset_cache!(cache) -> MaterialCache

Empty a [`MaterialCache`](@ref) and zero its counters. Needed whenever the
underlying microstructure changes in a way the key does not capture — a
different fracture network per material region, for instance, if the network id
is not part of the key.
"""
function reset_cache!(c::MaterialCache)
    empty!(c.entries)
    c.hits = 0
    c.misses = 0
    return c
end

reset_cache!(::Nothing) = nothing
