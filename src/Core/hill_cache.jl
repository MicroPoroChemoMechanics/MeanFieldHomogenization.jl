# =============================================================================
#  hill_cache.jl — Hill tensors computed once for a series of calls.
#
#  In an anisotropic matrix a Hill tensor is a cubature, of the order of a
#  millisecond, and it is most of the time of a scheme. A series of calls that
#  keeps the shapes and the matrix (a sweep of the fractions, of the moduli of
#  the inclusions) recomputes the same tensors at every call. Inside
#  `with_hill_cache`, each numerical Hill tensor is computed once and read back
#  afterwards, bit for bit. Opt-in and scoped: no global memo grows behind the
#  user's back, and a closed form, cheaper than a lookup, never consults it.
# =============================================================================

using Base.ScopedValues: ScopedValue, with

"""
    HillCache(; maxentries = 1024)

A store of Hill tensors for [`with_hill_cache`](@ref): each numerical Hill
tensor (anisotropic matrix: residues, nested Gauss–Kronrod, DECUHR) is computed
once per shape, frame, matrix, method and tolerances, and read back afterwards.
When it holds `maxentries` tensors it is emptied before storing the next one.
`hits` and `misses` count the reads and the computations. Thread-safe.
"""
struct HillCache
    store::Dict{Any, Any}
    lock::ReentrantLock
    maxentries::Int
    hits::Threads.Atomic{Int}
    misses::Threads.Atomic{Int}
end

function HillCache(; maxentries::Integer = 1024)
    maxentries ≥ 1 || throw(ArgumentError("HillCache: maxentries must be positive; got $maxentries"))
    return HillCache(Dict{Any, Any}(), ReentrantLock(), maxentries, Threads.Atomic{Int}(0), Threads.Atomic{Int}(0))
end

Base.length(c::HillCache) = lock(() -> length(c.store), c.lock)
Base.empty!(c::HillCache) = (lock(() -> empty!(c.store), c.lock); c)

const _HILL_CACHE = ScopedValue{Union{Nothing, HillCache}}(nothing)

"""
    with_hill_cache(f, cache::HillCache)

Run `f()` with the numerical Hill tensors read from `cache` when it holds them,
and stored in it otherwise. For a series of calls that keeps the shapes and the
anisotropic matrix:

```julia
cache = HillCache()
estimates = with_hill_cache(cache) do
    [homogenize(cell(f), MoriTanaka(), :C) for f in 0.01:0.01:0.3]
end
```

Each tensor is keyed by the exact values that determine it (the shape, its
semi-axes and frame, the canonical components of the matrix, the method and
the tolerances), and only `Float64` ones are stored: a dual or a symbolic input
is computed as usual, so that derivatives are unaffected. The results are the
bits the calls give without the cache.
"""
with_hill_cache(f, cache::HillCache) = with(f, _HILL_CACHE => cache)

# The tensor that `compute()` gives, read from the cache in scope under the key
# `keyof()` returns; the key is built only when a cache is in scope, and
# `nothing` (an input that is not `Float64`) bypasses it. The type read back is
# asserted to be the one `compute` returns, so that the call stays inferred.
@inline function _hill_cached(compute::F, keyof::K) where {F, K}
    cache = _HILL_CACHE[]
    cache === nothing && return compute()
    key = keyof()
    key === nothing && return compute()
    return _hill_cached(compute, key, cache)::Base.promote_op(compute)
end

@noinline function _hill_cached(compute, key, cache::HillCache)
    found = lock(() -> get(cache.store, key, nothing), cache.lock)
    if found !== nothing
        Threads.atomic_add!(cache.hits, 1)
        return found
    end
    Threads.atomic_add!(cache.misses, 1)
    value = compute()
    lock(cache.lock) do
        length(cache.store) ≥ cache.maxentries && empty!(cache.store)
        cache.store[key] = value
    end
    return value
end
