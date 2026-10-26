# =============================================================================
#  driver.jl — running a material point along a loading path.
#
#  At each time the components driven by their flux are matched by Newton's
#  method on the free gradients, with the consistent tangent as Jacobian; a
#  step whose local integration fails is retried in halves. Any material of the
#  Gauss-point contract can be driven, a law of this module or a homogenized
#  material.
# =============================================================================

"""
    PointDriverError(time, iterations, residual)

Thrown by [`point_response`](@ref) when the components driven by their flux
could not be matched at `time`, even after the step was cut as far as allowed.
"""
struct PointDriverError <: Exception
    time::Float64
    iterations::Int
    residual::Float64
end

Base.showerror(io::IO, e::PointDriverError) = print(
    io, "PointDriverError: the prescribed fluxes were not reached at t = ", e.time,
    " (", e.iterations, " iterations, relative residual ", e.residual,
    "). Use more times along the path, or raise max_cuts."
)

"""
    PointResponse

What [`point_response`](@ref) returns: the times `t`, and at each of them the
gradients `g`, the fluxes `F` (flat vectors, laid out as the law's fields) and
the state. Indexed by name:

- `r[:t]`, the times;
- a component of a field, by the names [`LoadingPath`](@ref) uses: `r[:ε11]`,
  `r[:σ12]` (tensor components, the √2 of the shears removed), `r[:p]`, `r[:φ]`;
- an internal variable, by the name the law's [`state_layout`](@ref) gives it:
  `r[:p]` would be shadowed by a field of that name, so a plastic law calls its
  cumulated strain `p` only when no field does; a six-entry one also by
  component, `r[:εᵖ11]`.
"""
struct PointResponse{T, M, S, G <: NamedTuple, F <: NamedTuple, L <: NamedTuple}
    t::Vector{T}
    g::Vector{SVector{M, T}}
    F::Vector{SVector{M, T}}
    states::Vector{S}
    gradients::G
    fluxes::F
    layout::L
end

Base.length(r::PointResponse) = length(r.t)

function Base.getindex(r::PointResponse, name::Symbol)
    name === :t && return r.t
    key = _canonical_name(name)
    for (fields, series) in ((r.gradients, r.g), (r.fluxes, r.F))
        hit = _series(series, fields, key)
        hit === nothing || return hit
    end
    if !isempty(r.layout)
        ys = [st.y for st in r.states]
        haskey(r.layout, key) && return [y[r.layout[key]] for y in ys]
        hit = _series(ys, r.layout, key)
        hit === nothing || return hit
    end
    throw(
        KeyError(
            "$name: a PointResponse knows :t, the components of the fields " *
                "$(keys(r.gradients)) and $(keys(r.fluxes)), and the internal variables $(keys(r.layout))"
        )
    )
end

# The time series of one named component of a flat vector, or `nothing`.
function _series(vectors, layout::NamedTuple, key::Symbol)
    for (name, range) in pairs(layout)
        n = _width(range)
        k = findfirst(==(key), _component_names(name, n))
        k === nothing && continue
        i = n == 1 ? range : range[k]
        f = _mandel_factors(n)[k]
        return [v[i] / f for v in vectors]
    end
    return nothing
end

"""
    point_response(material, path::LoadingPath; algorithm = LocalNewton(), state = initial_state(material),
                   rtol = 1e-10, maxiters = 25, max_cuts = 8, on_step = nothing) -> PointResponse

Drive a material point along `path` and return its response at every time of
the path.

`material` is a law of [`MaterialPoints`](@ref MeanFieldHomogenization.MaterialPoints)
or any material of the Gauss-point contract ([`AbstractMFHMaterial`](@ref)). At
each time the components the path drives by their flux are matched to a
relative residual `rtol` by Newton's method on the free gradients, with the
consistent tangent; when the local integration or this iteration fails, the
step is retried in two halves, down to `max_cuts` halvings.

`on_step(i, t, g, F, state)`, if given, is called after each time of the path,
for instance to write the response to a file as it is computed, so that an
interrupted run keeps what it had.

```julia
law = J2Plasticity(; E = 200e3, ν = 0.3, σ₀ = 250.0, H = 1e3)
r = point_response(law, LoadingPath(range(0, 1; length = 51); ε11 = t -> 5e-3 * t))
r[:σ11], r[:εᵖ11]
```
"""
function point_response(
        material::AbstractMFHMaterial, path::LoadingPath;
        algorithm::AbstractLocalAlgorithm = LocalNewton(),
        state::AbstractMaterialState = initial_state(material),
        rtol::Real = 1.0e-10, maxiters::Integer = 25, max_cuts::Integer = 8, on_step = nothing
    )
    gradients, fluxes, layout = _driver_fields(material)
    driven, targets, factors = _resolve(path, gradients, fluxes)
    drive = (; driven, targets, factors)
    opts = (; algorithm, rtol, maxiters, max_cuts)
    return _run(material, path.times, drive, state, opts, on_step, (gradients, fluxes, layout))
end

# Behind a function barrier: the resolved path is only known at run time.
function _run(m, times, drive, state, opts, on_step, fields)
    t₀ = first(times)
    g₀ = _starting_gradients(state, length(drive.driven))
    F, g, st = _advance(m, drive, t₀, t₀, g₀, state, opts, 0)
    T = promote_type(eltype(F), eltype(g), eltype(times))
    M = length(g)
    ts, gs, Fs, states = T[t₀], SVector{M, T}[g], SVector{M, T}[F], [st]
    on_step === nothing || on_step(1, t₀, g, F, st)
    for i in 2:length(times)
        F, g, st = _advance(m, drive, times[i - 1], times[i], g, st, opts, 0)
        push!(ts, times[i]); push!(gs, g); push!(Fs, F); push!(states, st)
        on_step === nothing || on_step(i, times[i], g, F, st)
    end
    return PointResponse(ts, gs, Fs, states, fields...)
end

_starting_gradients(st::PointState, M) = st.g
_starting_gradients(::AbstractMaterialState, M) = zero(SVector{M, Float64})

# The step from t₀ to t₁, retried in halves when it fails.
function _advance(m, drive, t₀, t₁, g, st, opts, depth)
    try
        return _mixed_step(m, drive, t₁, t₁ - t₀, g, st, opts)
    catch err
        (err isa LocalIntegrationError || err isa PointDriverError) || rethrow()
        depth < opts.max_cuts || rethrow()
        tₘ = (t₀ + t₁) / 2
        _, gₘ, stₘ = _advance(m, drive, t₀, tₘ, g, st, opts, depth + 1)
        return _advance(m, drive, tₘ, t₁, gₘ, stₘ, opts, depth + 1)
    end
end

# The free gradients matching the prescribed fluxes at time t: Newton's method on
# the flux residual of the free components, the rows of the driven ones being
# the identity.
function _mixed_step(m, drive, t, Δt, g_guess, st, opts)
    c = drive.driven
    target = SVector(map(f -> f(t), drive.targets)) .* drive.factors
    # The target is plain, the guess may carry derivatives: one element type for both.
    T = promote_type(eltype(target), eltype(g_guess))
    g = map((driven, a, b) -> driven ? T(a) : T(b), c, target, g_guess)
    it = 1
    F, 𝕂, st_new = _update(m, g, st, Δt, opts.algorithm)
    # Newton steps still to take: −1 while the values have not converged, then
    # the derivative steps, down to 0.
    left = _steps_left(F, 𝕂, g, c, target, t, it, opts)
    while left != 0
        # Row i is the identity where gᵢ is driven, the tangent elsewhere.
        g = g - ifelse.(c, one(𝕂), 𝕂) \ _flux_residual(F, c, target)
        it += 1
        F, 𝕂, st_new = _update(m, g, st, Δt, opts.algorithm)
        left = left > 0 ? left - 1 : _steps_left(F, 𝕂, g, c, target, t, it, opts)
    end
    return F, g, st_new
end

_flux_residual(F, c, target) = ifelse.(c, zero(F), F - target)

# Once the values have converged, the Newton steps that make the derivatives
# exact; −1 before; an error after `maxiters` iterations. Relative to the
# fluxes, with one floor: the rounding of 𝕂·g, below which no iteration can go.
# A residual exactly zero (every component driven by its gradient) is converged
# whatever the tangent, which may be infinite at a point.
function _steps_left(F, 𝕂, g, c, target, t, it, opts)
    nr = _primal_norm(_flux_residual(F, c, target))
    worst = nr / max(_primal_norm(F), _primal_norm(target), floatmin(Float64))
    if iszero(nr) || worst <= opts.rtol || nr <= 64 * eps(Float64) * _primal_norm(𝕂) * _primal_norm(g)
        return _derivative_steps(promote_type(eltype(F), eltype(g)))
    end
    it > opts.maxiters && throw(PointDriverError(Float64(TensND.primal(t)), it - 1, worst))
    return -1
end

_primal_norm(x) = norm(TensND.primal.(x))

# The fields of the material being driven.
_driver_fields(m::AbstractPointLaw) = (m.gradients, m.fluxes, m.layout)
# A material of the contract names its fields; the strain is a tensor and the
# other fields of the materials of the package (pore pressure, fluid content)
# are scalars.
function _driver_fields(m::AbstractMFHMaterial)
    gnames, fnames = Constitutive.gradient_names(m), Constitutive.flux_names(m)
    size_of(name) = name in (:ε, :σ) ? 6 : 1
    sizes(names) = NamedTuple{names}(map(size_of, names))
    return _layout(sizes(gnames)), _layout(sizes(fnames)), (;)
end

# One step of any material, on flat vectors.
_update(m::AbstractPointLaw, g, st, Δt, alg) = _point_update(m, g, st, Δt, alg)
function _update(m::AbstractMFHMaterial, g, st, Δt, alg)
    gl, fl, _ = _driver_fields(m)
    gradients = map(r -> _field_value(g[r]), gl)
    r = Constitutive.material_response(m, gradients, st, Δt)
    F = _flat(map(_field_components, NamedTuple{keys(fl)}(r.fluxes)), fl)
    rows = map(keys(fl), values(fl)) do f, rf
        reduce(hcat, map((gn, rg) -> _block_matrix(r.tangents[Symbol(f, gn)], _width(rf), _width(rg)), keys(gl), values(gl)))
    end
    return F, reduce(vcat, rows), Constitutive.state(r)
end

# A tangent block of the contract as a static matrix of the given shape.
_block_matrix(x::Number, nf, ng) = SMatrix{1, 1}(x)
_block_matrix(t::TensND.AbstractTens{4, 3}, nf, ng) = _mandel(t)
_block_matrix(t::TensND.AbstractTens{2, 3}, nf, ng) =
    nf == 6 ? SMatrix{6, 1}(_mandel(t)) : SMatrix{1, 6}(_mandel(t))
