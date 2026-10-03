# =============================================================================
#  sensitivities.jl — autodiff wrappers around `homogenize`.
#
#  ForwardDiff is now a strong dependency (it is also used by the
#  built-in Newton-Raphson SC solver in `_solve_sc(::NewtonDefault, …)`)
#  so the concrete methods live directly in this file rather than in a
#  weak extension.
# =============================================================================

"""
    derivative(rve, scheme, param::AbstractParameter;
               output = :C, indexer = identity, kw...) -> Number | AbstractTens

Scalar derivative of `homogenize(rve, scheme; property=output)` with respect
to the parameter selected by the lens `param`.

`indexer` is a function that extracts a *scalar* from the effective tensor
(for example `C -> get_data(C)[1] / 3` for the bulk modulus). Without an
`indexer`, the derivative of the full tensor is returned — this is valid as
long as ForwardDiff can propagate through the result (typically when the
output is itself a structured object whose components ForwardDiff handles
natively, via the `_data` Dual fields).

Extra `kw...` (e.g. `Chunk`, `Tag`) are forwarded to `ForwardDiff`.

The method is only available with `using ForwardDiff`; calling it without
the extension loaded raises an explicit error pointing at the extension.

See also [`gradient`](@ref), [`jacobian`](@ref), [`sensitivity`](@ref).
"""
function derivative(
        rve::AbstractHomogenizationCell, scheme::HomogenizationScheme,
        p::AbstractParameter;
        output::Symbol = :C, indexer = identity, kw...
    )
    x₀ = get_param(rve, p)
    f = x -> begin
        rve′ = set_param(rve, p, x)
        out = homogenize(rve′, scheme; property = output, kw...)
        return indexer(out)
    end
    return ForwardDiff.derivative(f, x₀)
end

"""
    gradient(rve, scheme, params::AbstractVector{<:AbstractParameter};
             output = :C, indexer, kw...) -> Vector

Gradient of a scalar functional (extracted via `indexer`) of the effective
tensor returned by `homogenize(rve, scheme; property=output)`, with respect
to the vector of lenses `params`.

Implementation: `ForwardDiff.gradient` with automatic chunking
(`ForwardDiff.Chunk(length(params))` via the standard `pickchunksize`
rule).

Without `using ForwardDiff`: error.
"""
function gradient(
        rve::AbstractHomogenizationCell, scheme::HomogenizationScheme,
        ps::AbstractVector{<:AbstractParameter};
        output::Symbol = :C, indexer = identity,
        chunk = nothing, kw...
    )
    x₀ = [get_param(rve, p) for p in ps]
    f = xs -> begin
        rve′ = _set_many(rve, ps, xs)
        out = homogenize(rve′, scheme; property = output, kw...)
        return indexer(out)
    end
    if chunk === nothing
        return ForwardDiff.gradient(f, x₀)
    else
        cfg = ForwardDiff.GradientConfig(f, x₀, chunk)
        return ForwardDiff.gradient(f, x₀, cfg)
    end
end

"""
    jacobian(rve, scheme, params::AbstractVector{<:AbstractParameter};
             output = :C, indexer = identity, kw...) -> Array

Jacobian (returned as a flat array `length(output_flat) × length(params)`)
of `homogenize(rve, scheme; property=output)` with respect to the vector of
lenses `params`.

`indexer` can be used to reduce the output to a sub-tensor before
flattening. Without an `indexer`, the full effective tensor is flattened
via `get_array` then `vec`.

Without `using ForwardDiff`: error.
"""
function jacobian(
        rve::AbstractHomogenizationCell, scheme::HomogenizationScheme,
        ps::AbstractVector{<:AbstractParameter};
        output::Symbol = :C, indexer = identity,
        chunk = nothing, kw...
    )
    x₀ = [get_param(rve, p) for p in ps]
    f = xs -> begin
        rve′ = _set_many(rve, ps, xs)
        out = homogenize(rve′, scheme; property = output, kw...)
        return _flatten_for_jacobian(indexer(out))
    end
    if chunk === nothing
        return ForwardDiff.jacobian(f, x₀)
    else
        cfg = ForwardDiff.JacobianConfig(f, x₀, chunk)
        return ForwardDiff.jacobian(f, x₀, cfg)
    end
end

# Single-parameter convenience: pass one AbstractParameter, return jacobian
# along that single dimension (still a Matrix of size (length(output_flat), 1)).
jacobian(
    rve::AbstractHomogenizationCell, scheme::HomogenizationScheme,
    p::AbstractParameter; kw...
) =
    jacobian(rve, scheme, [p]; kw...)

_flatten_for_jacobian(x::Number) = [x]
_flatten_for_jacobian(x::AbstractArray) = vec(x)
_flatten_for_jacobian(x::TensND.AbstractTens) = vec(collect(TensND.get_array(x)))

"""
    sensitivity(f, x₀; kind = :auto, kw...)

Generic ForwardDiff wrapper. `f(x)` is a user-supplied closure that builds
or perturbs an RVE, runs `homogenize`, and returns a scalar (or a tensor).

`kind`:
- `:derivative` — scalar `x₀`, scalar output; returns ``f'(x_0)``.
- `:gradient`   — vector `x₀`, scalar output; returns ``\\nabla f(x_0)``.
- `:jacobian`   — vector `x₀`, tensor output; returns the flattened Jacobian.
- `:auto` (default) — pick one of the above from the types of `x₀` and `f(x₀)`.

This entry point covers cases the `AbstractParameter` lenses cannot
express (compound parametrizations, user-defined inclusion types without
an exposed scalar field, etc.).

Without `using ForwardDiff`: error.
"""
function sensitivity(f, x₀; kind::Symbol = :auto, kw...)
    actual = kind === :auto ? _autoselect(f, x₀) : kind
    if actual === :derivative
        return ForwardDiff.derivative(f, x₀)
    elseif actual === :gradient
        return ForwardDiff.gradient(f, x₀)
    elseif actual === :jacobian
        wrapper = x -> _flatten_for_jacobian(f(x))
        return ForwardDiff.jacobian(wrapper, x₀)
    else
        throw(ArgumentError("sensitivity: unknown kind=:$(actual); expected :derivative, :gradient or :jacobian"))
    end
end

function _autoselect(f, x₀::Number)
    y = f(x₀)
    return y isa Number ? :derivative : :jacobian
end

function _autoselect(f, x₀::AbstractVector)
    y = f(x₀)
    return y isa Number ? :gradient : :jacobian
end

_autoselect(f, x₀) = throw(
    ArgumentError(
        "sensitivity: cannot autoselect kind for x₀ of type $(typeof(x₀)); pass `kind=:derivative|:gradient|:jacobian`"
    )
)
