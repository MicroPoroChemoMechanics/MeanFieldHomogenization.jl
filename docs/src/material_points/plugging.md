# [Plugging a law into a computation](@id mp-plugging)

A law of `MaterialPoints` is an [`AbstractMFHMaterial`](@ref): it answers the
same Gauss-point contract as a homogenized material, so every caller of that
contract accepts it unchanged. There are two doors, one convenient and one fast.

## The contract: TensND tensors, by name

[`material_response`](@ref) takes the gradients by name, as TensND tensors in
the global frame and numbers, and returns a [`MaterialResponse`](@ref): the
fluxes by name, the tangent as its blocks named *flux then gradient*, and the
new state.

```@example mp_plugging
using MeanFieldHomogenization, TensND, StaticArrays

law = J2Plasticity(; E = 200e3, ν = 0.3, σ₀ = 250.0, H = 2e3)
ε = from_mandel(SVector(3e-3, -1e-3, -1e-3, 0, 0, 0))     # a TensND strain
st = initial_state(law)

r = material_response(law, (ε = ε,), st, 0.0)
stress(r)          # a TensND tensor; tangent(r) is the 4th-order consistent tangent
```

A law of several fields takes and returns them all; its tangent blocks are
`σε`, `σp`, `φε`, `φp` for poroelasticity, which is exactly what the
[fractured-rock material](@ref fe-fractured-rock) of the finite-element section
returns. An element loop written for one serves the other:

```julia
states = [[initial_state(law) for _ in 1:nqp] for _ in 1:ncells]
for cell in cells, q in 1:nqp
    r = material_response(law, (ε = ε_at(cell, q),), states_old[cell][q], Δt)
    # assemble stress(r) into the residual, tangent(r) into the Jacobian
    states[cell][q] = state(r)
end
# once the global Newton has converged: states_old .= states
```

The old state is never mutated: a global iteration that fails is retried from
`states_old` without any bookkeeping.

## The fast door: flat static vectors

[`point_update`](@ref) is what `material_response` calls inside. It takes and
returns static vectors (strains and stresses as [`Mandel`](@ref) vectors, the
tangent as an `SMatrix`), is type-stable and **allocates nothing**:

```@example mp_plugging
g = SVector(3e-3, -1e-3, -1e-3, 0, 0, 0)
σ, 𝕂, st_new = point_update(law, g, st, 0.0)
bytes(law, g, st) = @allocated point_update(law, g, st, 0.0)   # measured inside a function,
bytes(law, g, st)                                             # not at global scope
```

An element loop that keeps its strains in Mandel form calls it directly and
pays only for the arithmetic of the law.

## Choosing the local algorithm

The integration algorithm is a separate object, passed by keyword. Its
tolerances are those of [`LocalNewton`](@ref):

```@example mp_plugging
σ_tight, _, _ = point_update(law, g, st, 0.0; algorithm = LocalNewton(; abstol = 1e-14, maxiters = 30))
maximum(abs, σ_tight - σ)
```

## Differentiating with respect to the parameters

Every number may be a `ForwardDiff.Dual`. Building the law with dual parameters
differentiates its response, exactly (see the
[theory](@ref mp-theory)): the internal variables are solved on the values,
then their derivatives come from the implicit-function theorem.

```@example mp_plugging
using ForwardDiff

σ11(θ) = point_update(J2Plasticity(; E = 200e3, ν = 0.3, σ₀ = θ[1], H = θ[2]), g, st, 0.0)[1][1]
ForwardDiff.gradient(σ11, [250.0, 2e3])          # ∂σ₁₁/∂σ₀, ∂σ₁₁/∂H
```

The same holds through a whole loading path: the response of
[`point_response`](@ref) differentiates with respect to the parameters, which
is what an identification by gradient descent needs.

```@example mp_plugging
path = LoadingPath(range(0, 1; length = 11); ε11 = t -> 5e-3 * t)
final_stress(θ) = point_response(J2Plasticity(; E = 200e3, ν = 0.3, σ₀ = θ[1], H = θ[2]), path)[:σ11][end]
ForwardDiff.gradient(final_stress, [250.0, 2e3])
```

In uniaxial tension with linear hardening the final stress is
``\sigma_0 + \frac{E H}{E + H}\,(\varepsilon - \sigma_0/E)``, whose derivatives
are ``1 - H/(E + H) = 0.990\ldots`` with respect to ``\sigma_0`` and
``E^2(\varepsilon - \sigma_0/E)/(E + H)^2 = 3.68\ldots\times10^{-3}`` with respect
to ``H``: the numbers above.

!!! tip "Parameters belong in `p`"
    A law reads its parameters from its `p`; a number captured elsewhere (in a
    closure) still carries its derivative, but keeping them in `p` makes
    [`law_parameters`](@ref) list them and is what the shipped laws do.
