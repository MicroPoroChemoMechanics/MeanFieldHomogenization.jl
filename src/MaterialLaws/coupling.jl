# =============================================================================
#  coupling.jl — what every finite-element code needs to call a material, and
#  nothing that depends on a particular one.
#
#  Three pieces. The states of the quadrature points, kept twice so that a
#  rejected step is undone by not committing it. The modeling hypotheses, which
#  say how the strain of an element (3-D, plane strain, axisymmetric) is a
#  strain of the material, and the reverse for the stress and the tangent. And a
#  quasi-static driver, Newton's method on the structure with the step cut when
#  it fails, which calls the assembly of the user's FE code through one function.
#  An adapter for a given FE library (Ferrite: the extension) only writes the
#  element loop and the assembly.
# =============================================================================

# ── States of the quadrature points ──────────────────────────────────────────

"""
    QuadratureStates(material, ncells, nqp)

The states of the quadrature points of a mesh, `nqp` per cell, kept twice:

- `states.old[cell][qp]`, the state at the last converged step, which the
  material reads;
- `states.new[cell][qp]`, the state the current iteration writes.

[`commit!`](@ref) makes the new states the old ones once a step has converged;
[`rollback!`](@ref) discards them when it is rejected. A material never mutates
a state, so nothing else is needed to retry a step.
"""
struct QuadratureStates{S}
    old::Vector{Vector{S}}
    new::Vector{Vector{S}}
end

function QuadratureStates(material::AbstractMFHMaterial, ncells::Integer, nqp::Integer)
    st = initial_state(material)
    return QuadratureStates([fill(st, nqp) for _ in 1:ncells], [fill(st, nqp) for _ in 1:ncells])
end

"""
    commit!(states::QuadratureStates)

The step has converged: the states it wrote become those the next step starts
from.
"""
function commit!(s::QuadratureStates)
    foreach(copyto!, s.old, s.new)
    return s
end

"""
    rollback!(states::QuadratureStates)

The step is rejected: the states it wrote are discarded.
"""
function rollback!(s::QuadratureStates)
    foreach(copyto!, s.new, s.old)
    return s
end

# ── Modeling hypotheses ──────────────────────────────────────────────────────

"""
    ModelingHypothesis

How the strain of a finite element is a strain of the material:

| hypothesis | element | strain of the material |
|:--|:--|:--|
| [`ThreeD`](@ref)`()` | 3-D, coordinates ``(x_1, x_2, x_3)`` | the element's |
| [`PlaneStrain`](@ref)`()` | 2-D, ``(x_1, x_2)`` | ``\\varepsilon_{33} = \\varepsilon_{13} = \\varepsilon_{23} = 0`` |
| [`Axisymmetric`](@ref)`()` | 2-D, ``(r, z)`` | the hoop strain ``\\varepsilon_{\\theta\\theta} = u_r/r`` as third component |

A hypothesis converts the strain of an element, and the virtual strain of each
shape function, into the `Mandel` strain of the material, so that the
element residual is ``\\int \\delta\\boldsymbol\\varepsilon : \\boldsymbol\\sigma`` and
its Jacobian ``\\int \\delta\\boldsymbol\\varepsilon : \\mathbb K : \\delta\\boldsymbol\\varepsilon'``
whatever the hypothesis.
"""
abstract type ModelingHypothesis end

"""
    ThreeD()

Three-dimensional elements: the strain of the element is that of the material.
"""
struct ThreeD <: ModelingHypothesis end

"""
    PlaneStrain()

Two-dimensional elements in plane strain: the out-of-plane strains vanish, and
the out-of-plane stress ``\\sigma_{33}`` is whatever the material returns.
Exact, unlike plane stress, which would have to condense the law at every
point.
"""
struct PlaneStrain <: ModelingHypothesis end

"""
    Axisymmetric()

Two-dimensional elements of an axisymmetric body, in the coordinates
``(r, z)``: the axes of the material are ``(r, z, \\theta)``, the hoop strain
``u_r / r`` being the third component. Volumes are measured per radian, with the
weight ``r``.
"""
struct Axisymmetric <: ModelingHypothesis end

const _SQRT2 = sqrt(2.0)

"""
    mandel_strain(hypothesis, ε, u = nothing, x = nothing) -> Mandel

The strain of the material for the strain `ε` of an element (a
`Tensors.SymmetricTensor{2, dim}`), in `Mandel` components. In
[`Axisymmetric`](@ref) the displacement `u` and the position `x` at the point
give the hoop strain ``u_r / r``. The same function gives the virtual strain of
a shape function, from its symmetric gradient and its value.
"""
mandel_strain(::ThreeD, ε::Tensors.SymmetricTensor{2, 3}, u = nothing, x = nothing) =
    Tensors.tomandel(SArray, ε)
mandel_strain(::PlaneStrain, ε::Tensors.SymmetricTensor{2, 2}, u = nothing, x = nothing) =
    Mandel(ε[1, 1], ε[2, 2], zero(eltype(ε)), zero(eltype(ε)), zero(eltype(ε)), _SQRT2 * ε[1, 2])
mandel_strain(::Axisymmetric, ε::Tensors.SymmetricTensor{2, 2}, u, x) =
    Mandel(ε[1, 1], ε[2, 2], u[1] / x[1], zero(eltype(ε)), zero(eltype(ε)), _SQRT2 * ε[1, 2])

"""
    volume_weight(hypothesis, x)

The factor of the volume element at the position `x`: ``r`` in
[`Axisymmetric`](@ref) (per radian), 1 otherwise (`true`, the unit of any number
type).
"""
volume_weight(::ModelingHypothesis, x) = true
volume_weight(::Axisymmetric, x) = x[1]

# ── One quadrature point, any material ───────────────────────────────────────
# The fluxes and the tangent of any material of the contract on flat vectors:
# the laws of this module directly, the other materials through
# `material_response` and the conversions of the contract.

function _mandel_update(m::AbstractPointLaw, g, st, Δt; algorithm, external, cache = nothing)
    return _point_update(_with_external(m, external), g, st, Δt, algorithm)
end

function _mandel_update(m::AbstractMFHMaterial, g, st, Δt; algorithm, external, cache = nothing)
    _no_external(m, external)
    gl, fl, _ = _driver_fields(m)
    gradients = map(r -> _field_value(g[r]), gl)
    r = Constitutive.material_response(m, gradients, st, Δt; cache)
    F = _flat(map(_field_components, NamedTuple{keys(fl)}(r.fluxes)), fl)
    rows = map(keys(fl), values(fl)) do f, rf
        reduce(hcat, map((gn, rg) -> _block_matrix(r.tangents[Symbol(f, gn)], _width(rf), _width(rg)), keys(gl), values(gl)))
    end
    return F, reduce(vcat, rows), Constitutive.state(r)
end

# A mechanical coupling serves a material of the strain alone.
function _check_mechanical(material, caller::Symbol)
    names = Constitutive.gradient_names(material)
    names == (:ε,) || throw(
        ArgumentError(
            "$caller couples a mechanical problem; $(nameof(typeof(material))) " *
                "has the gradients $names and needs a coupling of its fields"
        )
    )
    return nothing
end

# ── All the quadrature points of a mesh at once ──────────────────────────────

"""
    update_points!(F, 𝕂, G, states, Δt, material; algorithm = LocalNewton(),
                   external = (;), ntasks = Threads.nthreads()) -> (F, 𝕂)

The fluxes and the tangents of `material` at all the quadrature points of a
mesh, for a finite-element code that evaluates its gradients in one array:
column `p` of `G` holds the gradients of point `p`, stacked in the order of the
material, the points ordered cell by cell, `nqp` per cell as `states` (a
[`QuadratureStates`](@ref) of `n ÷ nqp` cells) holds them. Column `p` of `F`
receives the fluxes, stacked the same way, and `𝕂[:, :, p]` the consistent
tangent, ``\\partial F/\\partial G``; the states are read from `states.old` and
written into `states.new`, so that [`commit!`](@ref) and [`rollback!`](@ref)
manage the step as in [`quasistatic!`](@ref).

For a mechanical material `G` is the `6 × n` matrix of the `Mandel`
strains, `F` the stresses and `𝕂` the `6 × 6 × n` tangents. A material of
several fields stacks them: a poroelastic one ([`BiotPoroelasticity`](@ref),
[`ClosingCrackPoroelasticity`](@ref)) has the gradients ``(\\boldsymbol\\varepsilon, p)``,
7 rows, and the fluxes ``(\\boldsymbol\\sigma, \\phi)``, 7 rows.

The points are independent: they are shared among `ntasks` tasks by blocks of
cells, which run on as many threads as Julia was started with, and the result
does not depend on `ntasks`. A point whose law fails to integrate raises its
[`LocalIntegrationError`](@ref).
"""
function update_points!(
        F::AbstractMatrix, 𝕂::AbstractArray{<:Any, 3}, G::AbstractMatrix, states::QuadratureStates,
        Δt::Real, material; algorithm::AbstractLocalAlgorithm = LocalNewton(), external::NamedTuple = (;),
        ntasks::Integer = Threads.nthreads()
    )
    gradients, fluxes, _ = _driver_fields(material)
    ng, nf = sum(_width, values(gradients)), sum(_width, values(fluxes))
    ncells = length(states.old)
    nqp = ncells == 0 ? 0 : length(first(states.old))
    n = ncells * nqp
    size(G) == (ng, n) && size(F) == (nf, n) && size(𝕂) == (nf, ng, n) || throw(
        DimensionMismatch(
            "update_points!: $ncells cells of $nqp points of $(nameof(typeof(material))), of gradients " *
                "$(keys(gradients)) and fluxes $(keys(fluxes)), need G of size ($ng, $n), F of size ($nf, $n) " *
                "and 𝕂 of size ($nf, $ng, $n); got $(size(G)), $(size(F)) and $(size(𝕂))"
        )
    )
    ntasks ≥ 1 || throw(ArgumentError("update_points!: ntasks must be positive; got $ntasks"))
    blocks = Iterators.partition(1:ncells, max(1, cld(ncells, ntasks)))
    # `@sync` waits for every task, so that none still writes into the arrays,
    # before the error of the first that failed is raised as it was raised.
    try
        @sync for cells in blocks
            Threads.@spawn _update_cells!(F, 𝕂, G, states, Δt, material, cells, nqp, algorithm, external, Val(ng))
        end
    catch err
        throw(_first_failure(err))
    end
    return F, 𝕂
end

function _update_cells!(F, 𝕂, G, states, Δt, material, cells, nqp, algorithm, external, ::Val{ng}) where {ng}
    for cell in cells, qp in 1:nqp
        p = (cell - 1) * nqp + qp
        g = SVector{ng}(ntuple(k -> G[k, p], Val(ng)))
        Fp, Kp, st = _mandel_update(material, g, states.old[cell][qp], Δt; algorithm, external)
        F[:, p] .= Fp
        𝕂[:, :, p] .= Kp
        states.new[cell][qp] = st
    end
    return nothing
end

# The error a failed task raised, unwrapped from the `CompositeException` of
# `@sync`: a `LocalIntegrationError` stays one, for the caller that cuts the
# step. Any other exception (an interrupt) is passed on as it came.
_first_failure(err::CompositeException) = first(err).task.exception
_first_failure(err) = err

# ── A quasi-static driver for a structure ────────────────────────────────────

"""
    GlobalNewtonError(time, iterations, residual)

Thrown by [`quasistatic!`](@ref) when the global Newton iteration did not
converge at `time`, even after the step was cut as far as allowed.
"""
struct GlobalNewtonError <: Exception
    time::Float64
    iterations::Int
    residual::Float64
end

Base.showerror(io::IO, e::GlobalNewtonError) = print(
    io, "GlobalNewtonError: the structure did not reach equilibrium at t = ", e.time,
    " (", e.iterations, " iterations, relative residual ", e.residual,
    "). Use smaller load steps, or raise max_cuts; a slow convergence often points to a tangent that is not consistent."
)

"""
    quasistatic!(system!, prescribe!, u, states, times; rtol = 1e-8, maxiters = 25,
                 max_cuts = 8, on_step = nothing) -> history

Drive a structure through the load steps `times`, by Newton's method on its
equilibrium with the consistent tangent of its materials, independently of the
finite-element library.

- `prescribe!(u, t)` sets the prescribed degrees of freedom of `u` at time `t`;
- `system!(u, t, Δt)` assembles, at the displacement `u`, the Jacobian `K` and
  the residual `r` (internal minus external forces) of the step ending at `t`,
  the rows of the prescribed degrees of freedom set to those of the identity
  and to zero; it reads the states from `states.old` and writes them into
  `states.new`, and returns `(K, r)`;
- `u` is updated in place, `states` a [`QuadratureStates`](@ref).

A step is converged when ``\\lVert r \\rVert \\le`` `rtol` times the norm of
its first residual. It is then committed; otherwise, or when a material fails
to integrate ([`LocalIntegrationError`](@ref)), the step is rolled back and
retried in two halves, down to `max_cuts` halvings. `on_step(i, t, u, states)`
is called after each time of `times`. Returns, for each step taken (cut steps
included), the time and the norms of its successive residuals, from which the
quadratic convergence of a consistent tangent can be read.
"""
function quasistatic!(
        system!, prescribe!, u::AbstractVector, states::QuadratureStates, times::AbstractVector{<:Real};
        rtol::Real = 1.0e-8, maxiters::Integer = 25, max_cuts::Integer = 8, on_step = nothing
    )
    opts = (; rtol, maxiters, max_cuts)
    history = Tuple{Float64, Vector{Float64}}[]
    for i in 2:length(times)
        _qs_advance!(system!, prescribe!, u, states, times[i - 1], times[i], opts, 0, history)
        on_step === nothing || on_step(i, times[i], u, states)
    end
    return history
end

function _qs_advance!(system!, prescribe!, u, states, t₀, t₁, opts, depth, history)
    u₀ = copy(u)
    try
        push!(history, (Float64(t₁), _qs_newton!(system!, prescribe!, u, states, t₁, t₁ - t₀, opts)))
        commit!(states)
    catch err
        (err isa LocalIntegrationError || err isa GlobalNewtonError) || rethrow()
        depth < opts.max_cuts || rethrow()
        copyto!(u, u₀)
        rollback!(states)
        tₘ = (t₀ + t₁) / 2
        _qs_advance!(system!, prescribe!, u, states, t₀, tₘ, opts, depth + 1, history)
        _qs_advance!(system!, prescribe!, u, states, tₘ, t₁, opts, depth + 1, history)
    end
    return nothing
end

function _qs_newton!(system!, prescribe!, u, states, t, Δt, opts)
    prescribe!(u, t)
    K, r = system!(u, t, Δt)
    norms = [norm(r)]
    while norms[end] > opts.rtol * norms[1]
        length(norms) > opts.maxiters && throw(GlobalNewtonError(Float64(t), length(norms) - 1, norms[end] / norms[1]))
        u .-= K \ r
        K, r = system!(u, t, Δt)
        push!(norms, norm(r))
    end
    return norms
end

# ── Functions an FE adapter provides ─────────────────────────────────────────

"""
    material_element!(Ke, re, cv, material, hypothesis, ue, states, cell, Δt;
                      algorithm = LocalNewton(), external = (;), cache = nothing)

Assemble the element stiffness `Ke` and the internal-force vector `re` of one
cell of a small-strain mechanical problem, whose material is any material of
the Gauss-point contract (a law of `MaterialLaws` or a homogenized material),
under the [`ModelingHypothesis`](@ref) `hypothesis`; the states of the cell's
quadrature points are read from `states.old[cell]` and written into
`states.new[cell]`.

Methods are added by the adapter of a finite-element library; with Ferrite,
`using Ferrite` provides one, `cv` being its `CellValues` reinitialized on the
cell and `ue` the element's displacements.
"""
function material_element! end

"""
    material_system(dh, ch, cv, material, hypothesis, states; external_forces! = nothing, …)
        -> (system!, prescribe!)

The two functions [`quasistatic!`](@ref) calls, for a finite-element library:
the assembly of the Jacobian and the residual over the whole mesh with
[`material_element!`](@ref), and the application of the prescribed
displacements. Methods are added by the adapter of the library; with Ferrite,
`dh` is the `DofHandler`, `ch` the `ConstraintHandler` and `cv` the
`CellValues`.
"""
function material_system end

"""
    write_material_vtk(path, dh, u, law, states)

Write to a VTK file the displacement `u` and, cell by cell, the average over the
quadrature points of the committed states of a law of `MaterialLaws`: its
fluxes (the stress, by component `σ11`, …), its gradients (`ε11`, …) and its
internal variables by name (`p`, `εᵖ11`, …). Methods are added by the adapter of
a finite-element library; with Ferrite, `dh` is the `DofHandler`.
"""
function write_material_vtk end
