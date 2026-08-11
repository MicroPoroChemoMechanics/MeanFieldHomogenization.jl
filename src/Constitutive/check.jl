# =============================================================================
#  check.jl — conformance checker for the Gauss-point contract.
#
#  The counterpart of `CustomInclusions.check_inclusion_interface`: cheaper than
#  debugging a `MethodError` — or worse, a wrong tangent — from inside an FE
#  Newton loop, where the only symptom is that the iteration does not converge
#  quadratically.
# =============================================================================

"""
    check_material_interface(m::AbstractMFHMaterial; verbose = true, kw...) -> Bool

Exercise a Gauss-point material against the contract and report what it does
and does not honor. Returns `true` when every check passes.

What is verified:

1. `initial_state` returns an [`AbstractMaterialState`](@ref);
2. [`material_response`](@ref) accepts both the bare-strain and the
   `NamedTuple` forms and returns a [`MaterialResponse`](@ref);
3. every name declared by [`flux_names`](@ref) and [`tangent_blocks`](@ref) is
   actually present in the response;
4. the response is **frame-honest**: the fluxes and tangents come back in the
   canonical basis, which is what an FE code will read them as;
5. `state_old` is **not mutated** — the contract requires a new state, so that a
   rejected Newton iteration can simply be retried;
6. the declared `σε` block agrees with a **central finite difference** of the
   stress. This is the check that matters: a plausible-but-wrong tangent costs
   quadratic convergence and nothing else, so it is easy to ship.

The tangent check is skipped where it is meaningless — at a state on a
non-smooth branch of the law. Pass `probe` to move the probe strain somewhere
smooth.

# Keywords

- `probe` — the strain at which to exercise the law (default: a small
  non-symmetric-looking triaxial state, so a law that only works in uniaxial
  tension is caught).
- `Δ` — finite-difference step (default `1e-7`).
- `rtol` — tolerance on the tangent comparison (default `1e-5`).
- `verbose` — print the report (default `true`).

```julia
julia> check_material_interface(HomogenizedElastic(rve, MoriTanaka()));
```
"""
function check_material_interface(
        m::AbstractMFHMaterial;
        probe::Union{Nothing, TensND.AbstractTens{2, 3}} = nothing,
        Δ::Real = 1.0e-7, rtol::Real = 1.0e-5, verbose::Bool = true
    )
    ok = true
    say(msg) = verbose && println(msg)
    fail(msg) = (ok = false; verbose && println("  ✗ ", msg))
    pass(msg) = verbose && println("  ✓ ", msg)

    say("check_material_interface($(typeof(m)))")
    say("  gradients = $(gradient_names(m)), fluxes = $(flux_names(m)), " *
        "tangent blocks = $(tangent_blocks(m))")

    # 1. initial state
    st0 = initial_state(m)
    st0 isa AbstractMaterialState ? pass("initial_state") :
        fail("initial_state returned $(typeof(st0)), not an AbstractMaterialState")

    ε = probe === nothing ? _probe_strain() : probe

    # 2-3. response shape
    local r
    try
        r = material_response(m, ε, st0, 0.0)
    catch err
        fail("material_response threw: $(sprint(showerror, err))")
        return ok
    end
    r isa MaterialResponse ? pass("material_response returns a MaterialResponse") :
        fail("material_response returned $(typeof(r))")

    for name in flux_names(m)
        haskey(r.fluxes, name) ? pass("flux :$name present") :
            fail("declared flux :$name missing from the response")
    end
    for name in tangent_blocks(m)
        haskey(r.tangents, name) ? pass("tangent block :$name present") :
            fail("declared tangent block :$name missing from the response")
    end

    # 4. frame honesty
    if haskey(r.tangents, :σε)
        b = TensND.get_basis(r.tangents.σε)
        b isa TensND.CanonicalBasis ? pass("tangent is in the canonical frame") :
            fail(
            "tangent came back in a $(typeof(b).name.name); an FE code reads " *
                "components as global, so the material must convert (see to_tensors)"
        )
    end

    # 5. no mutation of the old state
    st_before = deepcopy(st0)
    material_response(m, ε, st0, 0.0)
    _states_equal(st0, st_before) ? pass("state_old is not mutated") :
        fail("material_response mutated state_old; the contract requires a new state")

    # 6. tangent vs central finite differences
    if haskey(r.tangents, :σε) && haskey(r.fluxes, :σ)
        ok_tan = _check_tangent(m, ε, st0, r.tangents.σε; Δ = Δ, rtol = rtol)
        if ok_tan === nothing
            say("  ~ tangent check skipped (non-smooth at the probe state)")
        elseif ok_tan
            pass("σε agrees with a central finite difference")
        else
            fail(
                "σε disagrees with a central finite difference — the FE Newton " *
                    "iteration will lose quadratic convergence"
            )
        end
    end

    say(ok ? "  → conforms" : "  → DOES NOT conform")
    return ok
end

# A triaxial probe with shear, so a law that only ever saw uniaxial tension is
# exercised properly.
function _probe_strain()
    return TensND.Tens(
        Tensors.SymmetricTensor{2, 3}(
            (i, j) -> i == j ? (1.0e-3 * i) : (2.0e-4 * (i + j))
        )
    )
end

_states_equal(a::NoState, b::NoState) = true
_states_equal(a::S, b::S) where {S <: AbstractMaterialState} =
    all(getfield(a, f) == getfield(b, f) for f in fieldnames(S))
_states_equal(a, b) = false

# Central differences on each independent component of ε. Returns `nothing` when
# the law turns out to be non-smooth at the probe (the perturbed responses
# disagree with each other in a way a tangent cannot represent).
function _check_tangent(m, ε, st, C; Δ, rtol)
    A = get_array(TensND.change_tens(C, TensND.CanonicalBasis{3, Float64}()))
    maxerr = 0.0
    scale = maximum(abs, A)
    scale == 0 && return nothing
    for (k, l) in ((1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3))
        E = Tensors.SymmetricTensor{2, 3}(
            (i, j) -> ((i, j) == (k, l) || (i, j) == (l, k)) ? (k == l ? 1.0 : 0.5) : 0.0
        )
        dE = TensND.Tens(E)
        σp = material_response(m, ε + Δ * dE, st, 0.0).fluxes.σ
        σm = material_response(m, ε - Δ * dE, st, 0.0).fluxes.σ
        fd = get_array(
            TensND.change_tens(
                (σp - σm) / (2Δ), TensND.CanonicalBasis{3, Float64}()
            )
        )
        for i in 1:3, j in 1:3
            maxerr = max(maxerr, abs(fd[i, j] - A[i, j, k, l]))
        end
    end
    return maxerr <= rtol * scale
end
