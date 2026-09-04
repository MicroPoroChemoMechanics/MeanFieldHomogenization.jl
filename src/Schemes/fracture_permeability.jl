# =============================================================================
#  fracture_permeability.jl — effective conductivity of a fracture network.
#
#  WHY NOT THE ORDINARY SELF-CONSISTENT SCHEME.  `SelfConsistent` on `:K` is
#  built for INSULATING cracks: its crack branch adds resistivity contributions
#  and assembles `𝔹 : 𝔸⁻¹`, where the volumetric accumulator `Σ f_α K_α A_α`
#  carries the flux. A flowing crack has no volume — its whole effect is the
#  `0 × ∞` product that accumulator drops — so with an impermeable matrix the
#  body collapses to `K = 0` identically, for any input. The estimate below is
#  therefore written out rather than routed through that machinery.
#
#  THE ESTIMATE.  Each family contributes its conductive limit evaluated in the
#  effective medium itself:
#
#      𝐊 = k_s 𝟏 + Σ_i (4π/3) d_i 𝕂_i(𝐊) ,
#
#  a self-consistent condition solved by the same fixed-point driver as every
#  other iterative scheme in the package. Reading `𝕂_i` in the effective medium
#  rather than in the matrix is what accounts for the fractures seeing one
#  another, and is what produces a percolation threshold: an isolated fracture
#  network conducts nothing until it connects.
# =============================================================================

"""
    fracture_permeability(k_matrix, cracks, densities; kw...) -> Tens{2,3}

Effective conductivity (permeability, diffusivity) of a solid of conductivity
`k_matrix` crossed by families of **flowing** fractures.

- `k_matrix` — the matrix conductivity, a scalar or a `Tens{2,3}`. For a
  fractured rock it is small but must be **non-zero**: with an exactly
  impermeable matrix the medium conducts nothing below the percolation
  threshold, and the estimate degenerates.
- `cracks` — the [`ConductiveCrack`](@ref MeanFieldHomogenization.Cracks.ConductiveCrack) families, each carrying its own
  normal and fracture conductivity.
- `densities` — the Budiansky crack density ``d_i`` of each family.

Solves the self-consistent condition

```math
\\boldsymbol{K}^{\\rm hom} = k_s\\,\\boldsymbol{1}
  + \\sum_i \\frac{4\\pi}{3}\\,d_i\\,\\boldsymbol{k}_i(\\boldsymbol{K}^{\\rm hom}) ,
```

each family being read **in the effective medium**, which is what lets the
fractures see one another and produces a percolation threshold.

# Keywords

`abstol`, `reltol`, `maxiters`, `damping`, `verbose` are passed to the
fixed-point driver shared with [`SelfConsistent`](@ref). Strong contrasts
usually need `damping < 1`.

```julia
cracks = (ConductiveCrack(1.0; conductivity = 6.7e-11, euler_angles = (π/2, 0.0)),
          ConductiveCrack(1.0; conductivity = 6.7e-11, euler_angles = (π/2, π/2)))
K = fracture_permeability(1.0e-18, cracks, (0.37, 0.37))
```

!!! note "Orientation lives in the crack, density in the argument"
    Same split as an [`RVE`](@ref): the family's normal is part of its geometry,
    its abundance is not.
"""
function fracture_permeability(
        k_matrix, cracks, densities;
        abstol = 0.0, reltol = 1.0e-8, maxiters = 500,
        damping = 1.0, verbose = false, kw...
    )
    length(cracks) == length(densities) || throw(
        ArgumentError(
            "$(length(cracks)) crack families but $(length(densities)) densities"
        )
    )
    K_s = _as_conductivity_tensor(k_matrix)
    # A plain relaxed Picard iteration rather than `_solve_sc`. That driver is
    # built for the `𝔹 : 𝔸⁻¹` scheme bodies — its damping and its behavior on
    # `maxiters` assume that structure — and the additive condition here is
    # simple enough that owning the loop is clearer than bending it to fit.
    #
    # The convergence test is RELATIVE, and `abstol` DEFAULTS TO ZERO on
    # purpose: conductivities span many decades — 10⁻¹⁸ m² for a tight rock —
    # so any fixed absolute floor is met by the very first iterate and the loop
    # returns the DILUTE estimate, silently, with no percolation threshold and
    # no dependence on the fracture conductivity at all. Pass `abstol` only with
    # a value scaled to the problem at hand.
    K = K_s
    converged = false
    for it in 1:maxiters
        K_new = _fracture_permeability_step(K_s, cracks, densities, K; kw...)
        Δ = sqrt(_frob_sq(K_new - K))
        scale = sqrt(_frob_sq(K_new))
        K = damping == 1 ? K_new : K + damping * (K_new - K)
        verbose && @info "fracture_permeability iter $it : ‖Δ‖/‖K‖ = $(Δ / scale)"
        if Δ <= abstol + reltol * scale
            converged = true
            break
        end
    end
    converged || @warn "fracture_permeability did not converge in $maxiters " *
        "iterations; a dense, strongly conducting network may be past its " *
        "percolation threshold, where the self-consistent estimate diverges."
    return K
end

function _fracture_permeability_step(K_s, cracks, densities, K_n; kw...)
    acc = K_s
    for (c, d) in zip(cracks, densities)
        𝕂 = MFH_Core.conductivity_contribution(c, K_n; kw...)
        acc = acc + MFH_Core.delta_conductivity(c, 𝕂, d)
    end
    return _major_symmetrize(acc)
end

_as_conductivity_tensor(k::Number) = TensND.TensISO{3}(k)
_as_conductivity_tensor(K::TensND.AbstractTens{2, 3}) = K

"""
    fracture_permeability(rve::RVE; solid = nothing, property = :K, kw...) -> Tens{2,3}

[`fracture_permeability`](@ref) reading the solid conductivity, the
[`ConductiveCrack`](@ref MeanFieldHomogenization.Cracks.ConductiveCrack) families and their densities straight off an
[`RVE`](@ref).

Phases that are not conductive cracks are ignored — a fracture network model has
no use for them, and silently folding them in would be worse than saying so.

`solid` names the phase that plays the intact skeleton; left unset, it is the
phase absorbing the volume complement.
"""
function fracture_permeability(
        rve::RVE; solid::Union{Nothing, Symbol} = nothing, property::Symbol = :K, kw...
    )
    s = host_phase_name(rve, solid, "fracture_permeability")
    K_s = phase_property(rve, s, property)
    cracks, densities = Any[], Any[]
    for name in inclusion_phase_names(rve, s)
        geom = rve.phases[name].geometry
        geom isa Cracks.ConductiveCrack || continue
        push!(cracks, geom)
        push!(densities, crack_density(rve, name))
    end
    isempty(cracks) && throw(
        ArgumentError("the RVE holds no ConductiveCrack phase")
    )
    return fracture_permeability(K_s, cracks, densities; kw...)
end
