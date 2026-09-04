# =============================================================================
#  voigt.jl — Voigt (uniform-strain) upper bound.
#
#       C_Voigt = Σ_i f_i  C_i
#
#  Cracks (`CrackDensity`) are ignored: their volume contribution → 0
#  in the penny limit. The matrix volume fraction is the implicit
#  complement `1 - Σ f_inc`.
# =============================================================================

"""
    _evaluate(rve, ::Voigt, ::Val{p}; kw...) -> AbstractTens

Voigt upper bound on the effective property `:p`:
``\\langle\\mathbb C\\rangle = \\sum_i f_i \\mathbb C_i``.

Phases carrying a [`CrackDensity`](@ref) instead of a
[`VolumeFraction`](@ref) are ignored (their volume contribution is
zero); use a Hill-tensor-aware scheme (e.g. [`Dilute`](@ref) or
[`MoriTanaka`](@ref)) to capture crack effects.

Reference: [hill1965](@cite).
"""
function _evaluate(rve::RVE, ::Voigt, ::Val{p}; kw...) where {p}
    names = _bound_phase_names(rve, "Voigt")
    ref = phase_property(rve, first(names), p)
    Ceff = zero(ref)
    for name in names
        Ceff += volume_fraction(rve, name) * _phase_voigt_property(rve, name, p, ref)
    end
    return Ceff
end

# The bounds sum over every phase that carries volume, with no distinguished
# one — a bound is an average, and averages need no reference medium.
#
# `ref` reaches `_layer_voigt` / `_layer_reuss` only to select the tensor order
# (4 for elasticity, 2 for transport); it is never used numerically. Seeding it
# from the first volume-carrying phase is therefore exact, where the old code
# happened to seed the accumulator with the matrix term instead.
function _bound_phase_names(rve::RVE, bound::AbstractString)
    names = Symbol[n for n in rve.phase_names if !(rve.amounts[n] isa CrackDensity)]
    isempty(names) && throw(
        ArgumentError(
            "the $bound bound needs at least one phase carrying volume; this RVE " *
                "holds only crack densities"
        )
    )
    return names
end
