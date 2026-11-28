# =============================================================================
#  density.jl — the mass density of a cell.
#
#  The density is additive: the mass of a volume is the sum of the masses of
#  its phases, so that ρ̄ = Σᵢ fᵢ ρᵢ over the phases that carry volume. A crack
#  carries none (its volume vanishes at fixed density of cracks), and adds
#  nothing. This is the Voigt average of the phase property `:ρ`, which the
#  `Voigt` scheme already computes for any property.
# =============================================================================

"""
    average_density(rve::RVE) → ρ̄

The mass density of the cell, ``\\bar\\rho = \\sum_i f_i\\,\\rho_i``, from the
property `:ρ` of each phase that carries volume; a phase declared by a crack
density adds nothing. It is `homogenize(rve, Voigt(), :ρ)`, exact for a
density whatever the microstructure. Numbers, dual numbers and symbols.
"""
average_density(rve::RVE) = homogenize(rve, Voigt(), :ρ)
