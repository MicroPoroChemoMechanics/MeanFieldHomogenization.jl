"""
    MeanFieldHomogenization.Harmonics

Harmonic functions in which the field around an inclusion is expanded, and the
formulas that carry an expansion from one center to another.

Every one-site scheme needs only the response of an inclusion to a uniform
remote field. A scheme that resolves the field around each inclusion of an
assembly, [`MultipoleExpansion`](@ref MeanFieldHomogenization.Schemes.MultipoleExpansion),
needs the whole family of fields an inclusion can receive and send back: the
regular and the decaying solid harmonics of every degree.

Contents
--------
- `solid_harmonics.jl` : `regular_harmonics`, `irregular_harmonics`, the table
                          that stores them and the normalization of
                          [kushch2013](@citet)
- `lattice_sums.jl`    : `lattice_harmonics`, the harmonics of a periodic array
                          of sources, by Ewald sums
"""
module Harmonics

using SpecialFunctions: logfactorial, erfc
using TensND: primal

include("solid_harmonics.jl")
include("lattice_sums.jl")

export HarmonicTable, regular_harmonics, irregular_harmonics, harmonic_norm
export lattice_harmonics

end # module
