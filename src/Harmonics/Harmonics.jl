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
- `vector_harmonics.jl`: the vector solid harmonics, the solutions of the Lamé
                          equation built on them and their translation
"""
module Harmonics

using SpecialFunctions: logfactorial, erfc
using StaticArrays: SVector
using TensND: primal

include("solid_harmonics.jl")
include("lattice_sums.jl")
include("vector_harmonics.jl")

export HarmonicTable, regular_harmonics, irregular_harmonics, harmonic_norm
export lattice_harmonics
export vector_regular_harmonic, vector_irregular_harmonic
export lame_regular, lame_irregular, lame_translation

end # module
