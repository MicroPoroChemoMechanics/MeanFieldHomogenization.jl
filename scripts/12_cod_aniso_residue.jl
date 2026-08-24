# =============================================================================
#  scripts/12_cod_aniso_residue.jl
#
#  Demonstrate the residue and DECUHR backends on a generic anisotropic
#  stiffness (triclinic-looking components).  Prints the 𝐁 tensor for a
#  penny, an ellipse (η=0.3) and a ribbon, for both algorithms.
# =============================================================================

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "docs"); io = devnull)

using MeanFieldHomogenization
using TensND
using LinearAlgebra
using Printf

Kcomponents = [
    0.388487  0.200301  0.13255   -0.0803777 -0.249878   0.038079
    0.200301  1.09373   0.178878  -0.369538  -0.161806   0.0734051
    0.13255   0.178878  0.387019  -0.210259  -0.249375   0.0735958
    -0.0803777 -0.369538 -0.210259  0.655779   0.123902  -0.227447
    -0.249878 -0.161806 -0.249375  0.123902   0.442613  -0.120333
    0.038079  0.0734051 0.0735958 -0.227447 -0.120333   0.448281
]
C_aniso = inv_KM(Kcomponents, CanonicalBasis{3, Float64}())

function showB(label, B)
    @printf "\n%-40s\n" label
    for i in 1:3
        @printf "  "
        for j in 1:3
            @printf "%12.6g " B[i, j]
        end
        @printf "\n"
    end
    return
end

pc = PennyCrack(1.0)
showB("Penny  — residue", cod_tensor(pc, C_aniso; method = :residues))
showB("Penny  — decuhr", cod_tensor(pc, C_aniso; method = :decuhr))

ec = EllipticCrack(1.0, 0.3)
showB("η=0.3  — residue", cod_tensor(ec, C_aniso; method = :residues))
showB("η=0.3  — decuhr", cod_tensor(ec, C_aniso; method = :decuhr))

r = RibbonCrack(1.0)
showB("Ribbon — residue", cod_tensor(r, C_aniso; method = :residues))
showB("Ribbon — decuhr", cod_tensor(r, C_aniso; method = :decuhr))
