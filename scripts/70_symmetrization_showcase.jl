# # Symmetrization showcase: exact average vs best-fit projection
#
# The didactic centerpiece of the symmetrization overhaul: the difference
# between the two mechanisms `echoes` (and now `MeanFieldHomogenization`) provide, on a
# **non-major-symmetric** strain-concentration tensor.
#
# - **(B) exact rotation-group average** — `transverse_isotropify` (re-exported from `TensND`) /
#   `IsoSymmetrize` / `TISymmetrize` inside scheme kernels. Preserves the
#   full axially-invariant content (`TensTI{4,T,8}`): ``\ell_3 \neq \ell_4``
#   and the antisymmetric azimuthal couplings ``\ell_7``, ``\ell_8``.
# - **(A) best-fit projection** — `best_fit_ti` (likewise from `TensND`). Forces major
#   symmetry (`TensTI{4,T,5}`), the `echoes` `.paramsym(sym=TI)` reporting
#   form.
#
# A concentration tensor
# ```math
# \mathbb{A} = \big[\mathbb{I} + \mathbb{P}:(\mathbb{C}_1-\mathbb{C}_0)\big]^{-1}
# ```
# generally lacks major symmetry; using (A) where (B) is required silently
# discards physical couplings. This script quantifies the gap.

import Pkg                                                          #jl
Pkg.activate(joinpath(@__DIR__, "..", "docs"); io = devnull)                 #jl

using MeanFieldHomogenization
using TensND
using Printf
using LinearAlgebra

# ## A genuinely non-major-symmetric tensor
#
# The dilute strain concentration of a stiff prolate spheroid in a softer
# isotropic matrix:

C₀ = TensISO{3}(3 * 20.0, 2 * 12.0)          # matrix
C₁ = TensISO{3}(3 * 90.0, 2 * 60.0)          # stiff inclusion
incl = Spheroid(6.0; euler_angles = (deg2rad(35.0), 0.0, 0.0))   # tilted needle
A = strain_strain_loc(incl, C₁, C₀)
arrA = get_array(A)

ismaj = maximum(abs, arrA .- permutedims(arrA, (3, 4, 1, 2)))
@printf "‖A − Aᵀ_major‖∞ = %.4e  (A is NOT major-symmetric)\n\n" ismaj

n = (0.0, 0.0, 1.0)

# ## (B) Exact azimuthal average about ``e_z``

A_exact = transverse_isotropify(A, n)         # TensTI{4,T,8}
ℓ = get_ℓ8(A_exact)
@printf "(B) exact TI average → TensTI{4,T,8}\n"
@printf "    ℓ₁..ℓ₈ = %s\n" join((@sprintf("%.4f", x) for x in ℓ), ", ")
@printf "    ℓ₃ − ℓ₄ (major-asymmetry) = %.4e\n" (ℓ[3] - ℓ[4])
@printf "    ℓ₇, ℓ₈  (antisym. couplings) = %.4e, %.4e\n\n" ℓ[7] ℓ[8]

# ## (A) Best-fit major-symmetric projection

A_fit = best_fit_ti(A, n)                     # TensTI{4,T,5}
@printf "(A) best-fit TI projection → TensTI{4,T,5}\n"
@printf "    (ℓ₁, ℓ₂, ℓ₃, ℓ₅, ℓ₆) = %s\n\n" join((@sprintf("%.4f", x) for x in get_data(A_fit)), ", ")

# ## The gap: what best-fit discards

gap = maximum(abs, get_array(A_exact) .- get_array(A_fit))
@printf "‖exact − best-fit‖∞ = %.4e  (content dropped by the projection)\n" gap
@printf "  → components lost : ℓ₃≠ℓ₄ split + the ℓ₇/ℓ₈ azimuthal couplings.\n\n"

# ## Consequence at scheme level: Mori–Tanaka homogenized stiffness
#
# Two RVEs identical except for the symmetrization mechanism applied to the
# tilted-needle phase. The exact average is what the scheme kernels use.

function homogenise_with(symmode)
    rve = RVE(:M)
    add_matrix!(rve, Ellipsoid(1.0), Dict(:C => C₀))
    add_phase!(
        rve, :I, incl, Dict(:C => C₁);
        fraction = 0.2, symmetrize = symmode
    )
    return homogenize(rve, MoriTanaka(), :C)
end

C_iso = homogenise_with(:iso)                       # full SO(3) average
C_ti = homogenise_with(TISymmetrize(n))             # exact azimuthal average
@printf "MT homogenised C₃₃₃₃ :  iso-average = %.4f   TI(ez)-average = %.4f\n" get_array(C_iso)[3, 3, 3, 3] get_array(C_ti)[3, 3, 3, 3]
println("\nDone.")
