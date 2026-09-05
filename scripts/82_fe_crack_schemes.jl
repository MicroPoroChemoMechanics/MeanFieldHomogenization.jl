# # A finite-element crack inside the homogenization schemes
#
# Script 81 checked the finite-element crack-opening-displacement tensor against
# its closed form. This one makes the point that matters for the contract: once
# `cod_tensor` exists, **nothing downstream knows or cares** how it was
# obtained. The finite-element crack is a drop-in replacement for
# `EllipticCrack` in every scheme, orientation averaging included.
#
# Requires `Ferrite`, `FerriteGmsh` and `Gmsh`.

import Pkg                                                          #jl
Pkg.activate(joinpath(@__DIR__, "fe"); io = devnull)                 #jl
Pkg.instantiate(; io = devnull)                                      #jl

using MeanFieldHomogenization
using TensND
using LinearAlgebra
using Printf

import Ferrite, FerriteGmsh, Gmsh                                    #jl

const νm = 0.3
const Em = 1.0
const C₀ = iso_stiffness(Em / (3 * (1 - 2νm)), Em / (2 * (1 + νm)))

c1111(C) = get_array(C)[1, 1, 1, 1]
c3333(C) = get_array(C)[3, 3, 3, 3]

# ## What the crack has to implement
#
# Exactly one method. `FEEllipticCrack` subtypes `AbstractCrack` and declares
# the standard `shape_trait`, so ℍ, ℕ, 𝐑, 𝐍_K, the bundled pair and the four
# `delta_*` — with the Budiansky ``4\pi/3`` prefactor — are all inherited.

fe = FEEllipticCrack(1.0, 0.25; htipdiv = 9.0)
an = EllipticCrack(1.0, 0.25)

println("="^78)
check_inclusion_interface(fe; amount = :density)
@printf "\ncrack_density_factor = %.6f   (4π/3 = %.6f)\n" MeanFieldHomogenization.Cracks.crack_density_factor(fe) 4π / 3

# ## Parallel cracks: the effective stiffness against crack density

function cracked(geom, ε; kw...)
    r = RVE(; distribution_shape = Ellipsoid(1.0))
    add_phase!(r, :M, Ellipsoid(1.0), Dict(:C => C₀); fraction = :rest)
    add_phase!(r, :cr, geom, Dict(:C => C₀); density = ε, kw...)
    return r
end

# Differences are reported relative to the **matrix** modulus rather than to
# the effective one: the dilute estimate of `C₃₃₃₃` crosses zero around
# ε ≈ 0.11 (and goes negative beyond), so a relative-to-itself error would blow
# up there for reasons that have nothing to do with the finite elements.

const C33_M = c3333(C₀)

println("\n", "="^78)
println("Parallel cracks (normal along e₃) — C₃₃₃₃,  Δ in % of the matrix modulus")
println("-"^78)
@printf "%-8s %14s %14s %10s   %14s %14s %10s\n" "ε" "Dilute FE" "Dilute ana" "Δ/C₀" "MT FE" "MT ana" "Δ/C₀"
for ε in (0.0, 0.02, 0.05, 0.1, 0.2)
    dfe = c3333(homogenize(cracked(fe, ε), Dilute(), :C))
    dan = c3333(homogenize(cracked(an, ε), Dilute(), :C))
    mfe = c3333(homogenize(cracked(fe, ε), MoriTanaka(), :C))
    man = c3333(homogenize(cracked(an, ε), MoriTanaka(), :C))
    @printf "%-8.2f %14.6f %14.6f %10.2f   %14.6f %14.6f %10.2f\n" ε dfe dan 100(dfe - dan) / C33_M mfe man 100(mfe - man) / C33_M
end

# The two columns track each other to a couple of percent of the matrix
# modulus — the residual is the finite-element discretization measured in
# script 81, propagated through the scheme, not a discrepancy in how the crack
# is coupled. (The dilute scheme itself loses meaning well before ε = 0.2,
# where it predicts a negative stiffness; that is a property of the scheme, not
# of the inclusion.)

# ## The one-shot schemes, at one density
#
# These evaluate the crack once, in the *matrix* — which is isotropic here.

println("\n", "="^78)
@printf "One-shot schemes at ε = 0.05 — C₃₃₃₃\n"
println("-"^78)
@printf "%-24s %14s %14s %10s\n" "scheme" "FE crack" "analytical" "Δ (%)"
for (name, sch) in (
        "Dilute" => Dilute(), "DiluteDual" => DiluteDual(),
        "MoriTanaka" => MoriTanaka(), "Maxwell" => Maxwell(),
        "PonteCastanedaWillis" => PonteCastanedaWillis(),
    )
    vfe = c3333(homogenize(cracked(fe, 0.05), sch, :C))
    van = c3333(homogenize(cracked(an, 0.05), sch, :C))
    @printf "%-24s %14.6f %14.6f %10.2f\n" name vfe van 100(vfe - van) / van
end

# ## Iterative schemes need an isotropic reference medium
#
# `SelfConsistent` and `DifferentialScheme` re-evaluate the crack in the
# *current estimate*, which for a family of **parallel** cracks is transversely
# isotropic. The corrected boundary condition uses the closed-form Kelvin
# dipole, so an anisotropic reference is out of scope — and the error says so
# rather than silently returning the isotropic projection:

try
    homogenize(cracked(fe, 0.05), SelfConsistent(), :C)
catch e
    println("\n", "="^78)
    println("SelfConsistent on *parallel* FE cracks:")
    println("  ", first(split(sprint(showerror, e), '\n')))
end

# With `IsoSymmetrize` the scheme hands the kernel an isotropic reference at
# every iteration, and the iterative schemes work:

println("\n", "="^78)
@printf "Iterative schemes at ε = 0.05, IsoSymmetrize — C₁₁₁₁\n"
println("-"^78)
@printf "%-24s %14s %14s %10s\n" "scheme" "FE crack" "analytical" "Δ (%)"
for (name, sch) in ("SelfConsistent" => SelfConsistent(), "Differential" => DifferentialScheme())
    vfe = c1111(homogenize(cracked(fe, 0.05; symmetrize = IsoSymmetrize()), sch, :C))
    van = c1111(homogenize(cracked(an, 0.05; symmetrize = IsoSymmetrize()), sch, :C))
    @printf "%-24s %14.6f %14.6f %10.2f\n" name vfe van 100(vfe - van) / van
end

# ## Randomly oriented cracks
#
# Orientation averaging is applied by the scheme, on the tensor `cod_tensor`
# returned — so it costs the finite-element crack nothing. And because the
# solve is memoized on the reference medium *expressed in the crack's own
# frame*, a whole family of orientations in an isotropic matrix shares **one**
# finite-element resolution.

fe_reset!(fe)
iso = homogenize(cracked(fe, 0.05; symmetrize = IsoSymmetrize()), MoriTanaka(), :C)
iso_an = homogenize(cracked(an, 0.05; symmetrize = IsoSymmetrize()), MoriTanaka(), :C)

println("\n", "="^78)
println("Isotropic orientation average, ε = 0.05")
println("-"^78)
println("  FE crack   : ", typeof(iso).name.name, "   C₁₁₁₁ = ", round(c1111(iso), digits = 6))
println("  analytical : ", typeof(iso_an).name.name, "   C₁₁₁₁ = ", round(c1111(iso_an), digits = 6))
@printf "  relative difference : %.2f %%\n" 100(c1111(iso) - c1111(iso_an)) / c1111(iso_an)
@printf "  finite-element assemblies performed : %d\n" fe_assembly_count(fe)

# One assembly for the whole isotropic average: the exact rotation-group
# average never changes the reference medium seen by the kernel, so the cache
# is hit every time.

# ## The cost of an iterative scheme
#
# A self-consistent iteration *does* change the reference medium at every step,
# so each distinct `C₀` costs one assembly plus six solves. This is the price
# of a finite-element inclusion, and the reason the one-shot schemes are the
# natural home for it.

fe_reset!(fe)
homogenize(cracked(fe, 0.05), MoriTanaka(), :C)
n_mt = fe_assembly_count(fe)
fe_reset!(fe)
homogenize(cracked(fe, 0.05; symmetrize = IsoSymmetrize()), SelfConsistent(), :C)
n_sc = fe_assembly_count(fe)

println("\n", "="^78)
@printf "assemblies — Mori-Tanaka: %d    self-consistent: %d\n" n_mt n_sc
