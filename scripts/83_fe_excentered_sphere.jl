# # A sphere with an off-center core, by axisymmetric Fourier finite elements
#
# The morphology of Adessina, Barthélémy, Lavergne & Ben Fraj (2017): a
# recycled-concrete aggregate — an old natural aggregate wrapped in a shell of
# adhered old mortar, embedded in fresh cement paste. The old aggregate sits
# **off center**, so the pattern is not the concentric composite sphere and has
# no closed form.
#
# The paper solves it in three dimensions. It is a solid of revolution, so here
# the fields are expanded in Fourier series in the azimuth ``\theta`` and each
# mode is solved on the **meridian half-plane** ``\rho \ge 0``:
#
# ```math
# u_\rho = \bar u_\rho(\rho,z)\cos m\theta, \quad
# u_\theta = \bar u_\theta(\rho,z)\sin m\theta, \quad
# u_z = \bar u_z(\rho,z)\cos m\theta .
# ```
#
# A transversely isotropic material does not couple the ``\cos`` and ``\sin``
# groups of the strain, so ``\int_0^{2\pi}`` leaves one constant per mode and
# the modes are independent. Four macroscopic loadings — two in mode 0, one in
# mode 1, one in mode 2 — span the whole transversely isotropic localization
# tensor.
#
# On top of that sits the same **first-order corrected boundary condition** as
# the finite-element crack, here in its general form: the polarization of the
# infinite-medium problem is its own consequence, so
#
# ```math
# \mathbb X = \bigl[\mathbb I - (\mathbb B^p - \mathbb C_0 : \mathbb A^p)\bigr]^{-1}
#             : (\mathbb B^E - \mathbb C_0 : \mathbb A^E), \qquad
# \mathbb A = \mathbb A^E + \mathbb A^p : \mathbb X .
# ```
#
# Requires a finite-element backend: `Ferrite`, `FerriteGmsh` and `Gmsh` as
# below, or `Gridap` and `GridapGmsh` with `backend = GridapBackend()` passed to
# the constructor. The two agree to round-off; Ferrite is the faster to run,
# Gridap states the weak form directly and is the easier to adapt.

import Pkg                                                          #jl
Pkg.activate(joinpath(@__DIR__, "fe"); io = devnull)                 #jl
Pkg.instantiate(; io = devnull)                                      #jl

using MeanFieldHomogenization
using TensND
using LinearAlgebra
using Printf

import Ferrite, FerriteGmsh, Gmsh                                    #jl

# Moduli in GPa: a stiff old aggregate, a soft adhered mortar, a fresh paste in
# between.
ce(E, ν) = iso_stiffness(E / (3 * (1 - 2ν)), E / (2 * (1 + ν)))

const C_AGG = ce(70.0, 0.2)          # old natural aggregate
const C_MORTAR = ce(8.0, 0.2)        # adhered old mortar
const C_PASTE = ce(20.0, 0.2)        # fresh cement paste, the reference medium
const W = 0.5                        # old aggregate / whole inclusion

mand(T) = MeanFieldHomogenization.Core.mandel66_minor(MeanFieldHomogenization.Core._C_array(T))
kelvinJ(M) = (M[1, 1] + 2M[1, 2]) / 3
kelvinK(M) = M[4, 4]
iso(C) = MeanFieldHomogenization.Core.isotropify(C)

# ## §1 The mesh
#
# Two-dimensional and therefore cheap: refining costs almost nothing compared
# with the three-dimensional cell the same problem would need.

incl0 = FEExcenteredSphere(
    1.0, (C_AGG, C_MORTAR); core_fraction = W, eccentricity = 0.0,
    nradial = 24, radius_ratio = 4.0
)
rep = fe_axi_mesh_report(incl0)
@printf(
    "mesh: %d triangles (%d core, %d shell, %d matrix), %d nodes\n",
    rep.ncells, rep.ncells_core, rep.ncells_shell, rep.ncells_matrix, rep.nnodes
)
@printf(
    "revolution volumes: core %.4f (exact %.4f), shell %.4f (exact %.4f)\n",
    rep.volume_core, rep.volume_core_exact, rep.volume_shell, rep.volume_shell_exact
)

# ## §2 The concentric limit against Hervé-Zaoui
#
# At zero eccentricity the pattern *is* the two-layer sphere, whose localization
# tensors `LayeredSphere` gives exactly. Everything has to be simultaneously
# right for this to match — the three modes, the axis conditions, the azimuthal
# projections, the dipole correction — and the answer must come out isotropic
# although it is assembled from three separate discrete problems.

A_fe, B_fe = fe_axi_localization(incl0, C_PASTE)
sph = LayeredSphere((cbrt(W), 1.0), (C_AGG, C_MORTAR))
A_ex = strain_strain_loc(sph, C_PASTE, C_PASTE)
B_ex = stress_strain_loc(sph, C_PASTE, C_PASTE)

println("\n  tensor   part    finite elements       exact        Δ")
for (nm, num, ex) in (("A", A_fe, A_ex), ("B", B_fe, B_ex))
    N, X = mand(num), mand(ex)
    for (pt, f) in (("𝕁", kelvinJ), ("𝕂", kelvinK))
        @printf(
            "     %s      %s      %12.6f  %12.6f   %+7.3f %%\n",
            nm, pt, f(N), f(X), 100 * (f(N) - f(X)) / f(X)
        )
    end
end
N = mand(A_fe)
@printf(
    "\n  isotropy of the assembled tensor: |A₁₁-A₃₃|/A₁₁ = %.2e, |A₄₄-A₆₆|/A₄₄ = %.2e\n",
    abs(N[1, 1] - N[3, 3]) / N[1, 1], abs(N[4, 4] - N[6, 6]) / N[4, 4]
)

# ## §3 What the boundary correction buys
#
# The truncated cell is biased by ``O((a/R)^3)``. Correcting it makes the
# answer flat in `R`: it is already at the discretization floor at `R = 1.5a`,
# where the uncorrected cell is off by 1 % — and by 14 % when the adhered
# mortar is badly degraded (see the manual page).

Jex = kelvinJ(mand(A_ex))
println("\n   R/a    u = E·x        corrected")
for R in (1.5, 2.0, 3.0, 4.0, 6.0, 10.0)
    incl = FEExcenteredSphere(
        1.0, (C_AGG, C_MORTAR); core_fraction = W, eccentricity = 0.0,
        nradial = 20, radius_ratio = R
    )
    r = fe_axi_breakdown(incl, C_PASTE)
    eu = 100 * (kelvinJ(mand(r.A_uncorrected)) - Jex) / Jex
    ec = 100 * (kelvinJ(mand(r.A)) - Jex) / Jex
    @printf("  %5.1f   %+8.3f %%     %+8.3f %%\n", R, eu, ec)
end

# ## §4 Moving the core off center
#
# The response loses its isotropy and becomes transversely isotropic about the
# eccentricity axis. `A₁₁` is the transverse localization, `A₃₃` the axial one.
# With this mortar (`E = 8 GPa`, only 2.5 times softer than the paste) the
# induced anisotropy stays under a hundredth of a percent; the effect on the
# *level* of localization, and hence on the effective moduli in §5, is an order
# of magnitude larger.

println("\n     α       A₁₁        A₃₃      A₃₃/A₁₁ − 1")
for α in (0.0, 0.2, 0.4, 0.6, 0.8)
    incl = FEExcenteredSphere(
        1.0, (C_AGG, C_MORTAR); core_fraction = W, eccentricity = α,
        nradial = 20, radius_ratio = 4.0
    )
    M = mand(fe_axi_localization(incl, C_PASTE)[1])
    @printf(
        "  %5.2f  %9.5f  %9.5f   %+8.4f %%\n",
        α, M[1, 1], M[3, 3], 100 * (M[3, 3] / M[1, 1] - 1)
    )
end

# ## §5 In a scheme
#
# The inclusion is heterogeneous, so it enters the schemes through *both*
# localization tensors, and every scheme that consumes them works. The
# concentric case must reproduce the composite sphere; the eccentric one is
# what the exercise is for.

println("\n  Mori-Tanaka, aggregate volume fraction 0.4:")
println("     α          k_eff        μ_eff        E_eff")
for α in (0.0, 0.4, 0.8)
    incl = FEExcenteredSphere(
        1.0, (C_AGG, C_MORTAR); core_fraction = W, eccentricity = α,
        nradial = 20, radius_ratio = 4.0
    )
    rve = RVE()
    add_phase!(rve, :paste, Ellipsoid(1.0), Dict(:C => C_PASTE); fraction = :rest)
    add_phase!(rve, :rca, incl, Dict(:C => C_PASTE); fraction = 0.4)
    k, μ = k_mu(iso(homogenize(rve, MoriTanaka(), :C)))
    @printf("  %5.2f   %10.4f   %10.4f   %10.4f\n", α, k, μ, 9k * μ / (3k + μ))
end

rve = RVE()
add_phase!(rve, :paste, Ellipsoid(1.0), Dict(:C => C_PASTE); fraction = :rest)
add_phase!(rve, :rca, sph, Dict(:C => C_PASTE); fraction = 0.4)
kx, μx = k_mu(homogenize(rve, MoriTanaka(), :C))
@printf(
    "  exact (Hervé-Zaoui, α = 0)  %10.4f   %10.4f   %10.4f\n",
    kx, μx, 9kx * μx / (3kx + μx)
)

# ## §6 Transport
#
# The same machinery with a scalar unknown and two modes instead of three. The
# eccentricity splits the equivalent conductivity of the particle into a
# transverse and an axial value.

const K_AGG = TensISO{3}(1.0)
const K_MORTAR = TensISO{3}(0.1)
const K_PASTE = TensISO{3}(0.3)

println("\n     α      k_eq transverse   k_eq axial")
for α in (0.0, 0.4, 0.8)
    incl = FEExcenteredSphere(
        1.0, (K_AGG, K_MORTAR); core_fraction = W, eccentricity = α,
        nradial = 24, radius_ratio = 4.0
    )
    A, B = fe_axi_localization(incl, K_PASTE)
    Am, Bm = TensND.components_canon(A), TensND.components_canon(B)
    @printf("  %5.2f   %14.5f   %10.5f\n", α, Bm[1, 1] / Am[1, 1], Bm[3, 3] / Am[3, 3])
end

sphk = LayeredSphere((cbrt(W), 1.0), (K_AGG, K_MORTAR))
ak = TensND.components_canon(gradient_gradient_loc(sphk, K_PASTE, K_PASTE))[1, 1]
bk = TensND.components_canon(flux_gradient_loc(sphk, K_PASTE, K_PASTE))[1, 1]
@printf("  exact (α = 0)  %14.5f\n", bk / ak)

# ## §7 Memoization
#
# One evaluation is three assemblies and eight solves; an iterative scheme asks
# for the tensors again at every iteration. The cache is keyed on the reference
# medium, so a repeat costs nothing.

incl = FEExcenteredSphere(
    1.0, (C_AGG, C_MORTAR); core_fraction = W, eccentricity = 0.4, nradial = 20
)
fe_axi_mesh_report(incl)                       # build the mesh out of the timing
t1 = @elapsed strain_strain_loc(incl, C_PASTE, C_PASTE)
t2 = @elapsed stress_strain_loc(incl, C_PASTE, C_PASTE)
t3 = @elapsed strain_strain_loc(incl, C_PASTE, ce(21.0, 0.2))
@printf(
    "\n  first solve %.2f s | same solve, stress side %.1e s | new medium %.2f s | assemblies %d\n",
    t1, t2, t3, fe_assembly_count(incl)
)
