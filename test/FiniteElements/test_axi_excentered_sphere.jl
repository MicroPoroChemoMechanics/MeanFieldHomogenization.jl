# =============================================================================
#  test_axi_excentered_sphere.jl
#
#  The sphere with an off-center spherical core, solved by axisymmetric Fourier
#  finite elements with the corrected boundary condition of Adessina et al.
#  (2017).  Requires the Ferrite stack; `runtests.jl` skips this file when it
#  is unavailable.
#
#  The decisive test is the **concentric limit**: at `eccentricity = 0` the
#  morphology is the two-layer sphere, for which `LayeredSphere` gives the
#  exact Hervé-Zaoui answer.  Everything else — the Fourier modes, the axis
#  conditions, the azimuthal projections, the dipole correction, the Kelvin
#  reassembly — has to be right simultaneously for that comparison to pass.
# =============================================================================

using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra

const AX_C_CORE = iso_stiffness(50.0, 30.0)     # (k, μ), stiff old aggregate
const AX_C_SHELL = iso_stiffness(2.0, 1.0)      # compliant adhered mortar
const AX_C_MAT = iso_stiffness(10.0, 6.0)       # fresh paste
const AX_K_CORE = TensISO{3}(10.0)
const AX_K_SHELL = TensISO{3}(1.0)
const AX_K_MAT = TensISO{3}(3.0)
const AX_W = 0.5                                 # core volume fraction

ax_mandel(T) = MeanFieldHomogenization.Core.mandel66_minor(MeanFieldHomogenization.Core._C_array(T))
ax_bulk(M) = M[1, 1] + 2M[1, 2]                  # 3× the 𝕁-eigenvalue
ax_shear(M) = M[4, 4]                            # the 𝕂-eigenvalue

@testset "Axisymmetric Fourier — mesh" begin
    incl = FEExcenteredSphere(
        1.0, (AX_C_CORE, AX_C_SHELL); core_fraction = AX_W,
        eccentricity = 0.4, nradial = 16, radius_ratio = 4.0
    )
    rep = fe_axi_mesh_report(incl)
    @test rep.ncells > 500
    @test rep.ncells_core > 0 && rep.ncells_shell > 0 && rep.ncells_matrix > 0
    # The revolution volumes are recovered from the meridian quadrature.
    @test rep.volume_core ≈ rep.volume_core_exact rtol = 5.0e-3
    @test rep.volume_shell ≈ rep.volume_shell_exact rtol = 5.0e-3
    @test rep.volume_cell ≈ rep.volume_cell_exact rtol = 5.0e-3
    # Geometry: `core_fraction` fixes the radius, `eccentricity` the offset.
    @test MeanFieldHomogenization.FiniteElements.core_radius(incl) ≈ cbrt(AX_W)
    @test MeanFieldHomogenization.FiniteElements.core_offset(incl) ≈ 0.4 * (1 - cbrt(AX_W))
    @test MeanFieldHomogenization.FiniteElements.tensor_order(incl) == 4
end

@testset "Modal boundary data equals the Cartesian closed form" begin
    # The whole Fourier construction hangs on resolving `u = E·x` and the Kelvin
    # dipole field on the cylindrical basis mode by mode. Both are checked here
    # against their Cartesian expressions, rebuilt from the modal amplitudes at
    # several azimuths — a failure here would otherwise surface only as a
    # plausible-looking few-percent drift in the localization tensors.
    FE = MeanFieldHomogenization.FiniteElements
    s2 = sqrt(2.0)
    basis = (
        (0, 1, [1.0 0 0; 0 1 0; 0 0 0] ./ s2),      # m₁, mode 0
        (0, 2, [0.0 0 0; 0 0 0; 0 0 1]),            # m₂, mode 0
        (1, 1, [0.0 0 1; 0 0 0; 1 0 0] ./ s2),      # m₃, mode 1
        (2, 1, [1.0 0 0; 0 -1 0; 0 0 0] ./ s2),     # m₄, mode 2
    )
    μ, ν, V = 0.8, 0.28, 1.7
    # (ū_ρ, ū_θ, ū_z) from the solved unknowns — mode 1 is stored as (p, q, ū_z).
    unpack(m, t) = m == 0 ? (t[1], 0.0, t[2]) :
        m == 1 ? (t[1] + t[2], -t[1] + t[2], t[3]) : (t[1], t[2], t[3])
    function to_cart(m, t, θ)
        ūρ, ūθ, ūz = unpack(m, t)
        return ūρ * cos(m * θ) .* [cos(θ), sin(θ), 0.0] .+
            ūθ * sin(m * θ) .* [-sin(θ), cos(θ), 0.0] .+ ūz * cos(m * θ) .* [0.0, 0, 1]
    end
    for (m, j, Mt) in basis, ρ in (0.3, 1.1, 2.0), z in (-1.7, 0.4, 2.2),
            θ in (0.0, 0.7, 1.9, 4.1)

        x = [ρ * cos(θ), ρ * sin(θ), z]
        @test to_cart(m, FE._axi_bc_affine(m, j, ρ, z), θ) ≈ Mt * x atol = 1.0e-12
        @test to_cart(m, FE._axi_bc_dipole(m, j, ρ, z, μ, ν, V), θ) ≈
            MeanFieldHomogenization.Core._dipole_displacement_iso(μ, ν, x, V * Mt) rtol = 1.0e-12
    end
end

@testset "Concentric limit reproduces LayeredSphere (elasticity)" begin
    incl = FEExcenteredSphere(
        1.0, (AX_C_CORE, AX_C_SHELL); core_fraction = AX_W,
        eccentricity = 0.0, nradial = 24, radius_ratio = 4.0
    )
    A, B = fe_axi_localization(incl, AX_C_MAT)

    sph = LayeredSphere((cbrt(AX_W), 1.0), (AX_C_CORE, AX_C_SHELL))
    Aex = ax_mandel(strain_strain_loc(sph, AX_C_MAT, AX_C_MAT))
    Bex = ax_mandel(stress_strain_loc(sph, AX_C_MAT, AX_C_MAT))
    Anum, Bnum = ax_mandel(A), ax_mandel(B)

    @test ax_bulk(Anum) ≈ ax_bulk(Aex) rtol = 2.0e-3
    @test ax_shear(Anum) ≈ ax_shear(Aex) rtol = 3.0e-3
    @test ax_bulk(Bnum) ≈ ax_bulk(Bex) rtol = 2.0e-3
    @test ax_shear(Bnum) ≈ ax_shear(Bex) rtol = 3.0e-3

    # At zero eccentricity the answer must come out *isotropic*, which is a
    # statement about the three Fourier modes agreeing with one another —
    # they are assembled from three separate discrete problems.
    @test Anum[1, 1] ≈ Anum[3, 3] rtol = 1.0e-4
    @test Anum[1, 2] ≈ Anum[1, 3] rtol = 1.0e-3
    @test Anum[4, 4] ≈ Anum[6, 6] rtol = 1.0e-4
    @test maximum(abs, Anum[1:3, 4:6]) < 1.0e-10

    # The contribution identity `N = B − C₀:A` must hold on the FE tensors.
    N = stiffness_contribution(incl, AX_C_MAT, AX_C_MAT)
    Nid = ax_mandel(B) - ax_mandel(AX_C_MAT) * ax_mandel(A)
    @test ax_mandel(N) ≈ Nid rtol = 1.0e-10
end

@testset "Concentric limit reproduces LayeredSphere (conduction)" begin
    incl = FEExcenteredSphere(
        1.0, (AX_K_CORE, AX_K_SHELL); core_fraction = AX_W,
        eccentricity = 0.0, nradial = 24, radius_ratio = 4.0
    )
    @test MeanFieldHomogenization.FiniteElements.tensor_order(incl) == 2
    A, B = fe_axi_localization(incl, AX_K_MAT)
    sph = LayeredSphere((cbrt(AX_W), 1.0), (AX_K_CORE, AX_K_SHELL))
    Aex = TensND.components_canon(gradient_gradient_loc(sph, AX_K_MAT, AX_K_MAT))
    Bex = TensND.components_canon(flux_gradient_loc(sph, AX_K_MAT, AX_K_MAT))
    Anum, Bnum = TensND.components_canon(A), TensND.components_canon(B)
    @test Anum[1, 1] ≈ Aex[1, 1] rtol = 1.0e-3
    @test Anum[3, 3] ≈ Aex[3, 3] rtol = 1.0e-3
    @test Bnum[1, 1] ≈ Bex[1, 1] rtol = 1.0e-3
    @test Anum[1, 1] ≈ Anum[3, 3] rtol = 1.0e-6      # isotropy of the two modes
end

@testset "The boundary correction is what makes R small" begin
    exact = ax_bulk(
        ax_mandel(
            strain_strain_loc(
                LayeredSphere((cbrt(AX_W), 1.0), (AX_C_CORE, AX_C_SHELL)),
                AX_C_MAT, AX_C_MAT
            )
        )
    )
    err_unc, err_cor = Float64[], Float64[]
    for R in (2.0, 4.0)
        incl = FEExcenteredSphere(
            1.0, (AX_C_CORE, AX_C_SHELL); core_fraction = AX_W,
            eccentricity = 0.0, nradial = 20, radius_ratio = R
        )
        r = fe_axi_breakdown(incl, AX_C_MAT)
        push!(err_unc, abs(ax_bulk(ax_mandel(r.A_uncorrected)) - exact) / exact)
        push!(err_cor, abs(ax_bulk(ax_mandel(r.A)) - exact) / exact)
    end
    # The truncated cell is biased by O((a/R)³): eight times worse at R = 2a
    # than at R = 4a.
    @test err_unc[1] / err_unc[2] > 4
    @test err_unc[1] > 0.02
    # The corrected result is accurate at R = 2a already, and does not move.
    @test err_cor[1] < 2.0e-3
    @test err_cor[2] < 2.0e-3
end

@testset "Eccentricity breaks isotropy the right way" begin
    ref = nothing
    prev_axial = nothing
    for α in (0.0, 0.4, 0.8)
        incl = FEExcenteredSphere(
            1.0, (AX_C_CORE, AX_C_SHELL); core_fraction = AX_W,
            eccentricity = α, nradial = 20, radius_ratio = 4.0
        )
        M = ax_mandel(fe_axi_localization(incl, AX_C_MAT)[1])
        if α == 0.0
            ref = M
            @test M[1, 1] ≈ M[3, 3] rtol = 1.0e-4
        else
            # Transverse isotropy about the eccentricity axis, not isotropy.
            @test !isapprox(M[1, 1], M[3, 3]; rtol = 1.0e-5)
            @test M[1, 1] ≈ M[2, 2] rtol = 1.0e-12      # exact by construction
            @test M[4, 4] ≈ M[5, 5] rtol = 1.0e-12
            @test M[6, 6] ≈ M[1, 1] - M[1, 2] rtol = 1.0e-10
        end
        prev_axial = M[3, 3]
    end
    # Moving the core off center softens the pattern along the axis.
    @test prev_axial < ref[3, 3]
end

@testset "Memoization and cache control" begin
    incl = FEExcenteredSphere(
        1.0, (AX_C_CORE, AX_C_SHELL); core_fraction = AX_W,
        eccentricity = 0.2, nradial = 12, radius_ratio = 4.0
    )
    @test fe_assembly_count(incl) == 0
    strain_strain_loc(incl, AX_C_MAT, AX_C_MAT)
    @test fe_assembly_count(incl) == 1
    # The stress-side tensor comes out of the *same* solve.
    stress_strain_loc(incl, AX_C_MAT, AX_C_MAT)
    @test fe_assembly_count(incl) == 1
    strain_strain_loc(incl, AX_C_MAT, iso_stiffness(10.0, 6.0))
    @test fe_assembly_count(incl) == 1              # same content, same key
    strain_strain_loc(incl, AX_C_MAT, iso_stiffness(11.0, 6.0))
    @test fe_assembly_count(incl) == 2
    fe_reset!(incl)
    @test fe_assembly_count(incl) == 0
end

@testset "Drop-in in the schemes" begin
    incl = FEExcenteredSphere(
        1.0, (AX_C_CORE, AX_C_SHELL); core_fraction = AX_W,
        eccentricity = 0.0, nradial = 16, radius_ratio = 4.0
    )
    sph = LayeredSphere((cbrt(AX_W), 1.0), (AX_C_CORE, AX_C_SHELL))

    for scheme in (Dilute(), MoriTanaka(), Maxwell(), PonteCastanedaWillis())
        rve_fe = RVE(; distribution_shape = Ellipsoid(1.0))
        add_phase!(rve_fe, :m, Ellipsoid(1.0), Dict(:C => AX_C_MAT); fraction = :rest)
        add_phase!(rve_fe, :i, incl, Dict(:C => AX_C_MAT); fraction = 0.2)
        rve_ex = RVE(; distribution_shape = Ellipsoid(1.0))
        add_phase!(rve_ex, :m, Ellipsoid(1.0), Dict(:C => AX_C_MAT); fraction = :rest)
        add_phase!(rve_ex, :i, sph, Dict(:C => AX_C_MAT); fraction = 0.2)
        # The finite-element estimate is isotropic in *content* but not in
        # type, so both are projected before comparison.
        kf, μf = k_mu(MeanFieldHomogenization.Core.isotropify(homogenize(rve_fe, scheme, :C)))
        ke, μe = k_mu(MeanFieldHomogenization.Core.isotropify(homogenize(rve_ex, scheme, :C)))
        @test kf ≈ ke rtol = 3.0e-3
        @test μf ≈ μe rtol = 5.0e-3
    end

    # The bounds too. A heterogeneous inclusion normally cannot enter one — a
    # bound averages the constituent properties and so needs the internal
    # volume fractions — but this geometry fixes them exactly, so `Voigt` and
    # `Reuss` are not merely available, they are *exact*: no finite-element
    # solve is involved at all.
    for scheme in (Voigt(), Reuss())
        rve_fe = RVE()
        add_phase!(rve_fe, :m, Ellipsoid(1.0), Dict(:C => AX_C_MAT); fraction = :rest)
        add_phase!(rve_fe, :i, incl, Dict(:C => AX_C_MAT); fraction = 0.2)
        rve_ex = RVE()
        add_phase!(rve_ex, :m, Ellipsoid(1.0), Dict(:C => AX_C_MAT); fraction = :rest)
        add_phase!(rve_ex, :i, sph, Dict(:C => AX_C_MAT); fraction = 0.2)
        @test k_mu(homogenize(rve_fe, scheme, :C))[1] ≈
            k_mu(homogenize(rve_ex, scheme, :C))[1] rtol = 1.0e-12
    end
end

@testset "Iterative schemes" begin
    # `SelfConsistent` and `AsymmetricSelfConsistent` feed the inclusion their
    # own current estimate. Two things have to hold for that to work: the
    # memoization must keep the cost finite, and the isotropy guard must be
    # loose enough to accept an estimate that inherits the discretization noise
    # of the finite-element tensors themselves.
    incl = FEExcenteredSphere(
        1.0, (AX_C_CORE, AX_C_SHELL); core_fraction = AX_W,
        eccentricity = 0.0, nradial = 10, radius_ratio = 4.0
    )
    sph = LayeredSphere((cbrt(AX_W), 1.0), (AX_C_CORE, AX_C_SHELL))
    res = Dict{Any, Any}()
    for (tag, geom) in ((:fe, incl), (:exact, sph)), scheme in
            (SelfConsistent(), AsymmetricSelfConsistent())

        rve = RVE()
        add_phase!(rve, :m, Ellipsoid(1.0), Dict(:C => AX_C_MAT); fraction = :rest)
        add_phase!(rve, :i, geom, Dict(:C => AX_C_MAT); fraction = 0.25)
        res[(tag, nameof(typeof(scheme)))] =
            k_mu(MeanFieldHomogenization.Core.isotropify(homogenize(rve, scheme, :C)))
    end
    for scheme in (:SelfConsistent, :AsymmetricSelfConsistent)
        @test res[(:fe, scheme)][1] ≈ res[(:exact, scheme)][1] rtol = 5.0e-3
        @test res[(:fe, scheme)][2] ≈ res[(:exact, scheme)][2] rtol = 5.0e-3
    end
    # The two schemes share a fixed point; only their iteration dynamics differ.
    @test res[(:fe, :SelfConsistent)][1] ≈
        res[(:fe, :AsymmetricSelfConsistent)][1] rtol = 1.0e-3
    @test res[(:fe, :SelfConsistent)][2] ≈
        res[(:fe, :AsymmetricSelfConsistent)][2] rtol = 1.0e-3
end

@testset "Guard rails" begin
    @test_throws ArgumentError FEExcenteredSphere(
        1.0, (AX_C_CORE, AX_C_SHELL); core_fraction = 0.0
    )
    @test_throws ArgumentError FEExcenteredSphere(
        1.0, (AX_C_CORE, AX_C_SHELL); core_fraction = 0.5, eccentricity = 1.0
    )
    @test_throws ArgumentError FEExcenteredSphere(
        1.0, (AX_C_CORE, AX_C_SHELL); core_fraction = 0.5, radius_ratio = 0.9
    )
    @test_throws ArgumentError FEExcenteredSphere(
        1.0, (AX_C_CORE, AX_C_SHELL); core_fraction = 0.5, order = 3
    )
    # An anisotropic reference medium has no closed-form dipole field.
    incl = FEExcenteredSphere(
        1.0, (AX_C_CORE, AX_C_SHELL); core_fraction = AX_W, nradial = 10
    )
    C_ti = TensTI{4}(20.0, 30.0, 4.0, 5.0, 8.0, (0, 0, 1))
    @test_throws ArgumentError fe_axi_localization(incl, C_ti)
    # ... and neither has a non-transversely-isotropic constituent.
    bad = FEExcenteredSphere(
        1.0, (C_ti, AX_C_SHELL); core_fraction = AX_W, nradial = 10
    )
    C_ortho = TensND.Tens(
        MeanFieldHomogenization.Core.array_from_mandel66(
            diagm([20.0, 25.0, 30.0, 8.0, 9.0, 10.0])
        )
    )
    worse = FEExcenteredSphere(
        1.0, (C_ortho, AX_C_SHELL); core_fraction = AX_W, nradial = 10
    )
    @test_throws ArgumentError fe_axi_localization(worse, AX_C_MAT)
    @test fe_axi_localization(bad, AX_C_MAT) isa Tuple   # TI core is allowed
end

@testset "Analytic sensitivity is refused, not silently zero" begin
    # `_replace_geom_field` copies non-numeric fields by reference, so a
    # perturbed geometry would share the original's `FECache` — whose key is
    # the reference medium alone — and hand back the unperturbed tensors. The
    # derivative would come out as exactly zero, with no warning. (It would be
    # zero anyway: the solve converts to `Float64` on entry, so a `Dual` loses
    # its perturbation at the door.) Refusing is the only honest answer.
    Sch = MeanFieldHomogenization.Schemes
    incl = FEExcenteredSphere(
        1.0, (AX_C_CORE, AX_C_SHELL); core_fraction = AX_W, nradial = 10
    )
    for f in (:a, :core_fraction, :eccentricity)
        e = @test_throws ErrorException Sch._replace_geom_field(incl, Val(f), nothing, 0.5)
        @test occursin("finite difference", e.value.msg)
    end
    crack = FEEllipticCrack(1.0, 0.5)
    @test_throws ErrorException Sch._replace_geom_field(crack, Val(:a), nothing, 2.0)
end
