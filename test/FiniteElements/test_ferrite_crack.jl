# =============================================================================
#  test_ferrite_crack.jl — the finite-element elliptical crack
#  (`MeanFieldHomogenizationFerriteExt`).
#
#  Slow by nature: every case meshes a ball and factorizes a ~10⁵-dof system.
#  Kept to penny-shaped cracks on deliberately coarse meshes; the accuracy
#  claim is carried by a *convergence* argument rather than by a loose absolute
#  tolerance.
#
#  Coverage:
#   1. Mesh: the crack front is welded, the two lips are separated, and their
#      area is πab.
#   2. The COD tensor converges to the closed form: the error is first order in
#      the element size, and the Richardson extrapolation to h → 0 lands within
#      3 % of the analytical value.
#   3. The boundary correction is wired correctly: ‖B_u‖ scales as (a/R)³.
#   4. Memoization: repeated calls with the same C₀ cost one assembly.
#   5. The crack is a drop-in replacement in the schemes, orientation
#      averaging included.
#   6. Out-of-scope inputs (anisotropic matrix, conduction) fail informatively.
# =============================================================================

using Test
using MeanFieldHomogenization
using TensND
using Tensors
using LinearAlgebra

@testset "FEEllipticCrack (Ferrite extension)" begin

    # E = 1, ν = 0.3, dimensionless — the SifAniso reference setting.
    ν_fe = 0.3
    E_fe = 1.0
    C₀_fe = iso_stiffness(E_fe / (3 * (1 - 2ν_fe)), E_fe / (2 * (1 + ν_fe)))

    a_fe = 1.0
    diag3(B) = diag(Matrix(get_array(B)))

    # One coarse and one finer penny crack, reused across the testsets so the
    # meshes are built once.
    crack4 = FEEllipticCrack(a_fe, a_fe; htipdiv = 4.0)
    crack6 = FEEllipticCrack(a_fe, a_fe; htipdiv = 6.0)
    B_ana = diag3(cod_tensor(EllipticCrack(a_fe, a_fe), C₀_fe))

    @testset "mesh: welded front, separated lips, exact area" begin
        r = fe_mesh_report(crack6)
        @test r.ncells > 1000
        @test r.nfacets_up == r.nfacets_dn          # the slit is symmetric
        @test r.area_up ≈ r.area_exact rtol = 0.02  # polygonal ellipse
        @test r.area_dn ≈ r.area_exact rtol = 0.02
        @test r.area_up ≈ r.area_dn rtol = 1.0e-12  # both lips discretized alike
        @test r.ndofs > r.nnodes                    # vector field
    end

    @testset "COD converges to the closed form (first order in h)" begin
        B4 = diag3(cod_tensor(crack4, C₀_fe))
        B6 = diag3(cod_tensor(crack6, C₀_fe))

        # A finite cell is stiffer than the infinite medium: the FE opening
        # approaches the analytical one from below, and refining must help.
        @test all(B4 .< B6 .< B_ana)

        # Error is O(h) with h ∝ 1/htipdiv, so Richardson-extrapolate:
        #   B(0) = B₆ + (B₆ - B₄)·h₆/(h₄ - h₆),   h = 1/htipdiv.
        h4, h6 = 1 / 4, 1 / 6
        B0 = B6 .+ (B6 .- B4) .* (h6 / (h4 - h6))
        @test B0 ≈ B_ana rtol = 0.03

        # ... and the extrapolation must be a genuine improvement over the
        # finest raw value, otherwise the "first order" claim is empty.
        @test maximum(abs, (B0 .- B_ana) ./ B_ana) <
            maximum(abs, (B6 .- B_ana) ./ B_ana)
    end

    @testset "the boundary correction scales as (a/R)³" begin
        # `B_u` is the cell's response to the crack's own dipole far field; its
        # magnitude is the size of the truncation bias being removed, and it
        # must fall like the cube of the domain radius.  This pins both the
        # magnitude and the sign of the dipole boundary condition.
        d5 = fe_cod_breakdown(FEEllipticCrack(a_fe, a_fe; htipdiv = 4.0, radius_ratio = 5.0), C₀_fe)
        d10 = fe_cod_breakdown(FEEllipticCrack(a_fe, a_fe; htipdiv = 4.0, radius_ratio = 10.0), C₀_fe)

        @test norm(d5.B_u) > 0
        @test norm(d5.B_u) / norm(d10.B_u) ≈ 8.0 rtol = 0.25

        # The correction is small at R = 5a but it goes the right way: it opens
        # the crack, since the truncated cell under-opens it.
        @test all(diag(d5.B_inf) .> diag(d5.B_s))
    end

    @testset "memoization" begin
        c = FEEllipticCrack(a_fe, a_fe; htipdiv = 4.0)
        @test fe_assembly_count(c) == 0
        cod_tensor(c, C₀_fe)
        @test fe_assembly_count(c) == 1
        cod_tensor(c, C₀_fe)
        cod_tensor(c, C₀_fe)
        @test fe_assembly_count(c) == 1            # same C₀ → no new solve
        cod_tensor(c, iso_stiffness(0.9, 0.4))
        @test fe_assembly_count(c) == 2            # different C₀ → one more
        fe_reset!(c)
        @test fe_assembly_count(c) == 0
    end

    @testset "the whole contribution chain is inherited" begin
        # Only `cod_tensor` is implemented; everything below follows from
        # `shape_trait == Penny`.
        @test MeanFieldHomogenization.shape_trait(crack6) === MeanFieldHomogenization.Penny
        @test MeanFieldHomogenization.Cracks.crack_density_factor(crack6) ≈ 4π / 3

        H = compliance_contribution(crack6, C₀_fe)
        N = stiffness_contribution(crack6, C₀_fe)
        @test N ≈ -(C₀_fe ⊡ H ⊡ C₀_fe)
        Hb, Nb = MeanFieldHomogenization.compliance_and_stiffness_contribution(crack6, C₀_fe)
        @test get_array(Hb) == get_array(H)
        @test get_array(Nb) == get_array(N)

        ε = 0.05
        @test get_array(delta_compliance(crack6, H, ε)) ≈ get_array((4π / 3) * ε * H)
        @test check_inclusion_interface(crack6; amount = :density, verbose = false)
    end

    @testset "drop-in replacement in the schemes" begin
        ε = 0.05
        function chom(geom, scheme; kw...)
            r = RVE()
            add_phase!(r, :M, Ellipsoid(1.0), Dict(:C => C₀_fe); fraction = :rest)
            add_phase!(r, :cr, geom, Dict(:C => C₀_fe); density = ε, kw...)
            return homogenize(r, scheme, :C)
        end

        ana = EllipticCrack(a_fe, a_fe)
        # The FE crack under-opens by ~9 % at htipdiv = 6, so the effective
        # moduli must agree with the analytical crack to that same order — the
        # point being that the *plumbing* is identical, not that a coarse mesh
        # is exact.
        for scheme in (Dilute(), DiluteDual(), MoriTanaka())
            Cfe = get_array(chom(crack6, scheme))[1, 1, 1, 1]
            Can = get_array(chom(ana, scheme))[1, 1, 1, 1]
            @test Cfe ≈ Can rtol = 0.02
            @test Cfe < get_array(C₀_fe)[1, 1, 1, 1]     # cracks soften
        end

        # Orientation averaging is applied by the scheme, after `cod_tensor`.
        Ciso = chom(crack6, MoriTanaka(); symmetrize = IsoSymmetrize())
        @test Ciso isa TensND.TensISO
    end

    @testset "out-of-scope inputs fail informatively" begin
        c = FEEllipticCrack(1.0, 0.5)
        # Anisotropic reference medium: the corrected boundary condition uses
        # the closed-form Kelvin dipole, which is isotropic-only.
        C_ti = TensTI{4}(20.0, 30.0, 4.0, 5.0, 8.0, (0.0, 0.0, 1.0))
        @test_throws ArgumentError cod_tensor(c, C_ti)
        # Conduction has no finite-element counterpart yet.
        @test_throws ErrorException cod_tensor(c, TensISO{3}(2.0))
    end

    @testset "isotropy is judged on content, not on the TensND type" begin
        # An iterative scheme hands the kernel a `TensCanonical`; when its
        # content is isotropic it must be accepted, otherwise `SelfConsistent`
        # under `IsoSymmetrize` would be refused for no reason.
        c = FEEllipticCrack(a_fe, a_fe; htipdiv = 4.0)
        C_canon = TensND.Tens(
            Tensors.SymmetricTensor{4, 3}((i, j, k, l) -> C₀_fe[i, j, k, l])
        )
        @test diag3(cod_tensor(c, C_canon)) ≈ diag3(cod_tensor(c, C₀_fe)) rtol = 1.0e-12
        @test fe_assembly_count(c) == 1        # and it is the *same* cache entry
    end

    @testset "iterative schemes work under IsoSymmetrize" begin
        # Parallel FE cracks make the self-consistent iterate transversely
        # isotropic — out of scope, and refused explicitly.
        rve_par = RVE()
        add_phase!(rve_par, :M, Ellipsoid(1.0), Dict(:C => C₀_fe); fraction = :rest)
        add_phase!(rve_par, :cr, crack4, Dict(:C => C₀_fe); density = 0.05)
        @test_throws ArgumentError homogenize(rve_par, SelfConsistent(), :C)

        # Isotropically averaged, the reference stays isotropic and it runs.
        rve_iso = RVE()
        add_phase!(rve_iso, :M, Ellipsoid(1.0), Dict(:C => C₀_fe); fraction = :rest)
        add_phase!(
            rve_iso, :cr, crack4, Dict(:C => C₀_fe);
            density = 0.05, symmetrize = IsoSymmetrize()
        )
        C_sc = homogenize(rve_iso, SelfConsistent(), :C)
        @test C_sc isa TensND.TensISO
        @test get_array(C_sc)[1, 1, 1, 1] < get_array(C₀_fe)[1, 1, 1, 1]
    end

    @testset "constructor guard rails" begin
        @test_throws ArgumentError FEEllipticCrack(1.0, 2.0)          # b > a
        @test_throws ArgumentError FEEllipticCrack(1.0, 0.0)          # b = 0
        @test_throws ArgumentError FEEllipticCrack(1.0, 0.5; order = 3)
        @test_throws ArgumentError FEEllipticCrack(1.0, 0.5; radius_ratio = 0.5)
        @test_throws ArgumentError FEEllipticCrack(1.0, 0.5; htipdiv = -1.0)
    end
end
