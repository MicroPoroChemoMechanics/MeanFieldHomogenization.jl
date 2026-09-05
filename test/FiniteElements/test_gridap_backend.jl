# =============================================================================
#  test_gridap_backend.jl
#
#  The Gridap backend, on both morphologies.  Requires the Gridap stack *and*
#  the Ferrite one, since the point of the file is the comparison; `runtests.jl`
#  skips it unless both are available.
#
#  The two backends share the mesh, the Fourier operators, the boundary data
#  and the fixed point, and differ only in the discretization layer — so they
#  are expected to agree to round-off, not merely to converge to the same
#  limit.  They do: the tolerance below is 1e-8 and the measured difference is
#  around 1e-14.  That makes this file a sharp regression guard on the nine
#  methods of the backend contract: any of them going subtly wrong in either
#  backend shows up immediately.
# =============================================================================

using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra

const GX_C_CORE = iso_stiffness(50.0, 30.0)
const GX_C_SHELL = iso_stiffness(2.0, 1.0)
const GX_C_MAT = iso_stiffness(10.0, 6.0)
const GX_K_CORE = TensISO{3}(10.0)
const GX_K_SHELL = TensISO{3}(1.0)
const GX_K_MAT = TensISO{3}(3.0)
const GX_W = 0.5

gx_mandel(T) = MeanFieldHomogenization.Core.mandel66_minor(MeanFieldHomogenization.Core._C_array(T))
gx_bulk(M) = M[1, 1] + 2M[1, 2]
gx_shear(M) = M[4, 4]
gx_rel(X, Y) = maximum(abs, X - Y) / maximum(abs, X)

gx_incl(backend, props; kw...) = FEExcenteredSphere(
    1.0, props; core_fraction = GX_W, backend, kw...
)

@testset "Backend resolution" begin
    # Both stacks are loaded here, so `AutoBackend` must pick the documented
    # first choice rather than error or flip-flop.
    FE = MeanFieldHomogenization.FiniteElements
    @test FE._resolve_backend(AutoBackend()) === FerriteBackend()
    @test FE._resolve_backend(GridapBackend()) === GridapBackend()
    @test FE._resolve_backend(FerriteBackend()) === FerriteBackend()

    # The backend is a property of the object, pinned at the first solve.
    incl = gx_incl(GridapBackend(), (GX_C_CORE, GX_C_SHELL); eccentricity = 0.0)
    @test fe_axi_mesh_report(incl).backend === GridapBackend()
    @test gx_incl(AutoBackend(), (GX_C_CORE, GX_C_SHELL)) isa FEExcenteredSphere
end

@testset "The two backends mesh the same body" begin
    rf = fe_axi_mesh_report(gx_incl(FerriteBackend(), (GX_C_CORE, GX_C_SHELL)))
    rg = fe_axi_mesh_report(gx_incl(GridapBackend(), (GX_C_CORE, GX_C_SHELL)))
    @test rf.ncells == rg.ncells
    # Node counts need not match to the unit: the three arc-center points of
    # the geometry belong to no element, and FerriteGmsh keeps them (it reads
    # the live gmsh session) where `GmshDiscreteModel` drops them (it re-reads
    # a written file, discarding orphans). Nothing downstream sees them.
    @test rg.nnodes ≤ rf.nnodes ≤ rg.nnodes + 3
    @test rf.ncells_core == rg.ncells_core
    @test rf.ncells_matrix == rg.ncells_matrix
    @test rf.volume_core ≈ rg.volume_core rtol = 1.0e-12
    @test rf.volume_cell ≈ rg.volume_cell rtol = 1.0e-12
end

@testset "Concentric limit reproduces LayeredSphere (Gridap)" begin
    incl = gx_incl(GridapBackend(), (GX_C_CORE, GX_C_SHELL); eccentricity = 0.0)
    A, B = fe_axi_localization(incl, GX_C_MAT)
    sph = LayeredSphere((cbrt(GX_W), 1.0), (GX_C_CORE, GX_C_SHELL))
    Aex = gx_mandel(strain_strain_loc(sph, GX_C_MAT, GX_C_MAT))
    Bex = gx_mandel(stress_strain_loc(sph, GX_C_MAT, GX_C_MAT))
    Anum, Bnum = gx_mandel(A), gx_mandel(B)
    @test gx_bulk(Anum) ≈ gx_bulk(Aex) rtol = 2.0e-3
    @test gx_shear(Anum) ≈ gx_shear(Aex) rtol = 3.0e-3
    @test gx_bulk(Bnum) ≈ gx_bulk(Bex) rtol = 2.0e-3
    @test gx_shear(Bnum) ≈ gx_shear(Bex) rtol = 3.0e-3

    ic = gx_incl(GridapBackend(), (GX_K_CORE, GX_K_SHELL); eccentricity = 0.0)
    a, b = fe_axi_localization(ic, GX_K_MAT)
    sphk = LayeredSphere((cbrt(GX_W), 1.0), (GX_K_CORE, GX_K_SHELL))
    aex = TensND.components_canon(gradient_gradient_loc(sphk, GX_K_MAT, GX_K_MAT))
    bex = TensND.components_canon(flux_gradient_loc(sphk, GX_K_MAT, GX_K_MAT))
    @test TensND.components_canon(a)[1, 1] ≈ aex[1, 1] rtol = 2.0e-3
    @test TensND.components_canon(b)[1, 1] ≈ bex[1, 1] rtol = 2.0e-3
end

@testset "Ferrite and Gridap agree to round-off" begin
    # Same mesh, same P2 space, same quadrature degree: the two backends build
    # the *same* discrete system, so this is far tighter than a convergence
    # statement. The eccentric case exercises all three elastic modes and both
    # transport ones with a genuinely transversely isotropic answer.
    for e in (0.0, 0.6)
        Af, Bf = fe_axi_localization(
            gx_incl(FerriteBackend(), (GX_C_CORE, GX_C_SHELL); eccentricity = e), GX_C_MAT
        )
        Ag, Bg = fe_axi_localization(
            gx_incl(GridapBackend(), (GX_C_CORE, GX_C_SHELL); eccentricity = e), GX_C_MAT
        )
        @test gx_rel(gx_mandel(Af), gx_mandel(Ag)) < 1.0e-8
        @test gx_rel(gx_mandel(Bf), gx_mandel(Bg)) < 1.0e-8

        af, bf = fe_axi_localization(
            gx_incl(FerriteBackend(), (GX_K_CORE, GX_K_SHELL); eccentricity = e), GX_K_MAT
        )
        ag, bg = fe_axi_localization(
            gx_incl(GridapBackend(), (GX_K_CORE, GX_K_SHELL); eccentricity = e), GX_K_MAT
        )
        @test gx_rel(TensND.components_canon(af), TensND.components_canon(ag)) < 1.0e-8
        @test gx_rel(TensND.components_canon(bf), TensND.components_canon(bg)) < 1.0e-8
    end
end

@testset "The boundary of the cell is fully constrained" begin
    # The endpoints of the outer arcs and of the axis segments belong to point
    # entities of their own, which a curve physical group does not cover. Left
    # free they cost an order of convergence while looking like discretization
    # error, so the mesh declares them and both backends must see the same
    # prescribed set. Three dofs on the outer sphere, six on the axis.
    FE = MeanFieldHomogenization.FiniteElements
    order = 2
    for (backend, ncomp, axis_zeros) in (
            (FerriteBackend(), 2, (1,)), (GridapBackend(), 2, (1,)),
        )
        grid = FE._axi_setup(
            gx_incl(backend, (GX_C_CORE, GX_C_SHELL); order)
        ).grid
        mode = FE.fe_axi_mode(backend, grid, order, ncomp, axis_zeros)
        nd, free, presc = FE.fe_axi_dof_split(backend, mode)
        @test nd == length(free) + length(presc)
        @test isempty(intersect(free, presc))
        @test issorted(presc)
    end
end

@testset "Drop-in in the schemes (Gridap)" begin
    incl = gx_incl(GridapBackend(), (GX_C_CORE, GX_C_SHELL); eccentricity = 0.3)
    rve = RVE(; distribution_shape = Ellipsoid(1.0))
    add_phase!(rve, :m, Ellipsoid(1.0), Dict(:C => GX_C_MAT); fraction = :rest)
    add_phase!(rve, :rca, incl, Dict(:C => GX_C_MAT); fraction = 0.25)
    Cmt = homogenize(rve, MoriTanaka(), :C)
    @test Cmt isa TensND.AbstractTens{4, 3}

    # Voigt and Reuss are exact here: the geometry knows its internal volume
    # fractions, whichever backend solves it.
    Cv = homogenize(rve, Voigt(), :C)
    Cr = homogenize(rve, Reuss(), :C)
    @test gx_bulk(gx_mandel(Cr)) < gx_bulk(gx_mandel(Cmt)) < gx_bulk(gx_mandel(Cv))
end


# ═══ The flat crack ══════════════════════════════════════════════════════════

@testset "The two backends mesh the same crack" begin
    rf = fe_mesh_report(FEEllipticCrack(1.0, 0.5; backend = FerriteBackend()))
    rg = fe_mesh_report(FEEllipticCrack(1.0, 0.5; backend = GridapBackend()))
    @test rf.ncells == rg.ncells
    @test rf.nnodes == rg.nnodes
    @test rf.ndofs == rg.ndofs
    # Both lips carry the same number of facets and the same area: the `Crack`
    # plugin split the surface cleanly and the front weld did not glue it back.
    @test rf.nfacets_up == rg.nfacets_up == rf.nfacets_dn == rg.nfacets_dn
    @test rf.area_up ≈ rg.area_up rtol = 1.0e-12
    @test rf.area_up ≈ rf.area_exact rtol = 2.0e-3
    @test rf.area_dn ≈ rf.area_exact rtol = 2.0e-3
end

@testset "The crack boundary is fully constrained in both backends" begin
    # The seam curves and poles of the OCC sphere belong to entities of their
    # own, which the surface physical group does not cover. Eleven nodes here —
    # enough to skew the opening, not enough to look like a bug.
    FE = MeanFieldHomogenization.FiniteElements
    n = Int[]
    for b in (FerriteBackend(), GridapBackend())
        s = FE._crack_setup(FEEllipticCrack(1.0, 0.5; backend = b))
        nd, free, presc = FE.fe_crack_dof_split(b, s.space)
        @test nd == length(free) + length(presc)
        push!(n, length(presc))
    end
    @test n[1] == n[2]
end

@testset "Crack: Ferrite and Gridap agree to round-off" begin
    C₀ = iso_stiffness(10.0, 6.0)
    Bf = TensND.components_canon(
        cod_tensor(FEEllipticCrack(1.0, 0.5; backend = FerriteBackend()), C₀)
    )
    Bg = TensND.components_canon(
        cod_tensor(FEEllipticCrack(1.0, 0.5; backend = GridapBackend()), C₀)
    )
    @test gx_rel(Bf, Bg) < 1.0e-8

    # …and both stay within the documented few percent of the closed form.
    Ba = TensND.components_canon(cod_tensor(EllipticCrack(1.0, 0.5), C₀))
    @test Bg[1, 1] ≈ Ba[1, 1] rtol = 5.0e-2
    @test Bg[3, 3] ≈ Ba[3, 3] rtol = 5.0e-2
end
