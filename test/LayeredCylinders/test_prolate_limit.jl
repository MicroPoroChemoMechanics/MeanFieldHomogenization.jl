# =============================================================================
#  test_prolate_limit.jl — the layered cylinder as the limit of elongated
#  inclusions of the package, which share none of its code.
#
#  - A homogeneous prolate `Ellipsoid` of aspect ratio ω tends to the one-layer
#    cylinder (Eshelby's tensor of the cylinder), with an error O(ln ω/ω²).
#  - A confocal `LayeredSpheroid` whose focal distance c grows at fixed disk radii
#    tends to coaxial layers. It is not a cylinder: its local radius varies along
#    the axis, b(z) = b √(1 − z²/a²). With perfect interfaces the answer of a
#    section does not depend on its scale, and the spheroid tends to the
#    cylinder itself. An imperfect interface carries a length (ρ, kˢ): each
#    section answers as the cylinder of its own radius, and the slender-body
#    limit is the average of the layered cylinder over the sections,
#        ⟨X⟩ = (3/4) ∫₋₁¹ (1 − ζ²) X_cyl(radii √(1 − ζ²)) dζ .
# =============================================================================
isdefined(@__MODULE__, :lc_ti) || include(joinpath(@__DIR__, "helpers.jl"))

lc_rel(a, b) = norm(lc_cc(a) .- lc_cc(b)) / norm(lc_cc(b))

@testset "LayeredCylinder — limit of an elongated ellipsoid" begin
    for (P_I, P_0) in (
            (TensISO{3}(3 * 30.0, 2 * 12.0), TensISO{3}(3 * 6.0, 2 * 2.5)),
            (lc_tensor(LC_F), TensISO{3}(3 * 6.0, 2 * 2.5)),
            (TensISO{3}(5.0), TensISO{3}(2.0)),
            (lc_K(5.0, 7.0), TensISO{3}(2.0)),
        )
        cyl = LayeredCylinder((1.0,), (P_I,))
        errors = map((1.0e2, 1.0e3, 1.0e4)) do ω
            lc_rel(gradient_gradient_concentration(Ellipsoid(1.0, 1.0, ω), P_I, P_0), gradient_gradient_concentration(cyl, P_I, P_0))
        end
        @test errors[3] < 1.0e-5
        # at the rate of the theory, O(ln ω/ω²): a factor of about 70 per decade
        @test errors[2] / errors[3] > 40 && errors[1] / errors[2] > 40
    end
end

@testset "LayeredCylinder — limit of an elongated confocal spheroid, perfect interfaces" begin
    c = 300.0
    disk = (0.6, 1.0)
    axis = map(b -> sqrt(b^2 + c^2), disk)
    K0 = TensISO{3}(2.0)
    layersK = (TensISO{3}(5.0), TensISO{3}(1.0))
    sp = LayeredSpheroid(axis, disk, layersK; Nseries = 10)
    cy = LayeredCylinder(disk, layersK)
    @test lc_rel(gradient_gradient_concentration(sp, K0, K0), gradient_gradient_concentration(cy, K0, K0)) < 2.0e-5
    @test lc_rel(flux_gradient_concentration(sp, K0, K0), flux_gradient_concentration(cy, K0, K0)) < 2.0e-5
    C0 = TensISO{3}(3 * 6.0, 2 * 2.5)
    layersC = (TensISO{3}(3 * 30.0, 2 * 12.0), TensISO{3}(3 * 4.0, 2 * 1.5))
    spC = LayeredSpheroid(axis, disk, layersC; Nseries = 10)
    cyC = LayeredCylinder(disk, layersC)
    pc = MeanFieldHomogenization.property_contribution
    @test lc_rel(pc(spC, C0, C0), pc(cyC, C0, C0)) < 1.0e-3
    # closer as the spheroid lengthens
    sp100 = LayeredSpheroid(map(b -> sqrt(b^2 + 100.0^2), disk), disk, layersC; Nseries = 10)
    @test lc_rel(pc(spC, C0, C0), pc(cyC, C0, C0)) < lc_rel(pc(sp100, C0, C0), pc(cyC, C0, C0))
end

@testset "LayeredCylinder — slender-body limit with imperfect interfaces" begin
    K0 = TensISO{3}(2.0)
    layers = (TensISO{3}(5.0), TensISO{3}(1.0))
    c = 300.0
    disk = (0.6, 1.0)
    function slender(f, interfaces; n = 2000)
        acc = zeros(3, 3)
        for i in 1:n
            ζ = -1 + (i - 0.5) * 2 / n
            s = sqrt(1 - ζ^2)
            acc .+= (1 - ζ^2) .* lc_cc(f(LayeredCylinder(disk .* s, layers; interfaces)))
        end
        return acc .* (3 / 4) .* (2 / n)
    end
    for interfaces in (
            (KapitzaInterface(0.2), PerfectInterface()),
            (PerfectInterface(), KapitzaInterface(0.2)),
            (SurfaceConductiveInterface(0.3), PerfectInterface()),
            (PerfectInterface(), SurfaceConductiveInterface(0.3)),
        )
        sp = LayeredSpheroid(map(b -> sqrt(b^2 + c^2), disk), disk, layers; interfaces, Nseries = 10)
        for f in (g -> gradient_gradient_concentration(g, K0, K0), g -> flux_gradient_concentration(g, K0, K0))
            ref = slender(f, interfaces)
            @test norm(lc_cc(f(sp)) .- ref) / norm(ref) < 5.0e-5
            # and the cylinder of the outer radius is not that limit
            @test norm(lc_cc(f(LayeredCylinder(disk, layers; interfaces))) .- ref) / norm(ref) > 1.0e-3
        end
    end
end
