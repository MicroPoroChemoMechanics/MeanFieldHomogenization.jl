using SymPyPythonCall   # top level on purpose: `@syms` is expanded before a nested `using` runs
import Symbolics

# =============================================================================
#  test_generic.jl — the layered cylinder on dual numbers (nested), extended
#  precision and symbols (Symbolics.Num and SymPy), against closed forms.
# =============================================================================
isdefined(@__MODULE__, :lc_ti) || include(joinpath(@__DIR__, "helpers.jl"))

# Plane-strain bulk and axial shear moduli of a fiber coated by its matrix,
# read on the composite cylinder itself.
function lc_coated_moduli(k_f, p_f, c)
    F, M = LC_F, LC_M
    cyl = LayeredCylinder((sqrt(c), one(c)), (lc_ti(k_f, F.l, F.m, F.n, p_f), lc_tensor(M)))
    sol = LCyl._elastic_solution(cyl, lc_tensor(LC_0))
    u, σ = sol.dilatation.outside[2]
    φ, τ = sol.antiplane.outside[2]
    return σ / (2u), τ / φ
end

@testset "LayeredCylinder — ForwardDiff, nested, against Hashin-Rosen" begin
    M = LC_M
    c, k_f, p_f = 0.36, 30.0, 14.0
    b = (1 - c) / (M.k + M.m)
    u = k_f - M.k
    # k* = k_m + c u/(1 + b u), and its first two derivatives in k_f
    @test ForwardDiff.derivative(x -> lc_coated_moduli(x, p_f, c)[1], k_f) ≈ c / (1 + b * u)^2
    @test ForwardDiff.derivative(y -> ForwardDiff.derivative(x -> lc_coated_moduli(x, p_f, c)[1], y), k_f) ≈ -2c * b / (1 + b * u)^3
    # p* = p_m (p_f (1 + c) + p_m (1 − c))/(p_f (1 − c) + p_m (1 + c)), in p_f and in c
    D = p_f * (1 - c) + M.p * (1 + c)
    @test ForwardDiff.derivative(x -> lc_coated_moduli(k_f, x, c)[2], p_f) ≈ 4 * M.p^2 * c / D^2
    @test ForwardDiff.derivative(x -> lc_coated_moduli(k_f, p_f, x)[2], c) ≈ 2 * M.p * (p_f^2 - M.p^2) / D^2
    @inferred lc_coated_moduli(k_f, p_f, c)
    cyl = LayeredCylinder((0.6, 1.0), (lc_tensor(LC_F), TensISO{3}(30.0, 8.0)))      # layers of two types
    @inferred LCyl._elastic_solution(cyl, lc_tensor(LC_0))
    @inferred LCyl._conduction_solution(LayeredCylinder((0.6, 1.0), (lc_K(5.0, 7.0), TensISO{3}(2.0))), lc_K(2.0, 3.0))
    # the public entry points, the pointwise fields and the layer averages
    C0 = lc_tensor(LC_0)
    bonded = LayeredCylinder((0.6, 1.0), (lc_tensor(LC_F), TensISO{3}(30.0, 8.0)); interfaces = (SpringInterface(20.0, 10.0), MembraneInterface(3.0, 1.2)))
    @inferred gradient_gradient_concentration(bonded, C0, C0)
    @inferred flux_gradient_concentration(bonded, C0, C0)
    @inferred MeanFieldHomogenization.property_contribution(bonded, C0)
    @inferred gradient_gradient_concentration(bonded, C0; layer = 2, external = true)
    @inferred LCyl.layer_property_average(bonded, Val(4))
    @inferred LCyl.layer_inverse_property_average(bonded, Val(4))
    fields = @inferred LayeredCylinderFields(bonded, C0)
    E = TensND.Tens([0.3 0.1 0.0; 0.1 -0.2 0.0; 0.0 0.0 0.5])
    @inferred local_gradient(fields, (0.3, 0.2, 0.1), E)
    @inferred local_flux(fields, (0.3, 0.2, 0.1), E)
    @inferred local_displacement(fields, (0.3, 0.2, 0.1), E)
end

@testset "LayeredCylinder — ForwardDiff in a layer, a radius, the interfaces" begin
    Cm, C0 = lc_tensor(LC_M), lc_tensor(LC_0)
    # One layer modulus, one radius and one parameter of each interface carry the
    # dual, the rest stay Float64: the promotions must reach every state.
    g(x) = lc_data(
        MeanFieldHomogenization.property_contribution(
            LayeredCylinder(
                (0.5 + x[1], 1.0), (lc_ti(30.0 + x[2], 12.0, 9.0, 150.0, 14.0), Cm);
                interfaces = (SpringInterface(; sn = 0.05 + x[3], st = 0.08), MembraneInterface(3.0 + x[4], 1.2))
            ), C0
        )
    )
    J = ForwardDiff.jacobian(g, zeros(4))
    h = 1.0e-6
    J_fd = reduce(hcat, [(g(h .* (1:4 .== j)) .- g(-h .* (1:4 .== j))) ./ 2h for j in 1:4])
    @test J ≈ J_fd rtol = 1.0e-6
    q(x) = lc_data(
        MeanFieldHomogenization.property_contribution(
            LayeredCylinder(
                (0.5, 1.0), (lc_K(5.0 + x[1], 7.0), lc_K(1.0, 2.0));
                interfaces = (KapitzaInterface(0.2 + x[2]), SurfaceConductiveInterface(0.3 + x[3]))
            ), lc_K(2.0, 3.0)
        )
    )
    Jq = ForwardDiff.jacobian(q, zeros(3))
    Jq_fd = reduce(hcat, [(q(h .* (1:3 .== j)) .- q(-h .* (1:3 .== j))) ./ 2h for j in 1:3])
    @test Jq ≈ Jq_fd rtol = 1.0e-6
    # a pointwise field differentiated in a modulus
    E = MeanFieldHomogenization.Core._tens_in([0.3 0.1 -0.2; 0.1 -0.4 0.25; -0.2 0.25 0.7], TensND.CanonicalBasis{3, Float64}())
    s(x) = vec(lc_cc(local_flux(LayeredCylinder((0.5, 1.0), (lc_ti(30.0 + x, 12.0, 9.0, 150.0, 14.0), Cm)), C0, (0.2, 0.1, 0.0), E)))
    @test ForwardDiff.derivative(s, 0.0) ≈ (s(h) .- s(-h)) ./ 2h rtol = 1.0e-6
end

@testset "LayeredCylinder — BigFloat" begin
    big_ti(L) = lc_ti(big(L.k), big(L.l), big(L.m), big(L.n), big(L.p); axis = (big(0.0), big(0.0), big(1.0)))
    cyl = LayeredCylinder(
        (big(0.6), big(1.0)), (big_ti(LC_F), big_ti(LC_M));
        interfaces = (SpringInterface(big(20.0), big(12.5)), PerfectInterface{BigFloat}()), axis = (big(0.0), big(0.0), big(1.0))
    )
    N = MeanFieldHomogenization.property_contribution(cyl, big_ti(LC_0))
    @test eltype(N) == BigFloat
    ref = LayeredCylinder((0.6, 1.0), (lc_tensor(LC_F), lc_tensor(LC_M)); interfaces = (SpringInterface(20.0, 12.5), PerfectInterface()))
    @test Float64.(lc_data(N)) ≈ lc_data(MeanFieldHomogenization.property_contribution(ref, lc_tensor(LC_0))) rtol = 1.0e-12
    d = lc_data(N)
    @test abs(d[3] - d[4]) < big(1.0e-60) * abs(d[3])       # major symmetry, to the precision
end

@testset "LayeredCylinder — Symbolics.Num closed forms" begin
    Symbolics.@variables k₁ l₁ m₁ n₁ p₁ k₂ l₂ m₂ n₂ p₂ a ρ κ R
    ez = (0, 0, 1)
    num(x, vals) = Float64(Symbolics.value(Symbolics.substitute(x, vals; fold = Val(true))))
    # Hashin-Rosen, symbolically: fiber of radius a in a coating of radius 1
    cyl = LayeredCylinder((a, one(a)), (lc_ti(k₁, l₁, m₁, n₁, p₁; axis = ez), lc_ti(k₂, l₂, m₂, n₂, p₂; axis = ez)))
    sol = LCyl._elastic_solution(cyl, lc_ti(k₂, l₂, m₂, n₂, p₂; axis = ez))
    u, σ = sol.dilatation.outside[2]
    φ, τ = sol.antiplane.outside[2]
    c = a^2
    k_HR = k₂ + c / (1 / (k₁ - k₂) + (1 - c) / (k₂ + m₂))
    p_HR = p₂ * (p₁ * (1 + c) + p₂ * (1 - c)) / (p₁ * (1 - c) + p₂ * (1 + c))
    @test isequal(Symbolics.simplify(σ / (2u) - k_HR; expand = true), 0)
    @test isequal(Symbolics.simplify(τ / φ - p_HR; expand = true), 0)
    # transverse shear and the axial coupling: against the numbers
    vals = Dict(k₁ => 30.0, l₁ => 12.0, m₁ => 9.0, n₁ => 150.0, p₁ => 14.0, k₂ => 4.0, l₂ => 2.0, m₂ => 1.5, n₂ => 7.0, p₂ => 1.5, a => 0.6)
    A_sym = gradient_gradient_concentration(cyl, lc_ti(k₂, l₂, m₂, n₂, p₂; axis = ez), lc_ti(k₂, l₂, m₂, n₂, p₂; axis = ez))
    A_num = gradient_gradient_concentration(LayeredCylinder((0.6, 1.0), (lc_tensor(LC_F), lc_tensor(LC_M))), lc_tensor(LC_M), lc_tensor(LC_M))
    @test [num(x, vals) for x in TensND.get_data(A_sym)] ≈ lc_data(A_num) rtol = 1.0e-10
    # conduction with a Kapitza resistance: the fiber k₁/(1 + ρ k₁/R)
    kc = LayeredCylinder((R,), (lc_K(k₁, l₁; axis = ez),); interfaces = (KapitzaInterface(ρ),))
    α = TensND.get_data(gradient_gradient_concentration(kc, lc_K(k₂, l₂; axis = ez), lc_K(k₂, l₂; axis = ez)))[1]
    @test isequal(Symbolics.simplify(α - 2k₂ / (k₂ + k₁ / (1 + ρ * k₁ / R)); expand = true), 0)
    # a conducting surface: k₁ + κ/R
    sc = LayeredCylinder((R,), (lc_K(k₁, l₁; axis = ez),); interfaces = (SurfaceConductiveInterface(κ),))
    N_sc = TensND.get_data(MeanFieldHomogenization.property_contribution(sc, lc_K(k₂, l₂; axis = ez)))
    kq = k₁ + κ / R
    @test isequal(Symbolics.simplify(N_sc[1] - (kq - k₂) * 2k₂ / (k₂ + kq); expand = true), 0)
    @test isequal(Symbolics.simplify(N_sc[2] - (l₁ + 2κ / R - l₂); expand = true), 0)
    # a pointwise field needs its region named: a symbolic radius does not compare
    f = LayeredCylinderFields(cyl, lc_ti(k₂, l₂, m₂, n₂, p₂; axis = ez))
    @test_throws ArgumentError local_gradient_gradient_concentration(f, a / 2, 0.0, 0.0)
    A_loc = local_gradient_gradient_concentration(f, a / 2, 0.3, 0.0; layer = 1)
    f_num = LayeredCylinderFields(LayeredCylinder((0.6, 1.0), (lc_tensor(LC_F), lc_tensor(LC_M))), lc_tensor(LC_M))
    @test [num(x, vals) for x in lc_cc(A_loc)] ≈ collect(lc_cc(local_gradient_gradient_concentration(f_num, 0.3, 0.3, 0.0))) rtol = 1.0e-10
    # a general tensor with symbolic entries is refused, not compared
    @test_throws "symbolic property" LCyl._ti_part(MeanFieldHomogenization.Core._tens_in(Symbolics.Num[k₁ 0 0; 0 k₁ 0; 0 0 l₁], TensND.CanonicalBasis{3, Symbolics.Num}()), ez)
end

@testset "LayeredCylinder — SymPy closed forms" begin
    @syms k1::positive l1::real m1::positive n1::positive p1::positive k2::positive l2::real m2::positive n2::positive p2::positive
    @syms a::positive sn::positive st::positive
    ez = (Sym(0), Sym(0), Sym(1))
    C2 = lc_ti(k2, l2, m2, n2, p2; axis = ez)
    cyl = LayeredCylinder((a, Sym(1)), (lc_ti(k1, l1, m1, n1, p1; axis = ez), C2))
    sol = LCyl._elastic_solution(cyl, C2)
    u, σ = sol.dilatation.outside[2]
    φ, τ = sol.antiplane.outside[2]
    c = a^2
    @test simplify(σ / (2u) - (k2 + c / (1 / (k1 - k2) + (1 - c) / (k2 + m2)))) == 0
    @test simplify(τ / φ - p2 * (p1 * (1 + c) + p2 * (1 - c)) / (p1 * (1 - c) + p2 * (1 + c))) == 0
    # a fiber bonded by springs answers as the fibers k₁/(1 + 2 sₙ k₁/a) and p₁/(1 + sₜ p₁/a)
    bonded = LayeredCylinder((a,), (lc_ti(k1, l1, m1, n1, p1; axis = ez),); interfaces = (SpringInterface(; sn, st),))
    solb = LCyl._elastic_solution(bonded, C2)
    ub, σb = solb.dilatation.outside[1]
    φb, τb = solb.antiplane.outside[1]
    @test simplify(σb / (2ub / a) - k1 / (1 + 2sn * k1 / a)) == 0
    @test simplify(τb / (φb / a) - p1 / (1 + st * p1 / a)) == 0
end
