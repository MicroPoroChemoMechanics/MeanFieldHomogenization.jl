using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using Random: Xoshiro
using StaticArrays: SMatrix
import ForwardDiff
import Symbolics

# =============================================================================
#  Aging viscoelastic operators stored by symmetry blocks (`VolterraOperator`),
#  histories (`TimeHistory`) and their grid (`TimeGrid`).
#
#  Reference: the dense path, which knows nothing of the blocks — the full
#  trapezoidal matrix of the same law (`trapezoidal_matrix`), its matrix
#  products and its block forward substitution (`volterra_inverse`) — in
#  randomly drawn axes and frames.
# =============================================================================

const VO_TIMES = [0.0, 0.4, 1.0, 2.0, 5.0, 9.0]
vo_rng = Xoshiro(20261031)
const VO_AXIS = Tuple(normalize(randn(vo_rng, 3)))
const VO_FRAME = random_basis(vo_rng)
vo_decay(t, tp) = exp(-(t - tp) / 2)

# One aging law per class, of order 4 and 2, in a random axis or frame.
vo_laws() = [
    "iso" => ViscoLaw((t, tp) -> TensISO{3}(3 * (2 + vo_decay(t, tp)), 2 * (1 + vo_decay(t, tp)))),
    "cubic" => ViscoLaw((t, tp) -> TensCubic(4 + vo_decay(t, tp), 2.0, 1 + 0.1tp, VO_FRAME)),
    "TI" => ViscoLaw((t, tp) -> TensTI{4}(5 + vo_decay(t, tp), 4.0, 1.0, 0.5 + 0.1tp, 2 + vo_decay(t, tp), 1.5, VO_AXIS)),
    "ortho" => ViscoLaw((t, tp) -> TensOrtho{4, Float64}((5.0, 6.0, 7.0, 1.0, 2.0, 3.0, 1.5, 2.5, 3.5) .* (1 + vo_decay(t, tp)), VO_FRAME)),
    "dense" => ViscoLaw((t, tp) -> (1 + vo_decay(t, tp)) * mandel(canonical_components(TensTI{4}(5.0, 4.0, 1.0, 2.0, 1.5, VO_AXIS)); symmetrize = true) + 0.1I),
    "iso order 2" => ViscoLaw((t, tp) -> TensISO{3}(2 + vo_decay(t, tp))),
    "TI order 2" => ViscoLaw((t, tp) -> TensTI{2}(2 + vo_decay(t, tp), 1.0 + 0.1tp, VO_AXIS)),
    "ortho order 2" => ViscoLaw((t, tp) -> TensOrtho{2, Float64}((1.0, 2.0, 3.0) .* (1 + vo_decay(t, tp)), VO_FRAME)),
    "scalar" => ViscoLaw((t, tp) -> 2 + vo_decay(t, tp)),
]
vo_dim(R) = size(R, 1) ÷ length(VO_TIMES)

@testset "VolterraOperator: the algebra against the dense matrices" begin
    for (name, law) in vo_laws()
        R = VolterraOperator(law, VO_TIMES)
        D = trapezoidal_matrix(law, VO_TIMES)
        B = vo_dim(R)
        @test Matrix(R) ≈ D rtol = 1.0e-13
        @test R ≈ D
        # The matrix interface: entries, product by a stacked vector.
        @test R[B + 1, 1] ≈ D[B + 1, 1]
        @test R[1, B + 1] == 0
        x = collect(range(-1.0, 1.0; length = size(D, 1)))
        @test R * x ≈ D * x rtol = 1.0e-13
        # Composition with itself and with an isotropic operator, sum, scaling, identity.
        S = VolterraOperator(vo_laws()[B == 6 ? 1 : (B == 3 ? 6 : 9)][2], VO_TIMES)
        @test Matrix(R * R) ≈ D * D rtol = 1.0e-12
        @test Matrix(R * S) ≈ D * Matrix(S) rtol = 1.0e-12
        @test Matrix(S * R) ≈ Matrix(S) * D rtol = 1.0e-12
        @test Matrix(2R - R + I) ≈ D + I rtol = 1.0e-13
        @test Matrix(-R + 3 * R) ≈ 2D rtol = 1.0e-13
        @test Matrix(one(R)) ≈ Matrix(1.0I, size(D)...) rtol = 1.0e-14
        @test iszero(Matrix(zero(R)))
        # The Volterra inverse, block by block, against the block forward substitution.
        @test Matrix(inv(R)) ≈ volterra_inverse(D; block_size = B) rtol = 1.0e-12
        @test Matrix(volterra_inverse(R)) ≈ Matrix(inv(R))
        @test_throws ArgumentError volterra_inverse(R; block_size = B + 1)
        # Back to a law of (t, t′), and to the same operator.
        L = operator_law(R)
        @test Matrix(VolterraOperator(L, VO_TIMES)) ≈ D rtol = 1.0e-12
        @test time_grid(R) == TimeGrid(VO_TIMES)
        @test occursin("VolterraOperator", sprint(show, R))
        @test occursin("VolterraOperator", sprint(show, MIME"text/plain"(), R))
        # A class operator compared with the dense one of the same values.
        @test R == VolterraOperator(Matrix(R), VO_TIMES) || !(R.class isa DenseClass)
    end
end

@testset "VolterraOperator: storage by blocks of the class" begin
    sizes(R) = map(B -> size(B, 1) ÷ length(VO_TIMES), R.blocks)
    expected = Dict(
        "iso" => (1, 1), "cubic" => (1, 1, 1), "TI" => (2, 1, 1), "ortho" => (3, 1, 1, 1), "dense" => (6,),
        "iso order 2" => (1,), "TI order 2" => (1, 1), "ortho order 2" => (1, 1, 1), "scalar" => (1,),
    )
    for (name, law) in vo_laws()
        @test sizes(VolterraOperator(law, VO_TIMES)) == expected[name]
    end
    # The value at a pair of instants is a tensor of the class.
    R = VolterraOperator(vo_laws()[3][2], VO_TIMES)
    @test operator_value(R, 3, 2) isa TensTI
    @test mandel(canonical_components(operator_value(R, 3, 2)); symmetrize = true) ≈
        trapezoidal_matrix(vo_laws()[3][2], VO_TIMES)[13:18, 7:12]
end

@testset "VolterraOperator: a matrix given on the grid, in a class" begin
    for (name, law) in vo_laws()[1:4]
        R = VolterraOperator(law, VO_TIMES)
        D = trapezoidal_matrix(law, VO_TIMES)
        # The dense matrix stored in the class of a tensor of that class.
        C = VolterraOperator(D, VO_TIMES; like = R.class)
        @test typeof(C.class) == typeof(R.class)
        @test Matrix(C) ≈ D rtol = 1.0e-12
        @test convert_class(R, DenseClass{4}()) ≈ D
    end
    # A matrix off the class is refused, not projected.
    D = trapezoidal_matrix(vo_laws()[3][2], VO_TIMES)
    @test_throws "does not belong to the isotropic class" VolterraOperator(D, VO_TIMES; like = TensISO{3}(1.0, 1.0))
    # Without a check, the projection is taken.
    @test convert_class(VolterraOperator(D, VO_TIMES), TensISO{3}(1.0, 1.0); check = false) isa VolterraOperator
    @test_throws DimensionMismatch VolterraOperator(zeros(7, 7), VO_TIMES)
end

@testset "VolterraOperator: mixed classes and constants" begin
    iso, ti = VolterraOperator(vo_laws()[1][2], VO_TIMES), VolterraOperator(vo_laws()[3][2], VO_TIMES)
    ortho = VolterraOperator(vo_laws()[4][2], VO_TIMES)
    # An isotropic operator joins the class of the other; two classes meet densely.
    @test (iso + ti).class isa TensTI
    @test (ti * iso).class isa TensTI
    @test (ti * ortho).class isa DenseClass
    @test Matrix(ti * ortho) ≈ Matrix(ti) * Matrix(ortho) rtol = 1.0e-12
    # Written as in elasticity.
    C = TensISO{3}(3.0, 2.0)
    @test Matrix(ti ⊡ C) ≈ Matrix(ti) * kron(Matrix(1.0I, 6, 6), mandel(C)) rtol = 1.0e-12
    @test Matrix(C ⊡ ti) ≈ kron(Matrix(1.0I, 6, 6), mandel(C)) * Matrix(ti) rtol = 1.0e-12
    @test Matrix(ti + C) ≈ Matrix(ti) + kron(Matrix(1.0I, 6, 6), mandel(C))
    @test Matrix(C - ti) ≈ kron(Matrix(1.0I, 6, 6), mandel(C)) - Matrix(ti)
    @test Matrix(ti ⊡ ti) ≈ Matrix(ti * ti)
    @test Matrix(I - ti) ≈ I - Matrix(ti)
    @test Matrix(ti + Matrix(ti)) ≈ 2Matrix(ti)
    @test Matrix(Matrix(ti) - ti) ≈ zeros(36, 36) atol = 1.0e-12
    @test ti * Matrix(iso) ≈ Matrix(ti) * Matrix(iso)
    @test Matrix(iso) * ti ≈ Matrix(iso) * Matrix(ti)
    @test isapprox(Matrix(ti), ti)
    k2 = VolterraOperator(vo_laws()[7][2], VO_TIMES)
    @test Matrix(k2 ⋅ k2) ≈ Matrix(k2) * Matrix(k2)
    @test Matrix(k2 + TensISO{3}(1.0)) ≈ Matrix(k2) + I
    # The order of the composition symbol is checked; operators of two orders do not mix.
    @test_throws "composes operators of order 4" k2 ⊡ k2
    @test_throws "composes operators of order 2" ti ⋅ ti
    @test_throws "do not combine" ti + k2
    # Two grids do not mix.
    other = VolterraOperator(vo_laws()[1][2], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    @test_throws "different time grids" iso + other
end

@testset "VolterraOperator: two frames of one class, norms" begin
    ti = VolterraOperator(vo_laws()[3][2], VO_TIMES)
    other = VolterraOperator(ViscoLaw((t, tp) -> TensTI{4}(5 + vo_decay(t, tp), 4.0, 1.0, 2.0, 1.5, (0.0, 0.0, 1.0))), VO_TIMES)
    # Two transversely isotropic operators about different axes meet in the full blocks.
    @test (ti * other).class isa DenseClass
    @test Matrix(ti * other) ≈ Matrix(ti) * Matrix(other) rtol = 1.0e-12
    @test Matrix(ti + other) ≈ Matrix(ti) + Matrix(other) rtol = 1.0e-13
    # The norm of the full matrix, from the blocks.
    for (name, law) in vo_laws()
        R = VolterraOperator(law, VO_TIMES)
        @test norm(R) ≈ norm(Matrix(R)) rtol = 1.0e-13
    end
end

@testset "TimeGrid and TimeHistory" begin
    g = TimeGrid(VO_TIMES)
    @test length(g) == 6 && g[2] == 0.4 && collect(g) == VO_TIMES && eltype(g) == Float64
    @test TimeGrid(g) === g
    @test sum(g) ≈ sum(VO_TIMES)
    @test occursin("6 instants", sprint(show, g))
    @test_throws "increase strictly" TimeGrid([0.0, 1.0, 1.0])
    @test_throws "no instant" TimeGrid(Float64[])
    Symbolics.@variables τ
    @test_throws "numbers that compare" TimeGrid([τ, 2τ])
    # A strain history, as a function or as values, and the response of each class.
    ε(t) = TensND.Tens(Symmetric([1.0e-3t 2.0e-4 0.0; 2.0e-4 -5.0e-4 1.0e-4sin(t); 0.0 1.0e-4sin(t) 3.0e-4]))
    h = TimeHistory(ε, VO_TIMES)
    @test h == h && length(h) == 6
    @test TimeHistory([ε(t) for t in VO_TIMES], g).values == h.values
    @test TimeHistory(Matrix(h), g).values == h.values
    @test time_grid(h) == g
    @test h[2] ≈ mandel(canonical_components(ε(0.4)); symmetrize = true)
    @test h[end] == h[6] && h[begin] == h[1] && g[end] == 9.0 && g[begin] == 0.0
    @test h ≈ TimeHistory(Matrix(h) .+ 1.0e-20, g)
    @test occursin("6 components on 6 instants", sprint(show, h))
    for (name, law) in vo_laws()
        R = VolterraOperator(law, VO_TIMES)
        d = vo_dim(R)
        hd = d == 6 ? h : TimeHistory(t -> d == 3 ? [t, 1.0, -t^2] : [t], VO_TIMES)
        y = R * hd
        @test vec(Matrix(y)) ≈ trapezoidal_matrix(law, VO_TIMES) * vec(Matrix(hd)) rtol = 1.0e-12
        d == 6 && @test (R ⊡ hd).values ≈ y.values
        d == 3 && (R.class isa ScalarClass || @test (R ⋅ hd).values ≈ y.values)
    end
    # The strain of a held stress, through the inverse.
    R = VolterraOperator(vo_laws()[3][2], VO_TIMES)
    σ = R ⊡ h
    @test (inv(R) ⊡ σ).values ≈ h.values rtol = 1.0e-10
    @test_throws DimensionMismatch R * TimeHistory(t -> [t, 0.0, 0.0], VO_TIMES)
    @test_throws DimensionMismatch TimeHistory(zeros(4, 6), VO_TIMES)
    @test_throws DimensionMismatch R * ones(5)
end

@testset "VolterraOperator: a law that changes class is refused" begin
    law = ViscoLaw((t, tp) -> t > 1 ? TensTI{4}(5.0, 4.0, 1.0, 2.0, 1.5, VO_AXIS) : TensISO{3}(3.0, 2.0))
    @test_throws "changes its symmetry class" VolterraOperator(law, VO_TIMES)
    law2 = ViscoLaw((t, tp) -> TensTI{4}(5.0, 4.0, 1.0, 2.0, 1.5, t > 1 ? (0.0, 0.0, 1.0) : (1.0, 0.0, 0.0)))
    @test_throws "changes its symmetry class" VolterraOperator(law2, VO_TIMES)
    @test_throws "6×6 (Kelvin-Mandel" VolterraOperator(ViscoLaw((t, tp) -> zeros(2, 2)), VO_TIMES)
    # A value off the class, given as a matrix, is refused; one in it is read.
    law3 = ViscoLaw((t, tp) -> t > 1 ? mandel(canonical_components(TensTI{4}(5.0, 4.0, 1.0, 2.0, 1.5, VO_AXIS)); symmetrize = true) : TensISO{3}(3.0, 2.0))
    @test_throws "changes its symmetry class" VolterraOperator(law3, VO_TIMES)
    law4 = ViscoLaw((t, tp) -> t > 1 ? mandel(canonical_components(TensISO{3}(3.0, 2.0)); symmetrize = true) : TensISO{3}(3.0, 2.0))
    @test VolterraOperator(law4, VO_TIMES).class isa TensISO
    @test Matrix(VolterraOperator(law4, VO_TIMES)) ≈ trapezoidal_matrix(law4, VO_TIMES) rtol = 1.0e-13
    # A law zero at first (a phase that forms later) takes the class of its later values.
    late = precipitated(vo_laws()[3][2], 1.5)
    @test VolterraOperator(late, VO_TIMES).class isa TensTI
    # A law of generic tensors (no class) gives a dense operator, at both orders.
    for C in (TensTI{4}(5.0, 4.0, 1.0, 2.0, 1.5, VO_AXIS), TensTI{2}(2.0, 1.0, VO_AXIS))
        law = ViscoLaw((t, tp) -> (1 + vo_decay(t, tp)) * generic_tens(C))
        R = VolterraOperator(law, VO_TIMES)
        @test R.class isa DenseClass
        @test Matrix(R) ≈ trapezoidal_matrix(law, VO_TIMES) rtol = 1.0e-13
    end
end

@testset "VolterraOperator: derivatives and symbols" begin
    # Derivative of the creep operator with respect to a modulus, against the same
    # derivative through the dense path.
    law(k) = ViscoLaw((t, tp) -> TensTI{4}(k + vo_decay(t, tp), 4.0, 1.0, 2.0, 1.5 * k, VO_AXIS))
    blocks(k) = Matrix(inv(VolterraOperator(law(k), VO_TIMES)))
    dense(k) = volterra_inverse(trapezoidal_matrix(law(k), VO_TIMES))
    @test ForwardDiff.derivative(blocks, 5.0) ≈ ForwardDiff.derivative(dense, 5.0) rtol = 1.0e-10
    # Nested: a second derivative.
    @test ForwardDiff.derivative(k -> ForwardDiff.derivative(blocks, k), 5.0) ≈
        ForwardDiff.derivative(k -> ForwardDiff.derivative(dense, k), 5.0) rtol = 1.0e-8
    # Symbols: the composition of two isotropic operators, entry by entry against the dense product.
    Symbolics.@variables a b
    times = [0.0, 1.0]
    R = VolterraOperator(TensISO{3}(a, b), times)
    S = VolterraOperator(ViscoLaw((t, tp) -> TensISO{3}(a + t, b * (1 + tp))), times)
    # The adapted basis enters with floating coefficients (1/√3, 1/√6): the
    # expressions are compared at values, against the same product at numbers.
    vals = Dict(a => 2.3, b => 1.7)
    num(x) = Symbolics.value(Symbolics.substitute(x, vals; fold = Val(true)))
    Rn = VolterraOperator(TensISO{3}(2.3, 1.7), times)
    Sn = VolterraOperator(ViscoLaw((t, tp) -> TensISO{3}(2.3 + t, 1.7 * (1 + tp))), times)
    @test num.(Matrix(R * S)) ≈ Matrix(Rn * Sn) rtol = 1.0e-14
    @test num.(Matrix(inv(S))) ≈ Matrix(inv(Sn)) rtol = 1.0e-13
    # Divided by a symbol, an operator keeps its class (the Symbolics extension).
    @test (R / a).class isa TensISO
    @test num.(Matrix(R / a)) ≈ Matrix(Rn) / 2.3
    @test num.(Matrix(a * R)) ≈ 2.3 * Matrix(Rn)
    # A symbolic operator is never compacted by a residual it cannot judge.
    @test MeanFieldHomogenization.Viscoelasticity._compact(VolterraOperator(Matrix(R), times), [TensISO{3}(1.0, 1.0)]).class isa DenseClass
    # The values of a symbolic operator are taken in its class as given.
    @test convert_class(VolterraOperator(Matrix(R), times), TensISO{3}(1.0, 1.0)).class isa TensISO
end

@testset "VolterraOperator: precipitation and dissolution" begin
    # On the operator, as on its law: the same discretized kernel.
    for (name, law) in vo_laws()[[1, 3, 6]]
        R = VolterraOperator(law, VO_TIMES)
        @test precipitated(R, 1.0) ≈ VolterraOperator(precipitated(law, 1.0), VO_TIMES) rtol = 1.0e-12
        @test dissolved(R, 2.0) ≈ VolterraOperator(dissolved(law, 2.0), VO_TIMES) rtol = 1.0e-12
        @test typeof(precipitated(R, 1.0).class) == typeof(R.class)
    end
    # A grain dissolving while a phase precipitates in its place: the sum of operators.
    g = VolterraOperator(TensISO{3}(30.0, 20.0), VO_TIMES)
    f = VolterraOperator(vo_laws()[1][2], VO_TIMES)
    both = ViscoLaw((t, tp) -> visco_eval(dissolved(heaviside_law(TensISO{3}(30.0, 20.0)), 2.0), t, tp) + visco_eval(precipitated(vo_laws()[1][2], 2.0), t, tp))
    @test dissolved(g, 2.0) + precipitated(f, 2.0) ≈ VolterraOperator(both, VO_TIMES) rtol = 1.0e-12
end

@testset "Conduction in the classes of the inclusions" begin
    V = MeanFieldHomogenization.Viscoelasticity
    times = VO_TIMES
    k = ViscoLaw((t, tp) -> TensISO{3}(2.0 + exp(tp - t)))
    angles = (0.4, 0.9, 0.2)
    Q = vecbasis(RotatedBasis(angles...))
    rotated(M) = (R = kron(Matrix(1.0I, length(times), length(times)), Q); R * M * transpose(R))
    function cell(shape; sym = nothing)
        r = RVE(; distribution_shape = Ellipsoid(1.0))
        add_phase!(r, :M, Ellipsoid(1.0), Dict(:C => k); fraction = :rest)
        add_phase!(r, :I, shape, Dict(:C => TensISO{3}(9.0)); fraction = 0.3, symmetrize = sym)
        return r
    end
    # A tilted spheroid: transversely isotropic about its axis; a tilted triaxial
    # ellipsoid: orthotropic in its frame. Each the aligned result, rotated.
    for (axes, class) in (((1.0, 1.0, 5.0), TensTI), ((2.0, 1.0, 0.4), TensOrtho))
        for scheme in (Dilute(), MoriTanaka(), Maxwell())
            tilted = homogenize_alv(cell(Ellipsoid(axes...; euler_angles = angles)), scheme, :C; times)
            @test tilted.class isa class
            aligned = Matrix(homogenize_alv(cell(Ellipsoid(axes...)), scheme, :C; times))
            scheme isa Maxwell || @test Matrix(tilted) ≈ rotated(aligned) rtol = 1.0e-12
        end
    end
    # The isotropic orientation average, read on the blocks, against the dense projector.
    avg = homogenize_alv(cell(Ellipsoid(1.0, 1.0, 5.0; euler_angles = angles); sym = :iso), MoriTanaka(), :C; times)
    @test avg.class isa TensISO
    P = V._hill_kernel_order2_op(Ellipsoid(1.0, 1.0, 5.0; euler_angles = angles), VolterraOperator(k, times))
    @test Matrix(V._iso_part2(P)) ≈ V._iso_project_blocks3(Matrix(P)) rtol = 1.0e-13
    # A frame that is not orthonormal: the full matrix, the dense kernel.
    skew = Ellipsoid(2.0, 1.0, 0.5, TensND.Basis([1.0 0.3 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]))
    K0 = VolterraOperator(k, times)
    @test V._depolarization_blocks(skew)[1] isa DenseClass
    @test Matrix(V._hill_kernel_order2_op(skew, K0)) ≈ V.hill_kernel_order2_at(skew, Matrix(K0)) rtol = 1.0e-13
    # Components that do not compare are kept full.
    Symbolics.@variables a
    @test V._depolarization_blocks(nothing, SMatrix{3, 3}(a, 0, 0, 0, a, 0, 0, 0, 2a), skew)[1] isa DenseClass
    # A reference of another class is refused, as by the dense kernel: the time-space
    # decoupling holds for an isotropic reference only.
    Kti = VolterraOperator(ViscoLaw((t, tp) -> TensTI{2}(2.0 + exp(tp - t), 3.0, (0.0, 0.0, 1.0))), times)
    @test_throws "only iso ALV matrix" V._hill_kernel_order2_op(Ellipsoid(1.0), Kti)
end

@testset "The modes of a law, at both orders" begin
    # `:property` and `:inverse_property` are `:relaxation` and `:creep` under the names that
    # hold at every order: a conductivity and a resistivity at order 2.
    times = VO_TIMES
    k(t, tp) = TensISO{3}(2.0 + exp(tp - t))
    @test visco_mode(ViscoLaw(k, :property)) === :relaxation
    @test visco_mode(ViscoLaw(k, :inverse_property)) === :creep
    @test visco_mode(ViscoLaw(k, :conductivity)) === :relaxation
    @test visco_mode(ViscoLaw(k, :resistivity)) === :creep
    @test_throws "mode must be one of" ViscoLaw(k, :stiffness_of_sorts)
    cell(law) = (
        r = RVE(); add_phase!(r, :M, Ellipsoid(1.0), Dict(:C => law); fraction = :rest);
        add_phase!(r, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(9.0)); fraction = 0.3); r
    )
    as_property = homogenize_alv(cell(ViscoLaw(k, :property)), MoriTanaka(), :C; times)
    @test Matrix(as_property) == Matrix(homogenize_alv(cell(ViscoLaw(k, :relaxation)), MoriTanaka(), :C; times))
    # A resistivity: its operator inverted is the conductivity the scheme reads.
    ρ = ViscoLaw(k, :resistivity)
    K = inv(VolterraOperator(ρ, times))
    @test Matrix(homogenize_alv(cell(ρ), MoriTanaka(), :C; times)) ≈ Matrix(homogenize_alv(cell(K), MoriTanaka(), :C)) rtol = 1.0e-13
end

@testset "VolterraOperator: inference" begin
    R = VolterraOperator(vo_laws()[3][2], VO_TIMES)
    D = trapezoidal_matrix(vo_laws()[3][2], VO_TIMES)
    # One class, or the full blocks when the frames differ (a run-time value): a
    # union of two concrete types.
    Dn = typeof(convert_class(R, DenseClass{4}()))
    @inferred Union{typeof(R), Dn} R * R
    @inferred Union{typeof(R), Dn} R + R
    @inferred inv(R)
    @inferred Matrix(R)
    @inferred VolterraOperator(vo_laws()[3][2], VO_TIMES)
    @inferred R * TimeHistory(D[1:6, 1:6], VO_TIMES)
end

@testset "homogenize_alv: operators as properties, compact results" begin
    times = VO_TIMES
    m = maxwell_iso(10.0, 6.0, 0.8, 2.5)
    E_I = heaviside_law(TensISO{3}(30.0, 20.0))
    function cell(Cm, Ci, geom)
        r = RVE()
        add_phase!(r, :M, Ellipsoid(1.0), Dict(:C => Cm); fraction = :rest)
        add_phase!(r, :I, geom, Dict(:C => Ci); fraction = 0.25)
        return r
    end
    # Spheres: an isotropic result, stored by its two blocks.
    R = homogenize_alv(cell(m, E_I, Ellipsoid(1.0)), MoriTanaka(), :C; times)
    @test R isa VolterraOperator && R.class isa TensISO
    # A phase given by its operator gives the same numbers, and the grid is read on it.
    Rm = VolterraOperator(m, times)
    R′ = homogenize_alv(cell(Rm, E_I, Ellipsoid(1.0)), MoriTanaka(), :C)
    @test Matrix(R′) ≈ Matrix(R) rtol = 1.0e-14
    @test time_grid(R′) == time_grid(R)
    @test has_visco_property(cell(Rm, E_I, Ellipsoid(1.0)))
    # A nested cell homogenized on the grid, as the explicit two steps.
    outer(Ci) = (
        r = RVE(); add_phase!(r, :M, Ellipsoid(1.0), Dict(:C => m); fraction = :rest);
        add_phase!(r, :A, Ellipsoid(1.0), Dict(:C => Ci); fraction = 0.4); r
    )
    nested = homogenize_alv(outer(Homogenized(cell(m, E_I, Ellipsoid(1.0)), MoriTanaka())), MoriTanaka(), :C; times)
    @test Matrix(nested) ≈ Matrix(homogenize_alv(outer(R), MoriTanaka(), :C; times)) rtol = 1.0e-13
    # Spheroids along e₃: transversely isotropic about e₃.
    @test homogenize_alv(cell(m, E_I, Spheroid(4.0)), MoriTanaka(), :C; times).class isa TensTI
    # Tilted spheroids of a phase transversely isotropic about their own axis: the
    # class of that phase.
    angles = (0.7, 0.4, 0.0)
    n = Tuple(vecbasis(RotatedBasis(angles...))[:, 3])
    E_TI = heaviside_law(TensTI{4}(40.0, 30.0, 8.0, 12.0, 9.0, n))
    Rt = homogenize_alv(cell(m, E_TI, Spheroid(4.0; euler_angles = angles)), MoriTanaka(), :C; times)
    @test Rt.class isa TensTI && collect(Rt.class.n) ≈ collect(n)
    # Tilted isotropic spheroids: transversely isotropic about their axis, computed
    # in that class; the result is the one of spheroids along e₃, rotated block by block.
    tilted = homogenize_alv(cell(m, E_I, Spheroid(4.0; euler_angles = angles)), MoriTanaka(), :C; times)
    @test tilted.class isa TensTI && abs(dot(collect(tilted.class.n), collect(n))) ≈ 1
    Q6 = TensND.mandel_rotation(vecbasis(RotatedBasis(angles...)))
    aligned = Matrix(homogenize_alv(cell(m, E_I, Spheroid(4.0)), MoriTanaka(), :C; times))
    @test Matrix(tilted) ≈ kron(Matrix(1.0I, 6, 6), Q6) * aligned * kron(Matrix(1.0I, 6, 6), transpose(Q6)) rtol = 1.0e-12
    # Conduction: the order-2 classes.
    k = ViscoLaw((t, tp) -> TensISO{3}(2 + vo_decay(t, tp)))
    @test homogenize_alv(cell(k, heaviside_law(TensISO{3}(5.0)), Ellipsoid(1.0)), MoriTanaka(), :C; times).class isa TensISO{2}
    # An elastic tensor (a conductivity at order 2) as a phase property: the kernel
    # C H(t - t′), in the class of C, as `heaviside_law(C)` gives it.
    for (Cm, C) in (
            (m, TensTI{4}(40.0, 30.0, 8.0, 12.0, 9.0, (0.0, 0.0, 1.0))),
            (k, TensTI{2}(5.0, 9.0, (0.0, 0.0, 1.0))),
        )
        direct = homogenize_alv(cell(Cm, C, Ellipsoid(1.0)), MoriTanaka(), :C; times)
        @test direct.class isa TensTI
        @test Matrix(direct) == Matrix(homogenize_alv(cell(Cm, heaviside_law(C), Ellipsoid(1.0)), MoriTanaka(), :C; times))
    end
    # In the self-consistent scheme, and as the matrix of a cell.
    sc_cell(Cm, Ci) = (
        r = RVE(); add_phase!(r, :M, Ellipsoid(1.0), Dict(:C => Cm); fraction = :rest);
        add_phase!(r, :I, Ellipsoid(1.0), Dict(:C => Ci); fraction = 0.4); r
    )
    E = TensISO{3}(3 * 30.0, 2 * 20.0)
    @test Matrix(homogenize_alv(sc_cell(m, E), SelfConsistent(), :C; times)) ≈
        Matrix(homogenize_alv(sc_cell(m, heaviside_law(E)), SelfConsistent(), :C; times)) rtol = 1.0e-14
    @test Matrix(homogenize_alv(sc_cell(E, m), MoriTanaka(), :C; times)) ≈
        Matrix(homogenize_alv(sc_cell(heaviside_law(E), m), MoriTanaka(), :C; times)) rtol = 1.0e-14
    # No grid, two grids.
    @test_throws "no time grid" homogenize_alv(cell(m, E_I, Ellipsoid(1.0)), MoriTanaka(), :C)
    @test_throws "different time grids" homogenize_alv(cell(Rm, E_I, Ellipsoid(1.0)), MoriTanaka(), :C; times = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0])
end
