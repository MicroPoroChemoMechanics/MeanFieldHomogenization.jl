using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using ForwardDiff

# =============================================================================
#  test_layered_cylinder_alv_schemes.jl — `LayeredCylinder` phases in the
#  schemes of `homogenize_alv`, orders 4 (elasticity) and 2 (conduction).
#
#  The references are independent of the code under test: the elastic
#  `homogenize` in the Heaviside limit, the homogeneous `Cylinder` through the
#  ALV Hill kernel, and exact identities (a homogeneous cell returns its law,
#  the self-consistent equation, the two formulations of the differential
#  scheme, Mori–Tanaka against Maxwell). Outside the elastic limit the laws age
#  and the grid is not uniform, so that the Volterra matrices do not commute and
#  a product written in the wrong order fails.
# =============================================================================

const LCS_TIMES = [0.0, 0.1, 0.35, 0.8]
const LCS_C = (M = TensISO{3}(9.0, 4.0), core = TensISO{3}(30.0, 12.0), coat = TensISO{3}(6.0, 3.0))
const LCS_K = (M = TensISO{3}(1.0), core = TensISO{3}(5.0), coat = TensISO{3}(0.5))
# Axis of `Elasticity.Cylinder`: column 1 of its basis, sent on e₃.
const LCS_R = [0.0 1.0 0.0; 0.0 0.0 1.0; 1.0 0.0 0.0]

lcs_moduli(prop::Symbol) = prop === :C ? LCS_C : LCS_K
# An aging law: the relaxation depends on the age t′ of loading.
lcs_aging(C, τ) = ViscoLaw((t, tp) -> t >= tp ? (1 + 0.5tp) * exp(-(t - tp) / τ) * C : 0.0 * C)
lcs_coated(core, coat) = LayeredCylinder((0.7, 1.0), (core, coat))
lcs_single(law) = LayeredCylinder((1.0,), (law,))
lcs_rel(a, b) = norm(Matrix(a) - Matrix(b)) / norm(Matrix(b))
# The relaxation matrix of an elastic tensor: the Heaviside kernel on the grid.
lcs_elastic(P, times) = Matrix(VolterraOperator(P, TimeGrid(times)))

# A host `host` of property `M` and one inclusion phase `geom` at 30 %.
function lcs_cell(prop::Symbol, host, M, geom; f = 0.3, symmetrize = nothing, dist = Ellipsoid(1.0, 1.0, 1.0), placeholder = M)
    r = RVE(; distribution_shape = dist)
    add_phase!(r, :M, host, Dict(prop => M); fraction = :rest)
    symmetrize === nothing ? add_phase!(r, :F, geom, Dict(prop => placeholder); fraction = f) :
        add_phase!(r, :F, geom, Dict(prop => placeholder); fraction = f, symmetrize)
    return r
end
# The cell whose every phase is a cylinder: a one-layer host and a second phase.
lcs_fibers(prop::Symbol, M, geom; kw...) = lcs_cell(prop, lcs_single(M), M, geom; kw...)

@testset "LayeredCylinder among other phases — closed-form schemes" begin
    schemes = (Voigt(), Reuss(), Dilute(), DiluteDual(), MoriTanaka(), Maxwell(), PonteCastanedaWillis())
    @testset "elastic limit, $prop, symmetrize = $sym" for prop in (:C, :K), sym in (nothing, :iso)
        P = lcs_moduli(prop)
        for scheme in schemes
            el = homogenize(lcs_cell(prop, Ellipsoid(1.0), P.M, lcs_coated(P.core, P.coat); symmetrize = sym), scheme, prop)
            R = homogenize_alv(
                lcs_cell(prop, Ellipsoid(1.0), heaviside_law(P.M), lcs_coated(heaviside_law(P.core), heaviside_law(P.coat)); symmetrize = sym),
                scheme, prop; times = LCS_TIMES
            )
            @test lcs_rel(R, lcs_elastic(el, LCS_TIMES)) < 1.0e-12
        end
    end

    @testset "one layer ≡ homogeneous Cylinder through its Hill kernel, aging, $prop" for prop in (:C, :K)
        P = lcs_moduli(prop)
        M, L = lcs_aging(P.M, 0.8), lcs_aging(P.core, 0.3)
        hom = MeanFieldHomogenization.Elasticity.Cylinder(1.0, 1.0, LCS_R)
        for scheme in (Dilute(), DiluteDual(), MoriTanaka(), Maxwell(), PonteCastanedaWillis())
            layered = homogenize_alv(lcs_cell(prop, Ellipsoid(1.0), M, lcs_single(L)), scheme, prop; times = LCS_TIMES)
            reference = homogenize_alv(lcs_cell(prop, Ellipsoid(1.0), M, hom; placeholder = L), scheme, prop; times = LCS_TIMES)
            @test lcs_rel(layered, reference) < 1.0e-12
            # Transversely isotropic about the axis, and not isotropic.
            @test layered.class isa TensTI
        end
        # A distribution of the shape of the inclusions: Maxwell is Mori–Tanaka.
        coated = lcs_coated(L, lcs_aging(P.coat, 1.3))
        mt = homogenize_alv(lcs_cell(prop, Ellipsoid(1.0), M, coated; dist = hom), MoriTanaka(), prop; times = LCS_TIMES)
        mx = homogenize_alv(lcs_cell(prop, Ellipsoid(1.0), M, coated; dist = hom), Maxwell(), prop; times = LCS_TIMES)
        @test lcs_rel(mx, mt) < 1.0e-12
    end
end

@testset "LayeredCylinder among other phases — schemes on the running estimate" begin
    running = (SelfConsistent(), AsymmetricSelfConsistent(), DifferentialScheme(), DifferentialScheme(; formulation = :compliance))

    @testset "elastic limit under an isotropic average, $prop" for prop in (:C, :K)
        P = lcs_moduli(prop)
        times = [0.0, 0.4, 1.5]
        for scheme in (running..., SelfConsistent(; algorithm = NewtonDefault()))
            el = homogenize(lcs_cell(prop, Ellipsoid(1.0), P.M, lcs_coated(P.core, P.coat); symmetrize = :iso), scheme, prop)
            R = homogenize_alv(
                lcs_cell(prop, Ellipsoid(1.0), heaviside_law(P.M), lcs_coated(heaviside_law(P.core), heaviside_law(P.coat)); symmetrize = :iso),
                scheme, prop; times
            )
            @test R.class isa TensISO
            @test lcs_rel(R, lcs_elastic(el, times)) < 1.0e-7
        end
    end

    @testset "a homogeneous aging cell returns its law, $prop" for prop in (:C, :K)
        M = lcs_aging(lcs_moduli(prop).M, 0.8)
        law = Matrix(VolterraOperator(M, TimeGrid(LCS_TIMES)))
        for scheme in running
            R = homogenize_alv(lcs_cell(prop, Ellipsoid(1.0), M, lcs_single(M); symmetrize = :iso), scheme, prop; times = LCS_TIMES)
            @test lcs_rel(R, law) < 1.0e-9
        end
    end

    @testset "refused without an isotropic average, $prop" for prop in (:C, :K)
        P = lcs_moduli(prop)
        cell = lcs_cell(prop, Ellipsoid(1.0), lcs_aging(P.M, 0.8), lcs_coated(lcs_aging(P.core, 0.3), heaviside_law(P.coat)))
        for scheme in (SelfConsistent(), AsymmetricSelfConsistent(), DifferentialScheme())
            @test_throws "out of the isotropic class" homogenize_alv(cell, scheme, prop; times = LCS_TIMES)
        end
    end

    @testset "a layered host among phases with a Hill kernel is refused" begin
        M = lcs_aging(LCS_C.M, 0.8)
        cell = lcs_cell(:C, lcs_single(M), M, Ellipsoid(1.0); placeholder = lcs_aging(LCS_C.core, 0.3))
        @test_throws "has no ALV Hill kernel" homogenize_alv(cell, SelfConsistent(), :C; times = LCS_TIMES)
    end

    @testset "derivative with respect to a layer modulus, differential scheme" begin
        # A sphere and an isotropically averaged cylinder; the dual number sits in
        # the core of the cylinder only.
        function estimate(x)
            M = lcs_aging(LCS_C.M, 0.8)
            r = lcs_cell(:C, Ellipsoid(1.0), M, lcs_coated(lcs_aging(x * LCS_C.core, 0.3), heaviside_law(LCS_C.coat)); f = 0.2, symmetrize = :iso)
            add_phase!(r, :S, Ellipsoid(1.0), Dict(:C => heaviside_law(LCS_C.core)); fraction = 0.1)
            return Matrix(homogenize_alv(r, DifferentialScheme(; abstol = 1.0e-11, reltol = 1.0e-11), :C; times = LCS_TIMES))
        end
        h = 1.0e-4
        fd = (estimate(1 + h) - estimate(1 - h)) / 2h
        @test norm(ForwardDiff.derivative(estimate, 1.0) - fd) < 1.0e-6 * norm(fd)
    end
end

@testset "Cell of LayeredCylinders — transversely isotropic estimate" begin
    running = (SelfConsistent(), AsymmetricSelfConsistent(), DifferentialScheme(), DifferentialScheme(; formulation = :compliance))
    tight = (; abstol = 1.0e-13, reltol = 1.0e-12)

    @testset "elastic limit, $prop" for prop in (:C, :K)
        P = lcs_moduli(prop)
        for scheme in running
            el = homogenize(lcs_fibers(prop, P.M, lcs_coated(P.core, P.coat)), scheme, prop)
            R = homogenize_alv(
                lcs_fibers(prop, heaviside_law(P.M), lcs_coated(heaviside_law(P.core), heaviside_law(P.coat))),
                scheme, prop; times = LCS_TIMES
            )
            @test R.class isa TensTI
            @test lcs_rel(R, lcs_elastic(el, LCS_TIMES)) < 1.0e-7
        end
    end

    @testset "a homogeneous aging cell returns its law, $prop" for prop in (:C, :K)
        # A transversely isotropic law about e₃: an elastic estimate of the fibers,
        # aging.
        P = lcs_moduli(prop)
        M = lcs_aging(homogenize(lcs_fibers(prop, P.M, lcs_coated(P.core, P.coat)), MoriTanaka(), prop), 0.8)
        law = Matrix(VolterraOperator(M, TimeGrid(LCS_TIMES)))
        for scheme in running
            # The property declared on the inclusion phase is a placeholder.
            R = homogenize_alv(lcs_fibers(prop, M, lcs_single(M); placeholder = heaviside_law(P.core)), scheme, prop; times = LCS_TIMES)
            @test R.class isa TensTI
            @test lcs_rel(R, law) < 1.0e-9
        end
        # The host alone: no inclusion phase to sum over.
        alone = RVE()
        add_phase!(alone, :M, lcs_single(M), Dict(prop => M); fraction = :rest)
        @test lcs_rel(homogenize_alv(alone, AsymmetricSelfConsistent(), prop; times = LCS_TIMES), law) < 1.0e-12
    end

    @testset "self-consistent equation, aging, $prop" for prop in (:C, :K)
        P = lcs_moduli(prop)
        M = lcs_aging(P.M, 0.8)
        fiber = lcs_coated(lcs_aging(P.core, 0.3), lcs_aging(P.coat, 1.3))
        cell = lcs_fibers(prop, M, fiber)
        X = Matrix(homogenize_alv(cell, SelfConsistent(; tight...), prop; times = LCS_TIMES))
        # The equation in the form Σ f Ñ(X) = 0, with Ñ = B̃ − X ∘ Ã the contribution of
        # each cylinder against the estimate, and the strain average Σ f Ã = 𝟙 it
        # implies, both from the public kernels of the cylinder.
        A_M = gradient_gradient_concentration_alv(lcs_single(M), X, LCS_TIMES)
        A_F = gradient_gradient_concentration_alv(fiber, X, LCS_TIMES)
        N = 0.7 * property_contribution_alv(lcs_single(M), X, LCS_TIMES) + 0.3 * property_contribution_alv(fiber, X, LCS_TIMES)
        @test norm(N) < 1.0e-9 * norm(X)
        @test 0.7 * A_M + 0.3 * A_F ≈ I rtol = 1.0e-9
        # The operators of this cell do not commute: the order of the products matters.
        @test norm(A_F * X - X * A_F) > 1.0e-4 * norm(X)
        # The options also come as keywords of `homogenize_alv`.
        @test lcs_rel(homogenize_alv(cell, SelfConsistent(), prop; times = LCS_TIMES, tight...), X) < 1.0e-10
    end

    @testset "one-layer cylinders: the asymmetric estimate is the self-consistent one, $prop" for prop in (:C, :K)
        # Homogeneous phases of one shape share their Hill operator P̃(X):
        # P̃ ∘ (X_i − X) ∘ A_i = 𝟙 − A_i, so that the self-consistent equation gives
        # Σ f A = 𝟙, and then X = Σ f X_i ∘ A_i = X_M + Σ_{i≠M} f_i (X_i − X_M) ∘ A_i.
        P = lcs_moduli(prop)
        cell = lcs_fibers(prop, lcs_aging(P.M, 0.8), lcs_single(lcs_aging(P.core, 0.3)))
        sc = homogenize_alv(cell, SelfConsistent(; tight...), prop; times = LCS_TIMES)
        asc = homogenize_alv(cell, AsymmetricSelfConsistent(; tight...), prop; times = LCS_TIMES)
        @test lcs_rel(asc, sc) < 1.0e-9
    end

    @testset "differential scheme: stiffness and compliance formulations, aging, $prop" for prop in (:C, :K)
        P = lcs_moduli(prop)
        cell = lcs_fibers(prop, lcs_aging(P.M, 0.8), lcs_coated(lcs_aging(P.core, 0.3), lcs_aging(P.coat, 1.3)))
        opts = (; abstol = 1.0e-11, reltol = 1.0e-11)
        stiff = homogenize_alv(cell, DifferentialScheme(; opts...), prop; times = LCS_TIMES)
        compl = homogenize_alv(cell, DifferentialScheme(; formulation = :compliance, opts...), prop; times = LCS_TIMES)
        @test lcs_rel(compl, stiff) < 1.0e-8
        # The first increment of the scheme is the dilute estimate.
        dilute = homogenize_alv(
            lcs_cell(prop, Ellipsoid(1.0), lcs_aging(P.M, 0.8), lcs_coated(lcs_aging(P.core, 0.3), lcs_aging(P.coat, 1.3)); f = 1.0e-6),
            Dilute(), prop; times = LCS_TIMES
        )
        small = lcs_fibers(prop, lcs_aging(P.M, 0.8), lcs_coated(lcs_aging(P.core, 0.3), lcs_aging(P.coat, 1.3)); f = 1.0e-6)
        M = Matrix(VolterraOperator(lcs_aging(P.M, 0.8), TimeGrid(LCS_TIMES)))
        @test norm(Matrix(homogenize_alv(small, DifferentialScheme(; opts...), prop; times = LCS_TIMES)) - Matrix(dilute)) < 1.0e-4 * norm(Matrix(dilute) - M)
    end

    @testset "derivative with respect to a layer modulus, $(nameof(typeof(scheme)))" for scheme in
        (SelfConsistent(; tight...), AsymmetricSelfConsistent(; tight...), DifferentialScheme(; abstol = 1.0e-11, reltol = 1.0e-11))
        estimate(x) = Matrix(
            homogenize_alv(
                lcs_fibers(:C, lcs_aging(LCS_C.M, 0.8), lcs_coated(lcs_aging(x * LCS_C.core, 0.3), heaviside_law(LCS_C.coat))),
                scheme, :C; times = LCS_TIMES
            )
        )
        h = 1.0e-4
        fd = (estimate(1 + h) - estimate(1 - h)) / 2h
        @test norm(ForwardDiff.derivative(estimate, 1.0) - fd) < 1.0e-6 * norm(fd)
    end

    @testset "refusals" begin
        M = lcs_aging(LCS_C.M, 0.8)
        L = lcs_aging(LCS_C.core, 0.3)
        tilted(law) = LayeredCylinder((1.0,), (law,); axis = (1.0, 0.0, 0.0))
        # Cylinders of two axes: no transversely isotropic estimate holds them.
        for scheme in (SelfConsistent(), AsymmetricSelfConsistent(), DifferentialScheme())
            @test_throws "must share one axis" homogenize_alv(lcs_fibers(:C, M, tilted(L)), scheme, :C; times = LCS_TIMES)
            @test_throws "must share one axis" homogenize_alv(lcs_cell(:C, tilted(M), M, lcs_single(L)), scheme, :C; times = LCS_TIMES)
        end
        for scheme in (AsymmetricSelfConsistent(), DifferentialScheme())
            # A host of several layers, or with an imperfect interface, has no single kernel.
            @test_throws "one-layer LayeredCylinder with a perfect interface" homogenize_alv(
                lcs_cell(:C, lcs_coated(M, L), M, lcs_single(L)), scheme, :C; times = LCS_TIMES
            )
            @test_throws "one-layer LayeredCylinder with a perfect interface" homogenize_alv(
                lcs_cell(:C, LayeredCylinder((1.0,), (M,); interfaces = (SpringInterface(1.0, 1.0),)), M, lcs_single(L)),
                scheme, :C; times = LCS_TIMES
            )
        end
        cell = lcs_fibers(:C, M, lcs_single(L))
        # The Newton march and a seed are those of an isotropic estimate.
        for scheme in (SelfConsistent(; algorithm = NewtonDefault()), AsymmetricSelfConsistent(; algorithm = NewtonDefault()))
            @test_throws "is not available for a cell of LayeredCylinder phases" homogenize_alv(cell, scheme, :C; times = LCS_TIMES)
        end
        @test_throws "is not supported for a viscoelastic (ALV) cell" homogenize_alv(cell, SelfConsistent(; init = :M), :C; times = LCS_TIMES)
        @test_throws "is not supported for a viscoelastic (ALV) cell" homogenize_alv(cell, AsymmetricSelfConsistent(; init = :M), :C; times = LCS_TIMES)
        @test_throws "formulation must be :stiffness or :compliance" homogenize_alv(cell, DifferentialScheme(; formulation = :creep), :C; times = LCS_TIMES)
        # A phase taking the volume complement next to a named matrix has no target.
        named = RVE()
        add_phase!(named, :M, lcs_single(M), Dict(:C => M); fraction = 0.6)
        add_phase!(named, :F, lcs_single(L), Dict(:C => M); fraction = :rest)
        @test_throws "has no target to integrate up to" homogenize_alv(named, DifferentialScheme(:M), :C; times = LCS_TIMES)
        # One cylinder with an orientation average: no longer the cell of aligned cylinders.
        @test_throws ArgumentError homogenize_alv(lcs_fibers(:C, M, lcs_single(L); symmetrize = :iso), SelfConsistent(), :C; times = LCS_TIMES)
    end
end

# ── Any axis ─────────────────────────────────────────────────────────────────
#
# The radial problems are solved in a frame of the axis. Two references that do
# not go through that change of basis: the elastic `homogenize`, which takes
# transversely isotropic tensors of any axis as they are, and covariance — the
# operator of a cylinder of axis n, read in a frame (t₁, t₂, n) built here from
# vectors, is the operator of the same cylinder of axis e₃ read in the canonical
# frame. The second holds outside the elastic limit, on aging laws.

const LCS_N = Tuple(normalize([1.0, -2.0, 2.0]))
# A frame (t₁, t₂, n) as the columns of a matrix.
function lcs_frame(n)
    t₁ = normalize(cross(collect(n), abs(n[1]) < 0.9 ? [1.0, 0.0, 0.0] : [0.0, 1.0, 0.0]))
    return hcat(t₁, cross(collect(n), t₁), collect(n))
end
# The Kelvin-Mandel basis tensors of the frame F, as vectors in canonical
# components ((11, 22, 33, 23, 13, 12), the shear terms weighted by √2).
function lcs_mandel_frame(F)
    km = ((1, 1), (2, 2), (3, 3), (2, 3), (1, 3), (1, 2))
    vec(S) = [S[1, 1], S[2, 2], S[3, 3], √2 * S[2, 3], √2 * S[1, 3], √2 * S[1, 2]]
    cols = map(km) do (i, j)
        D = F[:, i] * F[:, j]'
        vec(i == j ? D : (D + D') / √2)
    end
    return reduce(hcat, cols)
end
# Every b × b time block of X read in the frame whose basis vectors are the
# columns of P: Pᵀ Xᵢⱼ P.
function lcs_in_frame(X::AbstractMatrix, P::AbstractMatrix)
    b = size(P, 1)
    Y = similar(X)
    for i in 0:(size(X, 1) ÷ b - 1), j in 0:(size(X, 2) ÷ b - 1)
        rows, cols = (b * i + 1):(b * i + b), (b * j + 1):(b * j + b)
        Y[rows, cols] = P' * X[rows, cols] * P
    end
    return Y
end
lcs_P(prop::Symbol, F) = prop === :C ? lcs_mandel_frame(F) : F

# Transversely isotropic layers about `n`: a stiffness of Hill moduli, or a
# conductivity.
lcs_ti(prop::Symbol, s, n) = prop === :C ?
    TensND.ti_from_constants(s * 9.0 + s * 2.5, s * 9.0 - s * 2.5, s * 3.0, s * 25.0, s * 3.5, n) :
    TensND.TensTI{2}(s * 1.0, s * 2.5, n)

@testset "LayeredCylinder of any axis — aging" begin
    F = lcs_frame(LCS_N)
    e3 = (0.0, 0.0, 1.0)
    @testset "elastic limit against the elastic schemes, $prop" for prop in (:C, :K)
        cell(n) = begin
            core, coat, M = lcs_ti(prop, 3.0, n), lcs_ti(prop, 0.6, n), lcs_ti(prop, 1.0, n)
            r = RVE(; distribution_shape = Ellipsoid(1.0, 1.0, 1.0))
            add_phase!(r, :M, LayeredCylinder((1.0,), (M,); axis = n), Dict(prop => M); fraction = :rest)
            add_phase!(r, :F, LayeredCylinder((0.7, 1.0), (core, coat); axis = n), Dict(prop => M); fraction = 0.3)
            r
        end
        r = cell(LCS_N)
        for (alv, el) in (
                (SelfConsistent(; reltol = 1.0e-12), SelfConsistent(; reltol = 1.0e-12)),
                (AsymmetricSelfConsistent(; reltol = 1.0e-12), AsymmetricSelfConsistent(:M; reltol = 1.0e-12)),
                (DifferentialScheme(; abstol = 1.0e-11, reltol = 1.0e-11), DifferentialScheme(; abstol = 1.0e-11, reltol = 1.0e-11)),
            )
            R = homogenize_alv(r, alv, prop; times = LCS_TIMES)
            @test R.class isa TensTI && TensND.isparallel(TensND.axis(R.class), LCS_N)
            @test lcs_rel(R, lcs_elastic(homogenize(r, el, prop), LCS_TIMES)) < 1.0e-9
        end
    end

    @testset "covariance on aging laws, $prop" for prop in (:C, :K)
        P = lcs_P(prop, F)
        laws(n) = (lcs_aging(lcs_ti(prop, 3.0, n), 0.3), lcs_aging(lcs_ti(prop, 0.6, n), 1.5), lcs_aging(lcs_ti(prop, 1.0, n), 0.8))
        interfaces = prop === :C ? (SpringInterface(20.0, 10.0), MembraneInterface(0.4, 0.2)) :
            (KapitzaInterface(0.05), SurfaceConductiveInterface(0.3))
        coated(n) = (L = laws(n); LayeredCylinder((0.7, 1.0), (L[1], L[2]); interfaces, axis = n))
        ref(n) = laws(n)[3]
        # the operators of one cylinder: whole, per layer, contribution
        for f in (
                (c, R) -> gradient_gradient_concentration_alv(c, R, LCS_TIMES),
                (c, R) -> gradient_gradient_concentration_alv(c, R, LCS_TIMES; layer = 2, internal = true),
                (c, R) -> flux_gradient_concentration_alv(c, R, LCS_TIMES),
                (c, R) -> flux_gradient_concentration_alv(c, R, LCS_TIMES; layer = 1),
                (c, R) -> property_contribution_alv(c, R, LCS_TIMES),
            )
            X₃ = Matrix(f(coated(e3), ref(e3)))
            @test lcs_rel(lcs_in_frame(Matrix(f(coated(LCS_N), ref(LCS_N))), P), X₃) < 1.0e-12
        end
        # the derivative with respect to a layer modulus follows the same frame
        dcontribution(n) = ForwardDiff.derivative(
            s -> Matrix(
                property_contribution_alv(
                    LayeredCylinder((0.7, 1.0), (lcs_aging(lcs_ti(prop, s, n), 0.3), laws(n)[2]); interfaces, axis = n),
                    ref(n), LCS_TIMES
                )
            ), 3.0
        )
        @test lcs_rel(lcs_in_frame(dcontribution(LCS_N), P), dcontribution(e3)) < 1.0e-12
        # the pointwise operators, at a point of the coating and of the matrix
        for x in ([0.5, 0.6, 0.3], [1.1, -0.4, 2.0])
            X₃ = local_flux_gradient_concentration_alv(coated(e3), ref(e3), LCS_TIMES, x)
            Xₙ = local_flux_gradient_concentration_alv(coated(LCS_N), ref(LCS_N), LCS_TIMES, F * x)
            @test lcs_rel(lcs_in_frame(Xₙ, P), X₃) < 1.0e-12
        end
        # the three schemes on the running estimate
        cell(n) = begin
            L = laws(n)
            r = RVE(; distribution_shape = Ellipsoid(1.0, 1.0, 1.0))
            add_phase!(r, :M, LayeredCylinder((1.0,), (L[3],); axis = n), Dict(prop => L[3]); fraction = :rest)
            add_phase!(r, :F, coated(n), Dict(prop => L[3]); fraction = 0.3)
            r
        end
        for scheme in (
                SelfConsistent(; reltol = 1.0e-12), AsymmetricSelfConsistent(; reltol = 1.0e-12),
                DifferentialScheme(; abstol = 1.0e-11, reltol = 1.0e-11),
                DifferentialScheme(; formulation = :compliance, abstol = 1.0e-11, reltol = 1.0e-11),
            )
            X₃ = Matrix(homogenize_alv(cell(e3), scheme, prop; times = LCS_TIMES))
            @test lcs_rel(lcs_in_frame(Matrix(homogenize_alv(cell(LCS_N), scheme, prop; times = LCS_TIMES)), P), X₃) < 1.0e-9
        end
    end
end
