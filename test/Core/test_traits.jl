using Test
using MeanFieldHomogenization
using TensND

@testset "Core — traits" begin
    C_iso = TensISO{3}(3.0, 2.0)
    @test MeanFieldHomogenization.material_symmetry(C_iso) isa MeanFieldHomogenization.IsotropicSym

    # Analytical / Residue / DECUHR singletons
    @test MeanFieldHomogenization.Analytical() isa MeanFieldHomogenization.AbstractAlgorithm
    @test MeanFieldHomogenization.Residue() isa MeanFieldHomogenization.AbstractAlgorithm
    @test MeanFieldHomogenization.DECUHR() isa MeanFieldHomogenization.AbstractAlgorithm
end

# =============================================================================
#  `_resolve_algo` policy for a 3D anisotropic elastic reference.
#
#  `:auto` must pick a CUBATURE, never the residue algorithm: the residue
#  acoustic polynomial degenerates whenever the reference is anisotropic in
#  *type* and isotropic in *value* — the state the differential and
#  self-consistent schemes feed back at their first step — and returns NaN
#  there.  `Residue` stays reachable on an explicit `:residues`.
# =============================================================================

@testset "Core — `:auto` picks a cubature for a 3D anisotropic reference" begin
    C_iso = TensISO{3}(3 * 20.0, 2 * 8.0)
    C_i = TensISO{3}(3 * 50.0, 2 * 20.0)
    tri = Ellipsoid(1.0, 0.6, 0.3)

    # An anisotropically-TYPED tensor holding isotropic VALUES.
    N = property_contribution(tri, C_i, C_iso)
    aniso_typed = MeanFieldHomogenization.Schemes._diff_embed(C_iso, C_iso + N)
    @test !(aniso_typed isa TensND.TensISO)
    @test Array(aniso_typed) ≈ Array(C_iso) atol = 1.0e-10

    for C₀ in (aniso_typed, C_iso + 0.35 * N)
        # NestedQuadGK, whether or not the DECUHR extension is loaded (it is here).
        @test MeanFieldHomogenization.Core._resolve_algo(Val(:auto), tri, C₀) isa
            MeanFieldHomogenization.Core.NestedQuadGK
        @test MeanFieldHomogenization.Core._resolve_algo(Val(:residues), tri, C₀) isa
            MeanFieldHomogenization.Core.Residue
    end

    # …and the default actually returns the right number on the degenerate
    # reference, where the residue path throws.
    P = hill_tensor(tri, aniso_typed)
    P_ref = hill_tensor(tri, C_iso)                     # closed form
    @test maximum(abs, Array(P) .- Array(P_ref)) / maximum(abs, Array(P_ref)) < 1.0e-8
    # Explicitly asked for, the residue path refuses it and says why.
    @test_throws ArgumentError hill_tensor(tri, aniso_typed; method = :residues)

    # Non-Float64 coefficients keep the type-generic cubature.
    @test MeanFieldHomogenization.Core._aniso_default_algo(ComplexF64(1) * (C_iso + 0.35 * N)) isa
        MeanFieldHomogenization.Core.NestedQuadGK

    # A ribbon crack, whose reduced kernel is a 1-D integral: DECUHR integrates
    # over rectangles of dimension two or more and failed on every ribbon, which
    # made a ribbon in an anisotropic reference fail as soon as DECUHR was loaded.
    rib = RibbonCrack(1.0)
    B_ref = cod_tensor(rib, C_iso)                      # closed form
    for m in (:auto, :decuhr)
        B = cod_tensor(rib, aniso_typed; method = m)
        @test maximum(abs, Array(B) .- Array(B_ref)) / maximum(abs, Array(B_ref)) < 1.0e-8
    end
end

# Regression: `:nestedquadgk` used to be missing from the `TensISO`
# disambiguation rules, so asking for it with an isotropic matrix hit a
# method ambiguity instead of resolving to the closed form.
@testset "Core — every method symbol resolves for an isotropic reference" begin
    C_iso = TensISO{3}(3 * 20.0, 2 * 8.0)
    for incl in (Ellipsoid(1.0, 0.6, 0.3), Ellipsoid(1.0), PennyCrack(1.0))
        for m in (:auto, :analytical, :residues, :decuhr, :nestedquadgk)
            @test MeanFieldHomogenization.Core._resolve_algo(Val(m), incl, C_iso) isa
                MeanFieldHomogenization.Core.AbstractAlgorithm
        end
    end
    @test all(isfinite, get_array(hill_tensor(Ellipsoid(1.0, 0.6, 0.3), C_iso; method = :nestedquadgk)))
    @test all(isfinite, get_array(cod_tensor(PennyCrack(1.0), C_iso; method = :nestedquadgk)))
end
