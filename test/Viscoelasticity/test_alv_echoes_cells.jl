# =============================================================================
#  test_alv_echoes_cells.jl — every ALV scheme against Echoes on complex cells.
#
#  References: `echoes_alv_cells.json`, written by
#  `scripts/bench_echoes/alv_harness/make_test_references.py` (Echoes
#  `homogenize_visco`, 4 instants). Each cell is rebuilt from its description with
#  `scripts/bench_echoes/alv_harness/cells.jl`, the builder of the harness.
#
#  The cells have an isotropic aging matrix, as the ALV Hill kernel requires: at
#  order 4, stiff prolate spheroids and triaxial ellipsoids of an aging transversely
#  isotropic law, each oriented off the axes along its own frame, and two crack
#  families of different orientations; at order 2, the same without cracks. Voigt,
#  Reuss, dilute, dilute dual, Mori-Tanaka, Maxwell and Ponte Castañeda-Willis (on a
#  tilted oblate distribution) take the families as they are; the self-consistent,
#  asymmetric self-consistent and differential schemes, whose reference is the running
#  estimate, take them averaged in orientation. The `g` cases are the asymmetric
#  scheme with and without iso-averaged cracks, and at order 2.
#
#  Echoes integrates the differential scheme by explicit Euler steps: its reference is
#  the Richardson extrapolation from 400 and 800 steps, second order, hence the
#  looser tolerance of that scheme.
# =============================================================================

using Test
using MeanFieldHomogenization
using LinearAlgebra
using JSON

include(joinpath(pkgdir(MeanFieldHomogenization), "scripts", "bench_echoes", "alv_harness", "cells.jl"))

const _AEC_REFS = JSON.parsefile(joinpath(@__DIR__, "echoes_alv_cells.json"))

@testset "ALV schemes against Echoes — complex cells, orders 4 and 2" begin
    for row in _AEC_REFS["cases"]
        c = row["case"]
        rve, prop, m = ALVHarnessCells.mfh_rve(c, Dict{String, Any}())
        R = homogenize_alv(rve, ALVHarnessCells.SCHEMES[c["scheme"]](c, m), prop; times = Float64.(c["times"]))
        E = ALVHarnessCells.mat(row["effective"])
        rtol = c["scheme"] == "differential" ? 1.0e-6 : 1.0e-9
        @test isapprox(Matrix(R), E; rtol)
    end
end
