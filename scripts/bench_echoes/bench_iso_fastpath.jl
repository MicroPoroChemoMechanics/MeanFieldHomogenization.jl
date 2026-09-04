# Compare timing: iso fast path vs generic 6n×6n in homogenize_alv.
import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."); io = devnull)
using MeanFieldHomogenization, TensND, LinearAlgebra, Printf

# Setup similar to script 37 (whole_pores, all-iso phases).
const E0 = 1.0; const ν0 = 0.2
const k0 = E0 / (3 * (1 - 2ν0)); const μ0_ = E0 / (2 * (1 + ν0))
const η0 = 0.2; const γ0 = 0.133
const k1 = 5.0 / (3 * 0.4); const μ1 = 5.0 / 2.6
const η1 = 1.0; const γ1 = 1.67
const kp = 1.0e-8 / (3 * 0.6); const μp = 1.0e-8 / (2 * 1.2)

C0_law() = maxwell_iso(k0, μ0_, η0, γ0)
C1_law() = maxwell_iso(k1, μ1, η1, γ1)
C_p = TensISO{3}(3 * kp, 2 * μp)

function build_rve(N::Int)
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => C0_law()); fraction = :rest)
    add_phase!(
        rve, :PORE, Ellipsoid(1.0, 1.0, 1.0),
        Dict(:C => heaviside_law(C_p)); fraction = 0.1
    )
    for i in 1:N
        add_phase!(
            rve, Symbol("INC_$i"), Ellipsoid(1.0, 1.0, 1.0),
            Dict(:C => C1_law()); fraction = 0.3 / N
        )
    end
    return rve
end

# Run a quick timing comparison.  The iso fast path activates
# automatically when `_is_iso_block` returns true for every quantity.
println("=== Iso fast path vs generic 6n×6n in `homogenize_alv` (MT) ===")

const _iso_fastpath_disabled = Ref(false)
# Toggle: when true, every iso-form check returns false → forces the
# generic 6n×6n algebra everywhere.
function MeanFieldHomogenization.Viscoelasticity._is_iso_block(M::AbstractMatrix; tol::Real = 1.0e-12)
    _iso_fastpath_disabled[] && return false
    sz = size(M, 1)
    sz == size(M, 2) || return false
    sz % 6 == 0 || return false
    n = sz ÷ 6
    iszero(n) && return true
    scale = max(maximum(abs, M), one(real(eltype(M))))
    abstol = tol * scale
    α, β = MeanFieldHomogenization.Viscoelasticity.iso_params_from_blocks(M)
    @inbounds for i in 1:n, j in 1:n
        a = α[i, j]; b = β[i, j]
        diag_top = (a + 2b) / 3
        offdiag_top = (a - b) / 3
        rows = (6 * (i - 1) + 1):(6 * i)
        cols = (6 * (j - 1) + 1):(6 * j)
        for k in 1:3, l in 1:3
            expected = (k == l) ? diag_top : offdiag_top
            abs(M[rows[k], cols[l]] - expected) ≤ abstol || return false
        end
        for k in 1:3, l in 4:6
            abs(M[rows[k], cols[l]]) ≤ abstol || return false
        end
        for k in 4:6, l in 1:3
            abs(M[rows[k], cols[l]]) ≤ abstol || return false
        end
        for k in 4:6, l in 4:6
            expected = (k == l) ? b : zero(b)
            abs(M[rows[k], cols[l]] - expected) ≤ abstol || return false
        end
    end
    return true
end

for n_times in (11, 21, 41, 81, 161)
    T_grid = collect(range(0.5, 3.0; length = n_times))
    rve = build_rve(5)
    # Iso fast path.
    _iso_fastpath_disabled[] = false
    homogenize_alv(rve, MoriTanaka(), :C; times = T_grid)
    t_iso = @elapsed homogenize_alv(rve, MoriTanaka(), :C; times = T_grid)
    # Generic 6n×6n path.
    _iso_fastpath_disabled[] = true
    homogenize_alv(rve, MoriTanaka(), :C; times = T_grid)
    t_gen = @elapsed homogenize_alv(rve, MoriTanaka(), :C; times = T_grid)
    @printf "  n_times = %3d  → iso %.4f s, generic %.4f s, speedup ×%.2f\n" n_times t_iso t_gen (t_gen / t_iso)
end
