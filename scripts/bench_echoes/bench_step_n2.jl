import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."); io = devnull)
using MeanFieldHomogenization, TensND, LinearAlgebra, JSON, Printf

ref = JSON.parsefile(joinpath(@__DIR__, "bench_step_n2_python.json"))
T_grid = collect(Float64.(ref.T))
k0 = Float64(ref.k0); mu0 = Float64(ref.mu0)
eta0 = Float64(ref.eta0); gamma0 = Float64(ref.gamma0)
k1 = Float64(ref.k1); mu1 = Float64(ref.mu1)
eta1 = Float64(ref.eta1); gamma1 = Float64(ref.gamma1)
kp = Float64(ref.kp); mup = Float64(ref.mup)
t_set_inner = Float64(ref.t_set_inner); t_set_outer = Float64(ref.t_set_outer)

C_p_tens = TensISO{3}(3 * kp, 2 * mup)
R1 = maxwell_iso(k1, mu1, eta1, gamma1)

step_law(t_set) = ViscoLaw(
    function (t, tp)
        if t < tp
            return zero(C_p_tens)
        elseif tp ≥ t_set
            return R1.eval_fun(t, tp)
        else
            return C_p_tens
        end
    end, :relaxation
)

moduli = (step_law(t_set_inner), step_law(t_set_outer))
radii = (0.5^(1 / 3), 1.0)
sphere = LayeredSphere(radii, moduli)
C0_law = maxwell_iso(k0, mu0, eta0, gamma0)

α_jl = bulk_concentration_alv(sphere, C0_law, T_grid)
β_jl = shear_concentration_alv(sphere, C0_law, T_grid)
to_matrix(v) = reduce(hcat, [collect(Float64.(row)) for row in v])'

for k in 1:2
    α_py = Matrix(to_matrix(ref.layers_alpha[k]))
    β_py = Matrix(to_matrix(ref.layers_beta[k]))
    @printf "Layer %d: max |Δα|=%.3e (rel %.3e)  max |Δβ|=%.3e (rel %.3e)\n" k maximum(abs, α_jl[k] .- α_py) maximum(abs, α_jl[k] .- α_py) / maximum(abs, α_py) maximum(abs, β_jl[k] .- β_py) maximum(abs, β_jl[k] .- β_py) / maximum(abs, β_py)
    println("    β_jl: ", β_jl[k])
    println("    β_py: ", β_py)
end
