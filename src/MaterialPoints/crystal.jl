# =============================================================================
#  crystal.jl — single-crystal viscoplasticity on slip systems.
#
#  A crystal deforms plastically by slip on a finite set of systems, each a
#  plane of unit normal n and a direction m in it. The slip rate of a system is
#  driven by its resolved shear stress τ = σ : sym(m ⊗ n), the plastic strain is
#  the sum of the slips times the Schmid tensors sym(m ⊗ n), and each system
#  hardens with the slip accumulated on all of them.
#
#  The systems of the cubic lattices are not a table: they follow from their
#  definition (every direction of a family lying in every plane of another), and
#  are generated here from it, in the frame of the cube.
# =============================================================================

# The directions of a family {hkl}: every signed permutation of (h, k, l), one
# of each opposite pair, normalized.
function _family(h, k, l)
    dirs = SVector{3, Float64}[]
    for p in ((h, k, l), (h, l, k), (k, h, l), (k, l, h), (l, h, k), (l, k, h)), s in Iterators.product((1, -1), (1, -1), (1, -1))
        v = SVector{3, Float64}(p .* s)
        iszero(v) && continue
        v = v / norm(v)
        any(w -> w ≈ v || w ≈ -v, dirs) || push!(dirs, v)
    end
    return dirs
end

"""
    slip_systems(lattice::Symbol) -> Vector{Tuple{SVector{3}, SVector{3}}}

The slip systems of a cubic lattice, as pairs `(n, m)` of the unit normal to the
slip plane and the unit slip direction, in the frame of the cube:

- `:fcc`, face-centered cubic: the 12 systems ``\\{111\\}\\langle110\\rangle``;
- `:bcc`, body-centered cubic: the 12 systems ``\\{110\\}\\langle111\\rangle``.

They are generated from their definition, every direction of the second family
that lies in a plane of the first.
"""
function slip_systems(lattice::Symbol)
    planes, directions = lattice === :fcc ? (_family(1, 1, 1), _family(1, 1, 0)) :
        lattice === :bcc ? (_family(1, 1, 0), _family(1, 1, 1)) :
        throw(ArgumentError("slip_systems: lattice :fcc or :bcc; got :$lattice"))
    # Float64 lattice vectors built above, never symbolic: `dot` is safe here.
    return [(n, m) for n in planes for m in directions if abs(dot(n, m)) < 1.0e-12]
end

# The Schmid tensor sym(m ⊗ n) of each system, as the columns of a 6 × N matrix
# in Kelvin–Mandel components: the resolved shear stresses are its transpose
# times σ, and the plastic strain is it times the slips.
function _schmid_matrix(systems, R)
    cols = map(systems) do (n, m)
        n′, m′ = R * n, R * m
        P = (m′ * transpose(n′) + n′ * transpose(m′)) / 2
        Mandel(P[1, 1], P[2, 2], P[3, 3], sqrt(2.0) * P[2, 3], sqrt(2.0) * P[1, 3], sqrt(2.0) * P[1, 2])
    end
    return hcat(cols...)
end

"""
    CrystalViscoplasticity(; C₁₁, C₁₂, C₄₄, lattice = :fcc, frame = CanonicalBasis, K, n, τ₀,
                             Q = 0, b = 0, latent = 1)

Viscoplasticity of a single crystal by slip. The elasticity is cubic
(`C₁₁`, `C₁₂`, `C₄₄`), the slip systems those of `lattice` (see
[`slip_systems`](@ref)), and the crystal is oriented by `frame`, the TensND basis
whose vectors are its cube axes. On each system ``s`` of normal
``\\underline{n}_s`` and direction ``\\underline{m}_s``, with the Schmid tensor
``\\boldsymbol{P}_s = \\operatorname{sym}(\\underline{m}_s \\otimes \\underline{n}_s)``,

```math
\\tau_s = \\boldsymbol{\\sigma}:\\boldsymbol{P}_s,\\qquad
\\dot\\gamma_s = \\left\\langle\\frac{|\\tau_s| - r_s}{K}\\right\\rangle^{n}\\operatorname{sign}\\tau_s,\\qquad
\\dot v_s = |\\dot\\gamma_s|,
```

```math
\\boldsymbol{\\varepsilon}^p = \\sum_s \\gamma_s\\,\\boldsymbol{P}_s,\\qquad
r_s = \\tau_0 + Q \\sum_j h_{sj}\\left(1 - e^{-b\\,v_j}\\right),\\qquad
h_{sj} = \\text{latent} + (1 - \\text{latent})\\,\\delta_{sj},
```

``\\langle x \\rangle = \\max(x, 0)``. A system slips only beyond its critical
resolved shear stress ``r_s``, at a rate of power ``n`` in the excess; ``v_s``
is the slip it has accumulated, which hardens every system through the
interaction matrix ``h`` (`latent = 1`: all systems harden alike). The state is
`(γ = 12, v = 12)`; the steps are integrated by the backward Euler scheme, a step
on which no system exceeds its critical stress being elastic. The flow and
hardening rules have the form of the single-crystal model of
[meric1991](@citet), without its kinematic hardening.
"""
function CrystalViscoplasticity(;
        C₁₁, C₁₂, C₄₄, K, n, τ₀, Q = zero(τ₀), b = zero(τ₀), latent = one(τ₀),
        lattice::Symbol = :fcc, frame::TensND.OrthonormalBasis{3} = TensND.CanonicalBasis{3, Float64}()
    )
    n ≥ 1 || throw(DomainError(n, "CrystalViscoplasticity: the exponent n must be at least 1"))
    systems = slip_systems(lattice)
    N = length(systems)
    R = SMatrix{3, 3}(TensND.vecbasis(frame))
    P = SMatrix{6, N}(_schmid_matrix(systems, R))
    H = SMatrix{N, N}([i == j ? one(latent) : latent for i in 1:N, j in 1:N])
    ℂ = _mandel(TensND.tens_cubic(C₁₁, C₁₂, C₄₄, frame))
    return ResidualLaw(;
        name = :CrystalViscoplasticity, stress = _crystal_stress, residual = _crystal_residual,
        elastic = _crystal_elastic, layout = (γ = N, v = N),
        p = (; ℂ, P, H, K, n, τ₀, Q, b)
    )
end

_crystal_stress(ε, y, p) = p.ℂ * (ε - p.P * y.γ)
_crystal_resolved(ε, y, p) = transpose(p.P) * _crystal_stress(ε, y, p)
_crystal_critical(v, p) = p.τ₀ .+ p.Q .* (p.H * (1 .- exp.(-p.b .* v)))

# The slip rate of each system: zero below its critical stress.
_crystal_rate(τ, r, p) = (max.((abs.(τ) .- r) ./ p.K, 0)) .^ p.n .* sign.(τ)

function _crystal_residual(y, g, prev, Δt, p)
    τ = _crystal_resolved(g.ε, y, p)
    Δγ = y.γ - prev.y.γ
    return (
        γ = Δγ - Δt * _crystal_rate(τ, _crystal_critical(y.v, p), p),
        v = (y.v - prev.y.v) - abs.(Δγ),
    )
end

# No time, or no system beyond its critical stress at the trial state: no slip.
_crystal_elastic(g, prev, Δt, p) =
    iszero(Δt) || all(abs.(_crystal_resolved(g.ε, prev.y, p)) .<= _crystal_critical(prev.y.v, p))
