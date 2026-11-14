# =============================================================================
#  fe_oracle_layered.jl — an independent numerical reference for the layered
#  sphere and the layered cylinder in aging linear viscoelasticity.
#
#  Nothing of the transfer-matrix recurrences is used. The displacement (the
#  temperature in conduction) is written u = Σ_c U_c(r) g_c(x) + Σ_e a_e h_e(x),
#  g_c a pattern homogeneous of degree 0, h_e a prescribed field (the axial
#  strain of a cylinder), and the weak form of equilibrium
#      Σ_layers ∫ ε(v) : R̃ ∘ ε(u) dV + interface energies = boundary terms
#  is discretized by Lagrange finite elements of degree p in r for every U_c,
#  with the angular integrals done by quadrature on the sphere (circle), the
#  strains of the patterns obtained by automatic differentiation of their
#  Cartesian expressions. In time, the relaxation operators are the trapezoidal
#  Volterra matrices of the package (the shared time discretization): at instant
#  i, Σ_{j≤i} K(R̃_ij) q_j = loads. Interfaces do not age (diagonal in time):
#    spring      ∮ [v]·(k_n n⊗n + k_t (1 - n⊗n))·[u] dS
#    membrane    ∮ εˢ(v) : (λˢ 1ˢ⊗1ˢ + 2μˢ 𝕀ˢ) : εˢ(u) dS, εˢ = P·sym∇u·P, λˢ = κˢ - μˢ
#    Kapitza     ∮ [v][T]/ρ dS
#    surface     ∮ kˢ ∇ₛv·∇ₛT dS
#  The matrix beyond the truncation radius R_t carries u = u∞ + w, w spanned by
#  the decaying powers r^(-s) of each amplitude (s ∈ {2, 4} for the sphere,
#  {1, 3} for the cylinder: the exponents of the decaying solutions of the
#  biharmonic radial problems). w is discretized exactly by an exterior element
#  integrated on [R_t, ∞), and the uniform remote field enters by the traction it
#  exerts on r = R_t, which is exact because it is equilibrated.
#
#  Averages: ⟨ε⟩ over the ball of radius r_N read from outside, i.e. the material
#  average plus ∮ sym([u]⊗n) of the counted interfaces; ⟨σ⟩ the material average
#  plus ∮ σˢ dS of the counted membranes (∮ kˢ∇ₛT in conduction).
#
#  A numerical reference in Float64, on no differentiable path: its buffers are
#  Float64 and its loops are those of finite-element assembly.
# =============================================================================
module FEOracleLayered

using LinearAlgebra
using ForwardDiff

export Harmonic, OracleInterface, oracle_solve, compare, sphere_elastic, sphere_conduction,
    cylinder_elastic, cylinder_conduction

# ── Quadrature ───────────────────────────────────────────────────────────────

function gauss_legendre(n::Int)
    β = [k / sqrt(4k^2 - 1) for k in 1:(n - 1)]
    F = eigen(SymTridiagonal(zeros(n), β))
    return F.values, 2 .* F.vectors[1, :] .^ 2
end
function gauss_on(a, b, n)
    x, w = gauss_legendre(n)
    return (a + b) / 2 .+ (b - a) / 2 .* x, (b - a) / 2 .* w
end

function sphere_angles(nμ::Int, nφ::Int)
    μ, wμ = gauss_legendre(nμ)
    pts = Vector{Vector{Float64}}(); w = Float64[]
    for (m, wm) in zip(μ, wμ), k in 1:nφ
        φ = 2π * (k - 1) / nφ; s = sqrt(1 - m^2)
        push!(pts, [s * cos(φ), s * sin(φ), m]); push!(w, wm * 2π / nφ)
    end
    return pts, w
end
function circle_angles(nθ::Int)
    pts = [[cos(2π * (k - 1) / nθ), sin(2π * (k - 1) / nθ), 0.0] for k in 1:nθ]
    return pts, fill(2π / nθ, nθ)
end

# Mandel vector of the symmetric part of a 3×3 matrix (11, 22, 33, √2·23, √2·13, √2·12).
function mandel_sym(G::AbstractMatrix)
    S = (G + G') / 2
    return [S[1, 1], S[2, 2], S[3, 3], √2 * S[2, 3], √2 * S[1, 3], √2 * S[1, 2]]
end
mandel_to_matrix(v) = [
    v[1] v[6] / √2 v[5] / √2
    v[6] / √2 v[2] v[4] / √2
    v[5] / √2 v[4] / √2 v[3]
]

# ── A harmonic problem ───────────────────────────────────────────────────────

"""A remote loading: its gradient (Mandel 6-vector or 3-vector), the amplitudes ĉ of
the patterns (U_c = ĉ_c r far away) and the values a of the prescribed fields."""
struct Load
    name::String
    E::Vector{Float64}
    c::Vector{Float64}
    a::Vector{Float64}
end

struct Harmonic
    d::Int                          # 3: sphere (weight r²); 2: cylinder (weight r)
    elastic::Bool
    nc::Int
    ne::Int
    b::Int                          # 6 (Mandel) or 3 (gradient)
    w::Vector{Float64}              # angular weights
    nrm::Vector{Vector{Float64}}    # x̂ at the angular points
    gv::Matrix{Vector{Float64}}     # [c, q] pattern value (a 1-vector in conduction)
    A::Matrix{Vector{Float64}}      # [c, q] gradient per unit U_c'
    B::Matrix{Vector{Float64}}      # [c, q] gradient per unit U_c / r
    H::Matrix{Vector{Float64}}      # [e, q] gradient per unit a_e
    SB::Matrix{Matrix{Float64}}     # [c, q] surface gradient per unit U_c / r (3×3, or 3×1)
    SH::Matrix{Matrix{Float64}}     # [e, q] surface gradient per unit a_e
    powers::Vector{Int}             # decaying exponents of the exterior element
    loads::Vector{Load}
    patterns::Vector{Any}           # the g_c, kept for the pointwise fields
    prescribed::Vector{Any}         # the h_e
end

_radius(x, d) = d == 3 ? norm(x) : hypot(x[1], x[2])

function Harmonic(
        d::Int, elastic::Bool, patterns::Vector, prescribed::Vector,
        angles, powers::Vector{Int}, loads::Vector{Load}
    )
    pts, w = angles
    nc, ne, nq = length(patterns), length(prescribed), length(pts)
    b = elastic ? 6 : 3
    val(f, x) = elastic ? f(x) : [f(x)]
    grad(f, x) = elastic ? ForwardDiff.jacobian(f, x) : reshape(ForwardDiff.gradient(f, x), 1, 3)
    gv = [val(patterns[c], pts[q]) for c in 1:nc, q in 1:nq]
    A = Matrix{Vector{Float64}}(undef, nc, nq); B = similar(A)
    SB = Matrix{Matrix{Float64}}(undef, nc, nq)
    H = Matrix{Vector{Float64}}(undef, ne, nq); SH = Matrix{Matrix{Float64}}(undef, ne, nq)
    rng_shift = [0.37, -0.21, 0.53]
    for q in 1:nq
        x = pts[q]; P = I - x * x'
        for c in 1:nc
            G = grad(patterns[c], x)
            # degree-0 homogeneity (and independence of z for a cylinder)
            @assert isapprox(val(patterns[c], 2.3 .* x), gv[c, q]; atol = 1.0e-12)
            d == 2 && @assert isapprox(val(patterns[c], x .+ [0, 0, 0.7]), gv[c, q]; atol = 1.0e-12)
            if elastic
                A[c, q] = mandel_sym(gv[c, q] * x'); B[c, q] = mandel_sym(G)
                SB[c, q] = P * ((G + G') / 2) * P
            else
                A[c, q] = gv[c, q][1] .* x; B[c, q] = vec(G)
                SB[c, q] = reshape(P * vec(G), 3, 1)
            end
        end
        for e in 1:ne
            G = grad(prescribed[e], x)
            @assert isapprox(grad(prescribed[e], x .+ rng_shift), G; atol = 1.0e-12)   # uniform gradient
            if elastic
                H[e, q] = mandel_sym(G); SH[e, q] = P * ((G + G') / 2) * P
            else
                H[e, q] = vec(G); SH[e, q] = reshape(P * vec(G), 3, 1)
            end
        end
    end
    # every load is the uniform field it claims to be
    for L in loads, x in ([0.3, -0.7, 0.4], [1.1, 0.2, -0.9], [-0.4, 0.5, 2.0])
        r = _radius(x, d)
        u = sum((L.c[c] * r) .* val(patterns[c], x) for c in 1:nc; init = zeros(elastic ? 3 : 1)) +
            sum(L.a[e] .* val(prescribed[e], x) for e in 1:ne; init = zeros(elastic ? 3 : 1))
        ue = elastic ? mandel_to_matrix(L.E) * x : [dot(L.E, x)]
        @assert isapprox(u, ue; atol = 1.0e-12) "load $(L.name) is not the field its amplitudes describe"
    end
    return Harmonic(d, elastic, nc, ne, b, w, pts, gv, A, B, H, SB, SH, powers, loads, patterns, prescribed)
end

# ── The four problems ────────────────────────────────────────────────────────

const _ez = [0.0, 0.0, 1.0]
P2(μ) = (3μ^2 - 1) / 2

"""Sphere, elasticity: the spherical mode U₀ x̂ and the P₂ mode U₂ P₂ x̂ + W₂ dP₂/dθ e_θ."""
function sphere_elastic(; nμ = 12, nφ = 24)
    g0(x) = x / norm(x)
    g1(x) = (r = norm(x); P2(x[3] / r) * x / r)
    # dP₂/dθ e_θ = -3 cosθ sinθ e_θ, with sinθ e_θ = (z x, z y, -(x² + y²)) / r²
    g2(x) = (r = norm(x); -3 * (x[3] / r) * [x[3] * x[1], x[3] * x[2], -(x[1]^2 + x[2]^2)] / r^2)
    loads = [
        Load("spherical", [1.0, 1, 1, 0, 0, 0], [1.0, 0, 0], Float64[]),
        Load("deviatoric", [-1.0, -1, 2, 0, 0, 0], [0.0, 2, 1], Float64[]),
    ]
    return Harmonic(3, true, Any[g0, g1, g2], Any[], sphere_angles(nμ, nφ), [2, 4], loads)
end

"""Sphere, conduction: T = Θ(r) cosθ."""
function sphere_conduction(; nμ = 12, nφ = 24)
    g(x) = x[3] / norm(x)
    return Harmonic(3, false, Any[g], Any[], sphere_angles(nμ, nφ), [2], [Load("e3", [0.0, 0, 1], [1.0], Float64[])])
end

"""Cylinder about e₃, elasticity: axisymmetric (U e_r, with u_z = ε z prescribed),
transverse shear (U cos2θ e_r + V sin2θ e_θ) and antiplane shear (W cosθ e₃, with
z e₁ prescribed)."""
function cylinder_elastic(; nθ = 32)
    ρ(x) = hypot(x[1], x[2])
    er(x) = [x[1], x[2], 0] / ρ(x)
    eθ(x) = [-x[2], x[1], 0] / ρ(x)
    gax(x) = er(x)
    gU(x) = ((x[1]^2 - x[2]^2) / ρ(x)^2) * er(x)
    gV(x) = (2 * x[1] * x[2] / ρ(x)^2) * eθ(x)
    gW(x) = (x[1] / ρ(x)) * [0, 0, 1]
    hz(x) = [0, 0, x[3]]
    hx(x) = [x[3], 0, 0]
    loads = [
        Load("in-plane dilatation", [1.0, 1, 0, 0, 0, 0], [1.0, 0, 0, 0], [0.0, 0]),
        Load("axial", [0.0, 0, 1, 0, 0, 0], [0.0, 0, 0, 0], [1.0, 0]),
        Load("transverse shear", [1.0, -1, 0, 0, 0, 0], [0.0, 1, -1, 0], [0.0, 0]),
        Load("antiplane shear", [0.0, 0, 0, 0, √2, 0], [0.0, 0, 0, 1], [0.0, 1]),
    ]
    return Harmonic(2, true, Any[gax, gU, gV, gW], Any[hz, hx], circle_angles(nθ), [1, 3], loads)
end

"""Cylinder about e₃, conduction: T = Θ(r) cosθ, and the axial gradient z prescribed."""
function cylinder_conduction(; nθ = 32)
    g(x) = x[1] / hypot(x[1], x[2])
    h(x) = x[3]
    loads = [Load("transverse", [1.0, 0, 0], [1.0], [0.0]), Load("axial", [0.0, 0, 1], [0.0], [1.0])]
    return Harmonic(2, false, Any[g], Any[h], circle_angles(nθ), [1], loads)
end

# ── Interfaces (constant in time) ────────────────────────────────────────────

"""`kind` ∈ (:perfect, :spring, :membrane, :kapitza, :surface); `p` its parameters:
(kn, kt), (κs, μs), (ρ,), (ks,)."""
struct OracleInterface
    kind::Symbol
    p::Vector{Float64}
end
OracleInterface(kind::Symbol, p::Real...) = OracleInterface(kind, collect(Float64, p))
_continuous(itf::OracleInterface) = itf.kind in (:perfect, :membrane, :surface)

# ── Radial regions ───────────────────────────────────────────────────────────

struct Region
    r::Vector{Float64}      # quadrature radii
    w::Vector{Float64}      # quadrature weights × r^(d-1)
    N::Matrix{Float64}      # nq × nn
    dN::Matrix{Float64}
    node0::Int              # first global node
    law::Int                # index of the Volterra matrix
    withH::Bool             # prescribed fields counted (not in the exterior element)
    edges::Vector{Float64}  # element ends (the exterior element: [Rt])
    p::Int                  # degree (0 for the exterior element)
    powers::Vector{Int}     # decaying exponents of the exterior element
end
nnodes(R::Region) = size(R.N, 2)

function lagrange(ξn::Vector{Float64}, ξ::Float64)
    n = length(ξn)
    N = ones(n); dN = zeros(n)
    for k in 1:n
        for m in 1:n
            m == k && continue
            N[k] *= (ξ - ξn[m]) / (ξn[k] - ξn[m])
        end
        for j in 1:n
            j == k && continue
            t = 1 / (ξn[k] - ξn[j])
            for m in 1:n
                (m == k || m == j) && continue
                t *= (ξ - ξn[m]) / (ξn[k] - ξn[m])
            end
            dN[k] += t
        end
    end
    return N, dN
end

function fe_region(edges::Vector{Float64}, p::Int, d::Int, node0::Int, law::Int; nq = p + 4)
    nel = length(edges) - 1
    nn = nel * p + 1
    ξn = collect(range(-1.0, 1.0; length = p + 1))
    rs = Float64[]; ws = Float64[]; Ns = Vector{Vector{Float64}}(); dNs = similar(Ns)
    for e in 1:nel
        a, b = edges[e], edges[e + 1]
        rq, wq = gauss_on(a, b, nq)
        ξq, _ = gauss_legendre(nq)
        for (r, w, ξ) in zip(rq, wq, ξq)
            Nl, dNl = lagrange(ξn, ξ)
            row = zeros(nn); drow = zeros(nn)
            idx = ((e - 1) * p + 1):(e * p + 1)
            row[idx] = Nl; drow[idx] = dNl .* (2 / (b - a))
            push!(rs, r); push!(ws, w * r^(d - 1)); push!(Ns, row); push!(dNs, drow)
        end
    end
    return Region(
        rs, ws, permutedims(reduce(hcat, Ns)), permutedims(reduce(hcat, dNs)), node0, law, true,
        edges, p, Int[],
    )
end

# The exterior element on [Rt, ∞): shape functions ψ₁ = (Rt/r)^s₁ (value 1 at Rt) and
# ψ_k - ψ₁ (zero at Rt) for the other exponents; r = Rt/τ, τ ∈ (0, 1].
function exterior_region(Rt::Float64, powers::Vector{Int}, d::Int, node0::Int, law::Int; nq = 12)
    τq, wτ = gauss_on(0.0, 1.0, nq)
    r = Rt ./ τq
    w = wτ .* (Rt ./ τq .^ 2) .* r .^ (d - 1)
    N = zeros(nq, length(powers)); dN = zeros(nq, length(powers))
    for g in eachindex(r)
        N[g, :], dN[g, :] = exterior_shape(Rt, powers, r[g])
    end
    return Region(r, w, N, dN, node0, law, false, [Rt], 0, powers)
end
function exterior_shape(Rt, powers, r)
    τ = Rt / r
    ψ = [τ^s for s in powers]; dψ = [-s * τ^(s + 1) / Rt for s in powers]
    return vcat(ψ[1], ψ[2:end] .- ψ[1]), vcat(dψ[1], dψ[2:end] .- dψ[1])
end

# The shape functions of a region and their derivatives at the radius r.
function shape_at(R::Region, r)
    R.p == 0 && return exterior_shape(R.edges[1], R.powers, r)
    e = clamp(searchsortedlast(R.edges, r), 1, length(R.edges) - 1)
    a, b = R.edges[e], R.edges[e + 1]
    Nl, dNl = lagrange(collect(range(-1.0, 1.0; length = R.p + 1)), 2 * (r - a) / (b - a) - 1)
    N = zeros(nnodes(R)); dN = zeros(nnodes(R))
    idx = ((e - 1) * R.p + 1):(e * R.p + 1)
    N[idx] = Nl; dN[idx] = dNl .* (2 / (b - a))
    return N, dN
end

# ── Assembly ─────────────────────────────────────────────────────────────────

# Angular moments Σ_q w_q X_c Y_dᵀ, an array [c, d] of b×b matrices.
moments(h::Harmonic, X, Y) = [sum(h.w[q] .* (X[c, q] * Y[e, q]') for q in eachindex(h.w)) for c in axes(X, 1), e in axes(Y, 1)]

struct Moments
    AA; AB; BA; BB; AH; BH
    sA; sB; sH               # Σ_q w_q X (b-vectors)
end
function Moments(h::Harmonic)
    return Moments(
        moments(h, h.A, h.A), moments(h, h.A, h.B), moments(h, h.B, h.A), moments(h, h.B, h.B),
        moments(h, h.A, h.H), moments(h, h.B, h.H),
        [sum(h.w .* h.A[c, :]) for c in 1:h.nc], [sum(h.w .* h.B[c, :]) for c in 1:h.nc],
        [sum(h.w .* h.H[e, :]) for e in 1:h.ne],
    )
end
contract(M, Φ) = [sum(M .* Φ[c, e]) for c in axes(Φ, 1), e in axes(Φ, 2)]

struct Model
    h::Harmonic
    mom::Moments
    regions::Vector{Region}
    nlayers::Int
    radii::Vector{Float64}
    itfs::Vector{OracleInterface}
    itf_nodes::Vector{Tuple{Int, Int}}   # (inner node, outer node) at each r_k
    nnode::Int
    ndof::Int
    laws::Vector{Matrix{Float64}}        # layer Volterra matrices, then the reference
    nt::Int
    Rt::Float64
    ext_node::Int
end

dof(m::Model, node, c) = (node - 1) * m.h.nc + c
hdof(m::Model, e) = m.nnode * m.h.nc + e

"""
    build_model(h, radii, laws, itfs, ref; p, nel, shell = 0.0, nel_shell)

`laws` the Volterra matrices of the layers, `ref` that of the matrix (any 6n×6n or 3n×3n
block matrix: no symmetry is assumed), `nel` the number of elements per unit length in
each layer (at least 2 per layer) unless `nel_layers` gives the count of each layer (a
mesh that moves with the radii, for differences in a radius), `shell` the outer radius
of a meshed matrix shell (0: the exterior element starts at r_N).
"""
function build_model(
        h::Harmonic, radii, laws, itfs, ref; p = 3, nel = 8, shell = 0.0, nel_shell = 8, nel_layers = nothing
    )
    nl = length(radii)
    regions = Region[]; itf_nodes = Tuple{Int, Int}[]
    node = 1
    rin = 0.0
    for k in 1:nl
        ne_k = nel_layers === nothing ? max(2, ceil(Int, nel * (radii[k] - rin))) : nel_layers[k]
        R = fe_region(collect(range(rin, radii[k]; length = ne_k + 1)), p, h.d, node, k)
        push!(regions, R)
        last = node + nnodes(R) - 1
        node = _continuous(itfs[k]) ? last : last + 1
        push!(itf_nodes, (last, node))
        rin = radii[k]
    end
    Rt = radii[end]
    if shell > Rt
        edges = exp.(range(log(Rt), log(shell); length = nel_shell + 1))
        R = fe_region(edges, p, h.d, node, nl + 1)
        push!(regions, R)
        node += nnodes(R) - 1
        Rt = shell
    end
    E = exterior_region(Rt, h.powers, h.d, node, nl + 1)
    push!(regions, E)
    nnode = node + nnodes(E) - 1
    nt = size(ref, 1) ÷ h.b
    return Model(
        h, Moments(h), regions, nl, collect(Float64, radii), collect(itfs), itf_nodes, nnode,
        nnode * h.nc + h.ne, [laws..., ref], nt, Rt, node
    )
end

blk(m::Model, i) = ((i - 1) * m.h.b + 1):(i * m.h.b)
law_block(m::Model, ℓ, i, j) = m.laws[ℓ][blk(m, i), blk(m, j)]

# K_ij of the regions (no interface), and the exterior part alone.
function stiffness(m::Model, i, j)
    h, mom = m.h, m.mom
    K = zeros(m.ndof, m.ndof); Kext = zeros(m.ndof, m.ndof)
    for (ir, R) in enumerate(m.regions)
        M = law_block(m, R.law, i, j)
        mAA, mAB, mBA, mBB = contract(M, mom.AA), contract(M, mom.AB), contract(M, mom.BA), contract(M, mom.BB)
        ir_ = 1 ./ R.r
        RAA = R.dN' * (R.w .* R.dN); RAB = R.dN' * ((R.w .* ir_) .* R.N)
        RBA = R.N' * ((R.w .* ir_) .* R.dN); RBB = R.N' * ((R.w .* ir_ .^ 2) .* R.N)
        nn = nnodes(R)
        idx = dof(m, R.node0, 1):dof(m, R.node0 + nn - 1, h.nc)
        Kr = kron(RAA, mAA) + kron(RAB, mAB) + kron(RBA, mBA) + kron(RBB, mBB)
        K[idx, idx] .+= Kr
        ir == length(m.regions) && (Kext[idx, idx] .+= Kr)
        if R.withH && h.ne > 0
            mAH, mBH = contract(M, mom.AH), contract(M, mom.BH)
            rA = R.dN' * R.w; rB = R.N' * (R.w .* ir_)
            for a in 1:nn, c in 1:h.nc, e in 1:h.ne
                K[dof(m, R.node0 + a - 1, c), hdof(m, e)] += rA[a] * mAH[c, e] + rB[a] * mBH[c, e]
            end
        end
    end
    return K, Kext
end

# Interface terms (they do not age: on the diagonal of time only).
function interface_stiffness(m::Model)
    h = m.h
    K = zeros(m.ndof, m.ndof)
    for (k, itf) in enumerate(m.itfs)
        ρ = m.radii[k]; s = ρ^(h.d - 1)
        a, b = m.itf_nodes[k]
        if itf.kind in (:spring, :kapitza)
            S = zeros(h.nc, h.nc)
            for q in eachindex(h.w), c in 1:h.nc, e in 1:h.nc
                n = h.nrm[q]
                Ks = itf.kind == :spring ? itf.p[1] * n * n' + itf.p[2] * (I - n * n') : fill(1 / itf.p[1], 1, 1)
                S[c, e] += s * h.w[q] * dot(h.gv[c, q], Ks * h.gv[e, q])
            end
            ia = dof(m, a, 1):dof(m, a, h.nc); ib = dof(m, b, 1):dof(m, b, h.nc)
            K[ia, ia] .+= S; K[ib, ib] .+= S; K[ia, ib] .-= S; K[ib, ia] .-= S
        elseif itf.kind in (:membrane, :surface)
            X = [h.SB[c, q] ./ ρ for c in 1:h.nc, q in eachindex(h.w)]
            Y = vcat(X, h.SH)                                  # unknowns, then prescribed
            nY = size(Y, 1)
            S = zeros(nY, nY)
            for q in eachindex(h.w), c in 1:nY, e in 1:nY
                S[c, e] += s * h.w[q] * _surface_energy(itf, Y[c, q], Y[e, q])
            end
            ia = [dof(m, a, c) for c in 1:h.nc]; ih = [hdof(m, e) for e in 1:h.ne]
            all_ = vcat(ia, ih)
            K[all_, all_] .+= S
        end
    end
    return K
end
function _surface_energy(itf::OracleInterface, X, Y)
    if itf.kind == :membrane
        κs, μs = itf.p; λs = κs - μs
        return λs * tr(X) * tr(Y) + 2μs * dot(X, Y)
    else
        return itf.p[1] * dot(X, Y)
    end
end
# The load of the uniform remote field on the exterior element, per unit remote amplitude:
# -(K_ext n_ĉ + ∮_{Rt} v·σ∞·x̂), σ∞ = R̃₀ ∘ ε∞.
function remote_load(m::Model, L::Load, Kext, i, j)
    h, mom = m.h, m.mom
    F = zeros(m.ndof)
    nĉ = zeros(m.ndof)
    for c in 1:h.nc
        nĉ[dof(m, m.ext_node, c)] = L.c[c] * m.Rt
    end
    F .-= Kext * nĉ
    M0 = law_block(m, length(m.laws), i, j)
    s = m.Rt^(h.d - 1)
    for c in 1:h.nc
        v = 0.0
        for e in 1:h.nc
            v += L.c[e] * (sum(M0 .* mom.AA[c, e]) + sum(M0 .* mom.AB[c, e]))
        end
        for e in 1:h.ne
            v += L.a[e] * sum(M0 .* mom.AH[c, e])
        end
        F[dof(m, m.ext_node, c)] -= s * v
    end
    return F
end

# ── Averages ─────────────────────────────────────────────────────────────────

# ∫_layer ∇u dV as a b × ndof matrix.
function layer_integral(m::Model, k)
    h, mom = m.h, m.mom
    R = m.regions[k]
    G = zeros(h.b, m.ndof)
    rA = R.dN' * R.w; rB = R.N' * (R.w ./ R.r)
    for a in 1:nnodes(R), c in 1:h.nc
        G[:, dof(m, R.node0 + a - 1, c)] .+= rA[a] .* mom.sA[c] .+ rB[a] .* mom.sB[c]
    end
    for e in 1:h.ne
        G[:, hdof(m, e)] .+= sum(R.w) .* mom.sH[e]
    end
    return G
end
# ∮ sym([u]⊗n) dS (∮ [T] n dS) across interface k.
function jump_integral(m::Model, k)
    h = m.h
    G = zeros(h.b, m.ndof)
    a, b = m.itf_nodes[k]
    a == b && return G
    s = m.radii[k]^(h.d - 1)
    for c in 1:h.nc
        G[:, dof(m, b, c)] .+= s .* m.mom.sA[c]
        G[:, dof(m, a, c)] .-= s .* m.mom.sA[c]
    end
    return G
end
# ∮ σˢ dS of a membrane (∮ kˢ∇ₛT dS of a conducting surface).
function surface_integral(m::Model, k)
    h = m.h
    itf = m.itfs[k]
    G = zeros(h.b, m.ndof)
    itf.kind in (:membrane, :surface) || return G
    ρ = m.radii[k]; s = ρ^(h.d - 1)
    a, _ = m.itf_nodes[k]
    for q in eachindex(h.w)
        P = I - h.nrm[q] * h.nrm[q]'
        flux(X) = itf.kind == :membrane ?
            mandel_sym((itf.p[1] - itf.p[2]) * tr(X) * P + 2 * itf.p[2] * X) : itf.p[1] .* vec(X)
        for c in 1:h.nc
            G[:, dof(m, a, c)] .+= s * h.w[q] .* flux(h.SB[c, q] ./ ρ)
        end
        for e in 1:h.ne
            G[:, hdof(m, e)] .+= s * h.w[q] .* flux(h.SH[e, q])
        end
    end
    return G
end

ball_volume(m::Model, r) = m.h.d == 3 ? 4π * r^3 / 3 : π * r^2

# ── Solve ────────────────────────────────────────────────────────────────────

"""
    oracle_solve(h, radii, laws, itfs, ref; external = true, p, nel, shell, nel_shell)

Solve the problem for every load of `h` applied as a unit at each instant J (the
columns of the Volterra operators). Returns a named tuple of arrays indexed
`[load, i, J]` (each a b-vector): `eps`, `sig`, `N` of the whole inclusion (its outer
interface counted when `external`), `eps_layer[k]`, `sig_layer[k]` the material
averages over layer k, and `eps_layer_ext[k]` (`eps_layer_int[k]`) the material average
plus the jump across r_k (across r_{k-1}) divided by the volume of the layer.
"""
function oracle_solve(h::Harmonic, radii, laws, itfs, ref; external = true, kw...)
    m = build_model(h, radii, laws, itfs, ref; kw...)
    nt, nL = m.nt, length(h.loads)
    known = vcat(
        [dof(m, 1, c) for c in 1:h.nc],                 # U(0) = 0
        [hdof(m, e) for e in 1:h.ne],
    )
    free = setdiff(1:m.ndof, known)
    ncol = nL * nt
    col(ℓ, J) = (ℓ - 1) * nt + J
    Q = [zeros(m.ndof, ncol) for _ in 1:nt]
    for ℓ in 1:nL, J in 1:nt, e in 1:h.ne
        Q[J][hdof(m, e), col(ℓ, J)] = h.loads[ℓ].a[e]
    end
    Kitf = interface_stiffness(m)
    for i in 1:nt
        rhs = zeros(m.ndof, ncol)
        Kii = nothing
        for j in 1:i
            K, Kext = stiffness(m, i, j)
            j == i && (K .+= Kitf; Kii = K)
            j < i && (rhs .-= K * Q[j])
            for ℓ in 1:nL
                F = remote_load(m, h.loads[ℓ], Kext, i, j)
                rhs[:, col(ℓ, j)] .-= F
            end
        end
        rhs .-= Kii[:, known] * Q[i][known, :]
        Q[i][free, :] = Kii[free, free] \ rhs[free, :]
    end
    # averages
    Gl = [layer_integral(m, k) for k in 1:m.nlayers]
    Gj = [jump_integral(m, k) for k in 1:m.nlayers]
    Gs = [surface_integral(m, k) for k in 1:m.nlayers]
    counted = external ? (1:m.nlayers) : (1:(m.nlayers - 1))
    V = ball_volume(m, radii[end])
    Vl = [ball_volume(m, radii[k]) - (k == 1 ? 0.0 : ball_volume(m, radii[k - 1])) for k in 1:m.nlayers]
    epsA = Array{Vector{Float64}}(undef, nL, nt, nt); sigA = similar(epsA); NA = similar(epsA)
    epsL = [similar(epsA) for _ in 1:m.nlayers]; sigL = [similar(epsA) for _ in 1:m.nlayers]
    epsLe = [similar(epsA) for _ in 1:m.nlayers]; epsLi = [similar(epsA) for _ in 1:m.nlayers]
    for ℓ in 1:nL, J in 1:nt
        cl = col(ℓ, J)
        intg = [[Gl[k] * Q[i][:, cl] for i in 1:nt] for k in 1:m.nlayers]
        for i in 1:nt
            e = sum(intg[k][i] for k in 1:m.nlayers) + sum(Gj[k] * Q[i][:, cl] for k in counted)
            epsA[ℓ, i, J] = e ./ V
            σ = sum(Gs[k] * Q[i][:, cl] for k in counted; init = zeros(h.b))
            for k in 1:m.nlayers
                σk = sum(law_block(m, k, i, j) * intg[k][j] for j in 1:i)
                σ += σk
                epsL[k][ℓ, i, J] = intg[k][i] ./ Vl[k]
                epsLe[k][ℓ, i, J] = (intg[k][i] + Gj[k] * Q[i][:, cl]) ./ Vl[k]
                epsLi[k][ℓ, i, J] = (intg[k][i] + (k > 1 ? Gj[k - 1] * Q[i][:, cl] : zeros(h.b))) ./ Vl[k]
                sigL[k][ℓ, i, J] = σk ./ Vl[k]
            end
            sigA[ℓ, i, J] = σ ./ V
        end
        for i in 1:nt
            NA[ℓ, i, J] = sigA[ℓ, i, J] - sum(law_block(m, length(m.laws), i, j) * epsA[ℓ, j, J] for j in 1:i)
        end
    end
    return (;
        eps = epsA, sig = sigA, N = NA, eps_layer = epsL, sig_layer = sigL,
        eps_layer_ext = epsLe, eps_layer_int = epsLi, model = m, Q,
    )
end

"""
    point_fields(res, x, region) -> (; u, grad, flux)

The fields of the solution `res` at the point `x`, read in the region `region` (k ≤ N
the layer k, N + 1 the matrix): arrays `[load, i, J]` of the displacement (temperature,
a 1-vector), of the gradient (Mandel strain, or ∇T) and of the flux R̃ ∘ gradient (σ, or
+k·∇T) of that region. The amplitudes are interpolated by the shape functions of the
region; on an interface, `region` chooses the side.
"""
function point_fields(res, x::AbstractVector, region::Int)
    m, Q = res.model, res.Q
    h = m.h
    nt, nL = m.nt, length(h.loads)
    r = _radius(x, h.d)
    n = h.d == 3 ? x ./ r : [x[1], x[2], 0.0] ./ r
    val(f, y) = h.elastic ? f(y) : [f(y)]
    gr(f, y) = h.elastic ? ForwardDiff.jacobian(f, y) : reshape(ForwardDiff.gradient(f, y), 1, 3)
    gc = [val(f, n) for f in h.patterns]
    A = [h.elastic ? mandel_sym(gc[c] * n') : gc[c][1] .* n for c in 1:h.nc]
    B = [h.elastic ? mandel_sym(gr(h.patterns[c], n)) : vec(gr(h.patterns[c], n)) for c in 1:h.nc]
    H = [h.elastic ? mandel_sym(gr(f, x)) : vec(gr(f, x)) for f in h.prescribed]
    hv = [val(f, x) for f in h.prescribed]
    R = if region ≤ m.nlayers
        m.regions[region]
    else                                              # the meshed shell, or the exterior element
        length(m.regions) > m.nlayers + 1 && r ≤ m.Rt ? m.regions[m.nlayers + 1] : m.regions[end]
    end
    N, dN = shape_at(R, r)
    ψ1, dψ1 = R.p == 0 ? (N[1], dN[1]) : (0.0, 0.0)
    u = Array{Vector{Float64}}(undef, nL, nt, nt); g = similar(u); f = similar(u)
    for ℓ in 1:nL, J in 1:nt
        cl = (ℓ - 1) * nt + J
        L = h.loads[ℓ]
        for i in 1:nt
            q = Q[i][:, cl]
            Ei = i == J ? 1.0 : 0.0
            U = [sum(N[a] * q[dof(m, R.node0 + a - 1, c)] for a in eachindex(N)) for c in 1:h.nc]
            dU = [sum(dN[a] * q[dof(m, R.node0 + a - 1, c)] for a in eachindex(N)) for c in 1:h.nc]
            if R.p == 0                               # u = u∞ + w in the exterior element
                U .+= L.c .* (r - m.Rt * ψ1) .* Ei
                dU .+= L.c .* (1 - m.Rt * dψ1) .* Ei
            end
            a = [q[hdof(m, e)] for e in 1:h.ne]
            g[ℓ, i, J] = sum(dU[c] .* A[c] .+ (U[c] / r) .* B[c] for c in 1:h.nc) +
                sum(a[e] .* H[e] for e in 1:h.ne; init = zeros(h.b))
            u[ℓ, i, J] = sum(U[c] .* gc[c] for c in 1:h.nc) + sum(a[e] .* hv[e] for e in 1:h.ne; init = zero(gc[1]))
        end
        for i in 1:nt
            f[ℓ, i, J] = sum(law_block(m, R.law, i, j) * g[ℓ, j, J] for j in 1:i)
        end
    end
    return (; u, grad = g, flux = f, normal = n)
end

"""
    compare(pkg, arr, h; loads, rb = h.b, sign = 1) -> relative error

Compare a Volterra block matrix `pkg` of the package with an oracle array `arr`
(`[load, i, J]`): block (i, J) applied to the gradient of each load, against the
oracle's response at instant i to that load applied as a unit at instant J, times
`sign`. `rb` is the number of rows per instant (3 for a displacement, 1 for a
temperature). The error is the largest absolute difference over the largest absolute
entry of the package.
"""
function compare(pkg::AbstractMatrix, arr, h::Harmonic; loads = eachindex(h.loads), rb = h.b, sign = 1)
    _, nt, _ = size(arr)
    b = h.b
    err = 0.0; sc = 0.0
    for ℓ in loads, i in 1:nt, J in 1:nt
        P = pkg[((i - 1) * rb + 1):(i * rb), ((J - 1) * b + 1):(J * b)] * h.loads[ℓ].E
        err = max(err, maximum(abs, P - sign .* arr[ℓ, i, J])); sc = max(sc, maximum(abs, P))
    end
    return err / sc
end

end # module
