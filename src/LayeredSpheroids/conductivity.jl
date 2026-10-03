# =============================================================================
#  conductivity.jl — confocal spheroidal-harmonic transfer-matrix
#  recurrence for the N-layer spheroid conduction problem
#  (Barthélémy & Bignonnet, IJES 2020, §2).
#
#  Mirrors `LayeredSpheres/conductivity.jl`'s 2×2 state-vector transfer
#  for the sphere, but the spheroid's imperfect (LC/HC) interfaces
#  COUPLE DIFFERENT HARMONIC DEGREES (the sphere's do not), so the
#  "state" here is a `2𝒩`-vector of series coefficients
#  `Xℓ = [Aℓ; Bℓ]` (𝒩 = `Nseries` truncated odd degrees `1,3,…,2𝒩-1`,
#  eq:AlBlXl) rather than a 2-vector, and the transfer "matrix" `Rℓ` is
#  `2𝒩×2𝒩` (eq:aximat) built from the diagonal blocks `𝒥(k,q)`
#  (eq:matJ) plus, for LC/HC interfaces, the full coupling perturbation
#  `δ𝒥^{LC/HC}` (eq:axiLCdJ/axiHCdJ, eq:transLCdJ/transHCdJ) assembled
#  from the `I, J, K, L` matrices of `coupling.jl`.
#
#  Axial (`trans = false`, order `m = 0`, `H = H₃ê₃`) and transverse
#  (`trans = true`, order `m = 1`, `H = H₁ê₁`) problems share the exact
#  same machinery, differing only in which Legendre table (`P0/Q0` vs
#  `P1/Q1`) and which coupling matrix (`I` vs `J` for LC,
#  `J` vs `K+L/(q²-1)` for HC) enter — see `_transition`.
#
#  Convention: `KapitzaInterface(resistance)` ↔ paper's `αℓ` (LC,
#  temperature-jump resistance, entering the TEMPERATURE block);
#  `SurfaceConductiveInterface(conductance)` ↔ paper's `βℓ` (HC,
#  flux-jump surface conductance, entering the FLUX block) — the SAME
#  convention as `LayeredSphere`'s interfaces (not to be confused with
#  the inverse convention some `echoes` call sites use for the raw
#  `interf_prop` scalar; see the theory page).
# =============================================================================

@inline function _spheroid_layer_moduli(s::LayeredSpheroid{T, N}) where {T, N}
    return ntuple(k -> MFH_Core.extract_iso_conductivity(layer_modulus(s, k)), Val(N))
end

"""
    _as_scalar_k(k₀) -> Number

Normalize a matrix conductivity argument to its isotropic scalar,
accepting either a bare `Number` or a `TensISO{2,3}` (the natural way
to call the public API, matching `gradient_gradient_loc`'s own `K₀`
convention).
"""
@inline _as_scalar_k(k₀::Number) = k₀
@inline _as_scalar_k(k₀) = MFH_Core.extract_iso_conductivity(k₀)

"""
    _spheroid_tables(s::LayeredSpheroid) -> NamedTuple

Per-layer scalar conductivities and Legendre value/derivative tables
(`P0,Q0,dP0,dQ0` for the axial problem, `P1,Q1,dP1,dQ1` for the
transverse one), each evaluated once at layer ``k``'s own confocal
parameter ``q_k``.
"""
function _spheroid_tables(s::LayeredSpheroid{T, N, Q}) where {T, N, Q}
    𝒩 = s.Nseries
    k_layers = _spheroid_layer_moduli(s)
    P0 = Vector{Vector{Q}}(undef, N); dP0 = Vector{Vector{Q}}(undef, N)
    Q0 = Vector{Vector{Q}}(undef, N); dQ0 = Vector{Vector{Q}}(undef, N)
    P1 = Vector{Vector{Q}}(undef, N); dP1 = Vector{Vector{Q}}(undef, N)
    Q1 = Vector{Vector{Q}}(undef, N); dQ1 = Vector{Vector{Q}}(undef, N)
    for k in 1:N
        q = s.q[k]
        P0[k], dP0[k] = legendre_odd(:P0, q, 𝒩)
        Q0[k], dQ0[k] = legendre_odd(:Q0, q, 𝒩)
        P1[k], dP1[k] = legendre_odd(:P1, q, 𝒩)
        Q1[k], dQ1[k] = legendre_odd(:Q1, q, 𝒩)
    end
    return (; k_layers, P0, dP0, Q0, dQ0, P1, dP1, Q1, dQ1)
end

"""
    _J_ext(k, P, Qf, dP, dQ) -> Matrix

The ``2\\mathcal{N}\\times 2\\mathcal{N}`` diagonal block matrix ``\\mathcal{J}(k,q)`` (eq:matJ),
with ``P``, ``Q``, ``P'``, ``Q'`` the arguments `P`, `Qf`, `dP`, `dQ`:

```math
\\mathcal{J}(k,q) = \\begin{pmatrix}
\\mathrm{diag}(P) & \\mathrm{diag}(Q)\\\\
k\\,\\mathrm{diag}(P') & k\\,\\mathrm{diag}(Q')
\\end{pmatrix}.
```
"""
function _J_ext(k, P::Vector{Qx}, Qf::Vector{Qx}, dP::Vector{Qx}, dQ::Vector{Qx}) where {Qx}
    𝒩 = length(P)
    Tm = promote_type(typeof(k), Qx)
    M = zeros(Tm, 2𝒩, 2𝒩)
    for r in 1:𝒩
        M[r, r] = P[r]
        M[r, 𝒩 + r] = Qf[r]
        M[𝒩 + r, r] = k * dP[r]
        M[𝒩 + r, 𝒩 + r] = k * dQ[r]
    end
    return M
end

"""
    _J_int(k, coef, P, Qf, dP, dQ, M, dual, trans) -> Matrix

``\\mathcal{J}(k,q) + \\delta\\mathcal{J}^{\\mathrm{LC/HC}}``: the diagonal block `_J_ext` plus the full
coupling perturbation of generic row/column term
``[\\delta\\mathcal{J}]_{rs} = c\\,\\frac{4r-1}{2}\\,M_{rs}\\,\\mathcal{R}_{2s-1}``, ``c`` = `coef` (``\\mathcal{R} = P, Q`` for HC,
``\\mathcal{R} = P', Q'`` for LC), divided by ``(2r-1)(2r)`` for the transverse
problem (eq:axiLCdJ/axiHCdJ/transLCdJ/transHCdJ). `dual = true` (HC)
perturbs the FLUX (lower) block using unprimed ``\\mathcal{R}``; `dual = false`
(LC) perturbs the TEMPERATURE (upper) block using primed ``\\mathcal{R}'``.
"""
function _J_int(
        k, coef, P::Vector{Qx}, Qf::Vector{Qx}, dP::Vector{Qx}, dQ::Vector{Qx},
        M::AbstractMatrix, dual::Bool, trans::Bool,
    ) where {Qx}
    𝒩 = length(P)
    # `coef` carries the interface parameter (Kapitza resistance / surface
    # conductance), so it must enter the element type: `eltype(_J_ext(...))`
    # alone knows only about `k` and the Legendre tables, and a `Dual` `coef`
    # written into a `Float64` buffer is a `MethodError` — which is what made
    # every sensitivity with respect to an interface parameter unreachable.
    J0 = _J_ext(k, P, Qf, dP, dQ)
    Tm = promote_type(eltype(J0), typeof(coef), eltype(M))
    J = convert(Matrix{Tm}, J0)
    MP = zeros(Tm, 𝒩, 𝒩)
    MQ = zeros(Tm, 𝒩, 𝒩)
    for r in 1:𝒩
        i = 2r - 1
        ci = coef * (4r - 1) / 2
        trans && (ci = ci / (i * (i + 1)))
        Rp, Rq = dual ? (P, Qf) : (dP, dQ)
        for c in 1:𝒩
            MP[r, c] = M[r, c] * ci * Rp[c]
            MQ[r, c] = M[r, c] * ci * Rq[c]
        end
    end
    if dual
        @views J[(𝒩 + 1):(2𝒩), 1:𝒩] .+= MP
        @views J[(𝒩 + 1):(2𝒩), (𝒩 + 1):(2𝒩)] .+= MQ
    else
        @views J[1:𝒩, 1:𝒩] .+= MP
        @views J[1:𝒩, (𝒩 + 1):(2𝒩)] .+= MQ
    end
    return J
end

"""
    _transition(s, layer, trans, k_next, tables) -> Matrix

Transfer matrix ``R_\\ell = \\mathcal{J}(k_{\\ell+1}, q_\\ell)^{-1}\\,\\bigl(\\mathcal{J}(k_\\ell, q_\\ell) + \\delta\\mathcal{J}\\bigr)``
(eq:aximat) across the interface at the outer boundary of `layer`.
`k_next` is the conductivity on the other side (next layer's, or the
matrix ``k_0`` if `layer == N`).
"""
function _transition(
        s::LayeredSpheroid{T, N, Q}, layer::Int, trans::Bool, k_next, tables,
    ) where {T, N, Q}
    q = s.q[layer]
    kℓ = tables.k_layers[layer]
    intf = layer_interface(s, layer)
    P, Qf, dP, dQ = trans ?
        (tables.P1[layer], tables.Q1[layer], tables.dP1[layer], tables.dQ1[layer]) :
        (tables.P0[layer], tables.Q0[layer], tables.dP0[layer], tables.dQ0[layer])

    Jint = if intf isa PerfectInterface
        _J_ext(kℓ, P, Qf, dP, dQ)
    elseif intf isa KapitzaInterface
        Im, Jm, Km, Lm = coupling_matrices(q, s.Nseries)
        M = trans ? Jm : Im
        coef = kℓ * intf.resistance / s.c * sqrt(q^2 - 1)
        _J_int(kℓ, coef, P, Qf, dP, dQ, M, false, trans)
    elseif intf isa SurfaceConductiveInterface
        Im, Jm, Km, Lm = coupling_matrices(q, s.Nseries)
        M = trans ? (Km .+ Lm ./ (q^2 - 1)) : Jm
        coef = intf.conductance / (s.c * sqrt(q^2 - 1))
        _J_int(kℓ, coef, P, Qf, dP, dQ, M, true, trans)
    else
        throw(ArgumentError("LayeredSpheroid: unsupported interface type $(typeof(intf))"))
    end
    Jext_next = _J_ext(k_next, P, Qf, dP, dQ)
    return Jext_next \ Jint
end

"""
    spheroid_state_sequence(s, k₀, trans) -> Vector{Vector}

Series coefficient vectors ``X_\\ell = [A_\\ell; B_\\ell]`` (eq:AlBlXl), ``\\ell = 1, \\dots, N+1``
(the ``(N+1)``-th being the matrix), for the axial (`trans = false`) or
transverse (`trans = true`) problem in an isotropic matrix ``k_0``.
``B_1 = 0`` (core regularity, eq:axiBCbi) and
``A_{N+1} = (\\pm 1, 0, \\dots, 0)`` (unit remote field, ``+`` axial / ``-``
transverse, eq:axiBCai/eq:transBCai) are imposed exactly.
"""
function spheroid_state_sequence(s::LayeredSpheroid{T, N, Q}, k₀raw, trans::Bool) where {T, N, Q}
    k₀ = _as_scalar_k(k₀raw)
    tables = _spheroid_tables(s)
    𝒩 = s.Nseries
    # The layer conductivities and the interface parameters belong in this
    # promotion just as much as `q` and `k₀` do: `_J_ext`/`_J_int` widen locally
    # to whatever `kℓ` and `eltype(intf)` are, and a narrower buffer here then
    # refuses to store the widened transfer matrix. That is what made a single
    # `ForwardDiff.Dual` layer conductivity — or any sensitivity with respect to
    # a Kapitza resistance or a surface conductance — a `MethodError`.
    Qk = promote_type(
        Q, typeof(k₀),
        ntuple(ℓ -> typeof(tables.k_layers[ℓ]), Val(N))...,
        interfaces_eltype(s.interfaces)
    )

    R = Vector{Matrix{Qk}}(undef, N)
    for ℓ in 1:N
        k_next = ℓ < N ? tables.k_layers[ℓ + 1] : k₀
        R[ℓ] = _transition(s, ℓ, trans, k_next, tables)
    end
    S = Vector{Matrix{Qk}}(undef, N)
    S[1] = R[1]
    for ℓ in 2:N
        S[ℓ] = R[ℓ] * S[ℓ - 1]
    end

    Ainf = zeros(Qk, 𝒩)
    Ainf[1] = trans ? -one(Qk) : one(Qk)
    X1 = zeros(Qk, 2𝒩)
    @views X1[1:𝒩] = S[N][1:𝒩, 1:𝒩] \ Ainf

    X = Vector{Vector{Qk}}(undef, N + 1)
    X[1] = X1
    for ℓ in 1:N
        X[ℓ + 1] = S[ℓ] * X1
    end

    # The far-field condition IS `A_matrix = Ainf`: apart from the imposed
    # uniform remote field at degree 1, every GROWING amplitude in the matrix
    # vanishes identically, or the temperature would blow up at infinity.
    # `X1` was solved to make `S[N][1:𝒩,1:𝒩] * X1[1:𝒩]` equal `Ainf`, so
    # recomputing that product only reintroduces the linear solve's residual —
    # `O(1e-17)` instead of the exact zero.  Harmless in the near field, fatal
    # away from the particle: those residues multiply `P_{2r-1}(q) ~ q^{2r-1}`,
    # so at `Nseries = 12` and 500 particle radii a `1e-26` amplitude is
    # amplified past `1e40`, and raising `Nseries` made it worse rather than
    # better.  Write the boundary condition down instead of recovering it.
    @views X[N + 1][1:𝒩] .= Ainf
    return X
end

# ── Shape functions 𝒯ₐ, 𝒯ₜ, 𝒰ₐ, 𝒰ₜ (eq:TaTt, eq:VaVt) ─────────────────────

@inline _shape_Ta(q) = _arccoth(q) - 1 / q
@inline _shape_Tt(q) = _arccoth(q) - q / (q^2 - 1)
@inline _shape_Ua(q) = _arccoth(q) - q / (q^2 - 1)
@inline _shape_Ut(q) = _arccoth(q) + (2 - q^2) / (q * (q^2 - 1))

"""
    spheroid_ba_ratios(s, k₀) -> (ba_axial, ba_trans)

The two ratios ``b^0_{N+1,1}/a^0_{N+1,1}`` and ``b^1_{N+1,1}/a^1_{N+1,1}``
(eq:axiasb/eq:transasb) driving the volume-averaged concentration
tensors. Kept in their native (possibly complex, for the oblate
substitution ``q = \\mathrm{i}\\tau``) type — casting to real is only valid on the
FINAL shape-function product (``(b/a)\\,\\mathcal{T}(q_N)`` or ``(b/a)\\,\\mathcal{U}(q_N)``, done in
`spheroid_gradient_gradient` / `spheroid_flux_gradient` /
`_spheroid_concentration`), never on ``b/a`` alone.
"""
function spheroid_ba_ratios(s::LayeredSpheroid{T, N}, k₀) where {T, N}
    𝒩 = s.Nseries
    Xa = spheroid_state_sequence(s, k₀, false)
    Xt = spheroid_state_sequence(s, k₀, true)
    ba_a = Xa[end][𝒩 + 1] / Xa[end][1]
    ba_t = Xt[end][𝒩 + 1] / Xt[end][1]
    return ba_a, ba_t
end

"""
    spheroid_gradient_gradient(s, k₀) -> (αt, αa)

Real axial/transverse gradient-concentration scalars
``\\alpha_a = 1 + (b/a)|_a\\,\\mathcal{T}_a(q_N)``, ``\\alpha_t = 1 + (b/a)|_t\\,\\mathcal{T}_t(q_N)``
(eq:avgradTN/eq:transavgradTN, unit remote field), such that
``\\langle\\nabla T\\rangle_\\Omega = \\mathrm{diag}(\\alpha_t, \\alpha_t, \\alpha_a)\\cdot\\nabla T^{\\infty}`` in the spheroid's own (axis-aligned)
frame.
"""
function spheroid_gradient_gradient(s::LayeredSpheroid{T, N}, k₀) where {T, N}
    qN = s.q[N]
    ba_a, ba_t = spheroid_ba_ratios(s, k₀)
    αa = real(1 + ba_a * _shape_Ta(qN))
    αt = real(1 + ba_t * _shape_Tt(qN))
    return αt, αa
end

"""
    spheroid_flux_gradient(s, k₀) -> (βt, βa)

Real axial/transverse flux-concentration scalars
``\\beta_a = 1 + (b/a)|_a\\,\\mathcal{U}_a(q_N)``, ``\\beta_t = 1 + (b/a)|_t\\,\\mathcal{U}_t(q_N)``
(eq:avuN/eq:transavuN), such that
``\\langle\\boldsymbol{K}\\cdot\\nabla T\\rangle_\\Omega = k_0\\,\\mathrm{diag}(\\beta_t, \\beta_t, \\beta_a)\\cdot\\nabla T^{\\infty}`` in the spheroid's own frame
(before any surface-conductive-interface flux correction — none is
needed here: the HC surface-flux term is already folded into ``(b/a)``
through the transfer-matrix recurrence, unlike `LayeredSphere` where it
is a separate additive term).
"""
function spheroid_flux_gradient(s::LayeredSpheroid{T, N}, k₀) where {T, N}
    qN = s.q[N]
    ba_a, ba_t = spheroid_ba_ratios(s, k₀)
    βa = real(1 + ba_a * _shape_Ua(qN))
    βt = real(1 + ba_t * _shape_Ut(qN))
    return βt, βa
end
