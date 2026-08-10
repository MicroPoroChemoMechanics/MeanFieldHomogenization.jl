# =============================================================================
#  scheme_types.jl — concrete homogenization-scheme types and the
#  differential-trajectory hierarchy.
#
#  Each scheme is a subtype of `HomogenizationScheme`; bounds and one-shot
#  schemes are singletons (`Voigt`, `MoriTanaka`, …) while the iterative ones
#  (`SelfConsistent`, `AsymmetricSelfConsistent`, `DifferentialScheme`) carry
#  configuration. Solver algorithms for the self-consistent family are
#  themselves marker types (`AndersonDefault`, `NewtonDefault`, plus any
#  algorithm provided by the SciML `NonlinearSolve` weak extension).
# =============================================================================

"""
    HomogenizationScheme

Supertype for every mean-field homogenization scheme. Concrete subtypes:

- bounds — [`Voigt`](@ref), [`Reuss`](@ref) ;
- one-shot with matrix — [`Dilute`](@ref), [`DiluteDual`](@ref),
  [`MoriTanaka`](@ref), [`Maxwell`](@ref), [`PonteCastanedaWillis`](@ref) ;
- iterative — [`SelfConsistent`](@ref), [`AsymmetricSelfConsistent`](@ref) ;
- trajectory-based — [`DifferentialScheme`](@ref).
"""
abstract type HomogenizationScheme end

# ── Bounds ───────────────────────────────────────────────────────────────────

"""
    Voigt() <: HomogenizationScheme

Voigt (uniform-strain) upper bound: ``\\langle \\mathbb{C} \\rangle``.
"""
struct Voigt <: HomogenizationScheme end

"""
    Reuss() <: HomogenizationScheme

Reuss (uniform-stress) lower bound: ``\\langle \\mathbb S \\rangle^{-1}``.
"""
struct Reuss <: HomogenizationScheme end

# ── Exact periodic solutions (require an ordered cell, not an RVE) ───────────

"""
    Laminated() <: HomogenizationScheme

Exact solution of the periodic **multilayer** unit cell — a stack of parallel
layers of common normal `n`, with no matrix and no reference medium. Applies
to a `Laminate` cell, not to an [`RVE`](@ref).

Unlike every other scheme in this file it is not an estimate: for a laminate
it *is* the answer,

```math
\\mathbb{C}^{hom} = \\langle\\mathbb Q\\rangle
 + \\langle\\mathbb{C} : \\mathbb{P}\\rangle : \\langle\\mathbb{P}\\rangle^{\\dagger}
   : \\langle\\mathbb{P} : \\mathbb{C}\\rangle ,
```

with `ℙ_i` the flat-inclusion Hill tensor of layer `i` and
`ℚ_i = ℂ_i − ℂ_i:ℙ_i:ℂ_i`. Serves elasticity and transport, and accepts
imperfect interfaces of spring / membrane / Kapitza / surface-conductive
type. See the theory page on the laminate and [backus1962](@cite).
"""
struct Laminated <: HomogenizationScheme end

# ── One-shot schemes (require a matrix phase) ────────────────────────────────

"""
    Dilute() <: HomogenizationScheme

Dilute scheme: ``\\mathbb{C}_{\\mathrm{eff}} = \\mathbb{C}_0 + \\sum_i f_i \\mathbb N_i``
where ``\\mathbb N_i = (\\mathbb{C}_i - \\mathbb{C}_0):\\mathbb{A}_{\\varepsilon\\varepsilon}^{(i)}``
is the size-independent stiffness contribution
([Eshelby 1957](@cite eshelby1957);
[Kachanov & Sevostianov 2018](@cite kachanov2018)).
"""
struct Dilute <: HomogenizationScheme end

"""
    DiluteDual() <: HomogenizationScheme

Dual dilute scheme on the compliance:
``\\mathbb S_{\\mathrm{eff}} = \\mathbb S_0 + \\sum_i f_i \\mathbb H_i``,
returning ``\\mathbb{C}_{\\mathrm{eff}} = \\mathbb S_{\\mathrm{eff}}^{-1}``.
"""
struct DiluteDual <: HomogenizationScheme end

"""
    MoriTanaka() <: HomogenizationScheme

Mori-Tanaka scheme ([Mori & Tanaka 1973](@cite mori1973);
[Christensen 1990](@cite christensen1990)).
"""
struct MoriTanaka <: HomogenizationScheme end

"""
    Maxwell() <: HomogenizationScheme

Maxwell homogenization, using the RVE's distribution shape as the
reference for the Hill polarization tensor.
"""
struct Maxwell <: HomogenizationScheme end

"""
    PonteCastanedaWillis() <: HomogenizationScheme

Ponte-Castañeda & Willis 1995 scheme — distribution-shape-aware
generalization of Mori-Tanaka.
"""
struct PonteCastanedaWillis <: HomogenizationScheme end

# ── N-body schemes (act on a `ParticleAssembly`, not on an `RVE`) ────────────
#
# Both types are declared here so that the scheme hierarchy stays in one file
# and `SCHEME_ALIAS` can reach them, but their `_evaluate` methods live in the
# `Assemblies` sub-module — the same split as `Laminated`, whose kernel lives
# in `Laminates`.  Neither scheme accepts an `RVE`: a bag of volume fractions
# does not say where the inclusions are, and that is exactly the information
# these two models are about.

"""
    ClusterModel(; cluster_radius = nothing, kwargs...) <: HomogenizationScheme

Cluster model of [Molinari & El Mouden 1996](@cite molinari1996): an N-body
scheme in which the mean strain of every inclusion is solved for, accounting
for the pairwise interaction with every neighbor inside a cluster of radius
`R_c`, on top of the interaction with the matrix.

Acts on a [`ParticleAssembly`](@ref MeanFieldHomogenization.Assemblies.ParticleAssembly), not on an `RVE`. Writing
``\\delta\\mathbb{C}_K = \\mathbb{C}_K - \\mathbb{C}_m``, the localization tensors
of the families solve

```math
\\sum_K \\mathbb{M}_{IK} : \\mathbb{A}_K = \\mathbb{I} ,\\qquad
\\mathbb{M}_{IK} = \\begin{cases}
 \\mathbb{I} + \\big[(1-f_K)\\mathbb{P}_0 - \\bar{\\mathbb{T}}_{II}\\big] : \\delta\\mathbb{C}_I
   & K = I \\\\
 -\\big[\\bar{\\mathbb{T}}_{IK} + f_K \\mathbb{P}_0\\big] : \\delta\\mathbb{C}_K & K \\ne I
\\end{cases}
```

with ``\\bar{\\mathbb{T}}_{IK}`` the sum of the pairwise interaction tensors
([`interaction_tensor`](@ref MeanFieldHomogenization.Interactions.interaction_tensor)) over the cluster. `cluster_radius` overrides the
cutoff carried by the assembly's [`PeriodicBox`](@ref MeanFieldHomogenization.Assemblies.PeriodicBox).

Reducing the cluster to a single inclusion (`cluster_radius = 0`) makes every
``\\bar{\\mathbb{T}}`` vanish and the scheme degenerates **exactly** onto
[`MoriTanaka`](@ref) — the identity proved in Appendix C of the paper, and the
sharpest available check on an implementation.
"""
struct ClusterModel{K <: NamedTuple} <: HomogenizationScheme
    options::K
end
ClusterModel(; kwargs...) = ClusterModel(NamedTuple(kwargs))

"""
    EquivalentInclusion(; order = 0, kwargs...) <: HomogenizationScheme

Equivalent inclusion method in the variational (Galerkin) form of
[Brisard et al. 2014](@cite brisard2014) — a Galerkin discretization of the
weak form of the Lippmann-Schwinger equation, with the polarization taken
piecewise constant over each inclusion at `order = 0`.

Acts on a [`ParticleAssembly`](@ref MeanFieldHomogenization.Assemblies.ParticleAssembly). The polarizations solve

```math
\\Big[(\\mathbb{C}_a - \\mathbb{C}_0)^{-1} + \\mathbb{P}_a - f_a \\mathbb{P}_\\Omega\\Big] : \\boldsymbol{\\tau}_a
 - \\sum_{b \\ne a}\\Big[\\mathbb{T}^{ab} + f_b \\mathbb{P}_\\Omega\\Big] : \\tau_b = E ,
```

and the apparent stiffness follows from
``\\mathbb{C}^{app} : E = \\mathbb{C}_0 : E + \\sum_a f_a \\boldsymbol{\\tau}_a``. The term in
``\\mathbb{P}_\\Omega``, the Hill tensor of the SVE domain itself, is what
implements the mixed boundary conditions of the paper — with
[`MixedBC`](@ref MeanFieldHomogenization.Assemblies.MixedBC) no periodization and no conditionally convergent lattice sum
are needed.

Because the method minimizes a Hashin-Shtrikman functional over a
finite-dimensional space, the estimate is a **rigorous bound** on the apparent
stiffness whenever the matrix is stiffer (upper bound) or softer (lower bound)
than every inhomogeneity — see [`eim_bound_type`](@ref MeanFieldHomogenization.Assemblies.eim_bound_type).

Brisard et al. note in their §3.1 that at `order = 0` their influence
pseudotensors coincide with the interaction tensors of
[Berveiller et al. 1987](@cite berveiller1987) and
[Molinari & El Mouden 1996](@cite molinari1996); the two schemes of this
package accordingly share [`interaction_tensor`](@ref MeanFieldHomogenization.Interactions.interaction_tensor) and agree on a periodic
assembly with the same cutoff.
"""
struct EquivalentInclusion{K <: NamedTuple} <: HomogenizationScheme
    options::K
end
EquivalentInclusion(; kwargs...) = EquivalentInclusion(NamedTuple(kwargs))

# ── Self-consistent solvers (built-in markers) ───────────────────────────────

"""
    AndersonDefault()

Marker selecting the built-in Anderson-accelerated fixed-point solver
(default for [`SelfConsistent`](@ref)).  Pure Julia, Dual-safe.
"""
struct AndersonDefault end

"""
    NewtonDefault()

Marker selecting the built-in Newton-Raphson solver with ForwardDiff
Jacobian (alternative to [`AndersonDefault`](@ref)).
"""
struct NewtonDefault end

"""
    AutoNonlinear()

Marker selecting an **auto-resolving** nonlinear solver: at each call it
checks, at runtime, whether the weak extension
`MeanFieldHomogenizationNonlinearSolveExt` is active (`Base.get_extension`) — i.e.
whether `NonlinearSolve.jl` has been loaded into the session, whether by
an explicit `using NonlinearSolve` or transitively through some other
dependency (empirically, on a typical installation today the extension
is already active as soon as `MeanFieldHomogenization` itself is loaded — the
exact transitive path depends on the resolved dependency graph and is
not guaranteed across versions).

- If active, it dispatches to a globalized SciML algorithm
  (`NonlinearSolve.TrustRegion()` — chosen over a plain `NewtonRaphson()`
  for its better robustness near the self-consistent bifurcation),
  through the same ForwardDiff-safe path (implicit-function-theorem
  lift) as any other `NonlinearSolve.jl` algorithm passed explicitly.
- Otherwise (a slimmed-down dependency set, or a future `OrdinaryDiffEq`
  that no longer needs `NonlinearSolve.jl` internally) it falls back to
  the dependency-free built-in [`NewtonDefault`](@ref).

This gives a solver choice that always works, with or without
`NonlinearSolve.jl` explicitly requested.

`AutoNonlinear` is **not** the default of [`SelfConsistent`](@ref) /
[`AsymmetricSelfConsistent`](@ref) (which remains
[`AndersonDefault`](@ref)): this is a *numerical-robustness* choice, not
a dependency-cost one — the porous-percolation regime relies on the
Picard positive-definite guard and `select_best` to track the physical
branch through the bifurcation, a property a root-finder does not share.
Pass `algorithm = AutoNonlinear()` explicitly to opt in.
"""
struct AutoNonlinear end

"""
    SelfConsistent(; algorithm = AndersonDefault(), kwargs...) <: HomogenizationScheme

Self-consistent scheme. The `algorithm` selects the non-linear solver;
default is the built-in Anderson acceleration. Any solver from the
SciML `NonlinearSolve.jl` package can be passed once `using NonlinearSolve`
activates the weak extension `MeanFieldHomogenizationNonlinearSolveExt`.

Standard kwargs forwarded to the solver: `abstol`, `reltol`,
`maxiters`, `damping`, `verbose`.
"""
struct SelfConsistent{A, K <: NamedTuple} <: HomogenizationScheme
    algorithm::A
    options::K
end
SelfConsistent(; algorithm = AndersonDefault(), kwargs...) =
    SelfConsistent(algorithm, NamedTuple(kwargs))

"""
    AsymmetricSelfConsistent(; algorithm = AndersonDefault(), kwargs...) <: HomogenizationScheme

Asymmetric self-consistent scheme: iterates in stiffness or compliance
space depending on the matrix-vs-Voigt-bound contrast, providing a
better behavior than [`SelfConsistent`](@ref) in matrix-stiff /
inclusion-soft regimes.
"""
struct AsymmetricSelfConsistent{A, K <: NamedTuple} <: HomogenizationScheme
    algorithm::A
    options::K
end
AsymmetricSelfConsistent(; algorithm = AndersonDefault(), kwargs...) =
    AsymmetricSelfConsistent(algorithm, NamedTuple(kwargs))

# ── Differential scheme + trajectories ───────────────────────────────────────

"""
    DifferentialTrajectory

Supertype describing the path through the multi-phase volume-fraction
space used by the [`DifferentialScheme`](@ref) scheme. Concrete subtypes:

- [`Proportional`](@ref) (default) — every phase grows linearly with
  the fictitious incorporation time `τ ∈ [0, 1]`, all phases reaching
  their target simultaneously.
- [`Sequential`](@ref) — phases are introduced one after the other in
  the user-supplied order, each occupying a contiguous slice of `τ`.
- [`CustomPath`](@ref) — explicit per-phase trajectory as a vector of
  monotone non-decreasing values; piecewise-linear interpolated along
  `τ ∈ [0, 1]`.
- [`Path`](@ref) — explicit per-phase trajectory as a callable
  `τ -> f(τ)` (auto-differentiated by `ForwardDiff`); the natural API
  for the multi-phase incorporation-sequence ODE
  ([Norris 1985](@cite norris1985); the user's hand-written DEM note).
"""
abstract type DifferentialTrajectory end

"""
    Proportional() <: DifferentialTrajectory

All phases grow proportionally during the differential integration.
"""
struct Proportional <: DifferentialTrajectory end

"""
    Sequential(order::Vector{Symbol}) <: DifferentialTrajectory

Introduce phases in the given `order`. The first phase ramps from 0 to
its target fraction over the steps it owns, then is frozen; the next
phase ramps over its allotted steps; and so on.
"""
struct Sequential <: DifferentialTrajectory
    order::Vector{Symbol}
end
Sequential(first::Symbol, rest::Symbol...) = Sequential(Symbol[first, rest...])

"""
    CustomPath(path::Dict{Symbol, <:AbstractVector{<:Real}}) <: DifferentialTrajectory
    CustomPath(:phase => values, ...)

Explicit per-phase trajectory. `path[:phase]` must be a length-`N`
monotone vector with `path[:phase][1] = 0` and `path[:phase][end] = 1`,
where `N` is the number of differential steps.

The pair form is a convenience constructor for the same thing:
`CustomPath(:I1 => [0.0, 0.5, 1.0], :I2 => [0.0, 0.2, 1.0])`.
"""
struct CustomPath{D <: AbstractDict{Symbol, <:AbstractVector{<:Real}}} <: DifferentialTrajectory
    path::D
end
CustomPath(first::Pair{Symbol}, rest::Pair{Symbol}...) =
    CustomPath(Dict(first, rest...))

"""
    Path(path::Dict{Symbol, <:Function}) <: DifferentialTrajectory
    Path(:phase => τ -> f(τ), ...)

Explicit per-phase trajectory as a callable.  `path[:phase]` is a
function of the fictitious incorporation time `τ ∈ [0, 1]` returning
the **effective volume fraction ratio** `f_α(τ) / f_α^∞ ∈ [0, 1]` for
solid phases, or the **density ratio** `ε_α(τ) / ε_α^∞` for crack
phases — `f(0) = 0`, `f(1) = 1`, monotone non-decreasing.

The derivative `df/dτ` is computed by `ForwardDiff.derivative` at
each ODE step.  This is the natural API for the multi-phase DEM
incorporation-sequence ODE :

```math
\\frac{\\mathrm d \\mathbb{C}^{hom}}{\\mathrm d \\tau}
  = \\sum_i \\frac{\\mathrm d \\varphi_i}{\\mathrm d \\tau}
            (\\mathbb{C}_i - \\mathbb{C}^{hom}):\\mathbb{A}_i^{dil}(\\mathbb{C}^{hom})
```

with the volumetric balance `dφ = (𝟙 - f ⊗ 𝐔)^{-1} · df` (Sherman-
Morrison) inverted at each `τ` to translate user-supplied `f_α(τ)`
into the increments `dφ_α / dτ`.

The single-phase case is degenerate (`f₁` itself serves as `τ`) and
does not require a `Path` — the default [`Proportional`](@ref) is
sufficient.

The pair form is a convenience constructor for the same thing:

```julia
Path(:I1 => τ -> τ^2, :I2 => τ -> 2τ - τ^2)
```
"""
struct Path{D <: AbstractDict{Symbol, <:Function}} <: DifferentialTrajectory
    path::D
end
Path(first::Pair{Symbol}, rest::Pair{Symbol}...) = Path(Dict(first, rest...))

"""
    DifferentialScheme(; trajectory = Proportional(), nsteps::Int = 100,
                         abstol::Real = 1e-8, reltol::Real = 1e-6,
                         alg = nothing, formulation = :stiffness, kwargs...)

Differential scheme : integrates the Norris ODE on the fictitious
incorporation time `τ ∈ [0, 1]` ([Norris 1985](@cite norris1985)) :

```math
\\frac{\\mathrm d \\mathbb{C}^{hom}}{\\mathrm d \\tau}
  = \\sum_i \\frac{\\mathrm d \\varphi_i}{\\mathrm d \\tau}\\,
            (\\mathbb{C}_i - \\mathbb{C}^{hom}):\\mathbb{A}_i^{dil}(\\mathbb{C}^{hom})
```

with the volume balance `df = (𝟙 - f ⊗ 𝐔)·dφ` inverted by Sherman-
Morrison so the user supplies effective volume fractions `f_α(τ)`
along the chosen `trajectory`.

# Keyword arguments

- `trajectory` — one of [`Proportional`](@ref), [`Sequential`](@ref),
  [`CustomPath`](@ref), [`Path`](@ref).  Default `Proportional()`.
- `formulation` — `:stiffness` (default) integrates the ODE above;
  `:compliance` integrates its exact dual
  ``\\mathrm d \\mathbb S^{hom} / \\mathrm d \\tau =
  \\sum_i \\dot\\varphi_i \\, \\mathbb H_i(\\mathbb S^{hom})``
  and inverts the result, so both return the same declared property.
  The two agree analytically (`ℍ = −𝕊 : 𝐍 : 𝕊`) and differ only in
  which variable carries the solver's error control: prefer
  `:compliance` for a medium softening towards percolation (porous,
  cracked), `:stiffness` for a stiffening one.
- `nsteps` — density of save points along `τ` (passed as `saveat` to
  the SciML ODE solver).  The integration step is controlled by
  `abstol` / `reltol`, **not** by `nsteps`.  See
  [`differential_path`](@ref) to read the saved states back.
- `abstol`, `reltol` — ODE solver tolerances (forwarded to
  `OrdinaryDiffEq.solve`).
- `alg` — explicit ODE algorithm.  `nothing` selects `Tsit5()` (5th
  order adaptive Runge-Kutta).  Pass any `OrdinaryDiffEqAlgorithm`
  instance to override (e.g. `Vern9()` for higher accuracy).  Implicit
  algorithms must be built with a non-AD Jacobian
  (`Rosenbrock23(autodiff = AutoFiniteDiff())`): the RHS calls the
  Hill-tensor backends, which are not differentiable with respect to
  the ODE state.
- `kwargs...` — any other keyword is forwarded verbatim to
  `OrdinaryDiffEq.solve` (`maxiters`, `dtmax`, `dt`, `callback`, …).
"""
struct DifferentialScheme{P <: DifferentialTrajectory, K <: NamedTuple} <: HomogenizationScheme
    trajectory::P
    options::K
end
DifferentialScheme(;
    trajectory = Proportional(),
    nsteps::Int = 100,
    abstol::Real = 1.0e-8,
    reltol::Real = 1.0e-6,
    alg = nothing,
    formulation::Symbol = :stiffness,
    kwargs...
) =
    DifferentialScheme(
    trajectory,
    (; nsteps, abstol, reltol, alg, formulation, kwargs...)
)
