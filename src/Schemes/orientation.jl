# =============================================================================
#  orientation.jl — discretized orientation families.
#
#  A uniform spatial distribution of spheroid axes can be represented by a
#  finite set of polar-angle families, each exactly averaged over its
#  azimuthal orbit about the global axis (`TISymmetrize(axis)`), with
#  solid-angle weights.  This is the classical quasi-brittle strength [pichler2011](@cite)
#  discretization of hydrate-needle orientations (echoes' per-bin
#  `symmetrize=[TI]` phases).
# =============================================================================

"""
    polar_orientation_bins(N) -> Vector{@NamedTuple{θ, weight}}

Discretize the polar angle ``\\theta \\in [0, \\pi/2]`` into ``N`` families with
solid-angle weights on the hemisphere, following the quasi-brittle strength model of [pichler2011](@citet)
and the `disc_theta` convention of Echoes:

- family angles ``\\theta_i = \\frac{\\pi}{2}\\,\\frac{i-1}{N-1}`` (endpoints included),
- bin edges at the mid-points, ``\\theta_i^{\\pm} = \\frac{\\pi}{2}\\,\\frac{i-1\\pm 1/2}{N-1}`` clamped to
  ``[0, \\pi/2]``,
- weights ``w_i = \\cos\\theta_i^{-} - \\cos\\theta_i^{+}`` (``\\sum_i w_i = 1``).

Declare each family as a phase with the tilted geometry
(`Spheroid(ω; euler_angles = (θ_i, 0, 0))`), fraction ``f\\,w_i`` and
`symmetrize = TISymmetrize((0, 0, 1))` — the exact azimuthal average about
the **global** axis represents the uniform-in-azimuth orbit of the family.
As ``N \\to \\infty`` the family sum converges (in ``O(\\Delta\\theta^2)``) to the single
`IsoSymmetrize` phase.
"""
function polar_orientation_bins(N::Int)
    N ≥ 2 || throw(ArgumentError("polar_orientation_bins: N ≥ 2 required"))
    out = Vector{@NamedTuple{θ::Float64, weight::Float64}}(undef, N)
    for i in 1:N
        θ = (π / 2) * (i - 1) / (N - 1)
        θm = i == 1 ? 0.0 : (π / 2) * (i - 1.5) / (N - 1)
        θp = i == N ? π / 2 : (π / 2) * (i - 0.5) / (N - 1)
        out[i] = (θ = θ, weight = cos(θm) - cos(θp))
    end
    return out
end
