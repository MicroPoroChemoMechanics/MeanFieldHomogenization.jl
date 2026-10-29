# =============================================================================
#  test_hill_inference.jl — the Hill tensors and the auxiliary tensors of every
#  route are inferred with their order and their dimension.
#
#  Each route computes the components into an array and wraps them in the frame
#  of the inclusion. The wrapper used to read the dimension from the size of the
#  array, at run time: the result was inferred as a tensor of unknown dimension,
#  and every scheme built on it (a sum, a contraction, an inverse) was dispatched
#  at run time. The storage, symmetric or not, is still chosen from the values,
#  so a route may infer as the union of the two; never as an abstract tensor.
# =============================================================================

using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
import DECUHR, Integrals   # the `:decuhr` route lives in an extension

const _HI_CORE = MeanFieldHomogenization.Core

# A tensor of the given order and dimension, either storage.
_hi_tens(order, dim) = Union{
    TensND.TensCanonical{order, dim, Float64}, TensND.TensRotated{order, dim, Float64},
    TensND.TensISO{order, dim, Float64},
}

# The inferred return type of `f(args...)`, a single one.
_hi_inferred(f, args...) = only(Base.return_types(f, map(typeof, args)))

@testset "Hill tensors inferred with their order and dimension" begin
    ellipsoid = Ellipsoid(2.0, 1.0, 0.5; euler_angles = (0.4, 1.1, 0.3))
    cylinder = Cylinder(1.0, 0.6)
    disk = Ellipsoid(1.0, 0.5)
    C = TensND.from_mandel(KM(TensND.change_tens_canon(TensND.tens_cubic(168.4e3, 121.4e3, 75.4e3, TensND.RotatedBasis(0.4, 1.1, 0.3)))))
    C₂ = TensND.Tens(Array(TensISO{2}(15.0, 4.0)))
    K = TensND.Tens([3.0 0.5 0.2; 0.5 2.0 0.1; 0.2 0.1 1.0])
    cases = (
        (4, 3, (e, c) -> hill_tensor(e, c), (ellipsoid, C)),
        (4, 3, (e, c) -> hill_tensor(e, c; method = :decuhr), (ellipsoid, C)),
        (4, 3, (e, c) -> hill_tensor(e, c; method = :nestedquadgk), (ellipsoid, C)),
        (4, 3, (e, c) -> hill_tensor(e, c; method = :residues), (ellipsoid, C)),
        (4, 3, (e, c) -> hill_tensor(e, c), (cylinder, C)),
        (4, 2, (e, c) -> hill_tensor(e, c), (disk, C₂)),
        (4, 2, (e, c) -> hill_tensor(e, c), (disk, TensISO{2}(15.0, 4.0))),
        (2, 3, (e, c) -> hill_tensor(e, c), (ellipsoid, K)),
        (2, 3, (e, c) -> hill_tensor(e, c), (cylinder, K)),
        (2, 2, (e, c) -> hill_tensor(e, c), (disk, TensND.Tens([2.0 0.3; 0.3 1.0]))),
        (2, 2, (e, c) -> hill_tensor(e, c), (disk, TensISO{2}(2.0))),
        (2, 3, tens_IA, (ellipsoid,)), (2, 3, tens_IA, (cylinder,)), (2, 2, tens_IA, (disk,)),
        (4, 2, tens_UA, (disk,)), (4, 2, tens_VA, (disk,)),
        (2, 3, _HI_CORE.shape_tensor, (ellipsoid,)), (2, 3, _HI_CORE.shape_tensor, (cylinder,)),
    )
    for (order, dim, f, args) in cases
        R = _hi_inferred(f, args...)
        @test R <: _hi_tens(order, dim)
        @test f(args...) isa R
    end
end
