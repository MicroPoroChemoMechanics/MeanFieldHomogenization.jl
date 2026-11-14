"""
    MeanFieldHomogenization.LayeredCylinders

Infinite `n`-layer cylinder of circular section (core and coaxial coatings)
embedded in an infinite matrix, with layers and matrix isotropic or
transversely isotropic about the cylinder axis, and perfect or imperfect
interfaces (spring, surface-elastic, Kapitza, surface-conductive). Public
entry point: [`LayeredCylinder`](@ref).
"""
module LayeredCylinders

using LinearAlgebra
using TensND

import ..Core
using ..Core
const MFH_Core = Core

using ..Elliptic: is_hard_numeric

import ..Core: gradient_gradient_concentration, flux_gradient_concentration,
    property_contribution, is_homogeneous_inclusion

# The interface types and the layer accessors are those of `LayeredSpheres`.
using ..LayeredSpheres: AbstractInterface, PerfectInterface, SpringInterface,
    MembraneInterface, KapitzaInterface, SurfaceConductiveInterface,
    spring_compliances, interfaces_eltype
import ..LayeredSpheres: layer_count, layer_radius, layer_modulus, layer_interface,
    layer_volume_fraction, outer_radius, layer_property_average,
    layer_inverse_property_average, get_layer, region_stiffness,
    local_gradient_gradient_concentration, local_flux_gradient_concentration,
    local_gradient_flux_concentration, local_flux_flux_concentration,
    local_gradient, local_flux, local_displacement, local_temperature,
    layer_gradient_average, layer_flux_average
using ..LayeredSpheres: _locate_region, _resolve_region

include("geometry.jl")
include("kernel.jl")             # the radial problems, for scalars and operators
include("assembly.jl")           # averages over the cylinder and its layers
include("elastic.jl")            # moduli, solutions, packing in TI tensors
include("scheme_integration.jl") # concentration, flux, contribution, bounds
include("localfields.jl")        # pointwise fields
include("averages.jl")           # averages under a given remote loading

export LayeredCylinder, LayeredCylinderFields, LayeredCylinderTransportFields
export cylinder_gradient_average, cylinder_flux_average,
    cylinder_strain_average, cylinder_stress_average

end # module
