# [API — Localization & contribution](@id api-localization)

## Localization tensors

```@docs
MeanFieldHomogenization.strain_strain_loc
MeanFieldHomogenization.stress_strain_loc
MeanFieldHomogenization.strain_stress_loc
MeanFieldHomogenization.stress_stress_loc
MeanFieldHomogenization.gradient_gradient_loc
MeanFieldHomogenization.flux_gradient_loc
MeanFieldHomogenization.gradient_flux_loc
MeanFieldHomogenization.flux_flux_loc
```

## Contribution tensors

```@docs
MeanFieldHomogenization.stiffness_contribution
MeanFieldHomogenization.compliance_contribution
MeanFieldHomogenization.conductivity_contribution
MeanFieldHomogenization.resistivity_contribution
```

## The amount × contribution seam

```@docs
MeanFieldHomogenization.delta_stiffness
MeanFieldHomogenization.delta_compliance
MeanFieldHomogenization.delta_conductivity
MeanFieldHomogenization.delta_resistivity
```

For a flat inclusion the four three-argument seams share one geometric
prefactor, [`crack_density_factor`](@ref MeanFieldHomogenization.Cracks.crack_density_factor).

## Inclusion traits and bundled seams

```@docs
MeanFieldHomogenization.Core.is_homogeneous_inclusion
MeanFieldHomogenization.loc_and_stiffness
MeanFieldHomogenization.loc_and_stress_average
MeanFieldHomogenization.compliance_and_stiffness_contribution
```

## Custom (user-defined) inclusions

See [Custom inclusions](@ref man-custom-inclusions) for the tutorial and
[Adding a new inclusion](@ref dev-adding-inclusion) for the full contract.

```@docs
MeanFieldHomogenization.CustomInclusions
MeanFieldHomogenization.Core.AbstractCustomInclusion
MeanFieldHomogenization.CustomInclusion
MeanFieldHomogenization.CustomShape
MeanFieldHomogenization.check_inclusion_interface
```

## Finite-element inclusions

Requires a finite-element backend: `Ferrite`, `FerriteGmsh` and `Gmsh` serve
every morphology, while `Gridap` and `GridapGmsh` serve the crack and the
axisymmetric ones. **Five morphologies, one method** — the same corrected finite
Eshelby cell and the same entry into the schemes:
[Finite-element inclusions](@ref man-fe-inclusions) for the elliptical crack,
[A recycled-concrete aggregate](@ref app-recycled-aggregate) for the sphere with
an off-center core, [Concave pores](@ref app-concave-pores) for the
superspherical and superspheroidal cavities, and
[A layered spheroid, meshed](@ref tut-axi-layered-spheroid) for `N` nested
spheroids.

```@docs
MeanFieldHomogenization.FiniteElements
MeanFieldHomogenization.FECache
MeanFieldHomogenization.fe_assembly_count
MeanFieldHomogenization.fe_reset!
```

### Choosing a backend

```@docs
MeanFieldHomogenization.FEBackend
MeanFieldHomogenization.AutoBackend
MeanFieldHomogenization.FerriteBackend
MeanFieldHomogenization.GridapBackend
```

### Writing a backend

A backend is sixteen methods and nothing else — nine for the axisymmetric
solve, seven for the crack. The Fourier operators, the boundary data, the fixed
point of the corrected boundary condition and the memoization are shared, and
the driver closes the strain operator and the azimuthal projection over the
mode before handing them over, so an implementation never sees a Fourier mode
or a physics: only "this many scalar fields, this operator, this projection".

`ext/MeanFieldHomogenizationGridapExt/` is the shorter of the two implementations and the
one to read first.

### The axisymmetric solve

```@docs
MeanFieldHomogenization.FiniteElements._build_gmsh_axi_model
MeanFieldHomogenization.FiniteElements.fe_axi_grid
MeanFieldHomogenization.FiniteElements.fe_axi_grid_counts
MeanFieldHomogenization.FiniteElements.fe_axi_region_volume
MeanFieldHomogenization.FiniteElements.fe_axi_mode
MeanFieldHomogenization.FiniteElements.fe_axi_dof_split
MeanFieldHomogenization.FiniteElements.fe_axi_set_dirichlet!
MeanFieldHomogenization.FiniteElements.fe_axi_stiffness
MeanFieldHomogenization.FiniteElements.fe_axi_average
MeanFieldHomogenization.FiniteElements._resolve_backend
```

### The crack

```@docs
MeanFieldHomogenization.FiniteElements._build_gmsh_crack_model
MeanFieldHomogenization.FiniteElements._weld_msh_crack_front
MeanFieldHomogenization.FiniteElements.fe_crack_grid
MeanFieldHomogenization.FiniteElements.fe_crack_counts
MeanFieldHomogenization.FiniteElements.fe_crack_space
MeanFieldHomogenization.FiniteElements.fe_crack_dof_split
MeanFieldHomogenization.FiniteElements.fe_crack_set_dirichlet!
MeanFieldHomogenization.FiniteElements.fe_crack_stiffness
MeanFieldHomogenization.FiniteElements.fe_crack_mean_jump
```

### The three-dimensional cell

Eight methods for the third cell family. Two things distinguish it from the
crack's: the geometry is **quadratic**, the mid-edge nodes of the inclusion
boundary having been moved onto the exact shape, and one contract serves two
physics — a scalar temperature and a vector displacement — with only the
material and the two averages differing.

```@docs
MeanFieldHomogenization.FiniteElements._build_gmsh_cell_model
MeanFieldHomogenization.FiniteElements._snap_cell_surface_to_shape!
MeanFieldHomogenization.FiniteElements._snap_cell_surface_to_sphere!
MeanFieldHomogenization.FiniteElements.fe_cell_grid
MeanFieldHomogenization.FiniteElements.fe_cell_counts
MeanFieldHomogenization.FiniteElements.fe_cell_space
MeanFieldHomogenization.FiniteElements.fe_cell_dof_split
MeanFieldHomogenization.FiniteElements.fe_cell_set_dirichlet!
MeanFieldHomogenization.FiniteElements.fe_cell_stiffness
MeanFieldHomogenization.FiniteElements.fe_cell_mean_gradient
MeanFieldHomogenization.FiniteElements.fe_cell_mean_strain
MeanFieldHomogenization.FiniteElements._cell_close_dipole
MeanFieldHomogenization.FiniteElements._cell_conduction_localization
MeanFieldHomogenization.FiniteElements._cell_elastic_localization
MeanFieldHomogenization.FiniteElements._cell_kelvin_basis
MeanFieldHomogenization.FiniteElements._cell_outer_radius
MeanFieldHomogenization.FiniteElements._cell_solver
```

### Superspherical and superspheroidal pore

The inclusion type. It enters as a **heterogeneous** inclusion, which for a
cavity is the truthful answer rather than a convenience: `is_homogeneous_inclusion`
asks whether a single ``\mathbb C_1`` describes the interior, and `inv(0)` is
meaningless. The package's exact identities then take over, and with a cavity's
stress-side localization being *identically* zero they collapse to
``\mathbb N = -\mathbb C_0:\mathbb A`` and
``\mathbb H = \mathbb A:\mathbb S_0`` — the right answer for a pore, with no
``\mathbb C_1`` anywhere in it. So the phase property handed to `add_phase!` is
genuinely ignored, and no contribution tensor is overridden.

```@docs
MeanFieldHomogenization.FESupershapePore
MeanFieldHomogenization.SupershapePoreShape
MeanFieldHomogenization.fe_cell_localization
MeanFieldHomogenization.fe_cell_mesh_report
MeanFieldHomogenization.has_surrogate
MeanFieldHomogenization.pore_shape_params
MeanFieldHomogenization.FiniteElements._pore_surrogate_response
MeanFieldHomogenization.FiniteElements._rebuild_pore_shape
MeanFieldHomogenization.Schemes._geom_field
MeanFieldHomogenization.FEAxiSupershapePore
MeanFieldHomogenization.FEAxiLayeredSpheroid
MeanFieldHomogenization.FiniteElements.LayeredSpheroidShape
MeanFieldHomogenization.check_nested_spheroids
MeanFieldHomogenization.axi_layer_set
MeanFieldHomogenization.layer_volumes
MeanFieldHomogenization.layer_fractions
MeanFieldHomogenization.AxiSupershapePoreShape
MeanFieldHomogenization.fe_axi_pore_localization
MeanFieldHomogenization.fe_axi_pore_breakdown
MeanFieldHomogenization.fe_axi_pore_mesh_report
MeanFieldHomogenization.fe_axi_pore_boundary
MeanFieldHomogenization.FiniteElements._superspheroid_meridian
MeanFieldHomogenization.FiniteElements._fe_frame
```

### Elliptical crack (3-D)

```@docs
MeanFieldHomogenization.FEEllipticCrack
MeanFieldHomogenization.FEMeshOptions
MeanFieldHomogenization.fe_mesh_report
MeanFieldHomogenization.fe_cod_breakdown
```

### Sphere with an off-center core (axisymmetric Fourier)

```@docs
MeanFieldHomogenization.FEExcenteredSphere
MeanFieldHomogenization.FEAxiMeshOptions
MeanFieldHomogenization.fe_axi_localization
MeanFieldHomogenization.fe_axi_breakdown
MeanFieldHomogenization.fe_axi_mesh_report
MeanFieldHomogenization.FiniteElements.core_radius
MeanFieldHomogenization.FiniteElements.core_offset
MeanFieldHomogenization.FiniteElements.tensor_order
MeanFieldHomogenization.FiniteElements.ExcenteredSphereShape
```

### The three-dimensional cell around a non-ellipsoidal shape

The third cell family. The inclusion surface has no CAD representation, so it is
built analytically by [`shape_surface`](@ref
MeanFieldHomogenization.Superspheres.shape_surface) and handed to gmsh as a
*discrete* entity; the interior size follows an exact radial law, the cell being
star-shaped about its center.

Two volume measures rather than one, and the difference is the point: the flat
[`mesh_volume`](@ref MeanFieldHomogenization.Superspheres.mesh_volume) cannot
see a curved boundary at all, so
[`fe_cell_curved_volume`](@ref) is what shows what snapping the mid-edge nodes
onto the exact shape actually bought.

```@docs
MeanFieldHomogenization.FECellMeshOptions
MeanFieldHomogenization.fe_cell_size_estimate
MeanFieldHomogenization.fe_cell_curved_volume
MeanFieldHomogenization.fe_cell_meshed_volume
MeanFieldHomogenization.fe_available_gb
```

### Green function of the corrected boundary condition

```@docs
MeanFieldHomogenization.Core.green_gradient_iso
MeanFieldHomogenization.Core.dipole_displacement_iso
MeanFieldHomogenization.Core.green_gradient_iso2
MeanFieldHomogenization.Core.dipole_temperature_iso
```

## Neural-surrogate inclusions

The fourth route into the contract: the response comes out of a trained network.
See [Neural-surrogate inclusions](@ref man-neural-inclusions) for the tutorial.
Evaluating needs nothing beyond the package; *training* needs
`import Lux, Optimisers, Zygote`.

```@docs
MeanFieldHomogenization.NeuralInclusions
MeanFieldHomogenization.NeuralHillInclusion
MeanFieldHomogenization.NeuralLocalizationInclusion
MeanFieldHomogenization.NeuralInclusions.NeuralShape
MeanFieldHomogenization.NeuralInclusions.StrainLocTI
MeanFieldHomogenization.NeuralInclusions.StressLocTI
MeanFieldHomogenization.NeuralInclusions.StrainLocCubic
MeanFieldHomogenization.NeuralInclusions.GradLocISO2
MeanFieldHomogenization.NeuralInclusions.GradLocTI2
```

### The surrogate

```@docs
MeanFieldHomogenization.NeuralSurrogate
MeanFieldHomogenization.Provenance
MeanFieldHomogenization.worst_error
MeanFieldHomogenization.NeuralInclusions.check_domain
MeanFieldHomogenization.NeuralInclusions.predict_components
```

### What the network predicts

The symmetry class, the major symmetry, the homogeneity in the reference moduli
and the frame are *enforced* by these types rather than fitted — see
[What is exact, and what is fitted](@ref man-neural-inclusions).

```@docs
MeanFieldHomogenization.NeuralInclusions.AbstractHillClass
MeanFieldHomogenization.HillISO
MeanFieldHomogenization.HillTI
MeanFieldHomogenization.HillOrtho
MeanFieldHomogenization.HillISO2
MeanFieldHomogenization.HillTI2
MeanFieldHomogenization.NeuralInclusions.AbstractOutputSpec
MeanFieldHomogenization.DimensionlessHill
MeanFieldHomogenization.AnchoredHill
MeanFieldHomogenization.NeuralInclusions.anchor_baselines
MeanFieldHomogenization.NeuralInclusions.anchor_tensor
MeanFieldHomogenization.NeuralInclusions.encode
MeanFieldHomogenization.NeuralInclusions.spec_baseline
MeanFieldHomogenization.AffineHill
MeanFieldHomogenization.NeuralInclusions.ncomponents
MeanFieldHomogenization.NeuralInclusions.tensor_order
MeanFieldHomogenization.NeuralInclusions.nterms
MeanFieldHomogenization.NeuralInclusions.noutputs
MeanFieldHomogenization.NeuralInclusions.needs_nu
MeanFieldHomogenization.NeuralInclusions.build
MeanFieldHomogenization.NeuralInclusions.components
MeanFieldHomogenization.NeuralInclusions.decode
MeanFieldHomogenization.NeuralInclusions.material_coeffs
MeanFieldHomogenization.NeuralInclusions.dimensionless_scale
MeanFieldHomogenization.NeuralInclusions.hill_class
MeanFieldHomogenization.NeuralInclusions.output_spec
MeanFieldHomogenization.NeuralInclusions.apply_transform
MeanFieldHomogenization.NeuralInclusions.invert_transform
MeanFieldHomogenization.NeuralInclusions._feature
MeanFieldHomogenization.NeuralInclusions.raw_features
MeanFieldHomogenization.NeuralInclusions._class_frame
MeanFieldHomogenization.NeuralInclusions._canonical_axes
MeanFieldHomogenization.NeuralInclusions._spheroid_axis_index
```

### Sampling and labeling

```@docs
MeanFieldHomogenization.SampleBox
MeanFieldHomogenization.Dataset
MeanFieldHomogenization.generate_dataset
MeanFieldHomogenization.NeuralInclusions.sample_box
MeanFieldHomogenization.NeuralInclusions.grid_box
MeanFieldHomogenization.NeuralInclusions.halton
MeanFieldHomogenization.NeuralInclusions.feature_index
MeanFieldHomogenization.fit_scaling
MeanFieldHomogenization.validate_surrogate
MeanFieldHomogenization.report_surrogate
MeanFieldHomogenization.component_labels
```

### Training

`train_surrogate` is the seam of the `MeanFieldHomogenizationLuxExt` extension: the method
below is the fallback that raises when the extension is not loaded.

```@docs
MeanFieldHomogenization.TrainingOptions
MeanFieldHomogenization.train_surrogate
MeanFieldHomogenization.assemble_surrogate
MeanFieldHomogenization.NeuralInclusions.network_widths
```

### The network, and its serialization

```@docs
MeanFieldHomogenization.NeuralInclusions.MLP
MeanFieldHomogenization.NeuralInclusions.NNDense
MeanFieldHomogenization.NeuralInclusions.glorot_mlp
MeanFieldHomogenization.NeuralInclusions.softplus
MeanFieldHomogenization.NeuralInclusions.activation
MeanFieldHomogenization.NeuralInclusions.activation_name
MeanFieldHomogenization.NeuralInclusions.layer_widths
MeanFieldHomogenization.NeuralInclusions.layer_activations
MeanFieldHomogenization.NeuralInclusions.nparams
MeanFieldHomogenization.save_surrogate
MeanFieldHomogenization.load_surrogate
MeanFieldHomogenization.model_path
MeanFieldHomogenization.shipped_models
MeanFieldHomogenization.NeuralInclusions.SURROGATE_FORMAT
MeanFieldHomogenization.NeuralInclusions.MODEL_DIR
```
