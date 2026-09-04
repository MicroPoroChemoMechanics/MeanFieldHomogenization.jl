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

Requires a finite-element backend: `Ferrite`, `FerriteGmsh` and `Gmsh`, or —
for the axisymmetric morphology — `Gridap` and `GridapGmsh`. Two morphologies,
one method: see [Finite-element inclusions](@ref man-fe-inclusions) for the
elliptical crack and [A recycled-concrete aggregate](@ref app-recycled-aggregate)
for the sphere with an off-center core.

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

### Green function of the corrected boundary condition

```@docs
MeanFieldHomogenization.Core.green_gradient_iso
MeanFieldHomogenization.Core.dipole_displacement_iso
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
