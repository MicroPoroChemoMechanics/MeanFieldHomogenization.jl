# The script a model becomes: the conventions the studio exists to remove from
# the user's mind, laminates, sensitivities, scales, and the round trip through
# the embedded model. Each test says what the script must contain, as the
# tests of the Python studio did.

gen(m) = MS.generate(m; embed_model = false)
# equal as Python compares dicts: whatever the order of the keys, an integer
# apart from a float
samejson(a::AbstractDict, b::AbstractDict) = Set(keys(a)) == Set(keys(b)) && all(k -> samejson(a[k], b[k]), keys(a))
samejson(a::AbstractVector, b::AbstractVector) = length(a) == length(b) && all(samejson(x, y) for (x, y) in zip(a, b))
samejson(a, b) = typeof(a) == typeof(b) && isequal(a, b)
props(pairs...) = MS.jdict(pairs...)
P(; kw...) = MS.Property(; kw...)
pexpr(p) = MS.prop_expr(MS.CodeGen(MS.StudioModel(); embed = false), p)
count_of(needle, haystack) = length(findall(needle, haystack))

@testset "conventions" begin
    src = gen(MS.default_model())
    # the phase taking up the rest carries no number, and an RVE names no matrix
    @test occursin("fraction = :rest", src) && occursin("RVE()", src) && !occursin("add_matrix!", src)
    # physical moduli, never the raw (3k, 2μ) of TensISO{3}
    @test occursin("iso_stiffness(72.0, 32.0)", src) && !occursin("TensISO{3}(216", src)
    @test !occursin("euler_angles", src)
    m = MS.default_model()
    m.sweep.schemes = Any[props("name" => "SelfConsistent", "options" => props("abstol" => 1.0e-10, "maxiters" => 300, "select_best" => true))]
    src = gen(m)
    @test occursin("SelfConsistent(; abstol = 1.0e-10, maxiters = 300, select_best = true)", src)
    @test occursin("homogenize(cell, scheme, :C)", src)
    m.sweep.schemes = Any[props("name" => "MoriTanaka", "options" => MS.JDict()), props("name" => "Voigt", "options" => MS.JDict())]
    src = gen(m)
    @test occursin("(\"MoriTanaka\", MoriTanaka())", src) && occursin("(\"Voigt\", Voigt())", src)
    @test occursin("for (name, scheme) in SCHEMES", src)
    # a scheme option left empty is not written
    @test MS.scheme_expr("SelfConsistent", props("abstol" => nothing)) == "SelfConsistent()"
end

@testset "outputs" begin
    m = MS.default_model()
    m.sweep.mode = "single"
    src = gen(m)
    @test occursin("One homogenization with the amounts entered", src) && !occursin("set_param", src)
    @test occursin("homogenize(cell, MoriTanaka(), :C)", src)
    @test occursin("@printf \"MoriTanaka                ", src)
    m = MS.default_model()
    m.sweep.projection = "none"
    m.sweep.outputs = Any[props("kind" => "km", "i" => 1, "j" => 1), props("kind" => "km", "i" => 3, "j" => 3)]
    src = gen(m)
    @test occursin("KMC = mandel(C)", src) && occursin("KMC[1, 1]", src) && occursin("KMC[3, 3]", src)
    @test !occursin("k_mu", src)
    m = MS.default_model()
    @test !occursin("max(", gen(m))
    m.sweep.clamp_zero = true
    @test occursin("return (max(km[1], 0.0), max(km[2], 0.0))", gen(m))
    # a reduction is computed once, and named only when used twice
    m = MS.default_model()
    src = gen(m)
    @test occursin("km = k_mu(C)", src) && count_of("k_mu(C)", src) == 1
    m.sweep.outputs = Any[props("kind" => "k")]
    src = gen(m)
    @test !occursin("km = ", src) && occursin("k_mu(C)[1]", src) && occursin("return (k_mu(C)[1],)", src)
    m.sweep.outputs = Any[props("kind" => "km", "i" => 1, "j" => 1), props("kind" => "km", "i" => 3, "j" => 3), props("kind" => "k")]
    src = gen(m)
    @test occursin("KMC = mandel(C)", src) && count_of("mandel(C)", src) == 1 && occursin("k_mu(C)[1]", src)
    m.sweep.mode = "single"
    m.sweep.outputs = Any[props("kind" => "k"), props("kind" => "mu")]
    m.sweep.schemes = Any[props("name" => "MoriTanaka", "options" => MS.JDict()), props("name" => "Voigt", "options" => MS.JDict())]
    src = gen(m)
    @test occursin("km_MoriTanaka = k_mu(C_MoriTanaka)", src) && occursin("km_Voigt = k_mu(C_Voigt)", src)
    # every kind of output, and its label
    outs = Any[
        props("kind" => "E"), props("kind" => "nu"), props("kind" => "comp", "i" => 2, "j" => 3),
        props("kind" => "trace3"), props("kind" => "other"), props("kind" => "km", "i" => 1.0, "j" => "2"),
    ]
    @test [MS.output_expr(o) for o in outs] == ["E_nu(C)[1]", "E_nu(C)[2]", "Array(C)[2, 3]", "tr(Array(C)) / 3", "k_mu(C)[1]", "mandel(C)[1, 2]"]
    @test [MS.output_label(o) for o in outs] == ["E", "nu", "C23", "tr/3", "other", "KM12"]
    m.sweep.outputs = Any[]
    @test occursin("@printf \"MoriTanaka" * " "^16 * "value = %.6f\\n\" C_MoriTanaka", gen(m))
    m.sweep.mode = "sweep"
    m.sweep.plot = false
    src = gen(m)
    @test occursin("return (C,)", src) && !occursin("display(p)", src) && !occursin("using Plots", src)
end

@testset "properties" begin
    @test pexpr(P(; builder = "maxwell_iso", args = props("k" => 10.0, "mu" => 5.0, "eta_k" => 2.0, "eta_mu" => 3.0))) ==
        "maxwell_iso(10.0, 5.0, 2.0, 3.0)"
    @test pexpr(P(; builder = "kelvin_iso", args = props("k0" => 10.0, "mu0" => 5.0, "k1" => 20.0, "mu1" => 10.0, "tau_k" => 1.0, "tau_mu" => 2.0))) ==
        "kelvin_iso(10.0, 5.0, [20.0], [10.0], [1.0], [2.0])"
    # the `visco` field of older projects: one time per law
    @test pexpr(P(; visco = props("kind" => "maxwell_iso", "args" => props("k" => 10.0, "mu" => 5.0, "tau" => 2.0)))) ==
        "maxwell_iso(10.0, 5.0, 2.0, 2.0)"
    @test pexpr(P(; visco = props("kind" => "kelvin_iso", "args" => props("k" => 10.0, "mu" => 5.0, "tau" => 2.0)))) ==
        "kelvin_iso(Inf, Inf, [10.0], [5.0], [2.0], [2.0])"
    @test pexpr(P(; visco = props("kind" => "heaviside", "args" => props("k" => 1.0)))) == "heaviside_law(iso_stiffness(1.0, 5.0))"
    @test pexpr(P(; source = "visco", visco = nothing)) == "maxwell_iso(10.0, 5.0, 1.0, 1.0)"
    @test pexpr(P(; visco = props("kind" => "custom", "args" => props("expr" => "1 / t"), "mode" => "relaxation"))) ==
        "ViscoLaw((t, t′) -> 1 / t, :relaxation)"
    @test pexpr(P(; builder = "heaviside_law")) == "heaviside_law(iso_stiffness(10.0, 5.0))"
    @test pexpr(P(; builder = "ViscoLaw", args = props("expr" => "J(t)", "mode" => " :relaxation "))) == "ViscoLaw((t, t′) -> J(t), :relaxation)"
    @test pexpr(P(; builder = "ViscoLaw", args = props("mode" => "bogus"))) == "ViscoLaw((t, t′) -> iso_stiffness(10.0, 5.0), :creep)"
    @test pexpr(P(; builder = "iso_stiffness_E_nu", args = props("E" => 30))) == "iso_stiffness_E_nu(30.0, 0.2)"
    @test pexpr(P(; builder = "TensISO{3}", args = props("k" => "k_s"))) == "TensISO{3}(k_s)"
    @test pexpr(P(; builder = "unknown")) == "iso_stiffness(10.0, 5.0)"
    @test pexpr(P(; source = "expr", expr = "")) == "one(TensISO{3})"
    @test pexpr(P(; source = "cell", cell = "gone")) == "#= missing inner cell =# one(TensISO{3})"
    # the outer TensTI{2} constructor, and the axis of a TI tensor
    @test pexpr(P(; builder = "TensTI2", args = props("kt" => 1.0, "ka" => 5.0))) == "TensTI{2}(1.0, 5.0, (0.0, 0.0, 1.0))"
    @test pexpr(P(; builder = "TensTI2", args = props("kt" => 1.0, "ka" => 5.0, "axis" => Any[1, 0, 0]))) == "TensTI{2}(1.0, 5.0, (1.0, 0.0, 0.0))"
    @test pexpr(P(; builder = "TensDiag2", args = props("k1" => 1.0, "k2" => 2.0, "k3" => 5.0))) ==
        "Tens([1.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 5.0])"
    # anisotropic properties carry their own frame, not the shape's
    ang = Any["pi/4", 0.7, 0.0]
    @test pexpr(P(; builder = "TensTI2", args = props("kt" => 1.0, "ka" => 5.0), euler_angles = ang)) ==
        "TensTI{2}(1.0, 5.0, vecbasis(RotatedBasis(pi/4, 0.7, 0.0))[:, 3])"
    hoenig = pexpr(P(; builder = "hoenig_stiffness", euler_angles = ang))
    @test startswith(hoenig, "hoenig_stiffness(") && endswith(hoenig, "vecbasis(RotatedBasis(pi/4, 0.7, 0.0))[:, 3])")
    @test endswith(pexpr(P(; builder = "hoenig_stiffness")), "(0.0, 0.0, 1.0))")
    @test endswith(pexpr(P(; builder = "TensDiag2", args = props("k1" => 1.0, "k2" => 2.0, "k3" => 5.0), euler_angles = ang)), ", RotatedBasis(pi/4, 0.7, 0.0))")
    # fewer than three angles would make a 2-D basis
    @test endswith(pexpr(P(; builder = "TensOrtho", euler_angles = Any[0.3])), "RotatedBasis(0.3, 0.0, 0.0))")
    @test startswith(pexpr(P(; builder = "TensOrtho")), "TensOrtho(120.0, 90.0, 70.0, 40.0, 35.0, 30.0, 25.0, 22.0, 20.0, CanonicalBasis{3, Float64}()")
end

function layered_spheroid_model()
    layer(fr, k) = props("fraction" => fr, "property" => props("key" => ":K", "source" => "builder", "builder" => "TensISO{3}", "form" => "iso_conduction", "args" => props("k" => k)))
    geo = MS.Geometry(; kind = "layered_spheroid", args = props("omega" => 0.5, "radius" => 1.0, "Nseries" => 5), layers = Any[layer(0.3, 1.0), layer(0.7, 5.0)])
    c = MS.Cell(;
        name = "r", phases = [
            MS.Phase(; name = "M", amount_kind = "rest", properties = [P(; key = ":K", builder = "TensISO{3}", form = "iso_conduction", args = props("k" => 2.0))]),
            MS.Phase(; name = "I", amount = 0.2, geometry = geo),
        ]
    )
    return MS.StudioModel(; cells = [c])
end

oriented(geo) = MS.StudioModel(; cells = [MS.Cell(; name = "r", phases = [MS.Phase(; name = "M", amount_kind = "rest", properties = [P()]), MS.Phase(; name = "I", amount = 0.2, geometry = geo)])])

@testset "shapes" begin
    m = layered_spheroid_model()
    src = gen(m)
    @test occursin("layered_spheroid_from_fractions(0.5, 1.0, (0.3, 0.7)", src) && !occursin("LayeredSpheroid(", src)
    @test !occursin("axis", src)
    @test occursin("TensISO{3}(2.0)", src) && !occursin("TensISO{2, 3}", src)
    m.cells[1].phases[2].geometry.euler_angles = Any[0.7, 1.1]
    @test occursin("axis = Tuple(vecbasis(RotatedBasis(0.7, 1.1", gen(m))
    @test occursin("Spheroid(0.3; euler_angles = (0.7, 1.1))", gen(oriented(MS.Geometry(; args = props("omega" => 0.3), euler_angles = Any[0.7, 1.1]))))
    @test occursin("euler_angles = (1.2,)", gen(oriented(MS.Geometry(; args = props("omega" => 0.3), euler_angles = Any[1.2]))))
    # sizes and angles are floats: a Tuple{Int, Float64} matches no method
    @test occursin("euler_angles = (0.0, 1.1, 0.3)", gen(oriented(MS.Geometry(; kind = "ellipsoid", args = props("a" => 2, "b" => 1, "c" => 0.5), euler_angles = Any[0, 1.1, 0.3]))))
    lay(r, k, mu) = props("radius" => r, "property" => props("key" => ":C", "source" => "builder", "builder" => "iso_stiffness", "args" => props("k" => k, "mu" => mu)))
    sphere = MS.Geometry(; kind = "layered_sphere", layers = Any[lay(0.6, 1, 1), lay(1, 30, 12)])
    src = gen(oriented(sphere))
    @test occursin("LayeredSphere((0.6, 1.0)", src) && !occursin("(0.6, 1)", src)
    sphere.layers[1]["interface"] = props("kind" => "SpringInterface", "args" => props("kn" => 1.0, "kt" => 2.0))
    @test occursin("; interfaces = (SpringInterface(1.0, 2.0), PerfectInterface()))", gen(oriented(sphere)))
    shapes = [
        MS.Geometry(; kind = "cylinder", args = props("b" => 1, "c" => "c0")) => "Cylinder(1.0, c0)",
        MS.Geometry(; kind = "penny_crack", args = MS.JDict()) => "PennyCrack(1.0)",
        MS.Geometry(; kind = "elliptic_crack", args = props("a" => 2.0)) => "EllipticCrack(2.0, 0.5)",
        MS.Geometry(; kind = "ribbon_crack", args = props("b" => 0.1)) => "RibbonCrack(0.1)",
        MS.Geometry(; kind = "unknown") => "Spheroid(1.0)",
        MS.Geometry(; kind = "layered_sphere", layers = Any[]) => "LayeredSphere((), ())",
    ]
    for (geo, text) in shapes
        @test MS.geometry_expr(MS.CodeGen(MS.StudioModel()), MS.Phase(; geometry = geo)) == text
    end
    # a phase averaged over orientations, and a long call broken at its own `;`
    long = oriented(MS.Geometry(; kind = "ellipsoid", args = props("a" => 1.0, "b" => 0.5, "c" => 0.25), euler_angles = Any[0.1, 0.2, 0.3]))
    long.cells[1].phases[2].symmetrize = "ti"
    src = gen(long)
    @test occursin("    add_phase!(\n        rve, :I, Ellipsoid(1.0, 0.5, 0.25; euler_angles = (0.1, 0.2, 0.3)), Dict{Symbol, Any}();\n        fraction = 0.2, symmetrize = TISymmetrize()\n    )", src)
    @test MS.toplevel_kw_split("f(a; b); c") == 8
    @test MS.toplevel_kw_split("f(a; b)") === nothing
    g = MS.CodeGen(MS.StudioModel())
    MS.call!(g, "no_parenthesis_" * "x"^90)
    MS.call!(g, "f(" * "x"^90 * ")")
    @test all(l -> !occursin('\n', l), g.out) && length(g.out) == 2
    # `RVE` keywords, sorted
    opts = oriented(MS.Geometry())
    opts.cells[1].rve_options = props("T" => "ComplexF64", "hill_method" => ":analytical")
    @test occursin("rve = RVE(; T = ComplexF64, hill_method = :analytical)", gen(opts))
end

function bilayer()
    lam = MS.Cell(; name = "lam", kind = "laminate", layers = MS.default_layers())
    m = MS.StudioModel(; title = "bilayer", cells = [lam], root_cell = lam.id)
    m.sweep = MS.Sweep(;
        enabled = true, mode = "single", cell = lam.id, schemes = Any[props("name" => "Laminated", "options" => MS.JDict())],
        projection = "none", outputs = Any[props("kind" => "km", "i" => 3, "j" => 3)],
    )
    return m
end

@testset "laminates" begin
    src = gen(bilayer())
    @test occursin("lam = Laminate()", src)
    i = findfirst("add_layer!(lam, :A, Dict(:C => iso_stiffness(2.0, 0.8)); fraction = 0.3)", src)
    j = findfirst("add_layer!(lam, :B, Dict(:C => iso_stiffness(0.5, 0.2)); fraction = 0.7)", src)
    @test i !== nothing && j !== nothing && first(i) < first(j)
    @test !occursin("normal", src) && !occursin("PerfectInterface", src)
    m = bilayer()
    m.cells[1].frame_mode = "normal"
    m.cells[1].normal = Any[1.0, 0.0, 1.0]
    @test occursin("Laminate(; normal = (1.0, 0.0, 1.0))", gen(m))
    m.cells[1].frame_mode = "euler"
    m.cells[1].euler_angles = Any["π/4", 0.3]
    src = gen(m)
    @test occursin("Laminate(; euler_angles = (π/4, 0.3))", src) && !occursin("normal", src)
    m.cells[1].euler_angles = Any[0.5]
    @test occursin("Laminate(; euler_angles = (0.5,))", gen(m))
    m = bilayer()
    m.cells[1].layers[1].interface = props("kind" => "SpringInterface", "args" => props("kn" => 1.0e-3, "kt" => 2.0e-3))
    @test occursin("interface = SpringInterface(0.001, 0.002)", gen(m))
    m.cells[1].layers[1].interface = props("kind" => "KapitzaInterface", "args" => MS.JDict())
    @test occursin("interface = KapitzaInterface()", gen(m))
    @test MS.interface_expr(nothing) == ""
    m = bilayer()
    m.cells[1].layers[1].amount_kind = "thickness"
    @test any(p -> occursin("mixes absolute thicknesses", p), MS.validate(m))
    m = bilayer()
    m.cells[1].layers[1].amount = 0.5
    @test any(p -> occursin("sum to 1.2", p), MS.validate(m))
    @test !any(p -> occursin("matrix", p), MS.validate(bilayer()))
    # a layer carries the multiscale seam, and the inner scale comes first
    inner = MS.Cell(;
        name = "foam", phases = [
            MS.Phase(; name = "SOLID", amount_kind = "rest", properties = [P(; args = props("k" => 72.0, "mu" => 32.0))]),
            MS.Phase(; name = "PORE", amount = 0.3, properties = [P(; args = props("k" => 1.0e-6, "mu" => 1.0e-6))]),
        ]
    )
    lam = MS.Cell(;
        name = "lam", kind = "laminate", layers = [
            MS.Layer(; name = "A", amount = 0.4, properties = [P(; source = "cell", cell = inner.id, scheme = "MoriTanaka")]),
            MS.Layer(; name = "B", amount = 0.6, properties = [P(; args = props("k" => 0.5, "mu" => 0.2))]),
        ]
    )
    src = gen(MS.StudioModel(; title = "ms_lam", cells = [lam, inner], root_cell = lam.id))
    @test occursin("Homogenized(build_foam(), MoriTanaka())", src)
    @test first(findfirst("function build_foam", src)) < first(findfirst("function build_lam", src))
    @test occursin("# ── Scales ", src)
    # the 3-D view is built from the generated code
    m = bilayer()
    expr = MS.cell_expression(m, m.cells[1])
    @test startswith(expr, "let") && endswith(rstrip(expr), "end") && !occursin("return", expr) && count_of("add_layer!", expr) == 2
    m.sweep.mode = "sweep"
    m.sweep.lens = MS.Lens(; kind = "thickness", phase = "A")
    @test occursin("thickness(:A)", gen(m))
    m.sweep.lens = MS.Lens(; kind = "interface_param", index = 2, field_name = "kn")
    @test occursin("interface_param(2, :kn)", gen(m))
    m = bilayer()
    m.cells[1].normal = Any[0.0, 1.0, 1.0]
    m.cells[1].layers[2].interface = props("kind" => "KapitzaInterface", "args" => props("h" => 0.25))
    back = MS.model_from_dict(MS.extract_embedded(MS.generate(m)))
    @test samejson(MS.to_dict(back), MS.to_dict(m))
end

function sens_model(; kw...)
    m = MS.default_model()
    m.sweep.enabled = false
    m.sens = MS.Sens(; enabled = true, cell = m.cells[1].id, output = props("kind" => "k"), projection = "iso", lenses = [MS.Lens(; kind = "amount", phase = "PORE")], kw...)
    return m
end

@testset "sensitivities" begin
    src = gen(sens_model())
    @test occursin("const param = amount(:PORE)", src)
    @test occursin("derivative(cell, scheme, param; output = :C, indexer = C -> k_mu(best_fit_iso(C))[1])", src)
    @test occursin("@printf \"d(k) / d(%s) = %.8g\\n\" \"amount(PORE)\" D", src)
    @test !occursin("using Plots", src)
    m = sens_model(; kind = "gradient")
    m.sens.lenses = [MS.Lens(; kind = "amount", phase = "PORE"), MS.Lens(; kind = "property", phase = "SOLID", property = ":C", index = 1)]
    src = gen(m)
    @test occursin("const params = [", src) && occursin("amount(:PORE),", src) && occursin("property(:SOLID, :C, 1),", src)
    @test occursin("gradient(cell, scheme, params;", src) && occursin("for (name, g) in zip(LABELS, D)", src)
    src = gen(sens_model(; kind = "jacobian"))
    @test occursin("jacobian(cell, scheme, params; output = :C)", src) && !occursin("indexer", src)
    @test occursin("row = @view D[i, :]", src)
    m = sens_model()
    m.sens.projection = "none"
    @test occursin("indexer = C -> k_mu(C)[1])", gen(m))
    # every lens, written and labeled
    nested = MS.Lens(; kind = "nested", member = "FOAM", property = ":C", inner = MS.to_dict(MS.Lens(; kind = "amount", phase = "PORE")))
    lenses = [
        MS.Lens(; kind = "geometry", phase = "I", field_name = "semi_axes", index = 2),
        MS.Lens(; kind = "shape_param", field_name = "ω", index = 1),
        MS.Lens(; kind = "interface_param", index = 2, field_name = "kn"),
        MS.Lens(; kind = "thickness", phase = "A"), nested, MS.Lens(; kind = "other", phase = "X"),
    ]
    @test [MS.lens_expr(l) for l in lenses] == [
        "geometry(:I, :semi_axes, 2)", "shape_param(:ω, 1)", "interface_param(2, :kn)",
        "thickness(:A)", "nested(:FOAM, :C, amount(:PORE))", "amount(:X)",
    ]
    @test [MS.lens_label(l) for l in lenses] == [
        "I.semi_axes[2]", "shape.ω[1]", "interface 2.kn", "thickness(A)", "FOAM:C → amount(PORE)", "other",
    ]
    @test MS.lens_label(MS.Lens(; kind = "property", phase = "S", index = 3)) == "S:C[3]"
end

function two_scale()
    inner = MS.Cell(;
        name = "foam", phases = [
            MS.Phase(; name = "SOLID", amount_kind = "rest", properties = [P(; args = props("k" => 72.0, "mu" => 32.0))]),
            MS.Phase(; name = "PORE", amount = 0.3, properties = [P(; args = props("k" => 1.0e-6, "mu" => 1.0e-6))]),
        ]
    )
    outer = MS.Cell(;
        name = "paste", phases = [
            MS.Phase(; name = "FOAM", amount_kind = "rest", properties = [P(; source = "cell", cell = inner.id, scheme = "SelfConsistent", scheme_options = props("abstol" => 1.0e-10))]),
            MS.Phase(; name = "CLINKER", amount = 0.2, properties = [P(; args = props("k" => 100.0, "mu" => 50.0))]),
        ]
    )
    return MS.StudioModel(; title = "ms", cells = [inner, outer], root_cell = outer.id)
end

@testset "scales" begin
    src = gen(two_scale())
    @test occursin("Homogenized(build_foam(), SelfConsistent(; abstol = 1.0e-10))", src)
    @test first(findfirst("function build_foam", src)) < first(findfirst("function build_paste", src))
    m = two_scale()
    foam, paste = m.cells
    foam.phases[1].properties[1] = P(; source = "cell", cell = paste.id)
    ps = MS.validate(m)
    @test occursin("cycle", ps[1]) && occursin("foam", ps[1]) && occursin("paste", ps[1])
    # a cycle is named in the header, and the builders follow in their order
    src = gen(m)
    @test occursin("UNRESOLVED PROBLEMS", src) && first(findfirst("function build_foam", src)) < first(findfirst("function build_paste", src))
    m = two_scale()
    m.sweep = MS.Sweep(; enabled = true, cell = m.cells[2].id, lens = MS.Lens(; kind = "nested", member = "FOAM", property = ":C", inner = MS.to_dict(MS.Lens(; kind = "amount", phase = "PORE"))))
    @test occursin("nested(:FOAM, :C, amount(:PORE))", gen(m))
    m.alv.enabled = true
    @test any(p -> occursin("viscoelast", lowercase(p)), MS.validate(m))
    # builders taking parameters, and a builder kept under its own name
    m = two_scale()
    m.cells[1].params = ["φ"]
    m.cells[1].builder_name = "_foam"
    m.cells[1].phases[2].amount = "φ"
    m.params = [MS.Param(; name = "φ", value = "0.3", comment = "porosity")]
    src = gen(m)
    @test occursin("function _foam(φ)", src) && occursin("Homogenized(_foam(φ), SelfConsistent", src) && occursin("fraction = φ", src)
    @test occursin("const φ = 0.3  # porosity", src)
end

@testset "aging viscoelasticity" begin
    m = MS.default_model()
    m.alv.enabled = true
    src = gen(m)
    @test occursin("volterra_inverse(R; block_size = 6)", src) && occursin("homogenize_alv(", src)
    @test occursin("using Plots", src) && occursin("const times = range(0.0, 10.0; length = 41)", src)
    m.alv.log_time = true
    m.alv.component = Any[3, 3]
    m.alv.plot = false
    src = gen(m)
    @test occursin("const times = vcat(0.0, 10 .^ range(0.0, 10.0; length = 41))", src)
    @test occursin("6 * (a - 1) + 3, 6 * (b - 1) + 3", src) && !occursin("using Plots", src)
    m.alv.plot = true
    @test occursin("xscale = :log10)", gen(m))
end

@testset "round trip and text kept" begin
    m = MS.default_model()
    src = MS.generate(m)
    back = MS.extract_embedded(src)
    @test back !== nothing && MS.generate(MS.model_from_dict(back)) == src
    m = two_scale()
    m.cells[1].ui = props("x" => 123, "y" => 456)
    @test MS.model_from_dict(MS.extract_embedded(MS.generate(m))).cells[1].ui == props("x" => 123, "y" => 456)
    # a model with an infinite value is embedded as Python wrote it, and reads back
    m = MS.default_model()
    m.cells[1].phases[2].properties[1].args["k"] = Inf
    @test MS.model_from_dict(MS.extract_embedded(MS.generate(m))).cells[1].phases[2].properties[1].args["k"] == Inf
    @test MS.extract_embedded("no model here") === nothing
    @test MS.extract_embedded("#= mfhstudio-model v1\nno end") === nothing
    @test MS.extract_embedded("#= mfhstudio-model v1\nno brace\n=#") === nothing
    @test MS.extract_embedded("#= mfhstudio-model v1\n{broken\n=#") === nothing
    @test MS.extract_embedded("#= mfhstudio-model v1\n[1, 2]\n=#") === nothing
    # Python's special floats, outside strings only; integers stay integers
    v = MS.parse_python_json("{\"a\": [NaN, -Infinity, Infinity, 3, \"NaN \\\" Infinity\"]}")
    @test isnan(v["a"][1]) && v["a"][2] == -Inf && v["a"][3] == Inf && v["a"][4] === 3 && v["a"][5] == "NaN \" Infinity"
    m = MS.default_model()
    push!(m.params, MS.Param(; name = "T", value = "[1, 2]", origin = "const T = [\n    1,\n    2,\n]"))
    @test occursin("const T = [\n    1,\n    2,\n]", gen(m))
    m.params[end] = MS.Param(; name = "T", value = "99.0", origin = "const T = 1.0", edited = true)
    @test occursin("const T = 99.0", gen(m))
    # the description in the header, and the code kept verbatim in its order
    m.description = "two\nlines"
    m.opaque = [MS.OpaqueBlock(; source = "b = 2", order = 2), MS.OpaqueBlock(; source = "a = 1", order = 1, note = "kept as written: x")]
    src = gen(m)
    @test occursin("#  two\n#  lines", src)
    @test first(findfirst("a = 1", src)) < first(findfirst("b = 2", src)) && occursin("# kept as written: x\na = 1", src)
    # nothing to compute without a cell
    @test !occursin("Result", gen(MS.StudioModel()))
    @test MS.py_splitlines("a\r\nb\rc d\n") == ["a", "b", "c", "d"]
    @test MS.py_splitlines("") == String[] && MS.py_splitlines("x") == ["x"]
end
