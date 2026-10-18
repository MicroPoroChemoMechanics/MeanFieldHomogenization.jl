# Reading a script back: what the forms model is recognized, everything else is
# kept verbatim. The load-bearing check is the first: opening and saving any
# demo script of the package loses none of its lines.

const REPO = pkgdir(MeanFieldHomogenization)
demo_scripts = sort(filter(f -> endswith(f, ".jl"), readdir(joinpath(REPO, "scripts"); join = true)))

@testset "every demo script is preserved" begin
    @test !isempty(demo_scripts)
    lossy = String[]
    for f in demo_scripts
        src = read(f, String)
        # the nodes tile the file: joined, they give it back byte for byte
        @test MS.reconstruct(MS.parse_script(src)) == src
        m, report = MS.model_from_script(src)
        @test report["exact"] == false && report["opaque"] == length(m.opaque)
        out = MS.generate(m; embed_model = false)
        missing_lines = [strip(l) for l in split(src, '\n') if !isempty(strip(l)) && !startswith(strip(l), "#") && !occursin(strip(l), out)]
        isempty(missing_lines) || push!(lossy, basename(f) * ": " * first(missing_lines))
    end
    @test isempty(lossy)
end

@testset "a script the studio wrote reopens exactly" begin
    m = MS.default_model()
    src = MS.generate(m)
    back, report = MS.model_from_script(src)
    @test report["exact"] && report["recognized"] === nothing
    @test JSON.json(MS.to_dict(back)) == JSON.json(MS.to_dict(MS.model_from_dict(MS.extract_embedded(src))))
    @test MS.generate(back) == src
    # a damaged model block is a comment: the code around it is read, and kept
    damaged = replace(src, "\"title\": \"porous_benchmark\"" => "\"title\": porous_benchmark")
    back, report = MS.model_from_script(damaged)
    @test !report["exact"] && [c.name for c in back.cells] == ["rve"]
    @test any(b -> occursin("mfhstudio-model v1", b.source), back.opaque)
end

@testset "a laminate written by the studio is read back" begin
    lam = MS.Cell(; name = "lam", kind = "laminate", layers = MS.default_layers(), frame_mode = "normal", normal = Any[1.0, 0.0, 1.0])
    lam.layers[1].interface = MS.jdict("kind" => "SpringInterface", "args" => MS.jdict("kn" => 1.0e-3, "kt" => 2.0e-3))
    m = MS.StudioModel(; cells = [lam])
    back, report = MS.model_from_script(MS.generate(m; embed_model = false))
    @test [c.kind for c in back.cells] == ["laminate"] && report["cells"] == ["lam"]
    c = only(back.cells)
    @test c.frame_mode == "normal" && c.normal == [1.0, 0.0, 1.0]
    @test [l.name for l in c.layers] == ["A", "B"] && [l.amount for l in c.layers] == [0.3, 0.7]
    @test c.layers[1].interface["kind"] == "SpringInterface" && c.layers[1].interface["args"] == MS.jdict("kn" => 0.001, "kt" => 0.002)
    @test c.layers[2].interface["kind"] == "PerfectInterface"
    # the frame given by angles, and by nothing at all
    lam.frame_mode = "euler"
    lam.euler_angles = Any["π / 4", 0.3]
    c = only(first(MS.model_from_script(MS.generate(m; embed_model = false))).cells)
    @test c.frame_mode == "euler" && c.euler_angles == Any["π / 4", 0.3]
    # an expression Julia prints otherwise (`π/4` becomes `π / 4`) would not be
    # written back as it was: the cell is kept as written
    lam.euler_angles = Any["π/4", 0.3]
    back, _ = MS.model_from_script(MS.generate(m; embed_model = false))
    @test isempty(back.cells) && any(b -> occursin("would not reproduce", b.note), back.opaque)
    lam.euler_angles = Any[]
    c = only(first(MS.model_from_script(MS.generate(m; embed_model = false))).cells)
    @test c.frame_mode == "euler" && isempty(c.euler_angles) && c.normal == [0.0, 0.0, 1.0]
end

@testset "parameters, cells and the rest" begin
    src = """
    const k_s = 72.0
    const T = [
        1,
    ]

    function build_foam(φ)
        rve = RVE(; T = ComplexF64)
        add_phase!(
            rve, :SOLID, Spheroid(1.0), Dict(:C => iso_stiffness(k_s, 32.0));
            fraction = :rest
        )
        add_phase!(
            rve, :PORE, Spheroid(0.5; euler_angles = (0.1, 0.2)), Dict(:C => iso_stiffness(1.0e-6, 1.0e-6));
            fraction = φ, symmetrize = IsoSymmetrize()
        )
        add_phase!(rve, :X, PennyCrack(1.0), Dict{Symbol, Any}(); density = 0.1)
        return rve
    end

    # a comment between statements

    C = homogenize(build_foam(0.2), MoriTanaka(), :C)
    println(C)
    """
    m, report = MS.model_from_script(src)
    @test [p.name for p in m.params] == ["k_s", "T"] && m.params[2].origin == "const T = [\n    1,\n]"
    foam = only(m.cells)
    @test foam.name == "foam" && m.root_cell == foam.id && foam.params == ["φ"]
    @test foam.rve_options == MS.jdict("T" => "ComplexF64")
    @test foam.phases[2].amount == "φ" && foam.phases[2].symmetrize == "iso"
    @test foam.phases[2].geometry.args["omega"] == 0.5 && foam.phases[2].geometry.euler_angles == Any[0.1, 0.2]
    @test foam.phases[1].properties[1].args == MS.jdict("k" => "k_s", "mu" => 32.0)
    @test foam.phases[3].amount_kind == "density" && foam.phases[3].amount == 0.1 && isempty(foam.phases[3].properties)
    @test any(b -> occursin("a comment between statements", b.source), m.opaque)
    @test any(b -> occursin("homogenize(build_foam(0.2)", b.source), m.opaque)
    @test report["recognized"] == 4 && report["cells"] == ["foam"]
    # what is read back is written back: the script keeps every line
    out = MS.generate(m; embed_model = false)
    @test all(l -> occursin(strip(l), out), filter(l -> !isempty(strip(l)), split(src, '\n')))
    # whatever does not parse is kept as written
    m, report = MS.model_from_script("x = (\n")
    @test isempty(m.cells) && only(m.opaque).source == "x = (" && report["recognized"] == 0
end

@testset "the matrix and the scales, read back" begin
    # `fraction = :rest` marks the phase taking up the rest: read as a fraction
    # ":rest", it left the cell with no matrix
    back, _ = MS.model_from_script(MS.generate(MS.default_model(); embed_model = false))
    @test only(back.cells).phases[1].amount_kind == "rest" && isempty(MS.validate(back))
    # two scales written without their model block read back whole: the seam
    # names its builder, whatever its name, and keeps the options of its scheme
    inner = MS.Cell(;
        name = "foam", builder_name = "_foam", phases = [
            MS.Phase(; name = "SOLID", amount_kind = "rest", properties = [MS.Property(; args = MS.jdict("k" => 72.0, "mu" => 32.0))]),
            MS.Phase(; name = "PORE", amount = 0.3, properties = [MS.Property(; args = MS.jdict("k" => 1.0e-6, "mu" => 1.0e-6))]),
        ]
    )
    seam = MS.Property(; source = "cell", cell = inner.id, scheme = "SelfConsistent", scheme_options = MS.jdict("abstol" => 1.0e-10, "maxiters" => 300, "verbose" => true))
    outer = MS.Cell(; name = "paste", phases = [MS.Phase(; name = "FOAM", amount_kind = "rest", properties = [seam]), MS.Phase(; name = "C", amount = 0.2, properties = [MS.Property()])])
    m = MS.StudioModel(; title = "ms", cells = [inner, outer], root_cell = outer.id)
    src = MS.generate(m; embed_model = false)
    back, report = MS.model_from_script(src)
    @test report["cells"] == ["_foam", "paste"]
    got = back.cells[2].phases[1].properties[1]
    @test got.source == "cell" && got.cell == back.cells[1].id && got.scheme == "SelfConsistent"
    @test got.scheme_options == MS.jdict("abstol" => 1.0e-10, "maxiters" => 300, "verbose" => true)
    @test isempty(MS.validate(back))
    out = MS.generate(back; embed_model = false)
    @test all(l -> occursin(strip(l), out), filter(l -> !isempty(strip(l)), split(src, '\n')))
    # saved on Windows, the same script: its builders compare with what the
    # code generation writes line by line, not with a `\r` at each end
    crlf, crlf_report = MS.model_from_script(replace(src, "\n" => "\r\n"))
    @test crlf_report == report && MS.generate(crlf; embed_model = false) == out
    # an inner builder kept as written keeps the outer one as written too: the
    # seam is never rewritten into an expression that would break the script
    hand = replace(src, "function _foam()" => "function _foam(x::Float64 = 1.0)")
    back, report = MS.model_from_script(hand)
    @test isempty(back.cells) && count(b -> startswith(b.note, "kept as written"), back.opaque) == 2
    # a seam whose scheme takes a positional argument, or whose builder is not
    # a plain call, is an expression
    @test MS.parse_property(":C", "Homogenized(build_x(), Scheme(1))").source == "expr"
    @test MS.parse_property(":C", "Homogenized(cells[1], MoriTanaka())").source == "expr"
    @test MS.parse_property(":C", "Homogenized(build_x(), schemes[1])").source == "expr"
    @test MS.parse_property(":C", "Homogenized(build_x(), Voigt(; a = :b))").scheme_options == MS.jdict("a" => ":b")
end

@testset "what is kept as written, and why" begin
    kept(src) = only(first(MS.model_from_script(src)).opaque).note
    rve(body; sig = "build_a()") = "function $sig\n    rve = RVE()\n$body\n    return rve\nend"
    matrix = "    add_phase!(rve, :M, Spheroid(1.0), Dict(:C => iso_stiffness(1.0, 1.0)); fraction = :rest)"
    @test kept(rve(matrix; sig = "build_a(x::Float64)")) == "kept as written: the builder signature uses `x::Float64`, which the studio does not model"
    @test kept(rve("    add_phase!(rve, :M, Torus(1.0), Dict(:C => iso_stiffness(1.0, 1.0)); fraction = :rest)")) ==
        "kept as written: geometry `Torus(1.0)` is not one the studio can rebuild"
    @test kept(rve("    add_phase!(rve, :M, Spheroid(1.0), props; fraction = :rest)")) ==
        "kept as written: properties `props` are not a Dict the studio can rebuild"
    @test kept(rve(matrix * "\n    add_phase!(rve, :I, Spheroid(1.0), Dict(:C => iso_stiffness(1.0, 1.0)); fraction = 0.1, symmetrize = MySym())")) ==
        "kept as written: symmetrize = `MySym()` is not one the studio can rebuild"
    @test kept(rve(matrix * "\n    println(\"extra\")")) == "kept as written: writing it back would not reproduce the original text"
    lam(body; opts = "") = "function build_l()\n    lam = Laminate($opts)\n$body\n    return lam\nend"
    layer = "    add_layer!(lam, :A, Dict(:C => iso_stiffness(1.0, 1.0)); fraction = 1.0)"
    @test kept(lam(layer; opts = "; basis = B")) == "kept as written: `basis = …` on the Laminate is not one the studio models"
    @test kept(lam("")) == "kept as written: the laminate has no layer"
    @test kept(lam("    add_layer!(lam, :A, d; fraction = 1.0)")) == "kept as written: properties `d` are not a Dict the studio can rebuild"
    @test kept(lam("    add_layer!(lam, :A, Dict(:C => iso_stiffness(1.0, 1.0)); fraction = 1.0, extra = 2)")) ==
        "kept as written: `extra = …` on add_layer! is not one the studio models"
    @test kept(lam("    add_layer!(lam, :A, Dict(:C => iso_stiffness(1.0, 1.0)))")) == "kept as written: layer `A` gives neither a fraction nor a thickness"
    @test kept(lam("    add_layer!(lam, :A, Dict(:C => iso_stiffness(1.0, 1.0)); fraction = 1.0, interface = MyInterface(1))")) ==
        "kept as written: interface `MyInterface(1)` is not one the studio can rebuild"
    @test kept(lam(layer * "\n    add_layer!(lam, :B, Dict(:C => iso_stiffness(1.0, 1.0)); thickness = 1.0)")) ==
        "kept as written: the stack mixes fractions and absolute thicknesses"
    # a layer given by its thickness is read
    m, _ = MS.model_from_script(lam("    add_layer!(lam, :A, Dict(:C => iso_stiffness(1.0, 1.0)); thickness = 2.0)"))
    @test only(only(m.cells).layers).amount_kind == "thickness" && only(only(m.cells).layers).amount == 2.0
end

@testset "pieces" begin
    @test MS.py_float("1_000") == 1000.0 && MS.py_float("1_0.5") == 10.5
    @test MS.py_float("1__0") === nothing && MS.py_float("_1") === nothing && MS.py_float("1_") === nothing
    @test MS.py_float("0x10") === nothing && MS.py_float(" 2 ") == 2.0 && MS.py_float("π") === nothing
    @test MS.py_float("Infinity") == Inf && isnan(MS.py_float("nan"))
    @test MS.number_or_text(" 2 ") === 2.0 && MS.number_or_text(" 2π ") == "2π"
    shapes = [
        "Ellipsoid(1.0, 0.5, c)" => ("ellipsoid", MS.jdict("a" => 1.0, "b" => 0.5, "c" => "c")),
        "Cylinder(1, 2)" => ("cylinder", MS.jdict("b" => 1.0, "c" => 2.0)),
        "PennyCrack(1.0; euler_angles = (0.1,))" => ("penny_crack", MS.jdict("a" => 1.0)),
        "EllipticCrack(1.0, 0.5)" => ("elliptic_crack", MS.jdict("a" => 1.0, "b" => 0.5)),
        "RibbonCrack(0.2)" => ("ribbon_crack", MS.jdict("b" => 0.2)),
    ]
    for (expr, (kind, args)) in shapes
        g = MS.parse_geometry(expr)
        @test g.kind == kind && g.args == args
    end
    @test MS.parse_geometry("PennyCrack(1.0; euler_angles = (0.1,))").euler_angles == Any[0.1]
    @test MS.parse_geometry("LayeredSphere((1.0,), (x,))") === nothing
    @test MS.parse_property(":C", "iso_stiffness_E_nu(30.0, 0.2)").args == MS.jdict("E" => 30.0, "nu" => 0.2)
    @test MS.parse_property(":C", "hoenig_stiffness(1, 2, 3, 4, 5)").builder == "hoenig_stiffness"
    seam = MS.parse_property(":C", "Homogenized(build_inner(), Voigt())")
    @test seam.cell == "@builder:build_inner" && seam.scheme == "Voigt" && isempty(seam.scheme_options)
    @test MS.parse_property(":C", "Homogenized(build_inner())").source == "expr"
    @test MS.parse_property(":K", " TensISO{3}(2.0) ").expr == "TensISO{3}(2.0)"
    @test isempty(MS.parse_properties("props")) && isempty(MS.parse_properties("Dict{Symbol, Any}()"))
    ps = MS.parse_properties("Dict(:C => iso_stiffness(1.0, 2.0), :K => TensISO{3}(f(1, 2)), \"x\" => 1, junk)")
    @test [p.key for p in ps] == [":C", ":K"] && ps[2].expr == "TensISO{3}(f(1, 2))"
    @test MS.parse_interface("SpringInterface(1.0e-3, 2.0e-3)") == MS.jdict("kind" => "SpringInterface", "args" => MS.jdict("kn" => 0.001, "kt" => 0.002))
    @test MS.parse_interface("AnisotropicSpringInterface([1.0 0.0; 0.0 2.0])")["args"]["compliance"] == "[1.0 0.0; 0.0 2.0]"
    @test MS.parse_interface("PerfectInterface()") == MS.jdict("kind" => "PerfectInterface", "args" => MS.JDict())
    @test MS.parse_interface("SpringInterface(1.0)") === nothing
    @test MS.parse_interface("OtherInterface(1.0)") === nothing && MS.parse_interface("f(1)") === nothing
    @test MS.tuple_items("(1, (2, 3), [4, 5])") == ["1", "(2, 3)", "[4, 5]"] && MS.tuple_items("a, b") == ["a", "b"]
    @test MS.split_toplevel(" , a ,, b ") == ["a", "b"]
    @test MS.cell_name("build_foam") == "foam" && MS.cell_name("_custom") == "_custom"
    # keywords after a comma as after a semicolon
    @test MS.keyword_args(Meta.parse("f(a, b = 1; c = 2)")) == Dict("b" => "1", "c" => "2")
    # the matrix named on the constructor and `add_matrix!` of the scripts before 0.8
    legacy = """
    function build_old()
        rve = RVE(:SOLID)
        add_matrix!(rve, Spheroid(1.0), Dict(:C => iso_stiffness(72.0, 32.0)))
        add_matrix!(rve, Spheroid(1.0))
        add_phase!(rve, :PORE, Spheroid(1.0), Dict(:C => iso_stiffness(1.0, 1.0)), fraction = 0.1)
        return rve
    end
    """
    node = first(MS.parse_script(legacy)["nodes"])
    @test node["kind"] == "rve_builder" && node["matrix"] == "SOLID"
    @test [p["name"] for p in node["phases"]] == ["SOLID", "PORE"] && node["phases"][1]["is_matrix"]
    @test node["phases"][2]["options"] == Dict("fraction" => "0.1")
    c = MS.cell_from_node(node)
    @test c.matrix_name == "SOLID" && [p.amount_kind for p in c.phases] == ["rest", "fraction"]
end
