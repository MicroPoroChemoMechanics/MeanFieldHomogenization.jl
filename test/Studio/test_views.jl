# What the page offers (the catalog, read from the installed package) and what
# it draws (the shapes and stacks of the model, built by the package's own
# constructors, never by running text the page sent).

using LinearAlgebra: norm, normalize

module Fake
    # a scheme named like one whose options the package publishes, in a module
    # that publishes none
    struct SelfConsistent end
end

@testset "catalog" begin
    cat = MS.catalog()
    @test cat["introspected"] && cat["mfh_version"] == string(pkgversion(MeanFieldHomogenization))
    names = Set(s["name"] for s in cat["schemes"])
    for expected in (
            "Voigt", "Reuss", "MoriTanaka", "SelfConsistent", "AsymmetricSelfConsistent", "DifferentialScheme",
            "Dilute", "DiluteDual", "Maxwell", "PonteCastanedaWillis", "Laminated",
        )
        @test expected in names
    end
    # the schemes acting on positioned particles have no cell in the studio
    @test isempty(intersect(names, MS.ASSEMBLY_SCHEMES))
    for key in ("geometries", "properties", "symmetrize", "projections", "interfaces", "lenses", "visco", "sensitivities", "cell_kinds")
        @test !isempty(cat[key])
    end
    # the options a scheme reads, not what its keyword bag would accept
    sc = only(s for s in cat["schemes"] if s["name"] == "SelfConsistent")
    editable = Set(o["name"] for o in sc["options"] if o["editable"])
    @test "abstol" in editable && "select_best" in editable && !("nsteps" in editable)
    @test only(o["default"] for o in sc["options"] if o["name"] == "abstol") == 1.0e-12
    diff = only(s for s in cat["schemes"] if s["name"] == "DifferentialScheme")
    deditable = Set(o["name"] for o in diff["options"] if o["editable"])
    @test "nsteps" in deditable && !("select_best" in deditable)
    @test only(o["default"] for o in diff["options"] if o["name"] == "abstol") == 1.0e-8
    @test only(s for s in cat["schemes"] if s["name"] == "Voigt")["singleton"]
    @test MS.consumed_options(Fake.SelfConsistent) == Symbol[]
    @test MS.option_default(MeanFieldHomogenization.Voigt, :nothing_like_it) === nothing
    # the Hoenig defaults are not the isotropic point of the parametrization
    hoenig = only(f for f in MS.PROPERTIES if f.name == "ti_hoenig")
    d = Dict(f.name => f.default for f in hoenig.fields)
    @test !(d["h"] == 1.0 && d["nu1"] == d["nu2"] && d["gamma"] == 1.0)
    # what the page receives is JSON
    @test JSON.parse(JSON.json(cat))["geometries"][1]["fields"][1]["name"] == "omega"
end

@testset "numbers the page may type" begin
    m = MS.StudioModel(;
        params = [
            MS.Param(; name = "φ", value = "0.25"), MS.Param(; name = "a", value = "2 * φ"),
            MS.Param(; name = "x", value = "y"), MS.Param(; name = "y", value = "x"), MS.Param(; name = "v", value = "[1, 2]"),
        ]
    )
    @test MS.numeric(m, 2) === 2.0 && MS.numeric(m, true) === 1.0
    @test MS.numeric(m, "π / 4") ≈ π / 4 && MS.numeric(m, "sqrt(4) + 1e-3") ≈ 2.001
    @test MS.numeric(m, "a") ≈ 0.5 && MS.numeric(m, "cos(0) - φ") ≈ 0.75
    @test MS.numeric(m, "x") === nothing                  # parameters defined in a cycle
    @test MS.numeric(m, "v") === nothing && MS.numeric(m, "unknown") === nothing
    @test MS.numeric(m, "run(`rm -rf /`)") === nothing    # nothing typed is run
    @test MS.numeric(m, "1 +") === nothing && MS.numeric(m, nothing) === nothing
    @test MS.numeric_angles(m, Any[0.1, "", "π / 2"]) == (0.1, π / 2)
    @test MS.numeric_angles(m, Any["θ"]) === nothing
end

phase(kind, args; angles = Any[], layers = Any[]) =
    MS.Phase(; geometry = MS.Geometry(; kind, args = MS.jdict(args...), euler_angles = angles, layers))
conduction(k) = MS.jdict("key" => ":K", "source" => "builder", "builder" => "TensISO{3}", "args" => MS.jdict("k" => k))

@testset "shapes" begin
    m = MS.StudioModel()
    sphere = MS.scene(MS.shape(m, phase("spheroid", ("omega" => 0.4,))))
    @test Set(keys(sphere)) == Set(["data", "layout"]) && sphere["data"][1]["type"] == "surface"
    # guides do not clutter the legend
    @test all(t -> get(t, "showlegend", false) !== true, sphere["data"])
    for (kind, args, n) in (
            ("ellipsoid", ("a" => 2.0, "b" => 1, "c" => "0.5"), 3), ("cylinder", ("b" => 1.0, "c" => 0.5), 3),
            ("penny_crack", ("a" => 1.0,), 3), ("elliptic_crack", ("a" => 1.0,), 3), ("ribbon_crack", ("b" => 0.5,), 3),
        )
        g = MS.shape(m, phase(kind, args; angles = Any[0.3, 0.2]))
        @test length(MS.traces(g)) >= n
    end
    # a crack is drawn flat, with its normal
    crack = MS.traces(MS.shape(m, phase("penny_crack", ("a" => 1.0,))); guides = false)
    @test length(crack) == 3 && crack[3]["text"] == ["n"]
    # an unknown kind is the unit sphere the script writes
    @test MS.shape(m, phase("mystery", ())) isa MeanFieldHomogenization.Ellipsoid
    @test MS.shape(m, phase("spheroid", ("omega" => "ω",))) === nothing
    @test MS.shape(m, phase("spheroid", ("omega" => 1.0,); angles = Any["θ"])) === nothing
    # layered inclusions: concentric shells, cut away or whole
    lay(r) = MS.jdict("radius" => r, "property" => conduction(1.0))
    ls = MS.shape(m, phase("layered_sphere", (); layers = Any[lay(0.6), lay(1)]))
    @test [t["name"] for t in MS.traces(ls)] == ["layer 1 (r = 0.6)", "layer 2 (r = 1.0)"]
    @test length(MS.traces(ls; cutaway = false)[1]["x"]) == 41
    @test MS.shape(m, phase("layered_sphere", ())) === nothing
    @test MS.shape(m, phase("layered_sphere", (); layers = Any[lay("r")])) === nothing
    frac(f) = MS.jdict("fraction" => f, "property" => conduction(1.0))
    upright = MS.shape(m, phase("layered_spheroid", ("omega" => 0.5, "radius" => 1.0, "Nseries" => 5); layers = Any[frac(0.3), frac(0.7)]))
    @test MS.frame_matrix(upright)[:, 3] ≈ [0, 0, 1]
    @test length(MS.traces(upright; guides = true)) == 6          # two layers, four guides
    # a tilted layered spheroid is drawn tilted: its revolution axis is the
    # third vector of the frame its angles give
    θ, ϕ = 0.9, 0.4
    tilted = MS.shape(m, phase("layered_spheroid", ("omega" => 0.5, "Nseries" => 5); angles = Any[θ, ϕ], layers = Any[frac(0.3), frac(0.7)]))
    axis = [sin(θ) * cos(ϕ), sin(θ) * sin(ϕ), cos(θ)]
    @test abs(MS.frame_matrix(tilted)[:, 3]' * axis) ≈ 1
    # the outer layer's points span the plane normal to the axis
    out = last(MS.traces(tilted))
    pts = [[x, y, z] for (xs, ys, zs) in zip(out["x"], out["y"], out["z"]) for (x, y, z) in zip(xs, ys, zs)]
    @test maximum(abs(p' * axis) for p in pts) < maximum(norm(p) for p in pts)
    @test MS.shape(m, phase("layered_spheroid", ("Nseries" => 2.5,); layers = Any[frac(1.0)])) === nothing
    @test MS.shape(m, phase("layered_spheroid", (); layers = Any[])) === nothing
end

function bilayer()
    lam = MS.Cell(; name = "lam", kind = "laminate", layers = MS.default_layers())
    return MS.StudioModel(; cells = [lam])
end
slabs(sc) = [t for t in sc["data"] if t["type"] == "mesh3d"]
function stacking_direction(sc)
    a, c = slabs(sc)
    ca = [sum(a[k]) / length(a[k]) for k in ("x", "y", "z")]
    cc = [sum(c[k]) / length(c[k]) for k in ("x", "y", "z")]
    return normalize(cc - ca)
end

@testset "stacks" begin
    m = bilayer()
    c = only(m.cells)
    upright = MS.view_of(m, c.id, nothing)
    @test length(slabs(upright)) == 2 && [first(split(t["name"])) for t in slabs(upright)] == ["A", "B"]
    @test abs(stacking_direction(upright)[3]) > 0.99
    # the same tilt, by the normal or by the angles (θ off e₃ in the x–z plane)
    want = [1, 0, 1] / sqrt(2)
    c.frame_mode = "normal"
    c.normal = Any[1.0, 0.0, 1.0]
    @test abs(stacking_direction(MS.view_of(m, c.id, nothing))' * want) > 0.99
    c.frame_mode = "euler"
    c.euler_angles = Any["π / 4"]
    @test abs(stacking_direction(MS.view_of(m, c.id, nothing))' * want) > 0.99
    # a layer reading an inner scale is drawn: properties do not change the stack
    c.layers[1].properties = [MS.Property(; source = "cell", cell = "elsewhere")]
    @test length(slabs(MS.view_of(m, c.id, nothing; cutaway = false))) == 2
    # thicknesses, and the canonical normal given as a vector
    c.frame_mode = "normal"
    c.normal = Any[0, 0, 1]
    for l in c.layers
        l.amount_kind = "thickness"
    end
    @test length(MS.traces(MS.stack(m, c); guides = false)) == 2
    c.layers[1].amount = "h"
    @test MS.stack(m, c) === nothing
    c.normal = Any["n", 0, 1]
    @test MS.stack(m, c) === nothing
    c.frame_mode = "euler"
    c.euler_angles = Any["θ"]
    @test MS.stack(m, c) === nothing
    @test isempty(MS.traces(MS.stack(m, MS.Cell(; kind = "laminate"))))
    zero = MS.Cell(; kind = "laminate", layers = [MS.Layer(; amount = 0.0)])
    @test isempty(MS.traces(MS.stack(m, zero)))
end

@testset "what the page asks for" begin
    m = MS.default_model()
    c = only(m.cells)
    @test MS.view_of(m, c.id, "PORE")["data"][1]["type"] == "surface"
    @test MS.view_of(m, "gone", "PORE")["note"] == "no such cell"
    @test occursin("not numbers", MS.view_of(m, c.id, "NO SUCH PHASE")["note"])
    c.phases[2].geometry.args["omega"] = "ω"
    @test occursin("not numbers", MS.view_of(m, c.id, "PORE")["note"])
    # a shape the package refuses says why
    lay(r) = MS.jdict("radius" => r, "property" => conduction(1.0))
    c.phases[2].geometry = MS.Geometry(; kind = "layered_sphere", layers = Any[lay(1.0), lay(0.5)])
    refused = MS.view_of(m, c.id, "PORE")
    @test isempty(refused["data"]) && occursin("ascending", refused["note"])
    @test MS.scene(MeanFieldHomogenization.Spheroid(1.0))["layout"]["scene"]["aspectmode"] == "data"
end
