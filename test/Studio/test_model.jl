# The authoring model: its JSON round trip, the migration of models saved by
# older studios, the graph of scales, and every problem `validate` reports.

rt(m) = MS.model_from_dict(JSON.parse(JSON.json(MS.to_dict(m))))
iso(k, mu; key = ":C") = MS.Property(; key, args = MS.jdict("k" => k, "mu" => mu))
rve(name, phases...; kw...) = MS.Cell(; name, phases = collect(MS.Phase, phases), kw...)
matrix(name = "M") = MS.Phase(; name, amount_kind = "rest", properties = [iso(72.0, 32.0)])
inclusion(name = "I"; kw...) = MS.Phase(; name, amount = 0.2, properties = [iso(1.0, 1.0)], kw...)

@testset "round trip" begin
    m = MS.default_model()
    back = rt(m)
    @test JSON.json(MS.to_dict(back)) == JSON.json(MS.to_dict(m))
    @test back.cells[1].phases[1].amount_kind == "rest" && back.sweep.lens.phase == "PORE"
    @test MS.root(back).id == m.root_cell
    # integers stay integers, floats floats: the scripts write them apart
    @test back.cells[1].ui["x"] === 40 && back.cells[1].phases[2].amount === 0.1
    # the keys of a JSON object keep their order (the arguments of an interface do)
    lay = MS.Layer(; interface = MS.jdict("kind" => "SpringInterface", "args" => MS.jdict("kt" => 2.0, "kn" => 1.0)))
    c = MS.Cell(; kind = "laminate", layers = [lay])
    @test collect(keys(rt(MS.StudioModel(; cells = [c])).cells[1].layers[1].interface["args"])) == ["kt", "kn"]
    @test length(MS.new_id()) == 8 && MS.new_id() != MS.new_id()
    @test MS.builder(MS.Cell(; name = "x")) == "build_x"
    @test MS.builder(MS.Cell(; name = "x", builder_name = "_mine")) == "_mine"
    @test MS.builder(MS.Cell(; name = "x", builder_name = "")) == "build_x"
    @test length(MS.default_layers()) == 2 && sum(l.amount for l in MS.default_layers()) ≈ 1
end

@testset "models of older studios" begin
    old = MS.model_from_dict(
        JSON.parse(
            """{"cells": [{"name": "old", "phases": [{"name": "M", "is_matrix": true}, {"name": "I"}]},
                      {"name": "lam", "kind": "laminate", "normal": [1, 0, 1]},
                      {"name": "flat", "kind": "laminate", "normal": [0, 0, 1], "frame_mode": ""},
                      {"id": "", "ui": null, "normal": null, "euler_angles": null}],
            "sweep": {"scheme": "SelfConsistent", "scheme_options": {"abstol": 1e-9}, "outputs": ["k", "E"]},
            "sens": {"output": "nu", "kind": "derivative", "lenses": [{"kind": "amount"}, {"kind": "geometry"}]},
            "alv": {"component": [3, 3, 9]}}"""
        )
    )
    @test [p.amount_kind for p in old.cells[1].phases] == ["rest", "fraction"]
    @test old.cells[2].frame_mode == "normal"        # a tilted normal decides the mode
    @test old.cells[3].frame_mode == "euler"
    @test length(old.cells[4].id) == 8 && old.cells[4].ui["x"] == 40 && old.cells[4].normal == [0.0, 0.0, 1.0]
    @test old.sweep.schemes == [MS.jdict("name" => "SelfConsistent", "options" => MS.jdict("abstol" => 1.0e-9))]
    @test [o["kind"] for o in old.sweep.outputs] == ["k", "E"]
    @test old.sens.output["kind"] == "nu" && length(old.sens.lenses) == 1    # a derivative takes one
    @test old.alv.component == [3, 3]
    empty = MS.model_from_dict(
        JSON.parse(
            """{"sweep": {"schemes": [{"name": "Voigt"}], "outputs": []},
            "sens": {"kind": "gradient", "lenses": []}, "alv": {"component": []}}"""
        )
    )
    @test empty.sweep.schemes[1]["options"] == MS.JDict() && [o["kind"] for o in empty.sweep.outputs] == ["k", "mu"]
    @test length(empty.sens.lenses) == 1 && empty.alv.component == [1, 1]
    @test MS.model_from_dict(MS.JDict()).title == "model"
end

@testset "graph of scales" begin
    inner = rve("inner", matrix(), inclusion())
    outer = rve("outer", matrix(), inclusion("H"; properties = [MS.Property(; source = "cell", cell = inner.id)]))
    m = MS.StudioModel(; cells = [outer, inner])
    @test MS.uses_multiscale(m)
    @test [c.name for c in MS.topological_order(m)] == ["inner", "outer"]
    @test MS.dependencies(m, outer) == [inner.id]
    @test MS.cell_by_name(m, "inner") === inner && MS.cell_by_name(m, "none") === nothing
    @test MS.cell(m, nothing) === nothing && MS.cell(m, "none") === nothing
    @test MS.root(m) === inner                       # the last cell, without a root
    @test MS.root(MS.StudioModel()) === nothing
    # a cycle is named
    push!(inner.phases[2].properties, MS.Property(; source = "cell", cell = outer.id))
    e = try
        MS.topological_order(m)
    catch err
        err
    end
    @test e isa ArgumentError && e.msg == "multiscale cycle: outer → inner → outer"
    @test e.msg in MS.validate(m)
    # a seam to a cell that is not there is no dependency the order can follow
    lone = rve("a", matrix(), inclusion("H"; properties = [MS.Property(; source = "cell", cell = "gone")]))
    @test [c.name for c in MS.topological_order(MS.StudioModel(; cells = [lone]))] == ["a"]
    @test !MS.uses_multiscale(MS.StudioModel(; cells = [rve("b", matrix())]))
end

@testset "frames" begin
    @test !MS.tilted_normal(nothing) && !MS.tilted_normal(Any[]) && !MS.tilted_normal(Any[0, 0, 1])
    @test MS.tilted_normal(Any[1.0, 0.0, 1.0]) && MS.tilted_normal(Any["a", 0, 1]) && MS.tilted_normal(Any[0, 1])
    @test MS.is_canonical_frame(MS.Cell(; euler_angles = Any[0, 0.0, ""]))
    @test !MS.is_canonical_frame(MS.Cell(; euler_angles = Any["π / 4"]))
    @test MS.is_canonical_frame(MS.Cell(; frame_mode = "normal", normal = Any[0, 0, 1]))
    @test !MS.is_canonical_frame(MS.Cell(; frame_mode = "normal", normal = Any[1, 0, 0]))
end

problems(m) = MS.validate(m)
has(m, text) = any(p -> occursin(text, p), problems(m))

@testset "problems" begin
    @test isempty(problems(MS.default_model()))
    @test has(MS.StudioModel(; cells = [rve("a", matrix()), rve("a", matrix())]), "two cells are both named `a`")
    @test has(MS.StudioModel(; cells = [rve("a", inclusion())]), "cell `a` has no matrix phase")
    @test has(MS.StudioModel(; cells = [rve("a", matrix("M"), matrix("N"))]), "cell `a` has 2 matrix phases (M, N); a cell has one")
    @test has(MS.StudioModel(; cells = [rve("a", matrix(), inclusion("I"), inclusion("I"))]), "cell `a` has two phases named `I`")
    lam(layers...; kw...) = MS.Cell(; name = "lam", kind = "laminate", layers = collect(MS.Layer, layers), kw...)
    @test has(MS.StudioModel(; cells = [lam()]), "laminate `lam` has no layer")
    mixed = lam(MS.Layer(; name = "A"), MS.Layer(; name = "B", amount_kind = "thickness"))
    @test has(MS.StudioModel(; cells = [mixed]), "mixes absolute thicknesses")
    @test has(MS.StudioModel(; cells = [lam(MS.Layer(; amount = 0.3), MS.Layer(; name = "B", amount = 0.5))]), "sum to 0.8, not 1")
    @test has(MS.StudioModel(; cells = [lam(MS.Layer(; amount = 1, amount_kind = "thickness"), MS.Layer(; name = "B", amount = -1, amount_kind = "thickness"))]), "layer `B` has a negative thickness")
    @test has(MS.StudioModel(; cells = [lam(MS.default_layers()...; frame_mode = "normal", normal = Any[0, 0, 0])]), "has a null normal")
    @test !has(MS.StudioModel(; cells = [lam(MS.Layer(; amount = "f"), MS.Layer(; name = "B"))]), "sum to")
    @test has(MS.StudioModel(; cells = [lam(MS.Layer(), MS.Layer())]), "cell `lam` has two layers named `A`")
    # a scheme that needs a matrix, asked of a laminate
    m = MS.StudioModel(; cells = [lam(MS.default_layers()...)])
    m.sweep.enabled = true
    m.sweep.projection = "iso"
    @test has(m, "Sweep: `MoriTanaka` needs a matrix phase, so it does not apply to the laminate `lam`. A laminate takes `Laminated`, `Voigt`, `Reuss`.")
    m.sweep.schemes = Any[MS.jdict("name" => "Laminated", "options" => MS.JDict())]
    @test isempty(problems(m))
    m.sweep.projection = "none"
    @test has(m, "k, μ, E and ν are only defined for an isotropic result. This model")
    m.alv.enabled = true
    m.alv.property = ":K"
    m.alv.scheme = "Laminated"
    @test !has(m, "transport")
    m.cells[1].frame_mode = "normal"
    m.cells[1].normal = Any[1, 0, 0]
    @test has(m, "aging viscoelasticity of a laminate in transport (`:K`) requires the canonical frame")
    m.alv.enabled = false
    m.sens.enabled = true
    m.sens.scheme = "Voigt"
    m.sens.lenses = [MS.Lens(; kind = "amount"), MS.Lens(; kind = "thickness")]
    m.sens.kind = "gradient"
    @test has(m, "a laminate has no phase amount")
    m.sens.kind = "derivative"
    @test has(m, "a derivative is with respect to one parameter")
    m.sens.lenses = MS.Lens[]
    @test has(m, "the sensitivity needs at least one parameter")
    m.sens.lenses = [MS.Lens(; kind = "thickness")]
    m.sens.projection = "none"
    @test has(m, "Pick a reporting projection, or differentiate a Kelvin-Mandel component")
    m.sens.kind = "jacobian"
    @test !has(m, "Pick a reporting projection")
    r = MS.StudioModel(; cells = [rve("a", matrix(), inclusion())])
    r.sens.enabled = true
    r.sens.lenses = [MS.Lens(; kind = "interface_param")]
    @test has(r, "the `interface_param` parameter only exists on a laminate")
    r.sens.lenses = [MS.Lens(; kind = "amount")]
    r.sens.projection = "none"
    @test has(r, "Pick a reporting projection")     # phases with no orientation average
    # a scheme on a seam, aging viscoelasticity with a nested scale
    inner = lam(MS.default_layers()...)
    outer = rve("outer", matrix(), inclusion("H"; properties = [MS.Property(; source = "cell", cell = inner.id, scheme = "SelfConsistent")]))
    two = MS.StudioModel(; cells = [inner, outer])
    @test has(two, "`outer`/`H`:C: `SelfConsistent` needs a matrix phase")
    two.alv.enabled = true
    two.alv.scheme = "Laminated"
    @test has(two, "aging viscoelasticity cannot be combined with a nested scale")
    @test MS.py_str(nothing) == "None"
end
