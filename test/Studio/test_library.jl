# The worked examples: each validates, reopens exactly, and is what the code
# generation writes from its own model (its note kept), so that none drifts
# from what the studio writes.

samejson(a::AbstractDict, b::AbstractDict) = Set(keys(a)) == Set(keys(b)) && all(k -> samejson(a[k], b[k]), keys(a))
samejson(a::AbstractVector, b::AbstractVector) = length(a) == length(b) && all(samejson(x, y) for (x, y) in zip(a, b))
samejson(a, b) = typeof(a) == typeof(b) && isequal(a, b)

@testset "every example" begin
    files = MS.library_files()
    @test length(files) == 10
    for f in files
        text = read(f, String)
        embedded = MS.extract_embedded(text)
        m = MS.model_from_dict(embedded)
        @test isempty(MS.validate(m))
        @test samejson(JSON.parse(JSON.json(MS.to_dict(m))), JSON.parse(JSON.json(embedded)))
        @test MS.regenerate_example(text) == text
        @test last(MS.model_from_script(text))["exact"]
    end
end

@testset "regenerating the library" begin
    mktempdir() do dir
        for f in MS.library_files()
            cp(f, joinpath(dir, basename(f)))
        end
        @test isempty(MS.regenerate_library!(dir))
        # an edit of the code, not of the model, is written over from the model
        target = joinpath(dir, basename(first(MS.library_files())))
        original = read(target, String)
        write(target, replace(original, "using Printf" => "using Printf  # edited"; count = 1))
        @test MS.regenerate_library!(dir) == [target] && read(target, String) == original
    end
    @test_throws ArgumentError MS.regenerate_example("# no model here\n")
end
