# The tests of MFH Studio (src/studio/), in a module of their own so that their
# helpers stay apart from those of the package's other tests. Part of the
# `Studio` group of the suite; the Studio workflow runs them alone too, with a
# compute process.

module StudioTests

using MeanFieldHomogenization, Test
using MeanFieldHomogenization.Studio: HTTP, JSON, Malt

const MS = MeanFieldHomogenization.Studio

@testset "MFH Studio" begin
    @testset "text conventions" include("test_pyformat.jl")
    @testset "model" include("test_model.jl")
    @testset "code generation" include("test_codegen.jl")
    @testset "read-back" include("test_readback.jl")
    @testset "catalog and views" include("test_views.jl")
    @testset "library" include("test_library.jl")
    @testset "memory" include("test_memory.jl")
    @testset "environment" include("test_environment.jl")
    @testset "server" include("test_server.jl")
    @testset "routes" include("test_api.jl")
    @testset "launcher" include("test_launcher.jl")
    # The compute process needs an environment holding MeanFieldHomogenization,
    # TensND and Plots; CI builds one, and a local run names it.
    if haskey(ENV, "MFHSTUDIO_WORKER_PROJECT")
        @testset "compute" include("test_compute.jl")
    else
        @info "Compute-process tests skipped: set MFHSTUDIO_WORKER_PROJECT to an environment holding MeanFieldHomogenization."
    end
end

end
