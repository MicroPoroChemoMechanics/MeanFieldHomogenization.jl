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
    @testset "memory" include("test_memory.jl")
    @testset "environment" include("test_environment.jl")
    @testset "server" include("test_server.jl")
    @testset "launcher" include("test_launcher.jl")
end

end
