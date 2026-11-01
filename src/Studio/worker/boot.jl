# Included by the compute process at start, in its `Main`: the scripts the
# studio writes run here, each in a module of its own.

module StudioWorker

using MeanFieldHomogenization, TensND
using Logging: ConsoleLogger, with_logger

include(joinpath(@__DIR__, "run.jl"))

end
