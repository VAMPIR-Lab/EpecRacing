module EpecRacing

include("EPEC.jl")
using .EPEC
using GLMakie
using LinearAlgebra

include("racing.jl")
include("visualize_racing.jl")

export setup, solve_simulation, gen_road, animate

end # module EpecRacing
