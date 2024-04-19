module EpecRacing

include("EPEC.jl")
using .EPEC
using LinearAlgebra
import CairoMakie
using GLMakie
GLMakie.activate!()
using Random
using Dates
using JLD2
using Statistics
using Infiltrator

include("racing.jl")
include("visualize_racing.jl")
include("experiment_utils.jl")

export setup, solve_simulation, gen_road, animate, create_experiment, solve_experiment, animate, read_from_file, process_results, gen_all_tables, print_mean_etc, get_mean_running_vel_cost, get_mean_running_total_cost

end # module EpecRacing
