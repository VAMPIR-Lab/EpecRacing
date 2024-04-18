using EPEC
using JLD2
using Infiltrator
include("postprocess_helpers.jl")

probs = setup(; T=10,
    Δt=0.1,
    r=1.0,
    α1=1e-3,
    α2=1e-4,
    β=.5, #.5, # sensitive to high values
    cd=0.1, #0.25,
    d=2.0, # actual road width (±)
    u_max_nominal=1.0,
    u_max_drafting=2.5, #2.5, # sensitive to high difference over nominal 
    box_length=5.0,
    box_width=5.0,
    lat_max=4.5 # just used for visulization rn 2024-04-10 (should be fixed)
);

data_dir = "data"
x0s_filename = "x0s_50samples_2024-04-17_1312"
results_suffix = "_(x0s_50samples_2024-04-17_1312)_2024-04-17_1312_25steps";
init_file = jldopen("$(data_dir)/$(x0s_filename).jld2", "r")
x0s = init_file["x0s"];
roads = init_file["roads"];

modes = 1:10
results = Dict()

for i in modes
    file = jldopen("$(data_dir)/results_mode$(i)$(results_suffix).jld2", "r")
    results[i] = process(file["results"])
end

include("print_cost_table.jl")
include("plot_boxplot.jl")
include("plot_running_cost.jl")

# visualize
experiment = 2;
x0 = x0s[experiment];
road = roads[experiment];

mode = 9;
sim_results = solve_simulation(probs, 25; x0, road, mode);
EPEC.animate(probs, sim_results; save=false, mode, road);


