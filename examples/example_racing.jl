#using EpecRacing
#using Infiltrator
#using Random
#include("../racing/road.jl")

#probs = setup(; T=10,
#    Δt=0.1,
#    r=1.0,
#    α1=1e-3,
#    α2=1e-4,
#    β=.5, #.5, # sensitive to high values
#    cd=0.1, #0.25,
#    d=2.0, # actual road width (±)
#    u_max_nominal=1.0,
#    u_max_drafting=2.5, #2.5, # sensitive to high difference over nominal 
#    box_length=5.0,
#    box_width=5.0,
#    lat_max=4.5 # just used for visulization rn 2024-04-10 (should be fixed)
#);

x0 = [.75, 0, 1, π / 2, -0.75, 0, 2, π / 2]

road = gen_road();
# shift initial position wrt to road
road_ys = road |> keys |> collect
sortedkeys = sortperm((road_ys .- x0[2]) .^ 2)
x0[1] = x0[1] + road[road_ys[sortedkeys[1]]]
sortedkeys = sortperm((road_ys .- x0[6]) .^ 2)
x0[5] = x0[5] + road[road_ys[sortedkeys[1]]]

mode = 3;
sim_results = solve_simulation(probs, 1; x0, road, mode)
animate(probs, sim_results; save=false, mode, road);
