using EpecRacing

time_steps = 10;
mode = 3;
x0 = [0.75, 0, 1, π / 2, -0.75, 0, 2, π / 2]
road = gen_road(seed=nothing);

# shift initial position wrt to road
road_ys = road |> keys |> collect
sortedkeys = sortperm((road_ys .- x0[2]) .^ 2)
lat_shift = road[road_ys[sortedkeys[1]]];
x0[1] = x0[1] + lat_shift
x0[5] = x0[5] + lat_shift

probs = setup(;
    T=10,
    Δt=0.1,
    r=1.0,
    cost_α1=1e-3,
    cost_α2=1e-4,
    cost_β=1e-1,
    drag_coef=0.2,
    u_max_nominal=1.0,
    u_max_drafting=2.5,
    draft_box_length=5.0,
    draft_box_width=2.0,
    d=2.0,
    min_speed=-1.0,
    max_heading_offset=π / 2,
    max_heading_rate=1.0
);

sim_results = solve_simulation(probs, time_steps; x0, road, mode)

animate(probs.params, sim_results; save=false, mode, road);
