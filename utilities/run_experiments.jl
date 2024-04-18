using EPEC
using JLD2
using Infiltrator 
using Dates


include("../racing/random_racing_helper.jl")

probs = setup(; T=10,
    Δt=0.1,
    r=1.0,
    α1=1e-3,
    α2=1e-4,
    β=.5, #.5, # sensitive to high values
    cd=0.2, #0.25,
    d=2.0, # actual road width (±)
    u_max_nominal=1.0,
    u_max_drafting=2.5, #2.5, # sensitive to high difference over nominal 
    box_length=5.0,
    box_width=5.0,
    lat_max=4.5 # just used for visulization rn 2024-04-10 (should be fixed)
);

is_x0s_from_file = false;
is_results_from_file = false;
data_dir = "data"
init_filename = "x0s_500samples_2024-01-31_1723";
sample_size = 10;
time_steps = 25;
r_offset_max = 3.0; # maximum distance between P1 and P2
a_long_vel_max = 3.0; # maximum longitudunal velocity for a
b_long_vel_delta_max = 1.5 # maximum longitudunal delta velocity for a
lat_max = probs.params.d - probs.params.col_buffer; # might be better to make smaller
r_offset_min = probs.params.r + probs.params.col_buffer;
date_now = Dates.format(now(), "YYYY-mm-dd_HHMM");

if (is_x0s_from_file)
    # WARNING params not loaded from file
    init_file = jldopen("$(data_dir)/$(init_filename).jld2", "r")
    x0s = init_file["x0s"]
    roads = init_file["roads"]
else
    (x0s, roads) = generate_x0s(sample_size, lat_max, r_offset_min, r_offset_max, a_long_vel_max, b_long_vel_delta_max)
    init_filename = "x0s_$(sample_size)samples_$(date_now)"
    jldsave("$(data_dir)/$(init_filename).jld2"; x0s, roads, lat_max, r_offset_min, r_offset_max, a_long_vel_max, b_long_vel_delta_max)
    x0s
end


```
Experiments:
						P1:						
				SP NE P1-leader P1-follower
			SP  1              
P2:			NE  2  3
	 P2-Leader  4  5  6 
   P2-Follower  7  8  9			10
```
function solve_for_x0s(x0s, roads, mode)
    results = Dict()
    start = time()
    x0s_len = length(x0s)
    progress = 0

    for (index, x0) in x0s
        try
            res = solve_simulation(probs, time_steps; x0, road=roads[index], mode);
            results[index] = res
        catch er
            @info "errored $index"
            println(er)
        end
        #global progress
        progress += 1
        @info "Progress $(progress/x0s_len*100)%"
    end
    elapsed = time() - start
    jldsave("$(data_dir)/results_mode$(mode)_($(init_filename))_$(date_now)_$(time_steps)steps.jld2"; params=probs.params, results, elapsed)
end

for mode in 1:10
    solve_for_x0s(x0s, roads, mode)
end
