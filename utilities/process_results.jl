using EpecRacing

modes = [3, 9]
time_steps = 25;
results, x0s, roads, params = read_from_file(modes, "exp_n2_2024-04-18_0025", "2024-04-18_0028_$(time_steps)steps")

processed_results = Dict()
for (index, res) in results
    processed_results[index] = process_results(res, params)
end

(; modes_sorted, steps_table, total_cost_table, lane_cost_table, control_cost_table, velocity_cost_table) = gen_all_tables(processed_results);


println("		mean (±95% CI) [95% CI l, u]	std	min	max")

# needs all modes 1 to 10
#print("print_compressed_tables.jl") 

println("Total:")
for (k, v) in total_cost_table
    print_mean_etc(v; title=k, scale=10)
end

println("Lane:")
for (k, v) in lane_cost_table
    print_mean_etc(v; title=k, scale=10)
end

println("Control:")
for (k, v) in control_cost_table
    print_mean_etc(v; title=k, scale=10)
end

println("Velocity:")
for (k, v) in velocity_cost_table
    print_mean_etc(v; title=k, scale=10)
end

# rss 2024 plots
#include("gen_boxplot.jl")
include("gen_running_cost_plot.jl")

# to visualize:
mode = 3;
sample = 2;
road = roads[sample];
#EpecRacing.animate(params, results[mode][sample]; save=false, mode, road);
