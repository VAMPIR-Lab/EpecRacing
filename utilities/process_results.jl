using EpecRacing

modes = 1:10
time_steps = 25;
experiment_fname = "exp2024-04-18_1820_n100";
run_date = "2024-04-18_1821";
results, x0s, roads, params = read_from_file(modes, experiment_fname, run_date, time_steps; data_dir="data")
sample_size = length(x0s)

processed_results = Dict()
for (index, res) in results
    processed_results[index] = process_results(res, params)
end

(; modes_sorted, steps_table, total_cost_table, lane_cost_table, control_cost_table, velocity_cost_table) = gen_all_tables(processed_results);


println("		mean (±95% CI) [95% CI l, u]	std	min	max")

# needs all modes 1 to 10
#print("print_compressed_tables.jl") 

println("Steps:")
for (k, v) in steps_table
    print_mean_etc(v; title=k, scale=1)
end

println("Total:")
for (k, v) in total_cost_table
    print_mean_etc(v; title=k, scale=100)
end

#println("Lane:")
#for (k, v) in lane_cost_table
#    print_mean_etc(v; title=k, scale=100)
#end

#println("Control:")
#for (k, v) in control_cost_table
#    print_mean_etc(v; title=k, scale=100)
#end

println("Velocity:")
for (k, v) in velocity_cost_table
    print_mean_etc(v; title=k, scale=100)
end

# rss 2024 plots
#include("gen_boxplot.jl")
#include("gen_running_cost_plot.jl")

# draw all:
# 41
samples = rand(1:100, 4)
#samples = [51]
for s in samples
    EpecRacing.randomly_animate(3, results, roads, params, sample_size; sample=s, t=1)
    EpecRacing.randomly_animate(6, results, roads, params, sample_size; sample=s, t=1)
    EpecRacing.randomly_animate(9, results, roads, params, sample_size; sample=s, t=1)
end