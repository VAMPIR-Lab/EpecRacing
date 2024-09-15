using EpecRacing

modes = 1:10
time_steps = 25;
experiment_fname = "exp2024-04-20_0154_merged_n1000";
run_date = "2024-04-20_0154";
results, x0s, roads, params = read_from_file(modes, experiment_fname, run_date, time_steps; data_dir="data/merged")
sample_size = length(x0s)

processed_results = Dict()
for (index, res) in results
    processed_results[index] = process_results(res, params)
end


(; steps_mean, steps_CI, a_costs_mean, a_costs_CI, b_costs_mean, b_costs_CI, steps_named, total_named, lane_named, control_named, velocity_named) = gen_all_tables(processed_results);

# normalized
@info "Mean steps table:"
display(steps_mean)
@info "Mean a total costs table:"
display(a_costs_mean.total.*100)
@info "Mean b total costs table:"
display(b_costs_mean.total.*100)

@info "CI Mean steps table:"
display(steps_CI)
@info "CI Mean a total costs table:"
display(a_costs_CI.total.*100)
@info "CI Mean b total costs table:"
display(b_costs_CI.total.*100)


steps_mean_rel = (steps_mean .- steps_mean[1,1])./ steps_mean[1,1] .* 100
a_costs_mean_rel =  (a_costs_mean.total .- a_costs_mean.total[1,1]) ./ a_costs_mean.total[1,1] .* 100
b_costs_mean_rel = (b_costs_mean.total .- a_costs_mean.total[1,1]) ./ a_costs_mean.total[1,1] .* 100

#@info "CI steps table (normalized):"
#display(steps_CI_rel)
#@info "CI a total costs table (normalized):"
#display(a_costs_CI_rel)
#@info "CI b total costs table (normalized):"
#display(b_costs_CI_rel)

#@info "Mean steps table (normalized):"
#display(steps_mean_rel)
#@info "Mean a total costs table (normalized):"
#display(a_costs_mean_rel)
#@info "Mean b total costs table (normalized):"
#display(b_costs_mean_rel)


#steps_mean_rel_compr = zeros(4)
#a_costs_mean_rel_compr = zeros(4)
#b_costs_mean_rel_compr = zeros(4)
#steps_CI_rel_compr = zeros(4)
#a_costs_CI_rel_compr = zeros(4)
#b_costs_CI_rel_compr = zeros(4)

#for i in 1:4
#    steps_mean_rel_compr[i] = sum(steps_mean_rel[:, i] /4)
#    a_costs_mean_rel_compr[i] = sum(a_costs_mean_rel[:, i] /4)
#    b_costs_mean_rel_compr[i] = sum(b_costs_mean_rel[i, :] /4)
#    steps_CI_rel_compr[i] = sum(steps_CI_rel[:, i] /4)
#    a_costs_CI_rel_compr[i] = sum(a_costs_CI_rel[:, i] /4)
#    b_costs_CI_rel_compr[i] = sum(b_costs_CI_rel[i, :] /4)
#end

#@info "Mean steps table (compressed):"
#display(steps_mean_rel_compr)
#@info "Mean a total costs table (compressed):"
#display(a_costs_mean_rel_compr)
#@info "Mean b total costs table (compressed):"
#display(b_costs_mean_rel_compr)

#@info "CI steps table (compressed):"
#display(steps_CI_rel_compr)
#@info "CI a total costs table (compressed):"
#display(a_costs_CI_rel_compr)
#@info "CI b total costs table (compressed):"
#display(b_costs_CI_rel_compr)



println("		mean (±95% CI) [95% CI l, u]	std	min	max")

# needs all modes 1 to 10
include("print_compressed_tables.jl") 

#println("Steps:")
#for (k, v) in steps_named
#    print_mean_etc(v; title=k, scale=1)
#end

#println("Total:")
#for (k, v) in total_named
#    print_mean_etc(v; title=k, scale=100)
#end

# draw all:
# 41
#samples = rand(1:100, 4)
##samples = [51]
#for s in samples
#    EpecRacing.randomly_animate(3, results, roads, params, sample_size; sample=s, t=1)
#    EpecRacing.randomly_animate(6, results, roads, params, sample_size; sample=s, t=1)
#    EpecRacing.randomly_animate(9, results, roads, params, sample_size; sample=s, t=1)
#end

#EpecRacing.randomly_animate(3, results, roads, params, sample_size)