using EpecRacing

sample_size = 2;
time_steps = 5;
```
Experiments:
						P1:						
				SP NE P1-leader P1-follower
			SP  1              
P2:			NE  2  3
	 P2-Leader  4  5  6 
   P2-Follower  7  8  9			10
```
experiment_modes = [3, 9]

probs, x0s, roads, params, experiment_fname = create_experiment(sample_size)

date_now = EpecRacing.Dates.format(EpecRacing.Dates.now(), "YYYY-mm-dd_HHMM")
results = Dict()
elapsed = Dict()
for mode in experiment_modes
    @info "Mode $mode:"
    results[mode], elapsed[mode] = solve_experiment(probs, x0s, roads, time_steps, mode; experiment_fname, date_now)
end

# to visualize:
mode = experiment_modes[1];
sample = EpecRacing.rand(1:sample_size);
@info "Sample $sample"
road = roads[sample];
animate(params, results[mode][sample]; save=false, mode, road);