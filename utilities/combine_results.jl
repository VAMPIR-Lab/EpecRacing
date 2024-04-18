using JLD2

data_dir = "data/x0s_500samples_2024-01-31"
data_new_dir = "data/x0s_500samples_2024-01-31"
x0s1_filename = "x0s_1000samples_2024-01-31_1720"
x0s2_filename = "x0s_500samples_2024-01-31_1753"
results1_suffix = "_(x0s_1000samples_2024-01-31_1720)_2024-01-31_1720_100steps";
results2_suffix = "_(x0s_500samples_2024-01-31_1753)_2024-01-31_1753_100steps";
init1_file = jldopen("$(data_dir)/$(x0s1_filename).jld2", "r")
init2_file = jldopen("$(data_dir)/$(x0s2_filename).jld2", "r")
x0s1 = init1_file["x0s"]
x0s2 = init2_file["x0s"]

modes = 1:10
params = Dict()
results1 = Dict()
results2 = Dict()

for i in modes
    file = jldopen("$(data_dir)/results_mode$(i)$(results1_suffix).jld2", "r")
    results1[i] = file["results"]
	params[i] = file["params"]
end

for i in modes
    file = jldopen("$(data_dir)/results_mode$(i)$(results2_suffix).jld2", "r")
    results2[i] = file["results"]
end

x0s_merged = Dict()
results_merged = Dict()

x0s_merged = copy(x0s1)
len = length(x0s_merged)

for (key, value) in x0s2
	x0s_merged[key + len] = value
end

jldsave("$(data_new_dir)/x0s_1500samples_2024-01-31_1720.jld2"; x0s=x0s_merged, lat_max=init1_file["lat_max"], r_offset_min=init1_file["r_offset_min"], r_offset_max=init1_file["r_offset_max"], a_long_vel_max=init1_file["a_long_vel_max"], b_long_vel_delta_max=init1_file["b_long_vel_delta_max"])

for i in modes
	results_merged[i] = copy(results1[i])
	len = length(results_merged[i])

	for (key, value) in results2[i]
		results_merged[i][key + len] = value
	end

	jldsave("$(data_new_dir)/results_mode$(i)_(x0s_1500samples_2024-01-31_1720)_2024-01-31_1720_100steps.jld2"; params=params[i], results=results_merged[i])
end
