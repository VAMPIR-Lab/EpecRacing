data_dir = "data"
data_new_dir = "data/merged"
exp1_fname = "exp_n50_2024-04-18_0823"
exp2_fname = "exp_n50_2024-04-18_0848"
res1_suffix = "_(exp_n50_2024-04-18_0823)_2024-04-18_0824_25steps";
res2_suffix = "_(exp_n50_2024-04-18_0823)_2024-04-18_0824_25steps";
date_now = EpecRacing.Dates.format(EpecRacing.Dates.now(), "YYYY-mm-dd_HHMM")
modes = [3, 9]

(results1, x0s1, roads1, params1) = read_from_file(modes, exp1_fname, res1_suffix; data_dir)
(results2, x0s2, roads2, params2) = read_from_file(modes, exp1_fname, res1_suffix; data_dir)

@assert params1 === params2

# merge x0s
x0s_merged = copy(x0s1)
len = length(x0s_merged)

for (key, val) in x0s2
    x0s_merged[key+len] = val
    x0s_merged[key+len] = val
end

# merge roads
roads_merged = copy(roads1)

for (key, val) in roads2
    roads_merged[key+len] = val
    roads_merged[key+len] = val
end

# save merged experiment file
jldsave("$(data_new_dir)/exp_n100samples_merged_$(date_now).jld2";
    x0s=x0s_merged,
    roads=roads_merged,
    params=params1
)

# merge results
res_merged = Dict()

for i in modes
    res_merged[i] = copy(results1[i])
    len = length(res_merged[i])

    for (key, value) in results2[i]
        res_merged[i][key+len] = value
    end

    jldsave("$(data_new_dir)/res_mode$(i)_(x0s_n100_merged_$(date_now))_25steps.jld2"; results=res_merged[i])
end
