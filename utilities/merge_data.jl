using EpecRacing

modes = 1:10
time_steps = 25;
exp1_fname = "exp2024-04-19_0928_merged_n200"
exp2_fname = "exp2024-04-19_0838_n100"
results1, x0s1, roads1, params1 = read_from_file(modes, exp1_fname, "2024-04-19_0928", time_steps)
results2, x0s2, roads2, params2 = read_from_file(modes, exp2_fname, "2024-04-19_0838", time_steps)

data_new_dir = "data/merged"

date_now = EpecRacing.Dates.format(EpecRacing.Dates.now(), "YYYY-mm-dd_HHMM")
@assert params1 == params2

# merge x0s
x0s_merged = copy(x0s1)
shift = length(x0s_merged)

for (key, val) in x0s2
    x0s_merged[key+shift] = val
end

# merge roads
roads_merged = [copy(roads1); copy(roads1)]

for i in eachindex(roads2)
    roads_merged[i+shift] = roads2[i]
end

# save merged experiment file
EpecRacing.jldsave("$(data_new_dir)/exp$(date_now)_merged_n300.jld2";
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

    EpecRacing.jldsave("$(data_new_dir)/exp$(date_now)_merged_n300_run$(date_now)_mode$(i)_steps$(time_steps).jld2"; results=res_merged[i])
end
