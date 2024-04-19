using EpecRacing

sample_size = 100;
time_steps = 25;
experiment_modes = 4:10
probs, x0s, roads, params, experiment_fname = create_experiment(sample_size;
    experiment_fname="exp2024-04-19_0055_n100",
    T=10,
    Δt=0.1,
    r=1.0,
    cost_α1=1e-3,
    cost_α2=1e-4,
    cost_β=1e-1,
    drag_coef=0.1,
    d=2.0,
    u_max_nominal=1.0,
    u_max_drafting=3.0,
    max_heading_offset=π / 2,
    max_heading_rate=1.0,
    min_speed=-1.0,
    draft_box_length=5.0,
    draft_box_width=5.0,
    x0_ab_dist_max=3.0,
    x0a_long_vel_max=3.0,
    x0b_long_vel_delta_max=1.5
)

date_now = EpecRacing.Dates.format(EpecRacing.Dates.now(), "YYYY-mm-dd_HHMM")
results = Dict()
elapsed = Dict()

for mode in experiment_modes
    @info "$(EpecRacing.Dates.format(EpecRacing.Dates.now(), "YYYY-mm-dd HH:MM")) Mode $mode:"
    results[mode], elapsed[mode] = solve_experiment(probs, x0s, roads, time_steps, mode; experiment_fname, date_now)
    @info "Elapsed $(elapsed[mode])"
end
