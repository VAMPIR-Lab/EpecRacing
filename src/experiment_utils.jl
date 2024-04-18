function create_experiment(sample_size;
    data_dir="data",
    experiment_fname=nothing,
    saving_to_file=true,
    T=10,
    Δt=0.1,
    r=1.0,
    cost_α1=1e-3,
    cost_α2=1e-4,
    cost_β=1e-1,
    drag_coef=0.2,
    d=2.0,
    u_max_nominal=1.0,
    u_max_drafting=2.5,
    u_max_braking=u_max_drafting,
    max_heading_offset=π / 2,
    max_heading_rate=1.0,
    min_speed=-1.0,
    draft_box_length=5.0,
    draft_box_width=5.0,
    col_buffer=r / 4,
    x0_ab_dist_max=3.0,
    x0a_long_vel_max=3.0,
    x0b_long_vel_delta_max=1.5
)
    x0_lat_max = d - col_buffer
    x0_ab_dist_min = r + col_buffer

    if isnothing(experiment_fname)
        (x0s, roads) = generate_inits(
            sample_size,
            x0_lat_max,
            x0_ab_dist_min,
            x0_ab_dist_max,
            x0a_long_vel_max,
            x0b_long_vel_delta_max
        )
        if saving_to_file
            params = (; T,
                Δt,
                r,
                cost_α1,
                cost_α2,
                cost_β,
                drag_coef,
                d,
                u_max_nominal,
                u_max_drafting,
                u_max_braking,
                max_heading_offset,
                max_heading_rate,
                min_speed,
                draft_box_length,
                draft_box_width,
                col_buffer,
                x0_ab_dist_max,
                x0a_long_vel_max,
                x0b_long_vel_delta_max
            )
            date_now = Dates.format(now(), "YYYY-mm-dd_HHMM")
            experiment_fname = "exp_n$(sample_size)_$(date_now)"
            jldsave("$(data_dir)/$(experiment_fname).jld2";
                x0s,
                roads,
                params
            )
        end
    else
        experiment_file = jldopen("$(data_dir)/$(experiment_fname).jld2", "r")
        x0s = experiment_file["x0s"]
        roads = experiment_file["roads"]
        params = experiment_file["params"]
    end

    probs = setup(;
        params.T,
        params.Δt,
        params.r,
        params.cost_α1,
        params.cost_α2,
        params.cost_β,
        params.drag_coef,
        params.d,
        params.u_max_nominal,
        params.u_max_drafting,
        params.u_max_braking,
        params.max_heading_offset,
        params.max_heading_rate,
        params.min_speed,
        params.draft_box_length,
        params.draft_box_width,
        params.col_buffer
    )

    probs, x0s, roads, params, experiment_fname
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
function solve_experiment(probs, x0s, roads, time_steps, mode; saving_to_file=true, date_now=nothing, experiment_fname=nothing, data_dir="data")
    results = Dict()
    start = time()
    x0s_len = length(x0s)
    progress = 0

    for (index, x0) in x0s
        try
            res = solve_simulation(probs, time_steps; x0, road=roads[index], mode)
            results[index] = res
        catch er
            @info "errored $index"
            println(er)
        end
        #global progress
        progress += 1
        @info "Mode $(mode) Progress $(progress/x0s_len*100)%"
    end
    elapsed = time() - start

    if saving_to_file
        if isnothing(date_now)
            date_now = Dates.format(now(), "YYYY-mm-dd_HHMM")
        end

        if isnothing(experiment_fname)
            result_fname = "res_mode$(mode)_$(date_now)_$(time_steps)steps"
        else
            result_fname = "res_mode$(mode)_($(experiment_fname))_$(date_now)_$(time_steps)steps"
        end
        jldsave("$(data_dir)/$(result_fname).jld2"; params=probs.params, results, elapsed)
    end
    results, elapsed
end


#const xdim = 4
#const udim = 2
# generate x0s
# also generates roads
function generate_inits(sample_size, lat_max, r_offset_min, r_offset_max, a_long_vel_max, b_long_vel_delta_max)
    # choose random P1 lateral position inside the lane limits, long pos = 0
    #c, r = get_road(0; road);
    # solve quadratic equation to find x intercepts of the road
    #lax_max = sqrt((r - road_d)^2 - c[2]^2) + c[1]
    #lax_max = sqrt((r + road_d)^2 - c[2]^2) + c[1]
    #lat_max = min()
    roads = Vector{Dict{Float64,Float64}}(undef, sample_size)

    a_lat_pos0_arr = -lat_max .+ 2 * lat_max .* rand(MersenneTwister(), sample_size)  # .5 .* ones(sample_size)
    # fix P1 longitudinal pos at 0
    a_pos0_arr = hcat(a_lat_pos0_arr, zeros(sample_size, 1))
    b_pos0_arr = zeros(size(a_pos0_arr))
    # choose random radial offset for P2
    for i in 1:sample_size
        # shift initial position wrt to road
        roads[i] = gen_road(seed=nothing)

        road_ys = roads[i] |> keys |> collect
        sortedkeys = sortperm((road_ys .- 0) .^ 2)
        road_offset = roads[i][road_ys[sortedkeys[1]]]

        a_pos0_arr[i, 1] += road_offset

        r_offset = r_offset_min .+ (r_offset_max - r_offset_min) .* sqrt.(rand(MersenneTwister()))
        ϕ_offset = rand(MersenneTwister()) * 2 * π
        b_lat_pos0 = a_pos0_arr[i, 1] + r_offset * cos(ϕ_offset)
        # reroll until we b lat pos is inside the lane limits
        while b_lat_pos0 > lat_max + road_offset || b_lat_pos0 < -lat_max + road_offset
            r_offset = r_offset_min .+ (r_offset_max - r_offset_min) .* sqrt.(rand(MersenneTwister()))
            ϕ_offset = rand(MersenneTwister()) * 2 * π
            b_lat_pos0 = a_pos0_arr[i, 1] + r_offset * cos(ϕ_offset)
        end
        b_long_pos0 = a_pos0_arr[i, 2] + r_offset * sin(ϕ_offset)
        b_pos0_arr[i, :] = [b_lat_pos0, b_long_pos0]
    end

    @assert minimum(sqrt.(sum((a_pos0_arr .- b_pos0_arr) .^ 2, dims=2))) >= 1.0 # probs.params.r
    #@assert all(-lat_max .< b_pos0_arr[:, 1] .< lat_max)


    # keep lateral velocity zero

    a_long_vel_min = b_long_vel_delta_max
    a_vel0_arr = hcat(zeros(sample_size), a_long_vel_min .+ (a_long_vel_max - a_long_vel_min) .* rand(MersenneTwister(), sample_size))
    #a_vel0_arr = hcat(zeros(sample_size), a_long_vel_max .* rand(MersenneTwister(), sample_size))

    b_vel0_arr = zeros(size(a_vel0_arr))
    # choose random velocity offset for 
    for i in 1:sample_size
        b_long_vel0_offset = -b_long_vel_delta_max + 2 * b_long_vel_delta_max .* rand(MersenneTwister())
        b_long_vel0 = a_vel0_arr[i, 2] + b_long_vel0_offset
        ## reroll until b long vel is nonnegative
        #while b_long_vel0 < 0
        #    b_long_vel0_offset = -b_long_vel_delta_max + 2 * b_long_vel_delta_max .* rand(MersenneTwister())
        #    b_long_vel0 = a_vel0_arr[i, 2] + b_long_vel0_offset
        #end
        b_vel0_arr[i, 2] = b_long_vel0
    end

    #@infiltrate
    #x0_arr = hcat(a_pos0_arr, a_vel0_arr, b_pos0_arr, b_vel0_arr)
    # really simple since lateral vel is zero
    x0_arr = hcat(a_pos0_arr, a_vel0_arr[:, 2], ones(length(a_vel0_arr[:, 1])) .* π / 2, b_pos0_arr, b_vel0_arr[:, 2], ones(length(b_vel0_arr[:, 1])) .* π / 2)
    #@infiltrate
    x0s = Dict()

    for (index, row) in enumerate(eachrow(x0_arr))
        x0s[index] = row
    end
    (x0s, roads)
end

function read_from_file(modes, experiment_fname, suffix; data_dir="data")
    results_suffix = "_($experiment_fname)_$(suffix)"

    experiment_file = EpecRacing.jldopen("$(data_dir)/$(experiment_fname).jld2", "r")
    x0s = experiment_file["x0s"]
    roads = experiment_file["roads"]
    params = experiment_file["params"]

    results = Dict()

    for i in modes
        file = EpecRacing.jldopen("$(data_dir)/res_mode$(i)$(results_suffix).jld2", "r")
        results[i] = file["results"]
    end

    (results, x0s, roads, params)
end

function process_results(results, params; is_trimming=false, trim_steps=100)
    costs = Dict()
    steps = Dict()

    for (index, res) in results
        len = length(res)
        steps[index] = len

        if is_trimming
            if len >= trim_steps
                #costs[index] = compute_realized_cost(Dict(i => res[i] for i in 1:trim_steps))
                costs[index] = compute_realized_cost(res, params)
                #steps[index] = len
            end
        else
            costs[index] = compute_realized_cost(res, params)
        end
    end
    (; costs, steps)
end

# each player wants to make forward progress and stay in center of lane
# e = ego
# o = opponent
function f_ego_breakdown(T, X, U, X_opp, c, r; α1, α2, β)
    xdim = 4
    udim = 2
    cost = 0.0

    lane_cost_arr = zeros(T)
    control_cost_arr = zeros(T)
    velocity_cost_arr = zeros(T)

    for t in 1:T
        @inbounds x = @view(X[xdim*(t-1)+1:xdim*t])
        @inbounds x_opp = @view(X_opp[xdim*(t-1)+1:xdim*t])
        @inbounds u = @view(U[udim*(t-1)+1:udim*t])
        long_vel = x[3] * sin(x[4])
        long_vel_opp = x_opp[3] * sin(x_opp[4])
        p = x[1:2]

        lane_cost_arr[t] = α1^2 * ((p - c)' * (p - c) - r[1]^2)^2
        control_cost_arr[t] = α2 * u' * u
        velocity_cost_arr[t] = β * (long_vel_opp - 2 * long_vel)
    end

    lane_cost = sum(lane_cost_arr)
    control_cost = sum(control_cost_arr)
    velocity_cost = sum(velocity_cost_arr)
    total_cost = lane_cost + control_cost + velocity_cost
    total_cost_arr = lane_cost_arr .+ control_cost_arr .+ velocity_cost_arr
    final = (; total=total_cost, lane=lane_cost, control=control_cost, velocity=velocity_cost)
    running = (; total=total_cost_arr, lane=lane_cost_arr, control=control_cost_arr, velocity=velocity_cost_arr)
    (; final, running)
end

function f1_breakdown(z; α1=1.0, α2=0.0, β=1.0)
    T, Xa, Ua, Xb, Ub, x0a, x0b, ca, cb, ra, rb = view_z(z)

    f_ego_breakdown(T, Xa, Ua, Xb, ca, ra; α1, α2, β)
end

function f2_breakdown(z; α1=1.0, α2=0.0, β=1.0)
    T, Xa, Ua, Xb, Ub, x0a, x0b, ca, cb, ra, rb = view_z(z)

    f_ego_breakdown(T, Xb, Ub, Xa, cb, rb; α1, α2, β)
end

function compute_realized_cost(res, params)
    xdim = 4
    udim = 2
    pdim = 14
    T = length(res)
    Xa = zeros(xdim * T)
    Ua = zeros(udim * T)
    Xb = zeros(xdim * T)
    Ub = zeros(udim * T)

    for t in eachindex(res)
        Xa[xdim*(t-1)+1:xdim*t] = res[t].x0[1:4]
        Xb[xdim*(t-1)+1:xdim*t] = res[t].x0[5:8]

        #fieldnames(typeof(res))
        if hasproperty(res[t], :U1) # we need to check this because I forgot to add U1 U2 to failed timesteps 2024-04-16
            Ua[udim*(t-1)+1:udim*t] = res[t].U1[1, :]
            Ub[udim*(t-1)+1:udim*t] = res[t].U2[1, :]
        end
    end
    z = [Xa; Ua; Xb; Ub; zeros(pdim)] # making it work with f1(Z) and f2(Z)


    a_breakdown = f1_breakdown(z; α1=params.cost_α1, α2=params.cost_α2, β=params.cost_β)
    b_breakdown = f2_breakdown(z; α1=params.cost_α1, α2=params.cost_α2, β=params.cost_β)

    if isdefined(Base, :probs)
        a_cost = probs.OP1.f(z)
        b_cost = probs.OP2.f(z)

        # breakdowns were copy pasted so
        @assert(isapprox(a_cost, a_breakdown.final.total))
        @assert(isapprox(b_cost, b_breakdown.final.total))
    end

    (; a=a_breakdown, b=b_breakdown)
end

function gen_steps_table(processed_results, modes_sorted)
    steps_table_old = Dict()

    for mode in modes_sorted
        res = processed_results[mode]
        inds = res.costs |> keys |> collect |> sort
        a_steps = [res.steps[i] for i in inds]
        b_steps = [res.steps[i] for i in inds]
        steps_table_old[mode, "a"] = a_steps
        steps_table_old[mode, "b"] = b_steps
    end

    full_steps_table = Dict()
    ```
    Experiments:
                            P1:						
                    SP NE P1-leader P1-follower
                SP  1              
    P2:			NE  2  3
         P2-Leader  4  5  6 
       P2-Follower  7  8  9			10
    ```
    if haskey(steps_table_old, (1, "a"))
        full_steps_table["S", "S"] = steps_table_old[1, "a"]
    end
    if haskey(steps_table_old, (2, "a"))
        full_steps_table["S", "N"] = steps_table_old[2, "a"]
    end
    if haskey(steps_table_old, (4, "a"))
        full_steps_table["S", "L"] = steps_table_old[4, "a"]
    end
    if haskey(steps_table_old, (7, "a"))
        full_steps_table["S", "F"] = steps_table_old[7, "a"]
    end
    if haskey(steps_table_old, (2, "b"))
        full_steps_table["N", "S"] = steps_table_old[2, "b"]
    end
    if haskey(steps_table_old, (3, "a"))
        full_steps_table["N", "N"] = steps_table_old[3, "a"]
    end
    if haskey(steps_table_old, (5, "a"))
        full_steps_table["N", "L"] = steps_table_old[5, "a"]
    end
    if haskey(steps_table_old, (8, "a"))
        full_steps_table["N", "F"] = steps_table_old[8, "a"]
    end
    if haskey(steps_table_old, (4, "b"))
        full_steps_table["L", "S"] = steps_table_old[4, "b"]
    end
    if haskey(steps_table_old, (5, "b"))
        full_steps_table["L", "N"] = steps_table_old[5, "b"]
    end
    if haskey(steps_table_old, (6, "a"))
        full_steps_table["L", "L"] = steps_table_old[6, "a"]
    end
    if haskey(steps_table_old, (9, "a"))
        full_steps_table["L", "F"] = steps_table_old[9, "a"]
    end
    if haskey(steps_table_old, (7, "b"))
        full_steps_table["F", "S"] = steps_table_old[7, "b"]
    end
    if haskey(steps_table_old, (8, "b"))
        full_steps_table["F", "N"] = steps_table_old[8, "b"]
    end
    if haskey(steps_table_old, (9, "b"))
        full_steps_table["F", "L"] = steps_table_old[9, "b"]
    end
    if haskey(steps_table_old, (10, "a"))
        full_steps_table["F", "F"] = steps_table_old[10, "a"]
    end

    full_steps_table
end

function gen_costs_table(processed_results, modes_sorted; property=:total)
    cost_table_old = Dict()

    for mode in modes_sorted
        res = processed_results[mode]
        inds = res.costs |> keys |> collect |> sort
        a_steps = [res.steps[i] for i in inds]
        b_steps = [res.steps[i] for i in inds]
        a_costs = [getindex(res.costs[i].a.final, property) for i in inds]
        b_costs = [getindex(res.costs[i].b.final, property) for i in inds]
        if a_steps == 0 || b_steps == 0
            @infiltrate
        end
        cost_table_old[mode, "a"] = a_costs ./ a_steps
        cost_table_old[mode, "b"] = b_costs ./ b_steps
    end

    full_cost_table = Dict()
    ```
    Experiments:
                            P1:						
                    SP NE P1-leader P1-follower
                SP  1              
    P2:			NE  2  3
         P2-Leader  4  5  6 
       P2-Follower  7  8  9			10
    ```
    if haskey(cost_table_old, (1, "a"))
        full_cost_table["S", "S"] = cost_table_old[1, "a"]
    end
    if haskey(cost_table_old, (2, "a"))
        full_cost_table["S", "N"] = cost_table_old[2, "a"]
    end
    if haskey(cost_table_old, (4, "a"))
        full_cost_table["S", "L"] = cost_table_old[4, "a"]
    end
    if haskey(cost_table_old, (7, "a"))
        full_cost_table["S", "F"] = cost_table_old[7, "a"]
    end
    if haskey(cost_table_old, (2, "b"))
        full_cost_table["N", "S"] = cost_table_old[2, "b"]
    end
    if haskey(cost_table_old, (3, "a"))
        full_cost_table["N", "N"] = cost_table_old[3, "a"]
    end
    if haskey(cost_table_old, (5, "a"))
        full_cost_table["N", "L"] = cost_table_old[5, "a"]
    end
    if haskey(cost_table_old, (8, "a"))
        full_cost_table["N", "F"] = cost_table_old[8, "a"]
    end
    if haskey(cost_table_old, (4, "b"))
        full_cost_table["L", "S"] = cost_table_old[4, "b"]
    end
    if haskey(cost_table_old, (5, "b"))
        full_cost_table["L", "N"] = cost_table_old[5, "b"]
    end
    if haskey(cost_table_old, (6, "a"))
        full_cost_table["L", "L"] = cost_table_old[6, "a"]
    end
    if haskey(cost_table_old, (9, "a"))
        full_cost_table["L", "F"] = cost_table_old[9, "a"]
    end
    if haskey(cost_table_old, (7, "b"))
        full_cost_table["F", "S"] = cost_table_old[7, "b"]
    end
    if haskey(cost_table_old, (8, "b"))
        full_cost_table["F", "N"] = cost_table_old[8, "b"]
    end
    if haskey(cost_table_old, (9, "b"))
        full_cost_table["F", "L"] = cost_table_old[9, "b"]
    end
    if haskey(cost_table_old, (10, "a"))
        full_cost_table["F", "F"] = cost_table_old[10, "a"]
    end

    full_cost_table
end

function gen_all_tables(processed_results)
    modes_sorted = sort(collect(keys(processed_results)))
    steps_table = gen_steps_table(processed_results, modes_sorted)
    total_cost_table = gen_costs_table(processed_results, modes_sorted, property=:total)
    lane_cost_table = gen_costs_table(processed_results, modes_sorted, property=:lane)
    control_cost_table = gen_costs_table(processed_results, modes_sorted, property=:control)
    velocity_cost_table = gen_costs_table(processed_results, modes_sorted, property=:velocity)

    (; modes_sorted, steps_table, total_cost_table, lane_cost_table, control_cost_table, velocity_cost_table)
end

function print_mean_etc(vals; title="", scale=1.0, sigdigits=3)
    vals = vals .* scale
    CI = 1.96 * std(vals) / sqrt(length(vals))
    m = mean(vals)
    m95l = m - CI
    m95u = m + CI
    s = std(vals)

    println("$(title)	$(round(m; sigdigits)) (±$(round(CI; sigdigits))) [$(round(m95l; sigdigits)), $(round(m95u; sigdigits))]	$(round(s; sigdigits))	$(round(minimum(vals); sigdigits))	$(round(maximum(vals); sigdigits))")
end

function get_mean_running_vel_cost(processed_results, i; time_steps)
    vals = [Float64[] for _ in 1:time_steps]
    for (index, c) in processed_results[i].costs
        T = length(c.a.running.velocity)
        for t in 1:T
            push!(vals[t], c.a.running.velocity[t])
        end
    end
    avgs = map(vals) do val
        mean(val)
    end
    stderrs = map(vals) do val
        1.96 * std(val) / sqrt(length(val))
    end
    (avgs, stderrs)
end