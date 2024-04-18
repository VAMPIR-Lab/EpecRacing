function process(results; is_trimming=false, trim_steps=100)
    costs = Dict()
    steps = Dict()

    for (index, res) in results
        len = length(res)
        steps[index] = len

        if is_trimming
            if len >= trim_steps
                #costs[index] = compute_realized_cost(Dict(i => res[i] for i in 1:trim_steps))
                costs[index] = compute_realized_cost(res)
                #steps[index] = len
            end
        else
            costs[index] = compute_realized_cost(res)
        end
    end
    (; costs, steps)
end

# detailed cost
function view_z(z)
    n_param = 14
    xdim = 4
    udim = 2
    T = Int((length(z) - n_param) / (2 * (xdim + udim))) # 2 real players, 4 players total
    indices = Dict()
    idx = 0
    for (len, name) in zip([xdim * T, udim * T, xdim * T, udim * T, xdim, xdim, 2, 2, 1, 1], ["Xa", "Ua", "Xb", "Ub", "x0a", "x0b", "ca", "cb", "ra", "rb"])
        indices[name] = (idx+1):(idx+len)
        idx += len
    end
    @inbounds Xa = @view(z[indices["Xa"]])
    @inbounds Ua = @view(z[indices["Ua"]])
    @inbounds Xb = @view(z[indices["Xb"]])
    @inbounds Ub = @view(z[indices["Ub"]])
    @inbounds x0a = @view(z[indices["x0a"]])
    @inbounds x0b = @view(z[indices["x0b"]])
    @inbounds ca = @view(z[indices["ca"]])
    @inbounds cb = @view(z[indices["cb"]])
    @inbounds ra = @view(z[indices["ra"]])
    @inbounds rb = @view(z[indices["rb"]])
    (T, Xa, Ua, Xb, Ub, x0a, x0b, ca, cb, ra, rb, indices)
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

function f1_breakdown(z; α1=1.0, α2=0.0, α3=0.0, β=1.0)
    T, Xa, Ua, Xb, Ub, x0a, x0b, ca, cb, ra, rb = view_z(z)


    f_ego_breakdown(T, Xa, Ua, Xb, ca, ra; α1, α2, β)
end

function f2_breakdown(z; α1=1.0, α2=0.0, α3=0.0, β=1.0)
    T, Xa, Ua, Xb, Ub, x0a, x0b, ca, cb, ra, rb = view_z(z)

    f_ego_breakdown(T, Xb, Ub, Xa, cb, rb; α1, α2, β)
end

function compute_realized_cost(res)
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

    a_cost = probs.OP1.f(z)
    b_cost = probs.OP2.f(z)
    a_breakdown = f1_breakdown(z; probs.params.α1, probs.params.α2, probs.params.α3, probs.params.β)
    b_breakdown = f2_breakdown(z; probs.params.α1, probs.params.α2, probs.params.α3, probs.params.β)

    # breakdowns were copy pasted so
    @assert(isapprox(a_cost, a_breakdown.final.total))
    @assert(isapprox(b_cost, b_breakdown.final.total))
    (; a=a_breakdown, b=b_breakdown)
end