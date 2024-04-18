# Z := [xᵃ₁ ... xᵃₜ | uᵃ₁ ... uᵃₜ | xᵇ₁ ... xᵇₜ | uᵇ₁ ... uᵇₜ]
# xⁱₖ := [p1 p2 v1 v2]
# xₖ = dyn(xₖ₋₁, uₖ; Δt) (pointmass dynamics)
const xdim = 4
const udim = 2

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
function f_ego(T, X, U, X_opp, c, r; α1, α2, β)
    xdim = 4
    udim = 2
    cost = 0.0

    for t in 1:T
        @inbounds x = @view(X[xdim*(t-1)+1:xdim*t])
        @inbounds x_opp = @view(X_opp[xdim*(t-1)+1:xdim*t])
        @inbounds u = @view(U[udim*(t-1)+1:udim*t])
        long_vel = x[3] * sin(x[4])
        long_vel_opp = x_opp[3] * sin(x_opp[4])
        p = x[1:2]
        #cost += α1 * x[1]^2 + α2 * u' * u + β * (long_vel_opp - 2 * long_vel)
        #road_center = (p - c) ./ ((p - c)' * (p - c)) * r[1]
        #cost += α1 * (p[1] - road_center[1])^2 + α2 * u' * u + β * (long_vel_opp - 2 * long_vel)
        cost += α1^2 * ((p - c)' * (p - c) - r[1]^2)^2 + α2 * u' * u + β * (long_vel_opp - long_vel)
    end
    cost
end

# P1 wants to make forward progress and stay in center of lane.
function f1(z; α1, α2, β)
    T, Xa, Ua, Xb, Ub, x0a, x0b, ca, cb, ra, rb = view_z(z)

    f_ego(T, Xa, Ua, Xb, ca, ra; α1, α2, β)
end

# P2 wants to make forward progress and stay in center of lane.
function f2(z; α1, α2, β)
    T, Xa, Ua, Xb, Ub, x0a, x0b, ca, cb, ra, rb = view_z(z)

    f_ego(T, Xb, Ub, Xa, cb, rb; α1, α2, β)
end

function pointmass(x, u, Δt, cd)
    Δt2 = 0.5 * Δt * Δt
    a1 = u[1] - cd * x[3]
    a2 = u[2] - cd * x[4]
    [x[1] + Δt * x[3] + Δt2 * a1,
        x[2] + Δt * x[4] + Δt2 * a2,
        x[3] + Δt * a1,
        x[4] + Δt * a2]
end

function simplecraft(x, u, Δt, cd; k1=1.0, k2=1.0)
    # x = [p_lat, p_long, v, θ]
    a = k1 * u[1] - cd * x[3] # tangential acceleration
    ω = k2 * u[2] # heading rate
    v_next = x[3] + Δt * a # next velocity
    θ_next = x[4] + Δt * ω # next heading

    # backwards euler
    [x[1] + Δt * v_next * cos(θ_next),
        x[2] + Δt * v_next * sin(θ_next),
        v_next,
        θ_next]
end

function dyn(X, U, x0, Δt, cd)
    xdim = 4
    udim = 2
    T = Int(length(X) / xdim)
    x = x0
    mapreduce(vcat, 1:T) do t
        xx = X[(t-1)*xdim+1:t*xdim]
        u = U[(t-1)*udim+1:t*udim]
        diff = xx - simplecraft(x, u, Δt, cd)
        x = xx
        diff
    end
end

function col(Xa, Xb, r; col_buffer=0.0)
    xdim = 4
    T = Int(length(Xa) / xdim)
    mapreduce(vcat, 1:T) do t
        @inbounds xa = @view(Xa[(t-1)*xdim+1:t*xdim])
        @inbounds xb = @view(Xb[(t-1)*xdim+1:t*xdim])
        delta = xa[1:2] - xb[1:2]
        [delta' * delta - (r + col_buffer)^2,]
    end
end

function responsibility(Xa, Xb)
    xdim = 4
    T = Int(length(Xa) / xdim)
    mapreduce(vcat, 1:T) do t
        @inbounds xa = @view(Xa[(t-1)*xdim+1:t*xdim])
        @inbounds xb = @view(Xb[(t-1)*xdim+1:t*xdim])

        h = [xb[2] - xa[2],] # h is positive when xa is behind xb in second coordinate
    end
end

function accel_bounds(X, X_opp, u_max_nominal, u_max_drafting, box_length, box_width, c_opp, x0_opp)
    xdim = 4
    T = Int(length(X) / xdim)
    d = mapreduce(vcat, 1:T) do t
        @inbounds x = @view(X[(t-1)*xdim+1:t*xdim])
        @inbounds x_opp = @view(X_opp[(t-1)*xdim+1:t*xdim])

        #θ = atan(x_opp[2] - c_road[2], x_opp[1] - c_road[1]) + π / 2 # road curvature & opponent -- does not work
        #θ = atan(x[2] - c_road[2], x[1] - c_road[1]) + π / 2 # road curvature & ego -- does not work
        #θ = x_opp[4] # opponent -- this breaks bilevel 2024-04-10
        #θ = x[4] # ego -- does not work
        #θ = atan(x0_opp[2] - c_opp[2], x0_opp[1] - c_opp[1]) + π / 2 # road curvature & opponent x0 
        #θ = x0_opp[4]
        #θ = π / 2 # constant (works)
        #θ = π / 2 + .5 # constant (does not work?)
        # basis change using passive transformation matrix, unity if θ=π/2
        #R = [cos(θ - π / 2) sin(θ - π / 2)
        #    -sin(θ - π / 2) cos(θ - π / 2)]
        #dd = R * (x[1:2] - x_opp[1:2])
        #[dd[1] dd[2]]
        #[R[1,:]'*(x[1:2] - x_opp[1:2]) R[2,:]'*(x[1:2] - x_opp[1:2])]
        [x[1] - x_opp[1] x[2] - x_opp[2]]
    end

    #Δθ = mapreduce(vcat, 1:T) do t
    #    @inbounds x = @view(X[(t-1)*xdim+1:t*xdim])
    #    @inbounds x_opp = @view(X_opp[(t-1)*xdim+1:t*xdim])
    #    x[4] - x_opp[4]
    #end

    @assert size(d) == (T, 2)
    du = u_max_drafting - u_max_nominal
    # objective: if ego is in opponent's drafting box set u_max to u_max_drafting, otherwise set to u_max_nominal
    # bonus objective: du = 0 if the players move in perpendicular directions

    u_max_1 = du * sigmoid.(d[:, 2] .+ box_length, 10.0, 0) .+ u_max_nominal
    u_max_2 = du * sigmoid.(-d[:, 2], 10.0, 0) .+ u_max_nominal
    u_max_3 = du * sigmoid.(d[:, 1] .+ box_width / 2 .* (-d[:, 2] ./ box_length), 10.0, 0) .+ u_max_nominal
    u_max_4 = du * sigmoid.(-d[:, 1] .+ box_width / 2 .* (-d[:, 2] ./ box_length), 10.0, 0) .+ u_max_nominal
    #u_max_3 = du * sigmoid.(d[:, 1] .+ box_width / 2, 10.0, 0) .+ u_max_nominal
    #u_max_4 = du * sigmoid.(-d[:, 1] .+ box_width / 2, 10.0, 0) .+ u_max_nominal
    (u_max_1, u_max_2, u_max_3, u_max_4)
end

function sigmoid(x, a, b)
    xx = x * a + b
    1.0 / (1.0 + exp(-xx))
end

# lower bound function -- above zero whenever h ≥ 0, below zero otherwise
function l(h; a=5.0, b=4.5)
    sigmoid(h, a, b) - sigmoid(0, a, b)
end

# road defined by checkpoints: y=>x
function get_road(y_ego; road=Dict(0 => 0, 1 => 0, 2 => 0.1, 3 => 0.1, 4 => 2, 5 => 0, 6 => -1, 7 => 0, 8 => 0), displaying=false)
    # choose closest 3 based on shortest vertical distance
    n = 3
    road_ys = road |> keys |> collect
    sortedkeys = sortperm((road_ys .- y_ego) .^ 2)
    closest_inds = sortedkeys[1:n]
    ys = road_ys[closest_inds]
    xs = mapreduce(vcat, ys) do y
        road[y]
    end
    A = hcat(2 * xs, 2 * ys, ones(n))
    b = mapreduce(vcat, zip(xs, ys)) do (x, y)
        (x^2 + y^2)
    end

    # if user makes a mistake and provides three colinear points, "regularize" by perturbation 
    if cond(A) > 1e6
        A[1] += randn() * 1e-2
    end

    c₁, c₂, r̂ = A \ b
    c = [c₁, c₂]
    r² = r̂ + c' * c
    r = sqrt(r²)

    # visualize for debug
    if displaying
        r = sqrt(r²)
        d = 0.2
        circ = mapreduce(vcat, 0:0.1:(2π+0.1)) do t
            c[1] + r * cos(t), c[2] + r * sin(t)
        end
        circ_left = mapreduce(vcat, 0:0.1:(2π+0.1)) do t
            c[1] + (r + d) * cos(t), c[2] + (r + d) * sin(t)
        end
        circ_right = mapreduce(vcat, 0:0.1:(2π+0.1)) do t
            c[1] + (r - d) * cos(t), c[2] + (r - d) * sin(t)
        end

        f = Figure()
        ax = Axis(f[1, 1], aspect=DataAspect())
        GLMakie.scatter!(ax, xs, ys)
        GLMakie.scatter!(ax, c[1], c[2])
        GLMakie.lines!(ax, circ)
        GLMakie.lines!(ax, circ_left)
        GLMakie.lines!(ax, circ_right)
        display(f)
    end

    (c, r)
end

# road constraint
function road_left(X, c, r; d, col_buffer=0.0)
    xdim = 4
    T = Int(length(X) / xdim)
    mapreduce(vcat, 1:T) do t
        @inbounds x = @view(X[(t-1)*4+1:t*4])
        p = x[1:2]
        [(p - c)' * (p - c) - (r[1] - d + col_buffer)^2,] # ≥ 0
    end
end

function road_right(X, c, r; d, col_buffer=0.0)
    xdim = 4
    T = Int(length(X) / xdim)
    mapreduce(vcat, 1:T) do t
        @inbounds x = @view(X[(t-1)*4+1:t*4])
        p = x[1:2]
        [(r[1] + d - col_buffer)^2 - (p - c)' * (p - c),] # ≥ 0
    end
end

# e = ego
# o = opponent
function g_ego(X, U, X_opp, x0, x0_opp, c_road, r_road, c_opp; Δt, r, cd, d, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer)
    xdim = 4
    udim = 2

    g_dyn = dyn(X, U, x0, Δt, cd)
    g_col = col(X, X_opp, r; col_buffer)
    g_road_left = road_left(X, c_road, r_road; d, col_buffer)
    g_road_right = road_right(X, c_road, r_road; d, col_buffer)
    h_col = responsibility(X, X_opp)
    u_max_1, u_max_2, u_max_3, u_max_4 = accel_bounds(X,
        X_opp,
        u_max_nominal,
        u_max_drafting,
        box_length,
        box_width,
        c_opp,
        x0_opp)
    as = @view(U[1:udim:end]) # rate of tangential velocity
    ωs = @view(U[2:udim:end]) # rate of heading 
    speed = @view(X[3:xdim:end])
    heading = @view(X[4:xdim:end])

    [
        g_dyn
        g_col - l.(h_col)
        g_road_left
        g_road_right
        speed
        heading
        as
        ωs
        as - u_max_1
        as - u_max_2
        as - u_max_3
        as - u_max_4
    ]
end

function g1(z; Δt, r, cd, d, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer)
    T, Xa, Ua, Xb, Ub, x0a, x0b, ca, cb, ra, rb = view_z(z)

    g_ego(Xa, Ua, Xb, x0a, x0b, ca, ra, cb; Δt, r, cd, d, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer)
end

function g2(z; Δt, r, cd, d, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer)
    T, Xa, Ua, Xb, Ub, x0a, x0b, ca, cb, ra, rb = view_z(z)

    g_ego(Xb, Ub, Xa, x0b, x0a, cb, rb, ca; Δt, r, cd, d, u_max_nominal, u_max_drafting, box_length, box_width, col_buffer)
end

function setup(;
    T=10,
    Δt=0.1,
    r=1.0,
    cost_α1=1e-3,
    cost_α2=1e-4,
    cost_β=1e-1,
    drag_coef=0.2,
    u_max_nominal=1.0,
    u_max_drafting=2.5,
    draft_box_length=5.0,
    draft_box_width=2.0,
    d=1.0,
    u_max_braking=u_max_drafting,
    min_speed=-1.0,
    max_heading_offset=π / 2,
    max_heading_rate=1.0,
    col_buffer=r / 4
)
    xdim = 4

    lb = [fill(0.0, xdim * T)
        fill(0.0, T)
        fill(0.0, T)
        fill(0.0, T)
        fill(min_speed, T)
        fill(π / 2 - max_heading_offset, T)
        fill(-u_max_braking, T)
        fill(-max_heading_rate, T)
        fill(-Inf, xdim * T)
    ]

    ub = [fill(0.0, xdim * T)
        fill(Inf, T)
        fill(Inf, T)
        fill(Inf, T)
        fill(Inf, T)
        fill(π / 2 + max_heading_offset, T)
        fill(Inf, T)
        fill(max_heading_rate, T)
        fill(0.0, xdim * T)
    ]

    f1_pinned = (z -> f1(z; α1=cost_α1, α2=cost_α2, β=cost_β))
    f2_pinned = (z -> f2(z; α1=cost_α1, α2=cost_α2, β=cost_β))
    g1_pinned = (z -> g1(z;
        Δt,
        r=r,
        cd=drag_coef,
        d=d,
        u_max_nominal,
        u_max_drafting,
        box_length=draft_box_length,
        box_width=draft_box_width,
        col_buffer)
    )
    g2_pinned = (z -> g2(z;
        Δt,
        r=r,
        cd=drag_coef,
        d=d,
        u_max_nominal,
        u_max_drafting,
        box_length=draft_box_length,
        box_width=draft_box_width,
        col_buffer)
    )

    n_param = 14 # 8 x0, 4 c, 2 r2
    OP1 = OptimizationProblem(12 * T + n_param, 1:6*T, f1_pinned, g1_pinned, lb, ub)
    OP2 = OptimizationProblem(12 * T + n_param, 1:6*T, f2_pinned, g2_pinned, lb, ub)

    sp_a = create_epec((1, 0), OP1)
    gnep = [OP1 OP2]
    bilevel = [OP1; OP2]

    function extract_gnep(θ)
        z = θ[gnep.x_inds]
        T, Xa, Ua, Xb, Ub = view_z([z; zeros(n_param)])
        (; Xa, Ua, Xb, Ub)
    end
    function extract_bilevel(θ)
        z = θ[bilevel.x_inds]
        T, Xa, Ua, Xb, Ub = view_z([z; zeros(n_param)])
        (; Xa, Ua, Xb, Ub)
    end

    problems = (;
        sp_a,
        gnep,
        bilevel,
        extract_gnep,
        extract_bilevel,
        OP1,
        OP2,
        params=(;
            T,
            Δt,
            r=r,
            cd=drag_coef,
            d=d,
            u_max_nominal,
            u_max_drafting,
            u_max_braking,
            α1=cost_α1,
            α2=cost_α2,
            β=cost_β,
            box_length=draft_box_length,
            box_width=draft_box_width,
            min_speed,
            max_heading_offset,
            col_buffer
        )
    )
end

function attempt_solve(prob, init)
    success = true
    result = init
    try
        result = solve(prob, init)
    catch err
        #println(err)
        success = false
    end
    (success, result)
end

function solve_seq_adaptive(probs, x0, road; only_want_gnep=false, only_want_sp=false, try_bilevel_first=false, try_gnep_first=true)
    T = probs.params.T
    Δt = probs.params.Δt
    cd = probs.params.cd
    Xa = []
    Ua = []
    Xb = []
    Ub = []
    x0a = x0[1:4]
    x0b = x0[5:8]
    xa = x0a
    xb = x0b
    for t in 1:T
        ua = [0; 0]#-cd * xa[3:4]
        ub = [0; 0]#-cd * xb[3:4]
        xa = simplecraft(xa, ua, Δt, cd)
        xb = simplecraft(xb, ub, Δt, cd)
        append!(Ua, ua)
        append!(Ub, ub)
        append!(Xa, xa)
        append!(Xb, xb)
    end

    # compute road
    ca, ra = get_road(x0a[2]; road)
    cb, rb = get_road(x0b[2]; road)

    # dummy init
    Z = (; Xa, Ua, Xb, Ub)
    valid_Z = Dict()
    valid_Z[8] = Z
    #dummy_init = zeros(probs.gnep.top_level.n)
    #dummy_init = [Xa; Ua; Xb; Ub]
    #show_me(dummy_init, x0; T=probs.params.T, lat_pos_max=probs.params.lat_max + sqrt(probs.params.r) / 2)

    bilevel_success = false
    gnep_success = false
    sp_success = false

    θ_bilevel = []
    θ_gnep = []
    θ_sp_a = []
    θ_sp_b = []

    if only_want_gnep
        want_gnep = true
        want_bilevel = false
    else
        want_gnep = false
        want_bilevel = true
    end

    if !try_bilevel_first && !try_gnep_first
        want_sp = true
    else
        want_sp = false
    end

    if only_want_sp
        want_sp = true
        want_gnep = false
        want_bilevel = false
        try_bilevel_first = false
        try_gnep_first = false
    end

    # preference order
    # 1. bilevel
    # 2. gnep->bilevel
    # 3. sp->gnep->bilevel
    # 4. sp->bilevel
    # 5. gnep
    # 6. sp->gnep
    # 7. sp
    # 8. dummy
    if try_bilevel_first && want_bilevel
        # initialized from dummy:
        bilevel_init = zeros(probs.bilevel.top_level.n)
        bilevel_init[probs.gnep.x_inds] = [Xa; Ua; Xb; Ub]
        bilevel_init = [bilevel_init; x0; ca; cb; ra; rb]
        #@info "(1) bilevel..."
        bilevel_success, θ_bilevel = attempt_solve(probs.bilevel, bilevel_init)

        if bilevel_success
            #@info "bilevel success 1"
            valid_Z[1] = probs.extract_bilevel(θ_bilevel)
        else
            want_gnep = true
        end
    elseif want_bilevel
        want_gnep = true
    end

    if try_gnep_first && want_gnep
        # initialized from dummy:
        gnep_init = zeros(probs.gnep.top_level.n)
        gnep_init[probs.gnep.x_inds] = [Xa; Ua; Xb; Ub]
        gnep_init = [gnep_init; x0; ca; cb; ra; rb]
        #@info "(5) gnep..."
        gnep_success, θ_gnep = attempt_solve(probs.gnep, gnep_init)

        if gnep_success
            #@info "gnep success 5"
            valid_Z[5] = probs.extract_gnep(θ_gnep)

            if want_bilevel
                # initialized from gnep:
                bilevel_init = zeros(probs.bilevel.top_level.n)
                bilevel_init[probs.bilevel.x_inds] = θ_gnep[probs.gnep.x_inds]
                bilevel_init[probs.bilevel.inds["λ", 1]] = θ_gnep[probs.gnep.inds["λ", 1]]
                bilevel_init[probs.bilevel.inds["s", 1]] = θ_gnep[probs.gnep.inds["s", 1]]
                bilevel_init[probs.bilevel.inds["λ", 2]] = θ_gnep[probs.gnep.inds["λ", 2]]
                bilevel_init[probs.bilevel.inds["s", 2]] = θ_gnep[probs.gnep.inds["s", 2]]
                bilevel_init = [bilevel_init; x0; ca; cb; ra; rb]
                #@info "(2) gnep->bilevel..."
                bilevel_success, θ_bilevel = attempt_solve(probs.bilevel, bilevel_init)

                if bilevel_success
                    #@info "gnep->bilevel success 2"
                    valid_Z[2] = probs.extract_bilevel(θ_bilevel)
                else
                    want_sp = true
                end
            end
        else
            want_sp = true
        end
    end

    if want_sp
        xu_dim = 6
        n_param = 14
        # initialized from dummy:
        # [!] need to be changed in problems.jl so this isn't such a mess
        # !!! 348 vs 568 breaks it
        # !!! Ordering of x_w changes because:
        # 1p: [x1=>1:60 λ1,s1=>61:280, w=>281:348]
        # 2p: [x1,x2=>1:120, λ1,λ2,s1,s2=>121:560 w=>561:568
        sp_a_init = zeros(probs.gnep.top_level.n)
        sp_a_init[probs.sp_a.x_inds] = [Xa; Ua]
        sp_a_init[probs.sp_a.top_level.n+1:probs.sp_a.top_level.n+xu_dim*T] = [Xb; Ub]
        sp_a_init[probs.sp_a.top_level.n+6*T+1:probs.sp_a.top_level.n+6*T+n_param] = [x0; ca; cb; ra; rb] # right now parameters are expected to be contiguous

        #@info "(7a) sp_a..."
        #@infiltrate
        sp_a_success, θ_sp_a = attempt_solve(probs.sp_a, sp_a_init)
        #show_me([θ_sp_a[probs.sp_a.x_inds]; θ_sp_a[probs.sp_a.top_level.n+1:probs.sp_a.top_level.n+60]], x0; T=probs.params.T, lat_pos_max=probs.params.lat_max + sqrt(probs.params.r) / 2)
        # if it fails:
        #show_me([safehouse.θ_out[probs.sp_a.x_inds]; safehouse.w[1:60]], safehouse.w[61:68]; T=probs.params.T, lat_pos_max=probs.params.lat_max + sqrt(probs.params.r) / 2)
        # swapping b for a:
        sp_b_init = zeros(probs.gnep.top_level.n)
        sp_b_init[probs.sp_a.x_inds] = [Xb; Ub]
        sp_b_init[probs.sp_a.top_level.n+1:probs.sp_a.top_level.n+6*T] = [Xa; Ua]
        sp_b_init[probs.sp_a.top_level.n+6*T+1:probs.sp_a.top_level.n+6*T+n_param] = [x0[5:8]; x0[1:4]; cb; ca; rb; ra]

        #@info "(7b) sp_b..."
        sp_b_success, θ_sp_b = attempt_solve(probs.sp_a, sp_b_init)
        #show_me([θ_sp_b[probs.sp_b.top_level.n+1:probs.sp_b.top_level.n+60]; θ_sp_b[probs.sp_b.x_inds]], x0; T=probs.params.T, lat_pos_max=probs.params.lat_max + sqrt(probs.params.r) / 2)
        # if it fails:
        #show_me([safehouse.w[1:60]; safehouse.θ_out[probs.sp_b.x_inds]], [safehouse.w[65:68]; safehouse.w[61:64]]; T=probs.params.T, lat_pos_max=probs.params.lat_max + sqrt(probs.params.r) / 2)    

        if only_want_sp
            sp_success = sp_a_success # assume ego is P1
        else
            sp_success = sp_a_success && sp_b_success # consider sp success to be bilateral success
        end

        if sp_success
            #@info "sp success 7"
            gnep_like = zeros(probs.gnep.top_level.n)

            if sp_a_success && sp_b_success
                gnep_like[probs.gnep.x_inds] = [θ_sp_a[probs.sp_a.x_inds]; θ_sp_b[probs.sp_a.x_inds]]
            elseif sp_a_success
                gnep_like[probs.gnep.x_inds] = [θ_sp_a[probs.sp_a.x_inds]; Xb; Ub]
            elseif sp_b_success
                gnep_like[probs.gnep.x_inds] = [Xa; Ua; θ_sp_b[probs.sp_a.x_inds]]
            end
            gnep_like = [gnep_like; x0]
            valid_Z[7] = probs.extract_gnep(gnep_like)

            if want_gnep
                # initialized from θ_sp_a and θ_sp_b:
                gnep_init = zeros(probs.gnep.top_level.n)
                gnep_init[probs.gnep.x_inds] = [θ_sp_a[probs.sp_a.x_inds]; θ_sp_b[probs.sp_a.x_inds]]
                gnep_init[probs.bilevel.inds["λ", 1]] = θ_sp_a[probs.gnep.inds["λ", 1]]
                gnep_init[probs.bilevel.inds["s", 1]] = θ_sp_a[probs.gnep.inds["s", 1]]
                gnep_init[probs.bilevel.inds["λ", 2]] = θ_sp_b[probs.gnep.inds["λ", 1]]
                gnep_init[probs.bilevel.inds["s", 2]] = θ_sp_b[probs.gnep.inds["s", 1]]
                gnep_init = [gnep_init; x0; ca; cb; ra; rb]
                #@info "(6) sp->gnep..."
                gnep_success, θ_gnep = attempt_solve(probs.gnep, gnep_init)

                if gnep_success
                    #@info "sp->gnep success 6"
                    want_gnep = false
                    valid_Z[6] = probs.extract_gnep(θ_gnep)

                    if want_bilevel
                        # initialized from gnep which was initialized from sp:
                        bilevel_init = zeros(probs.bilevel.top_level.n)
                        bilevel_init[probs.bilevel.x_inds] = θ_gnep[probs.gnep.x_inds]
                        bilevel_init[probs.bilevel.inds["λ", 1]] = θ_gnep[probs.gnep.inds["λ", 1]]
                        bilevel_init[probs.bilevel.inds["s", 1]] = θ_gnep[probs.gnep.inds["s", 1]]
                        bilevel_init[probs.bilevel.inds["λ", 2]] = θ_gnep[probs.gnep.inds["λ", 2]]
                        bilevel_init[probs.bilevel.inds["s", 2]] = θ_gnep[probs.gnep.inds["s", 2]]
                        bilevel_init = [bilevel_init; x0; ca; cb; ra; rb]
                        #@info "(3) sp->gnep->bilevel..."
                        bilevel_success, θ_bilevel = attempt_solve(probs.bilevel, bilevel_init)

                        if bilevel_success
                            #@info "sp->gnep->bilevel success 3"
                            valid_Z[3] = probs.extract_bilevel(θ_bilevel)
                        end
                    end
                else
                    if want_bilevel
                        # initialized from sp:
                        bilevel_init = zeros(probs.bilevel.top_level.n)
                        bilevel_init[probs.bilevel.x_inds] = θ_gnep[probs.gnep.x_inds]
                        bilevel_init[probs.bilevel.inds["λ", 1]] = θ_sp_a[probs.gnep.inds["λ", 1]]
                        bilevel_init[probs.bilevel.inds["s", 1]] = θ_sp_a[probs.gnep.inds["s", 1]]
                        bilevel_init[probs.bilevel.inds["λ", 2]] = θ_sp_b[probs.gnep.inds["λ", 1]]
                        bilevel_init[probs.bilevel.inds["s", 2]] = θ_sp_b[probs.gnep.inds["s", 1]]
                        bilevel_init = [bilevel_init; x0; ca; cb; ra; rb]
                        #bilevel_init = [bilevel_init; x0]
                        #@info "(4) sp->bilevel..."
                        bilevel_success, θ_bilevel = attempt_solve(probs.bilevel, bilevel_init)

                        if bilevel_success
                            #@info "sp->bilevel success 4"
                            valid_Z[4] = probs.extract_bilevel(θ_bilevel)
                        end
                    end
                end
            end
        end
    end

    sorted_Z = sort(collect(valid_Z), by=x -> x[1])
    lowest_preference, Z = sorted_Z[1] # best pair

    if lowest_preference < 8
        print("Success $lowest_preference ")
    else
        print("Fail $lowest_preference ")
    end

    P1 = [Z.Xa[1:4:end] Z.Xa[2:4:end] Z.Xa[3:4:end] Z.Xa[4:4:end]]
    U1 = [Z.Ua[1:2:end] Z.Ua[2:2:end]]
    P2 = [Z.Xb[1:4:end] Z.Xb[2:4:end] Z.Xb[3:4:end] Z.Xb[4:4:end]]
    U2 = [Z.Ub[1:2:end] Z.Ub[2:2:end]]

    (; P1, P2, U1, U2, lowest_preference, sorted_Z)
end


# Solve mode:
#						P1:						
#				SP  NE   P1-leader  P1-follower
#			 SP 1              
# P2:		 NE 2   3
#	  P2-Leader 4   5   6 
#   P2-Follower 7   8   9		    10
#
function solve_simulation(probs, T; x0=[0, 0, 0, 7, 0.1, -2.21, 0, 7], road=Dict(0 => 0, 1 => 0, 2 => 0.1), mode=1)
    status = "ok"
    x0a = x0[1:4]
    x0b = x0[5:8]
    P1_buffer = repeat(x0', 10, 1)
    P2_buffer = repeat(x0', 10, 1)
    U1_buffer = repeat([0, 0]', 10, 1)
    U2_buffer = repeat([0, 0]', 10, 1)

    results = Dict()
    for t = 1:T
        #@info "Sim timestep $t:"
        print("Sim timestep $t: ")
        # check initial condition feasibility
        ca, ra = get_road(x0a[2]; road)
        cb, rb = get_road(x0b[2]; road)
        pa = x0a[1:2]
        pb = x0b[1:2]
        d = probs.params.d
        is_x0_infeasible = false

        if col(x0a, x0b, probs.params.r)[1] <= 0 - 1e-4
            status = "Infeasible initial condition: Collision"
            is_x0_infeasible = true
        elseif (pa - ca)' * (pa - ca) - (ra[1] - d)^2 < -1e-4 || (ra[1] + d)^2 - (pa - ca)' * (pa - ca) < -1e-4 ||
               (pb - cb)' * (pb - cb) - (rb[1] - d)^2 < -1e-4 || (rb[1] + d)^2 - (pb - cb)' * (pb - cb) < -1e-4
            status = "Infeasible initial condition: Out of lanes"
            is_x0_infeasible = true
        elseif x0a[3] < probs.params.min_speed - 1e-4 || x0b[3] < probs.params.min_speed - 1e-4
            status = "Infeasible initial condition: Invalid speed"
            is_x0_infeasible = true
        elseif x0a[4] < π / 2 - probs.params.max_heading_offset - 1e-4 || x0a[4] > π / 2 + probs.params.max_heading_offset + 1e-4 || x0b[4] < π / 2 - probs.params.max_heading_offset - 1e-4 || x0b[4] > π / 2 + probs.params.max_heading_offset + 1e-4
            status = "Infeasible initial condition: Invalid heading"
            is_x0_infeasible = true
        end

        if is_x0_infeasible
            # currently status isn't saved
            print(status)
            print("\n")
            results[t] = (; x0, P1=P1_buffer, P2=P2_buffer, U1=U1_buffer, U2=U2_buffer)
            break
        end

        # replace P1 and P2
        x0_swapped = copy(x0)
        x0_swapped[1:4] = x0[5:8]
        x0_swapped[5:8] = x0[1:4]

        if mode == 1 # P1 SP, P2 SP
            a_res = solve_seq_adaptive(probs, x0, road; only_want_gnep=false, only_want_sp=true)
            b_res = solve_seq_adaptive(probs, x0_swapped, road; only_want_gnep=false, only_want_sp=true)
            r_P1 = a_res.P1
            r_U1 = a_res.U1
            r_P2 = b_res.P1
            r_U2 = b_res.U1
            r = (; P1=r_P1, U1=r_U1, P2=r_P2, U2=r_U2)
        elseif mode == 3 # P1 NE, P2 NE
            a_res = solve_seq_adaptive(probs, x0, road; only_want_gnep=true, only_want_sp=false)
            r_P1 = a_res.P1
            r_U1 = a_res.U1
            r_P2 = a_res.P2
            r_U2 = a_res.U2
            r = (; P1=r_P1, U1=r_U1, P2=r_P2, U2=r_U2)
        elseif mode == 9 # P1 Leader, P2 Follower
            a_res = solve_seq_adaptive(probs, x0, road; only_want_gnep=false, only_want_sp=false)
            r_P1 = a_res.P1
            r_U1 = a_res.U1
            r_P2 = a_res.P2
            r_U2 = a_res.U2
            r = (; P1=r_P1, U1=r_U1, P2=r_P2, U2=r_U2)
        elseif mode == 2 # P1 SP, P2 NE
            a_res = solve_seq_adaptive(probs, x0, road; only_want_gnep=false, only_want_sp=true)
            b_res = solve_seq_adaptive(probs, x0, road; only_want_gnep=true, only_want_sp=false)
            r_P1 = a_res.P1
            r_U1 = a_res.U1
            r_P2 = b_res.P2
            r_U2 = b_res.U2
            r = (; P1=r_P1, U1=r_U1, P2=r_P2, U2=r_U2)
        elseif mode == 4 # P1 SP, P2 Leader
            a_res = solve_seq_adaptive(probs, x0, road; only_want_gnep=false, only_want_sp=true)
            b_res = solve_seq_adaptive(probs, x0_swapped, road; only_want_gnep=false, only_want_sp=false)
            r_P1 = a_res.P1
            r_U1 = a_res.U1
            r_P2 = b_res.P1
            r_U2 = b_res.U1
            r = (; P1=r_P1, U1=r_U1, P2=r_P2, U2=r_U2)
        elseif mode == 5 # P1 NE, P2 Leader
            a_res = solve_seq_adaptive(probs, x0, road; only_want_gnep=true, only_want_sp=false)
            b_res = solve_seq_adaptive(probs, x0_swapped, road; only_want_gnep=false, only_want_sp=false)
            r_P1 = a_res.P1
            r_U1 = a_res.U1
            r_P2 = b_res.P1
            r_U2 = b_res.U1
            r = (; P1=r_P1, U1=r_U1, P2=r_P2, U2=r_U2)
        elseif mode == 6 # P1 Leader, P2 Leader
            a_res = solve_seq_adaptive(probs, x0, road; only_want_gnep=false, only_want_sp=false)
            b_res = solve_seq_adaptive(probs, x0_swapped, road; only_want_gnep=false, only_want_sp=false)
            r_P1 = a_res.P1
            r_U1 = a_res.U1
            r_P2 = b_res.P1
            r_U2 = b_res.U1
            r = (; P1=r_P1, U1=r_U1, P2=r_P2, U2=r_U2)
        elseif mode == 7 # P1 SP, P2 Follower
            a_res = solve_seq_adaptive(probs, x0, road; only_want_gnep=false, only_want_sp=true)
            b_res = solve_seq_adaptive(probs, x0, road; only_want_gnep=false, only_want_sp=false)
            r_P1 = a_res.P1
            r_U1 = a_res.U1
            r_P2 = b_res.P2
            r_U2 = b_res.U2
            r = (; P1=r_P1, U1=r_U1, P2=r_P2, U2=r_U2)
        elseif mode == 8 # P1 NE, P2 Follower 
            a_res = solve_seq_adaptive(probs, x0, road; only_want_gnep=true, only_want_sp=false)
            b_res = solve_seq_adaptive(probs, x0, road; only_want_gnep=false, only_want_sp=false)
            r_P1 = a_res.P1
            r_U1 = a_res.U1
            r_P2 = b_res.P2
            r_U2 = b_res.U2
            r = (; P1=r_P1, U1=r_U1, P2=r_P2, U2=r_U2)
        elseif mode == 10 # P1 Follower, P2 Follower 
            a_res = solve_seq_adaptive(probs, x0_swapped, road; only_want_gnep=false, only_want_sp=false)
            b_res = solve_seq_adaptive(probs, x0, road; only_want_gnep=false, only_want_sp=false)
            r_P1 = a_res.P2
            r_U1 = a_res.U2
            r_P2 = b_res.P2
            r_U2 = b_res.U2
            r = (; P1=r_P1, U1=r_U1, P2=r_P2, U2=r_U2)
        end

        print("\n")

        ##costs = [OP.f(z) for OP in probs.gnep.OPs]
        #lowest_preference, Z = r.sorted_Z[1]
        #z = [Z.Xa; Z.Ua; Z.Xb; Z.Ub; x0]
        #feasible_arr = [[OP.l .- 1e-4 .<= OP.g(z) .<= OP.u .+ 1e-4] for OP in probs.gnep.OPs]
        #feasible = all(all(feasible_arr[i][1]) for i in 1:2) # I think this is fine

        #if !feasible || any(r.P1[:, 4] .< probs.params.min_long_vel - 1e-4) || any(r.P2[:, 4] .< probs.params.min_long_vel - 1e-4) || any(r.P1[:, 1] .< -lat_max - 1e-4) || any(r.P2[:, 1] .< -lat_max - 1e-4) || any(r.P1[:, 1] .> 1e-4 + lat_max) || any(r.P2[:, 1] .> lat_max + 1e-4)
        #    if (feasible)
        #        # this must never trigger
        #        @infiltrate
        #    end
        #    status = "Invalid solution"
        #    @info "Invalid solution"
        #end

        # clamp controls and check feasibility
        xa = r.P1[1, :]
        xb = r.P2[1, :]
        ua = r.U1[1, :]
        ub = r.U2[1, :]

        ua_maxes = accel_bounds(xa,
            xb,
            probs.params.u_max_nominal,
            probs.params.u_max_drafting,
            probs.params.box_length,
            probs.params.box_width,
            ca,
            x0)

        ub_maxes = accel_bounds(xb,
            xa,
            probs.params.u_max_nominal,
            probs.params.u_max_drafting,
            probs.params.box_length,
            probs.params.box_width,
            cb,
            x0)

        ua[1] = minimum([maximum([ua[1], -probs.params.u_max_braking]), ua_maxes[1][1], ua_maxes[2][1], ua_maxes[3][1], ua_maxes[4][1]])
        ub[1] = minimum([maximum([ub[1], -probs.params.u_max_braking]), ub_maxes[1][1], ub_maxes[2][1], ub_maxes[3][1], ub_maxes[4][1]])
        ua[2] = minimum([maximum([ua[2], π / 2 + probs.params.max_heading_offset]), π / 2 - probs.params.max_heading_offset])
        ua[2] = minimum([maximum([ub[2], π / 2 + probs.params.max_heading_offset]), π / 2 - probs.params.max_heading_offset])

        x0a = simplecraft(xa, ua, probs.params.Δt, probs.params.cd)
        x0b = simplecraft(xb, ub, probs.params.Δt, probs.params.cd)

        results[t] = (; x0, r.P1, r.P2, r.U1, r.U2)
        P1_buffer = vcat(r.P1[3:end, :], r.P1[9:end, :]) # for visualization if x0 is not feasible in the next sim step
        P2_buffer = vcat(r.P2[3:end, :], r.P2[9:end, :])
        U1_buffer = vcat(r.U1[3:end, :], r.U1[9:end, :]) # for visualization if x0 is not feasible in the next sim step
        U2_buffer = vcat(r.U2[3:end, :], r.U2[9:end, :])
        x0 = [x0a; x0b]
    end
    results
end

function gen_road(;
    seed=nothing,
    #straight:
    #road_base=Dict(0 => 0, 2 => 0.01, 4 => 0, 6 => 0.01, 8 => 0, 10 => 0.01, 12 => 0, 14 => 0.01, 16 => 0, 18 => 0.01, 20 => 0, 22 => 0.01, 24 => 0, 26 => 0.01, 28 => 0, 30 => 0.01)
    # turns:
    road_base=Dict(
        0 => 0,
        2 => 0.01,
        4 => 0,
        6 => 0.01,
        6.5 => 0.09, # begin right turn
        7 => 0.25,
        7.5 => 0.5,
        8 => 0.8,
        9 => 1.6,
        10 => 2.3,
        10.5 => 2.45,
        11.0 => 2.5, # right peak
        11.5 => 2.45,
        12 => 2.3,
        13 => 1.6,
        14 => 0.8,
        14.5 => 0.5,
        15 => 0.25,
        15.5 => 0.09, # end right turn
        16 => 0.0,
        18 => 0.01,
        18.5 => -0.09, # begin left turn
        19 => -0.25,
        19.5 => -0.5,
        20 => -0.8,
        21 => -1.6,
        22 => -2.3,
        22.5 => -2.45,
        23 => -2.5, # left peak
        23.5 => -2.45,
        24 => -2.3,
        25 => -1.6,
        26 => -0.8,
        26.5 => -0.5,
        27 => -0.25,
        27.5 => -0.09 # end left turn
    )
)
    # randomize road
    road_sorted = road_base |> sort
    road_len = (road_sorted |> keys |> collect |> last) - (road_sorted |> keys |> collect |> first)
    road_y_offset = (0.5 - rand(MersenneTwister(seed))) * road_len
    #road_y_offset = 0
    road_offset = Dict{Float64,Float64}()

    for (y, x) in road_sorted
        road_offset[y+road_y_offset] = x
        road_offset[y+road_y_offset-road_len-1] = x
        road_offset[y+road_y_offset+road_len+1] = x
    end
    road_offset
end