function animate(probs, sim_results; save=false, filename="test.mp4", sleep_duration=1e-1, mode=1, road=nothing)
    mode_str = [
        1 => "Blue: S, Red: S"
        2 => "Blue: S, Red: N"
        3 => "Blue: N, Red: N"
        4 => "Blue: S, Red: L"
        5 => "Blue: N, Red: L"
        6 => "Blue: L, Red: L"
        7 => "Blue: S, Red: F"
        8 => "Blue: N, Red: F"
        9 => "Blue: L, Red: F"
        10 => "Blue: F, Red: F"
    ]
    rad = probs.params.r / 2
    lat = probs.params.lat_max + rad
    if isnothing(road)
        (f, ax, XA, XB, lat) = visualize(; rad=rad, lat=lat)
    else
        (f, ax, XA, XB, lat) = visualize(road; rad=rad, lat=lat, d=probs.params.d)
    end
    display(f)
    T = length(sim_results)

    if save
        # extra 5 frames at the end because powerpoint is being weird
        record(f, filename, 1:T+10; framerate=10) do t
            if t <= T
                update_visual!(ax, XA, XB, sim_results[t].x0, sim_results[t].P1, sim_results[t].P2; T=probs.params.T, lat=lat)
                ax.title = "$(mode_str[mode][2])\nTime step = $(string(t))"
            end
        end
    else
        for t in 1:T
            update_visual!(ax, XA, XB, sim_results[t].x0, sim_results[t].P1, sim_results[t].P2; T=probs.params.T, lat=lat)
            ax.title = "$(mode_str[mode][2])\nTime step = $(string(t))"
            sleep(sleep_duration)
        end
    end
end

function visualize(; T=10, rad=0.5, lat=6.0)
    f = Figure(size=(500, 1000), grid=false)
    ax = Axis(f[1, 1], aspect=DataAspect())
    #resize_to_layout!(f)

    carsymbol_string = "M 7 9 C 5.897 9 5 9.897 5 11 L 5 11.4375 C 6.328 12.0205 7.324 13.1585 7.75 14.5625 C 7.917 14.8145 8 15.245 8 16 L 8 18 L 11 18 L 11 19 L 13 19 L 13 18 L 15 18 C 16.103 18 17 17.103 17 16 L 17 11 C 17 9.897 16.103 9 15 9 L 7 9 z M 34 10 C 32.897 10 32 10.897 32 12 L 32 16 C 32 17.103 32.897 18 34 18 L 35 18 L 35 19 L 37 19 L 37 18 L 38 18 L 38 19 L 40 19 L 40 18 L 41 18 C 42.103 18 43 17.103 43 16 L 43 12 C 43 10.897 42.103 10 41 10 L 34 10 z M 1 13 C 0.448 13 0 13.448 0 14 L 0 36 C 0 36.553 0.448 37 1 37 L 3 37 C 4.654 37 6 35.654 6 34 L 6 16 C 6 14.346 4.654 13 3 13 L 1 13 z M 45 13 C 44.447 13 44 13.448 44 14 L 44 18.625 C 47.242 19.242 49.80475 20.10125 49.96875 20.15625 C 49.98475 20.16125 49.985 20.1825 50 20.1875 L 50 14 C 50 13.448 49.553 13 49 13 L 45 13 z M 20 15 C 19.649 15 19.33625 15.199 19.15625 15.5 L 16.4375 20 L 8 20 L 8 30 L 16.4375 30 L 19.15625 34.5 C 19.33625 34.8 19.649 35 20 35 L 27 35 C 27.208 35 27.42475 34.9345 27.59375 34.8125 L 34.3125 30 L 38 30 C 43.05 30 49.0605 28.0215 49.3125 27.9375 C 49.7235 27.8025 50 27.432 50 27 L 50 23 C 50 22.57 49.7205 22.1995 49.3125 22.0625 C 49.0635 21.9785 43.157 20 38 20 L 34.3125 20 L 27.59375 15.1875 C 27.42475 15.0665 27.208 15 27 15 L 20 15 z M 22 22 L 27 22 C 28.657 22 30 23.343 30 25 C 30 26.657 28.657 28 27 28 L 22 28 L 22 22 z M 50 29.8125 C 49.981 29.8195 49.9565 29.83775 49.9375 29.84375 C 49.7705 29.89875 47.219 30.72875 44 31.34375 L 44 36 C 44 36.553 44.447 37 45 37 L 49 37 C 49.553 37 50 36.553 50 36 L 50 29.8125 z M 11 31 L 11 32 L 8 32 L 8 34 C 8 34.496 7.97425 34.809 7.90625 35 C 7.58025 36.61 6.469 37.9185 5 38.5625 L 5 39 C 5 40.105 5.895 41 7 41 L 15 41 C 16.105 41 17 40.105 17 39 L 17 34 C 17 32.895 16.105 32 15 32 L 13 32 L 13 31 L 11 31 z M 35 31 L 35 32 L 34 32 C 32.897 32 32 32.897 32 34 L 32 38 C 32 39.103 32.897 40 34 40 L 41 40 C 42.103 40 43 39.103 43 38 L 43 34 C 43 32.897 42.103 32 41 32 L 40 32 L 40 31 L 38 31 L 38 32 L 37 32 L 37 31 L 35 31 z"
    carsymbol = BezierPath(carsymbol_string, fit=true, flipy=true)

    GLMakie.lines!(ax, [-lat, -lat], [-10.0, 1000.0], color=:black)
    GLMakie.lines!(ax, [+lat, +lat], [-10.0, 1000.0], color=:black)

    XA = Dict(t => [Observable(0.0), Observable(0.0)] for t in 0:T)
    XB = Dict(t => [Observable(0.0), Observable(0.0)] for t in 0:T)

    circ_x = [rad * cos(t) for t in 0:0.1:(2π+0.1)]
    circ_y = [rad * sin(t) for t in 0:0.1:(2π+0.1)]
    GLMakie.lines!(ax, @lift(circ_x .+ $(XA[0][1])), @lift(circ_y .+ $(XA[0][2])), color=:blue, linewidth=5)
    GLMakie.lines!(ax, @lift(circ_x .+ $(XB[0][1])), @lift(circ_y .+ $(XB[0][2])), color=:red, linewidth=5)

    a_rot = map(atan, @lift(($(XA[0][2]) - $(XA[1][2]))), @lift(($(XA[0][1]) - $(XA[1][1]))))
    b_rot = map(atan, @lift(($(XB[0][2]) - $(XB[1][2]))), @lift(($(XB[0][1]) - $(XB[1][1]))))

    GLMakie.scatter!(ax, @lift([0.0, 0.0] .+ $(XA[0][1])), @lift([0.0, 0.0] .+ $(XA[0][2])), rotations=@lift($(a_rot) .+ pi), marker=carsymbol, markersize=80, color=:blue)
    GLMakie.scatter!(ax, @lift([0.0, 0.0] .+ $(XB[0][1])), @lift([0.0, 0.0] .+ $(XB[0][2])), rotations=@lift($(b_rot) .+ pi), marker=carsymbol, markersize=80, color=:red)

    for t in 1:T
        GLMakie.lines!(ax, @lift(circ_x .+ $(XA[t][1])), @lift(circ_y .+ $(XA[t][2])), color=:blue, linewidth=2, linestyle=:dash)
        GLMakie.lines!(ax, @lift(circ_x .+ $(XB[t][1])), @lift(circ_y .+ $(XB[t][2])), color=:red, linewidth=2, linestyle=:dash)
    end

    return (f, ax, XA, XB, lat)
end

function visualize(road; T=10, rad=0.5, lat=6.0, d=1.0)
    f = Figure(size=(500, 1000), grid=false)
    ax = Axis(f[1, 1], aspect=DataAspect())

    carsymbol_string = "M 7 9 C 5.897 9 5 9.897 5 11 L 5 11.4375 C 6.328 12.0205 7.324 13.1585 7.75 14.5625 C 7.917 14.8145 8 15.245 8 16 L 8 18 L 11 18 L 11 19 L 13 19 L 13 18 L 15 18 C 16.103 18 17 17.103 17 16 L 17 11 C 17 9.897 16.103 9 15 9 L 7 9 z M 34 10 C 32.897 10 32 10.897 32 12 L 32 16 C 32 17.103 32.897 18 34 18 L 35 18 L 35 19 L 37 19 L 37 18 L 38 18 L 38 19 L 40 19 L 40 18 L 41 18 C 42.103 18 43 17.103 43 16 L 43 12 C 43 10.897 42.103 10 41 10 L 34 10 z M 1 13 C 0.448 13 0 13.448 0 14 L 0 36 C 0 36.553 0.448 37 1 37 L 3 37 C 4.654 37 6 35.654 6 34 L 6 16 C 6 14.346 4.654 13 3 13 L 1 13 z M 45 13 C 44.447 13 44 13.448 44 14 L 44 18.625 C 47.242 19.242 49.80475 20.10125 49.96875 20.15625 C 49.98475 20.16125 49.985 20.1825 50 20.1875 L 50 14 C 50 13.448 49.553 13 49 13 L 45 13 z M 20 15 C 19.649 15 19.33625 15.199 19.15625 15.5 L 16.4375 20 L 8 20 L 8 30 L 16.4375 30 L 19.15625 34.5 C 19.33625 34.8 19.649 35 20 35 L 27 35 C 27.208 35 27.42475 34.9345 27.59375 34.8125 L 34.3125 30 L 38 30 C 43.05 30 49.0605 28.0215 49.3125 27.9375 C 49.7235 27.8025 50 27.432 50 27 L 50 23 C 50 22.57 49.7205 22.1995 49.3125 22.0625 C 49.0635 21.9785 43.157 20 38 20 L 34.3125 20 L 27.59375 15.1875 C 27.42475 15.0665 27.208 15 27 15 L 20 15 z M 22 22 L 27 22 C 28.657 22 30 23.343 30 25 C 30 26.657 28.657 28 27 28 L 22 28 L 22 22 z M 50 29.8125 C 49.981 29.8195 49.9565 29.83775 49.9375 29.84375 C 49.7705 29.89875 47.219 30.72875 44 31.34375 L 44 36 C 44 36.553 44.447 37 45 37 L 49 37 C 49.553 37 50 36.553 50 36 L 50 29.8125 z M 11 31 L 11 32 L 8 32 L 8 34 C 8 34.496 7.97425 34.809 7.90625 35 C 7.58025 36.61 6.469 37.9185 5 38.5625 L 5 39 C 5 40.105 5.895 41 7 41 L 15 41 C 16.105 41 17 40.105 17 39 L 17 34 C 17 32.895 16.105 32 15 32 L 13 32 L 13 31 L 11 31 z M 35 31 L 35 32 L 34 32 C 32.897 32 32 32.897 32 34 L 32 38 C 32 39.103 32.897 40 34 40 L 41 40 C 42.103 40 43 39.103 43 38 L 43 34 C 43 32.897 42.103 32 41 32 L 40 32 L 40 31 L 38 31 L 38 32 L 37 32 L 37 31 L 35 31 z"
    carsymbol = BezierPath(carsymbol_string, fit=true, flipy=true)

    # real road
    road_sorted = road |> sort;
    road_center_y = road_sorted |> keys |> collect;
    road_center_x = road_sorted |> values |> collect;
    GLMakie.scatter!(ax, road_center_x, road_center_y, color=:black)
    GLMakie.lines!(ax, road_center_x .- d .- rad, road_center_y, color=:black, linewidth=2)
    GLMakie.lines!(ax, road_center_x .+ d .+ rad, road_center_y, color=:black, linewidth=2)

    XA = Dict(t => [Observable(0.0), Observable(0.0)] for t in 0:T)
    XB = Dict(t => [Observable(0.0), Observable(0.0)] for t in 0:T)

    # modeled road
    function draw_road_left(y)
        c, r = get_road(y; road)
        circ_left = mapreduce(vcat, -π-0.1:1e-2:π) do t
            c[1] + (r + d + rad) * cos(t), c[2] + (r + d + rad) * sin(t)
        end
        circ_left
    end

    function draw_road_right(y)
        c, r = get_road(y; road)
        circ_right = mapreduce(vcat, -π-0.1:1e-2:π) do t
            c[1] + (r - d - rad) * cos(t), c[2] + (r - d - rad) * sin(t)
        end
        circ_right
    end

    circ_x = [rad * cos(t) for t in 0:0.1:(2π+0.1)]
    circ_y = [rad * sin(t) for t in 0:0.1:(2π+0.1)]
    GLMakie.lines!(ax, @lift(circ_x .+ $(XA[0][1])), @lift(circ_y .+ $(XA[0][2])), color=:blue, linewidth=5)
    GLMakie.lines!(ax, @lift(circ_x .+ $(XB[0][1])), @lift(circ_y .+ $(XB[0][2])), color=:red, linewidth=5)
    GLMakie.lines!(ax, @lift(draw_road_left($(XA[0][2]))), color=:black, linestyle=:dash, linewidth=1)
    GLMakie.lines!(ax, @lift(draw_road_right($(XA[0][2]))), color=:black, linestyle=:dash, linewidth=1)

    a_rot = map(atan, @lift(($(XA[0][2]) - $(XA[1][2]))), @lift(($(XA[0][1]) - $(XA[1][1]))))
    b_rot = map(atan, @lift(($(XB[0][2]) - $(XB[1][2]))), @lift(($(XB[0][1]) - $(XB[1][1]))))

    GLMakie.scatter!(ax, @lift([0.0, 0.0] .+ $(XA[0][1])), @lift([0.0, 0.0] .+ $(XA[0][2])), rotations=@lift($(a_rot) .+ pi), marker=carsymbol, markersize=40, color=:blue)
    GLMakie.scatter!(ax, @lift([0.0, 0.0] .+ $(XB[0][1])), @lift([0.0, 0.0] .+ $(XB[0][2])), rotations=@lift($(b_rot) .+ pi), marker=carsymbol, markersize=40, color=:red)

    for t in 1:T
        GLMakie.lines!(ax, @lift(circ_x .+ $(XA[t][1])), @lift(circ_y .+ $(XA[t][2])), color=:blue, linewidth=2, linestyle=:dash)
        GLMakie.lines!(ax, @lift(circ_x .+ $(XB[t][1])), @lift(circ_y .+ $(XB[t][2])), color=:red, linewidth=2, linestyle=:dash)
    end

    return (f, ax, XA, XB, lat)
end

function update_visual!(ax, XA, XB, x0, P1, P2; T=10, lat=6.0)
    XA[0][1][] = x0[1]
    XA[0][2][] = x0[2]
    XB[0][1][] = x0[5]
    XB[0][2][] = x0[6]

    for l in 1:T
        XA[l][1][] = P1[l, 1]
        XA[l][2][] = P1[l, 2]
        XB[l][1][] = P2[l, 1]
        XB[l][2][] = P2[l, 2]
    end

    GLMakie.xlims!(ax, -lat, lat)
    avg_y_pos = (x0[2] + x0[6]) / 2
    GLMakie.ylims!(ax, avg_y_pos - lat, avg_y_pos + 3 * lat)
end

function show_me(θ, x0; T=10, t=0, lat_pos_max=1.0)
    x_inds = 1:12*T
    function extract(θ; x_inds=x_inds, T=T)
        Z = θ[x_inds]
        @inbounds Xa = @view(Z[1:4*T])
        @inbounds Ua = @view(Z[4*T+1:6*T])
        @inbounds Xb = @view(Z[6*T+1:10*T])
        @inbounds Ub = @view(Z[10*T+1:12*T])
        (; Xa, Ua, Xb, Ub)
    end
    Z = extract(θ)

    (f, ax, XA, XB, lat) = visualize(; T=T, lat=lat_pos_max)
    display(f)

    P1 = [Z.Xa[1:4:end] Z.Xa[2:4:end] Z.Xa[3:4:end] Z.Xa[4:4:end]]
    U1 = [Z.Ua[1:2:end] Z.Ua[2:2:end]]
    P2 = [Z.Xb[1:4:end] Z.Xb[2:4:end] Z.Xb[3:4:end] Z.Xb[4:4:end]]
    U2 = [Z.Ub[1:2:end] Z.Ub[2:2:end]]

    update_visual!(ax, XA, XB, x0, P1, P2; T=T, lat=lat_pos_max)

    if t > 0
        ax.title = string(t)
    end
end