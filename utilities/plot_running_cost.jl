function get_mean_running_cost(results, i; T = 50)
    vals = [ Float64[] for _ in 1:T]
    for (index, c) in results[i].costs
        T = length(c.a.running.velocity)
        for t in 1:T
            push!(vals[t], c.a.running.velocity[t])# + c.a.running.velocity[t])
            # CI = 1.96*std(vals)/sqrt(length(vals));
        end
    end
    avgs = map(vals) do val
        mean(val)
    end
    stderrs = map(vals) do val
        1.96*std(val) / sqrt(length(val))
    end
    (avgs, stderrs)
end

avgs_1, stderrs_1 = get_mean_running_cost(results, 1)
avgs_3, stderrs_3 = get_mean_running_cost(results, 3)
avgs_9, stderrs_9 = get_mean_running_cost(results, 9)
avgs_6, stderrs_6 = get_mean_running_cost(results, 6)
avgs_10, stderrs_10 = get_mean_running_cost(results, 10)

#, yaxis=(formatter=y->string(round(Int, y / 10^-4)))
#, yaxis=(formatter=y->round(y; sigdigits=4)

#Plots.plot(layout=(2,1))

p = Plots.plot(avgs_3, ribbon = stderrs_3, fillalpha = 0.3, linewidth=3, label = "Nash competition (N-N)")
Plots.plot!(p, avgs_9, ribbon = stderrs_9, fillalpha = 0.3, linewidth=3, label = "Bilevel competition (L-F)")
annotate!([(3, 8.5e-3, Plots.text(L"\times10^{-3}", 12, :black, :center))])
Plots.plot!(p, size=(500,400), xlabel="Simulation steps", ylabel="Mean running cost per time step", yaxis=(formatter=y->round(y*10.0; sigdigits=4)))
savefig("./figures/plot_3_v_9_running_cost.pdf")