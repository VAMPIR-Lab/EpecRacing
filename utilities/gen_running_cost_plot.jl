using StatsPlots
using LaTeXStrings

#avgs_1, stderrs_1 = get_mean_running_vel_cost(processed_results, 1)
avgs_3, stderrs_3 = get_mean_running_vel_cost(processed_results, 3; time_steps)
avgs_9, stderrs_9 = get_mean_running_vel_cost(processed_results, 9; time_steps)
#avgs_6, stderrs_6 = get_mean_running_vel_cost(processed_results, 6)
#avgs_10, stderrs_10 = get_mean_running_vel_cost(processed_results, 10)

#, yaxis=(formatter=y->string(round(Int, y / 10^-4)))
#, yaxis=(formatter=y->round(y; sigdigits=4)

#Plots.plot(layout=(2,1))

p = Plots.plot(avgs_3, ribbon = stderrs_3, fillalpha = 0.3, linewidth=3, label = "Nash competition (N-N)")
Plots.plot!(p, avgs_9, ribbon = stderrs_9, fillalpha = 0.3, linewidth=3, label = "Bilevel competition (L-F)")
annotate!([(3, 8.5e-3, Plots.text(L"\times10^{-3}", 12, :black, :center))])
Plots.plot!(p, size=(500,400), xlabel="Simulation steps", ylabel="Mean running cost per time step", yaxis=(formatter=y->round(y*10.0; sigdigits=4)))
savefig("./output/plot_3_v_9_running_cost.pdf")