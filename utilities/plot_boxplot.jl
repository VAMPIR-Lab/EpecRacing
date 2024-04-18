using StatsPlots
using LaTeXStrings

#p = Plots.boxplot(["S" "N" "L" "F"], [total_cost_table.compressed["S"], total_cost_table.compressed["N"], total_cost_table.compressed["L"], total_cost_table.compressed["F"]], legend=false)

p = Plots.boxplot(["Nash competition, N-N" "Bilevel competition, L-F "], [
        velocity_cost_table.full["N", "N"],
        velocity_cost_table.full["L", "F"]]
    #total_cost_table.full["F", "F"],
    #total_cost_table.full["L", "L"]]
    , legend=false, outliers=false)
annotate!([(0.2, 9e-2, Plots.text(L"\times10^{-3}", 12, :black, :center))])
Plots.plot!(p, size=(500, 400), xlabel="Competition type", ylabel="Mean running cost per time step", yaxis=(formatter = y -> round(y * 10.0; sigdigits=4)))
savefig("./figures/boxplot_running_cost.pdf")