using StatsPlots
using LaTeXStrings

#p = Plots.boxplot(["S" "N" "L" "F"], [total_cost_table.compressed["S"], total_cost_table.compressed["N"], total_cost_table.compressed["L"], total_cost_table.compressed["F"]], legend=false)

p = Plots.boxplot(["SP competition, S-S" "Nash competition, N-N" "Bilevel competition, L-F "],
    [
        velocity_named["S", "S", "a"],
        velocity_named["N", "N", "a"],
        velocity_named["L", "F", "a"]
    ],
    #[
    #    total_cost_table["F", "F"],
    #    total_cost_table["L", "L"]
    #], 
    legend=false, outliers=false)
#annotate!([(0.25, 1.1, Plots.text(L"\times10^{-2}", 12, :black, :center))])
Plots.plot!(p, size=(500, 400), xlabel="Competition type", ylabel="Mean velocity cost per simulation step", yaxis=(formatter = y -> round(y; sigdigits=4)))
savefig("./output/boxplot_running_cost.pdf")
#savefig("./output/boxplot_running_cost_$(date_now).pdf")