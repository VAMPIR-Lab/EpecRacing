compressed_steps_table = Dict()
for strat in ["S", "N", "L", "F"]
    #@infiltrate
    compressed_steps_table[strat] = (steps_table[strat, "S"] + steps_table[strat, "N"] + teps_table[strat, "F"] + steps_table[strat, "L"]) / 4
end
compressed_cost_table = Dict()
for strat in ["S", "N", "L", "F"]
    compressed_cost_table[strat] = (total_cost_table[strat, "S"] + total_cost_table[strat, "N"] + total_cost_table[strat, "F"] + total_cost_table[strat, "L"]) / 4
end

println("Steps:")
for (k, v) in steps_table.compressed
    print_mean_etc(v; title=k, scale=1)
end

println("Total:")
for (k, v) in total_cost_table.compressed
    print_mean_etc(v; title=k, scale=10)
end