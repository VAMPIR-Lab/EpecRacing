steps_table_compressed = Dict()
for strat in ["S", "N", "L", "F"]
    #@infiltrate
    steps_table_compressed[strat] = (steps_named[strat, "S"] + steps_named[strat, "N"] + steps_named[strat, "F"] + steps_named[strat, "L"]) / 4
end
a_cost_table_compressed = Dict()
for strat in ["S", "N", "L", "F"]
    a_cost_table_compressed[strat] = (total_named[strat, "S", "a"] + total_named[strat, "N", "a"] + total_named[strat, "F", "a"] + total_named[strat, "L", "a"]) / 4
end

b_cost_table_compressed = Dict()
for strat in ["S", "N", "L", "F"]
    b_cost_table_compressed[strat] = (total_named[strat, "S", "b"] + total_named[strat, "N", "b"] + total_named[strat, "F", "b"] + total_named[strat, "L", "b"]) / 4
end

println("steps compressed:")
for (k, v) in steps_table_compressed
    print_mean_etc(v; title=k, scale=1)
end

println("a costs compressed:")
for (k, v) in a_cost_table_compressed
    print_mean_etc(v; title=k, scale=1)
end

println("b costs compressed:")
for (k, v) in b_cost_table_compressed
    print_mean_etc(v; title=k, scale=1)
end