using Statistics

function process_steps(results, modes_sorted)
    steps_table_old = Dict()

    for mode in modes_sorted
        res = results[mode]
        inds = res.costs |> keys |> collect |> sort
		a_steps = [res.steps[i] for i in inds]
		b_steps = [res.steps[i] for i in inds]
        steps_table_old[mode, "a"] = a_steps
        steps_table_old[mode, "b"] = b_steps
    end

    full_steps_table = Dict()
    full_steps_table["S", "S"] = steps_table_old[1, "a"]
    full_steps_table["S", "N"] = steps_table_old[2, "a"]
    full_steps_table["S", "L"] = steps_table_old[4, "a"]
    full_steps_table["S", "F"] = steps_table_old[7, "a"]
    full_steps_table["N", "S"] = steps_table_old[2, "b"]
    full_steps_table["N", "N"] = steps_table_old[3, "a"]
    full_steps_table["N", "L"] = steps_table_old[5, "a"]
    full_steps_table["N", "F"] = steps_table_old[8, "a"]
    full_steps_table["L", "S"] = steps_table_old[4, "b"]
    full_steps_table["L", "N"] = steps_table_old[5, "b"]
    full_steps_table["L", "L"] = steps_table_old[6, "a"]
    full_steps_table["L", "F"] = steps_table_old[9, "a"]
    full_steps_table["F", "S"] = steps_table_old[7, "b"]
    full_steps_table["F", "N"] = steps_table_old[8, "b"]
    full_steps_table["F", "L"] = steps_table_old[9, "b"]
    full_steps_table["F", "F"] = steps_table_old[10, "a"]
    #display(cost_table)

    compressed_table = Dict()
    for strat in ["S", "N", "L", "F"]
        #@infiltrate
        compressed_table[strat] = (full_steps_table[strat, "S"] + full_steps_table[strat, "N"] + full_steps_table[strat, "F"] + full_steps_table[strat, "L"])/4
    end
	(;full=full_steps_table, compressed=compressed_table)
end

function process_costs(results, modes_sorted; property=:total)
    cost_table_old = Dict()

    for mode in modes_sorted
        res = results[mode]
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

    full_table = Dict()
    full_table["S", "S"] = cost_table_old[1, "a"]
    full_table["S", "N"] = cost_table_old[2, "a"]
    full_table["S", "L"] = cost_table_old[4, "a"]
    full_table["S", "F"] = cost_table_old[7, "a"]
    full_table["N", "S"] = cost_table_old[2, "b"]
    full_table["N", "N"] = cost_table_old[3, "a"]
    full_table["N", "L"] = cost_table_old[5, "a"]
    full_table["N", "F"] = cost_table_old[8, "a"]
    full_table["L", "S"] = cost_table_old[4, "b"]
    full_table["L", "N"] = cost_table_old[5, "b"]
    full_table["L", "L"] = cost_table_old[6, "a"]
    full_table["L", "F"] = cost_table_old[9, "a"]
    full_table["F", "S"] = cost_table_old[7, "b"]
    full_table["F", "N"] = cost_table_old[8, "b"]
    full_table["F", "L"] = cost_table_old[9, "b"]
    full_table["F", "F"] = cost_table_old[10, "a"]
    #display(cost_table)

    compressed_table = Dict()
    for strat in ["S", "N", "L", "F"]
        compressed_table[strat] = (full_table[strat, "S"] + full_table[strat, "N"] + full_table[strat, "F"] + full_table[strat, "L"]) / 4
    end
    (; full=full_table, compressed=compressed_table)
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


modes_sorted = sort(collect(keys(results)))
steps_table = process_steps(results, modes_sorted)
total_cost_table = process_costs(results, modes_sorted, property=:total)
lane_cost_table = process_costs(results, modes_sorted, property=:lane)
control_cost_table = process_costs(results, modes_sorted, property=:control)
velocity_cost_table = process_costs(results, modes_sorted, property=:velocity)

println("		mean (±95% CI) [95% CI l, u]	std	min	max")

println("Steps:")
for (k, v) in steps_table.compressed
    print_mean_etc(v; title=k, scale=1)
end

println("Total:")
for (k, v) in total_cost_table.compressed
    print_mean_etc(v; title=k, scale=10)
end

#println("Lane:")
#for (k, v) in lane_cost_table.compressed
#    print_mean_etc(v; title=k, scale=100)
#end

#println("Control:")
#for (k, v) in control_cost_table.compressed
#    print_mean_etc(v; title=k, scale=100)
#end

#println("Velocity:")
#for (k, v) in velocity_cost_table.compressed
#    print_mean_etc(v; title=k, scale=100)
#end

#println("Steps:")
#for (k, v) in steps_table.full
#    print_mean_etc(v; title=k, scale=1)
#end

println("Total:")
for (k, v) in total_cost_table.full
    print_mean_etc(v; title=k, scale=10)
end

println("Lane:")
for (k, v) in lane_cost_table.full
    print_mean_etc(v; title=k, scale=10)
end

println("Control:")
for (k, v) in control_cost_table.full
    print_mean_etc(v; title=k, scale=10)
end

println("Velocity:")
for (k, v) in velocity_cost_table.full
    print_mean_etc(v; title=k, scale=10)
end

#function print_mean_min_max(Δcost)
#    println("		mean		stderrmin			max")
#    println("P1 Δcost abs :  $(mean(Δcost.P1_abs))  $(std(Δcost.P1_abs)/sqrt(length(Δcost.P1_abs)))  $(minimum(Δcost.P1_abs))  $(maximum(Δcost.P1_abs))")
#    println("P2 Δcost abs :  $(mean(Δcost.P2_abs))  $(std(Δcost.P2_abs)/sqrt(length(Δcost.P2_abs)))  $(minimum(Δcost.P2_abs))  $(maximum(Δcost.P2_abs))")
#    println("P1 Δcost rel%:  $(mean(Δcost.P1_rel) * 100)  $(std(Δcost.P1_rel)/sqrt(length(Δcost.P1_rel)) * 100)  $(minimum(Δcost.P1_rel) * 100)  $(maximum(Δcost.P1_rel) * 100)")
#    println("P2 Δcost rel%:  $(mean(Δcost.P2_rel) * 100)  $(std(Δcost.P2_rel)/sqrt(length(Δcost.P2_rel)) * 100)  $(minimum(Δcost.P2_rel) * 100)  $(maximum(Δcost.P2_rel) * 100)")
#end

#@info "terminal cost mean CI min max"
#for mode in modes_sorted
#	res = results[mode]
#	inds = res.costs |> keys |> collect |> sort
#	b_steps = [res.steps[i] for i in inds]
#	b_total_costs = [res.costs[i].b.final.terminal for i in inds]
#	a_steps = [res.steps[i] for i in inds]
#	a_total_costs = [res.costs[i].a.final.terminal for i in inds]
#	print_mean_min_max(a_total_costs ./ a_steps; title="mode $(mode) a",scale=100)
#	print_mean_min_max(b_total_costs ./ b_steps; title="mode $(mode) b",scale=100)
#end

#@info "velocity cost mean CI min max"
#for mode in modes_sorted
#	res = results[mode]
#	inds = res.costs |> keys |> collect |> sort
#	b_steps = [res.steps[i] for i in inds]
#	b_total_costs = [res.costs[i].b.final.velocity for i in inds]
#	a_steps = [res.steps[i] for i in inds]
#	a_total_costs = [res.costs[i].a.final.velocity for i in inds]
#	print_mean_min_max(a_total_costs ./ a_steps; title="mode $(mode) a",scale=100)
#	print_mean_min_max(b_total_costs ./ b_steps; title="mode $(mode) b",scale=100)
#end

#@info "control cost mean CI min max"
#for mode in modes_sorted
#	res = results[mode]
#	inds = res.costs |> keys |> collect |> sort
#	b_steps = [res.steps[i] for i in inds]
#	b_total_costs = [res.costs[i].b.final.control for i in inds]
#	a_steps = [res.steps[i] for i in inds]
#	a_total_costs = [res.costs[i].a.final.control for i in inds]
#	print_mean_min_max(a_total_costs ./ a_steps; title="mode $(mode) a",scale=100)
#	print_mean_min_max(b_total_costs ./ b_steps; title="mode $(mode) b",scale=100)
#end
