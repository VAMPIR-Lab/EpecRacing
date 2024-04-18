struct OptimizationProblem
    n::Int # domain dimension
    dvars::Vector{Int}
    f::Function
    g::Function
    l::Vector{Float64}
    u::Vector{Float64}
end

function Base.hvcat(blocks_per_row::Tuple{Vararg{Int}}, OPs::OptimizationProblem...)
    create_epec(blocks_per_row, OPs...; use_z_slacks=false)
end

function Base.vcat(OPs::OptimizationProblem...)
    length(OPs) == 0 && error("Can't create equilibrium problem from no optimization problems")
    length(OPs) == 1 && return create_epec((1, 0), OPs...; use_z_slacks=false)
    length(OPs) == 2 && return create_epec((1, 1), OPs...; use_z_slacks=false)
    length(OPs) == 3 && error("L-level optimization problems are not supported for L > 2")
end

function Base.hcat(OPs::OptimizationProblem...)
    create_epec((length(OPs), 0), OPs...; use_z_slacks=false)
end

function create_epec(players_per_row::Tuple{Vararg{Int}}, OPs::OptimizationProblem...; use_z_slacks=false)
    @assert length(players_per_row) == 2
    @assert allequal(OP.n for OP in OPs)
    N_top, N_low = players_per_row
    N_all = N_top + N_low
    @assert N_all == length(OPs)

    top_OPs = OPs[1:N_top]
    low_OPs = OPs[N_top+1:end]

    n_privates = [length(OP.dvars) for OP in OPs]
    m_privates = [length(OP.l) for OP in OPs]

    # n = sum(n_privates) + n_param 
    n = first(OPs).n
    n_param = n - sum(n_privates)

    dim_z_low = sum(n_privates[i] + 2 * m_privates[i] for i in N_top+1:N_all; init=0)
    dim_total = sum(n_privates[i] + 2 * m_privates[i] + 4 * dim_z_low for i in 1:N_top; init=0)

    # Each top-level player: privates + duals on private cons + slacks on
    # private cons + slacks on z cons + duals on z cons + duals on z agreement
    # + copy of z
    #
    # θ := [x₁ ... xₙ | z₁ ... zₙ | λ₁ ... λₙ | s₁ ... sₙ | ψ₁ ... ψₙ | r₁ ... rₙ | γ₁ ... γₙ] 
    # For n players: 
    # 1. x₁ ... xₙ: privates
    # 2. z₁ ... zₙ: copy of z
    # 3. λ₁ ... λₙ: duals on private cons
    # 4. s₁ ... sₙ: slacks on private cons
    # 5. ψ₁ ... ψₙ: slacks on z cons
    # 6, r₁ ... rₙ: duals on z cons
    # 7. γ₁ ... γₙ: duals on z agreement
    θall = Symbolics.@variables(θ[1:dim_total+n_param])[1] |> Symbolics.scalarize
    θ = θall[1:dim_total]
    w = θall[dim_total+1:end]

    vars = Dict()
    inds = Dict()

    ind = 0
    for (key_base, len_itr) in zip(["x", "z", "λ", "s", "ψ", "r", "γ"],
        [n_privates, dim_z_low, m_privates, m_privates, dim_z_low, dim_z_low, dim_z_low])
        lens = length(len_itr) == 1 ? fill(len_itr[1], N_top) : len_itr
        for i in 1:N_top
            len = lens[i]
            vars[key_base, i] = θ[ind+1:ind+len]
            inds[key_base, i] = ind+1:ind+len
            ind += len
        end
    end

    z = vars["z", 1] # any player's copy of z would do (i.e. z2, z3, ..., or zₙ)
    ind = 0
    for (key_base, len_itr) in zip(["x", "λ", "s"], [n_privates, m_privates, m_privates])
        for i in N_top+1:N_all
            len = len_itr[i]
            vars[key_base, i] = z[ind+1:ind+len]
            inds[key_base, i] = inds["z", 1][ind+1:ind+len]
            ind += len
        end
    end
    inds["w", 0] = dim_total+1:dim_total+n_param

    x = vcat((vars["x", i] for i in 1:N_all)...)
    x_inds = vcat((inds["x", i] for i in 1:N_all)...)
    x_w = [x; w]
    x_w_inds = [x_inds; inds["w", 0]]

    # construct F for low-level MCP
    grad_lags = mapreduce(vcat, N_top+1:N_all; init=Num[]) do i
        Lag = OPs[i].f(x_w) - vars["λ", i]' * OPs[i].g(x_w)
        grad_lag = Symbolics.gradient(Lag, vars["x", i])
    end
    cons_s = mapreduce(vcat, N_top+1:N_all; init=Num[]) do i
        OPs[i].g(x_w) - vars["s", i]
    end
    λs = vcat((vars["λ", i] for i in N_top+1:N_all)...)

    F = Num[grad_lags; cons_s; λs]

    F! = Symbolics.build_function(F, θall; expression=Val(false))[2]
    J = Symbolics.sparsejacobian(F, z)
    (J_rows, J_cols, J_vals) = findnz(J)
    J_vals! = Symbolics.build_function(J_vals, θall; expression=Val(false))[2]

    l = fill(-Inf, length(grad_lags) + length(cons_s))
    u = fill(+Inf, length(grad_lags) + length(cons_s))
    for i in N_top+1:N_all
        append!(l, OPs[i].l)
        append!(u, OPs[i].u)
    end

    low_level = (; F!, J_rows, J_cols, J_vals!, z_inds=inds["z", 1], l, u, n=length(l))

    # reminder :  θ := [x₁ ... xₙ₁ | z₁ ... zₙ | λ₁ ... λₙ | s₁ ... sₙ | ψ₁ ... ψₙ | r₁ ... rₙ | γ₁ ... γₙ] 

    grad_lags_x = mapreduce(vcat, 1:N_top) do i
        Lag = OPs[i].f(x_w) - vars["λ", i]' * OPs[i].g(x_w) - vars["ψ", i]' * F
        grad_lag = Symbolics.gradient(Lag, vars["x", i])
    end
    grad_lags_z = mapreduce(vcat, 1:N_top) do i
        Lag = OPs[i].f(x_w) - vars["λ", i]' * OPs[i].g(x_w) - vars["ψ", i]' * F #- vars["γ", i]'*vars["z", 1]
        if use_z_slacks
            Lag -= vars["γ", i]' * vars["z", 1]
        end
        grad_lag = Symbolics.gradient(Lag, vars["z", 1])
    end
    cons_s_top = mapreduce(vcat, 1:N_top) do i
        OPs[i].g(x_w) - vars["s", i]
    end
    λs_top = vcat((vars["λ", i] for i in 1:N_top)...)
    cons_r = mapreduce(vcat, 1:N_top) do i
        F - vars["r", i]
    end
    ψs = vcat((vars["ψ", i] for i in 1:N_top)...)
    cons_z = mapreduce(vcat, 1:(N_top-1); init=Num[]) do i
        -vars["z", i+1] + vars["z", i]
    end
    append!(cons_z, sum(vars["γ", i] for i in 1:N_top))
    #append!(cons_z, -vars["z",1] + vars["z", N1])

    Ftotal = [grad_lags_x; grad_lags_z; cons_s_top; λs_top; cons_r; ψs; cons_z]
    ltotal = [fill(-Inf, length(grad_lags_x))
        fill(-Inf, length(grad_lags_z)) # will get overwritten by templates
        fill(-Inf, length(cons_s_top))
        vcat((OPs[i].l for i in 1:N_top)...)
        fill(-Inf, length(cons_r))
        fill(-Inf, length(ψs)) # will get overwritten by templates
        fill(-Inf, length(cons_z))]
    utotal = [fill(+Inf, length(grad_lags_x))
        fill(+Inf, length(grad_lags_z)) # will get overwritten by templates
        fill(+Inf, length(cons_s_top))
        vcat((OPs[i].u for i in 1:N_top)...)
        fill(+Inf, length(cons_r))
        fill(+Inf, length(ψs)) # will get overwritten by templates
        fill(+Inf, length(cons_z))]

    # these are needed for assigning bounds from low-level solutions
    ind = 0
    base = length(grad_lags_x)
    z_inds_top = map(1:N_top) do i
        local_inds = ((ind+1):(ind+dim_z_low)) .+ base
        ind += dim_z_low
        local_inds
    end
    ind = 0
    base = length(grad_lags_x) + length(grad_lags_z) + length(cons_s_top) + length(λs_top) + length(cons_r)
    r_inds_top = map(1:N_top) do i
        local_inds = ((ind+1):(ind+dim_z_low)) .+ base
        ind += dim_z_low
        local_inds
    end

    f_dict = Dict()
    idx = 0
    for (item, ame) in zip([grad_lags_x, grad_lags_z, cons_s_top, λs_top, cons_r, ψs, cons_z], ["grad_lags_x", "grad_lags_z", "cons_s_top", "λs_top", "cons_r", "ψs", "cons_z"])
        f_dict[ame] = (idx+1):(idx+length(item))
        idx += length(item)
    end

    Ftotal! = Symbolics.build_function(Ftotal, θall; expression=Val(false))[2]
    Jtotal = Symbolics.sparsejacobian(Ftotal, θ)
    (Jtotal_rows, Jtotal_cols, Jtotal_vals) = findnz(Jtotal)
    Jtotal_vals! = Symbolics.build_function(Jtotal_vals, θall; expression=Val(false))[2]

    top_level = (; (F!)=Ftotal!,
        J_rows=Jtotal_rows,
        J_cols=Jtotal_cols,
        (J_vals!)=Jtotal_vals!,
        z_inds=z_inds_top,
        r_inds=r_inds_top,
        l=ltotal,
        u=utotal,
        n=length(ltotal),
        n_param)

    (; low_level, top_level, x_inds, inds, f_dict, OPs)
end

function solve(epec; tol=1e-6)
    solve(epec, zeros(epec.top_level.n); tol)
end

function solve(epec, θ; tol=1e-6, max_iters=30)
    low_level = epec.low_level
    top_level = epec.top_level

    iters = 0
    converged = false
    while !converged
        iters += 1

        (; status, info) = solve_low_level!(low_level, θ) # this should be redundant after the initial iteration
        solution_graph = get_local_solution_graph(low_level, θ)
        converged = true
        errored = false
        #@info "Solution graph has $(length(solution_graph)) pieces."
        for S in solution_graph
            bounds = convert_recipe(low_level, S)
            #try
            (; dθ, status, info) = solve_top_level(top_level, bounds, θ, epec.x_inds, epec.inds, epec.f_dict)
            if iters > max_iters
                #@infiltrate
                throw(error("Solver failure"))
                return
            end
            if (norm(dθ) > tol)
                converged = false
                θ += dθ
                break
            end
            #catch e
            #    println(e)
            #    errored = true
            #end
        end
        if errored
            error("One or more of the subpieces resulted in no solution.")
        end
    end
    return θ
end

function solve_top_level(mcp, bounds, θ, x_inds, inds, f_dict; silent=true)
    n = mcp.n
    nnz_total = length(mcp.J_rows)
    J_shape = sparse(mcp.J_rows, mcp.J_cols, ones(Cdouble, nnz_total), n, n)
    J_col = J_shape.colptr[1:end-1]
    J_len = diff(J_shape.colptr)
    J_row = J_shape.rowval
    @assert length(J_col) == n
    @assert length(J_len) == n
    @assert length(J_row) == nnz_total

    θF = copy(θ)
    x = θ[1:n] # WARNING assumes that first n elements are non-parameter values.
    w = θ[n+1:end]
    #@infiltrate

    function F(n, θ, result)
        result .= 0.0
        θF[1:n] .= θ
        mcp.F!(result, θF)
        Cint(0)
    end
    function J(n, nnz, θ, col, len, row, data)
        @assert nnz == nnz_total == length(data) == length(row)
        data .= 0.0
        θF[1:n] .= θ
        mcp.J_vals!(data, θF)
        col .= J_col
        len .= J_len
        row .= J_row
        Cint(0)
    end

    l = mcp.l
    u = mcp.u

    for ind_set in mcp.z_inds
        l[ind_set] .= bounds.lz
        u[ind_set] .= bounds.uz
    end
    for ind_set in mcp.r_inds
        l[ind_set] .= bounds.lf
        u[ind_set] .= bounds.uf
    end

    f = zeros(n)
    F(n, x, f)
    already_solved = check_mcp_sol(f, x, l, u)
    if already_solved
        return (; dθ=zero(θF), status=:success, info="problem solved at initialization")
    end

    PATHSolver.c_api_License_SetString("2830898829&Courtesy&&&USR&45321&5_1_2021&1000&PATH&GEN&31_12_2025&0_0_0&6000&0_0")
    status, θ_out, info = PATHSolver.solve_mcp(
        F,
        J,
        l,
        u,
        x;
        silent,
        nnz=nnz_total,
        jacobian_structure_constant=true,
        jacobian_data_contiguous=true,
        cumulative_iteration_limit=50_000,
        convergence_tolerance=1e-7,
        lemke_rank_deficiency_iterations=100 # fixes silent crashes " ** SOLVER ERROR ** Lemke: invertible basis could not be computed."
    )

    if status != PATHSolver.MCP_Solved
        #@infiltrate
        throw(error("Top-level Solver failure"))
    end

    θF[1:n] .= θ_out

    dθ = θF - θ
    (; dθ, status, info)
end


function solve_low_level!(mcp, θ; silent=true)
    n = length(mcp.l)
    n == 0 && return (; status=:success, info="problem of zero dimension")
    θF = copy(θ)
    nnz_total = length(mcp.J_rows)
    J_shape = sparse(mcp.J_rows, mcp.J_cols, Vector{Cdouble}(undef, nnz_total), n, n)
    J_col = J_shape.colptr[1:end-1]
    J_len = diff(J_shape.colptr)
    J_row = J_shape.rowval
    z = θF[mcp.z_inds]

    function F(n, z, result)
        result .= 0.0
        θF[mcp.z_inds] .= z
        mcp.F!(result, θF)
        Cint(0)
    end
    function J(n, nnz, z, col, len, row, data)
        @assert nnz == nnz_total
        θF[mcp.z_inds] .= z
        data .= 0.0
        mcp.J_vals!(data, θF)
        col .= J_col
        len .= J_len
        row .= J_row
        Cint(0)
    end
    f = zero(z)
    F(n, z, f)
    already_solved = check_mcp_sol(f, z, mcp.l, mcp.u)
    if already_solved
        return (; status=:success, info="problem solved at initialization")
    end

    PATHSolver.c_api_License_SetString("2830898829&Courtesy&&&USR&45321&5_1_2021&1000&PATH&GEN&31_12_2025&0_0_0&6000&0_0")
    status, z_out, info = PATHSolver.solve_mcp(
        F,
        J,
        mcp.l,
        mcp.u,
        z;
        silent=true,
        nnz=nnz_total,
        jacobian_structure_constant=true,
        jacobian_data_contiguous=true,
        cumulative_iteration_limit=50_000,
        convergence_tolerance=1e-7,
        lemke_rank_deficiency_iterations=100,
        preprocess=0
    )

    #if status != PATHSolver.MCP_Solved && silent
    #    return solve_low_level!(mcp, θ; silent=false)
    #end

    if status != PATHSolver.MCP_Solved
        #@infiltrate
        throw(error("Low-level Solver failure"))
    end
    #@infiltrate status != PATHSolver.MCP_Solved

    θ[mcp.z_inds] .= z_out

    (; status, info)
end

#f_out = zeros(n);    F(n, θ_out, f_out);
function check_mcp_fail(f, z, l, u; tol=1e-6)
    n = length(f)
    fails = falses(n)
    for i in 1:n
        if isapprox(l[i], u[i]; atol=tol)
            continue
        elseif f[i] ≥ tol && z[i] < l[i] + tol
            continue
        elseif -tol < f[i] < tol && z[i] < l[i] + tol
            continue
        elseif -tol < f[i] < tol && l[i] + tol ≤ z[i] ≤ u[i] - tol
            continue
        elseif -tol < f[i] < tol && z[i] > u[i] - tol
            continue
        elseif f[i] ≤ -tol && z[i] > u[i] - tol
            continue
        else
            fails[i] = true
        end
    end
    return findall(fails)
end

function check_mcp_sol(f, z, l, u; tol=1e-6)
    n = length(f)
    sol = true
    for i in 1:n
        if isapprox(l[i], u[i]; atol=tol)
            continue
        elseif f[i] ≥ tol && z[i] < l[i] + tol
            continue
        elseif -tol < f[i] < tol && z[i] < l[i] + tol
            continue
        elseif -tol < f[i] < tol && l[i] + tol ≤ z[i] ≤ u[i] - tol
            continue
        elseif -tol < f[i] < tol && z[i] > u[i] - tol
            continue
        elseif f[i] ≤ -tol && z[i] > u[i] - tol
            continue
        else
            sol = false
            break
        end
    end
    return sol
end

function get_local_solution_graph(mcp, θ; tol=1e-6)
    l = mcp.l
    u = mcp.u
    n = length(l)
    f = zeros(n)
    mcp.F!(f, θ)
    z = @view θ[mcp.z_inds]
    J = Dict{Int,Vector{Int}}()
    for i in 1:n
        Ji = Int[]
        if isapprox(l[i], u[i]; atol=2 * tol)
            push!(Ji, 4)
        elseif f[i] ≥ tol && z[i] < l[i] + tol
            push!(Ji, 1)
        elseif -tol < f[i] < tol && z[i] < l[i] + tol
            push!(Ji, 1)
            push!(Ji, 2)
        elseif -tol < f[i] < tol && l[i] + tol ≤ z[i] ≤ u[i] - tol
            push!(Ji, 2)
        elseif -tol < f[i] < tol && z[i] > u[i] - tol
            push!(Ji, 2)
            push!(Ji, 3)
        elseif f[i] ≤ -tol && z[i] > u[i] - tol
            push!(Ji, 3)
        end
        J[i] = Ji
    end
    valid_solution = !any(isempty.(Ji for Ji in values(J)))
    !valid_solution && begin
        #@infiltrate
        error("Not a valid solution!")
    end
    recipes = get_all_recipes(J)
end

function get_all_recipes(J)
    Ks = Vector{Dict{Int,Set{Int}}}()
    N = length(J)
    multiples = [i for i in 1:N if length(J[i]) > 1]
    singles = setdiff(1:N, multiples)
    It = Iterators.product([J[i] for i in multiples]...)
    for assignment in It
        K = Dict(j => Set{Int}() for j in 1:4)
        for (e, ej) in enumerate(assignment)
            push!(K[ej], multiples[e])
        end
        for e in singles
            push!(K[J[e][1]], e)
        end
        push!(Ks, K)
    end
    Ks
end

function convert_recipe(mcp, recipe)
    K = recipe
    n = length(mcp.l)

    lf = zeros(n)
    uf = zeros(n)
    lz = zeros(n)
    uz = zeros(n)

    for i in K[1]
        lf[i] = 0.0
        uf[i] = Inf
        lz[i] = mcp.l[i]
        uz[i] = mcp.l[i]
    end
    for i in K[2]
        lf[i] = 0.0
        uf[i] = 0.0
        lz[i] = mcp.l[i]
        uz[i] = mcp.u[i]
    end
    for i in K[3]
        lf[i] = -Inf
        uf[i] = 0.0
        lz[i] = mcp.u[i]
        uz[i] = mcp.u[i]
    end
    for i in K[4]
        lf[i] = -Inf
        uf[i] = Inf
        lz[i] = mcp.l[i]
        uz[i] = mcp.u[i]
    end
    (; lf, uf, lz, uz)
end
