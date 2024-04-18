module EPEC

using LinearAlgebra
using PATHSolver
using SparseArrays
using Symbolics

include("problems.jl")

export OptimizationProblem, create_epec, solve

end # module EPEC
