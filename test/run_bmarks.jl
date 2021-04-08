
using Pipal
using CUTEst
using SolverBenchmark
using NLPModelsIpopt, Percival
using Plots

# set of problems
probnames = CUTEst.select(min_var = 1,
                          max_var = 3,
                          min_con = 1,
                          max_con = 1)
problems = (CUTEstModel(probname) for probname in probnames)

# set of solvers
TOL = 1e-6
MAXITER = 3000

solvers = Dict(:pipal => nlp -> pipal(nlp, tol=TOL, max_iter=MAXITER),
               :percival => nlp -> percival(nlp, atol=TOL, rtol=0, ctol=TOL, max_iter=MAXITER),
               :ipopt => nlp -> ipopt(nlp, tol=TOL, max_iter=MAXITER, print_level=0),
               )

# warm up
nlp = CUTEstModel(probnames[rand(1:length(probnames))])
for solver ∈ keys(solvers)
  out = solvers[solver](nlp)
  print(out)
end
finalize(nlp)

# run benchmarks
stats = bmark_solvers(solvers, problems)

# print statistics
for solver ∈ keys(stats)
  @info "$solver statuses" count_unique(stats[solver].status)
end

# plot performance profiles
cost(df) = (df.status .!= :first_order) * Inf + df.elapsed_time
performance_profile(stats, cost)
