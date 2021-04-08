
using Pipal
using CUTEst
using SolverBenchmark

# set of problems
probnames = CUTEst.select(min_var = 1,
                          max_var = 3,
                          min_con = 1,
                          max_con = 1)
problems = (CUTEstModel(probname) for probname in probnames)

# set of solvers
TOL = 1e-6
MAXITER = 3000

solver = nlp -> pipal(nlp, tol=TOL, max_iter=MAXITER)

# warm up
nlp = CUTEstModel(probnames[rand(1:length(probnames))])
out = pipal(nlp)
print(out)
finalize(nlp)

# run benchmarks
stats = solve_problems(solver, problems)

# print statistics
@info "PIPAL statuses" count_unique(stats.status)
