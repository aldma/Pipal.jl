module Pipal

export pipal

using NLPModels
using SolverCore

include("types.jl")
include("parameters.jl")
include("input.jl")
include("counters.jl")
include("iterate.jl")
include("direction.jl")
include("acceptance.jl")
include("output.jl")

"""
    pipal(nlp; kwargs...)

Implementation of a solver for nonlinear optimization problems of the form
	min   f(x)
	s.t.  cL ≤ c(x) ≤ cU,
		  xL ≤   x  ≤ xU,
where `f` and `c` are assumed to be continuously differentiable in `R^n`. It is
based on the penalty-interior-point method described in

    Frank E. Curtis,
    A Penalty-Interior-Point Algorithm for Nonlinear Constrained Optimization,
    Mathematical Programming Computation 4, 181--209, 2012.
    https://doi.org/10.1007/s12532-012-0041-4

Input:
- `nlp :: AbstractNLPModel`: Nonlinear model created using `NLPModels`.
The following keyword parameters can be passed:
- `algorithm :: Int` : PIPAL flavour: conservative => 0, adaptive => 1 (default: 1)
- `tol :: Real` : Tolerance used in assessing primal and dual feasibility (default: 1e-6)
- `max_iter :: Int` : Maximum number of iterations (default: 3000)
"""
function pipal(nlp::AbstractNLPModel; kwargs...)
    # set up
    p = Parameters(;kwargs...)
    i = Input(p, nlp)
    c = Counters()
    z = Iterate(p, i, c)
    d = Direction(i)
    a = Acceptance()
    #
    printBreak(p, c)
    # iteration loop
    while checkTermination(z, p, i, c) == 0
        printIterate(c, z)
        evalStep(d, p, i, c, z, a)
        #printDirection()
        lineSearch(a, p, i, c, z, d)
        #printAcceptance()
        updateIterate(z, p, i, c, d, a)
        incrementIterationCount(c)
        printBreak(p, c)
    end
    # output
    return getOutput(z, p, i, c)
end

"""
    collect data and prepare output structure
    getOutput(z, p, i, c)
"""
function getOutput(z::PIPAL_iterate, p::PIPAL_parameters, i::PIPAL_input, c::PIPAL_counters)
    # status
    b = checkTermination(z, p, i, c)
    status = if b == 1
        :first_order
    elseif b == 2 || b == 4
        :infeasible
    elseif b == 3
        :max_iter
    elseif b == 5
        :exception
    else
        :unknown
    end
    # output
    return GenericExecutionStats(
        status,
        i.nlp,
        solution = evalXOriginal(z, i),
        objective = z.fu,
        dual_feas = z.kkt[2],
        primal_feas = z.v,
        multipliers = evalLambdaOriginal(z, i),
        iter = c.k,
        elapsed_time = evalElapsedTime(c),
        solver_specific = Dict(:algorithm => p.algorithm),
    )
end

end # module
