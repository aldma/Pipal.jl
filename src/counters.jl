"""
    constructor
    c = Counters()
"""
function Counters()
    return PIPAL_counters(0, 0, 0, 0, 0, time())
end

"""
    Function evaluation counter incrementor
"""
function incrementFunctionCount(c::PIPAL_counters)
    c.f += 1
end

"""
    Gradient evaluation counter incrementor
"""
function incrementGradientCount(c::PIPAL_counters)
    c.g += 1
end

"""
    Hessian evaluation counter incrementor
"""
function incrementHessianCount(c::PIPAL_counters)
    c.H += 1
end

"""
    Matrix factorization counter incrementor
"""
function incrementFactorizationCount(c::PIPAL_counters)
    c.M += 1
end

"""
    Iteration counter incrementor
"""
function incrementIterationCount(c::PIPAL_counters)
    c.k += 1
end

"""
    Elapsed time evaluator
"""
function evalElapsedTime(c::PIPAL_counters)
    return time() - c.t0
end
