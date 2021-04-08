using LinearAlgebra
using SparseArrays
using LDLFactorizations

"""
    constructor
    z = Iterate(p, i)
"""
function Iterate(p::PIPAL_parameters, i::PIPAL_input, c::PIPAL_counters)
    T = eltype(i.x0)
    # initialize quantities
    z = PIPAL_iterate(
        copy(i.x0), # x
        p.rho_init, # rho
        p.rho_init, # rho_
        p.mu_init, # mu
        zero(T), # f
        zero(T), # fu
        zeros(T, i.nV), # g
        zeros(T, i.nE), # r1
        zeros(T, i.nE), # r2
        zeros(T, i.nE), # cE
        spzeros(T, i.nE, i.nV), # JE
        0, # JEnnz
        zeros(T, i.nE), # lE
        zeros(T, i.nI), # s1
        zeros(T, i.nI), # s2
        zeros(T, i.nI), # cI
        spzeros(T, i.nI, i.nV), # JI
        0, # JInz
        0.5 * ones(T, i.nI), # lI
        spzeros(T, i.nV, i.nV), # H
        0, # Hnnz
        zero(T), # v
        zero(T), # vu
        one(T), # v0
        zero(T), # phi
        ldl(spzeros(T, i.nA, i.nA)), # Af
        0, # Annz
        zero(T), # shift
        zeros(T, i.nA), # b
        zeros(T, 3), # kkt
        Inf * ones(T, p.opt_err_mem), # kkt_
        0, # err
        one(T), # fs
        ones(T, i.nE), # cEs
        zeros(T, i.nE), # cEu
        ones(T, i.nI), # cIs
        zeros(T, i.nI), # cIu
        spzeros(T, i.nA, i.nA), # A
        zero(T), # shift22
        zero(T), # v_
        false, # cut_
    )
    evalScalings(z, p, i, c)
    evalFunctions(z, i, c)
    evalGradients(z, i, c)
    evalDependent(z, p, i)
    z.v0 = one(T)
    evalInfeasibility(z, i)
    z.v0 = z.v
    evalInfeasibility(z, i)
    z.v_ = z.v
    evalHessian(z, i, c)
    z.Hnnz = nnz(z.H)
    z.JEnnz = nnz(z.JE)
    z.JInnz = nnz(z.JI)
    initNewtonMatrix(z, i)
    evalNewtonMatrix(z, p, i, c)
    return z
end

"""
    Function evaluator
    evalFunctions(z, i, c)
"""
function evalFunctions(z::PIPAL_iterate, i::PIPAL_input, c::PIPAL_counters)
    T = eltype(i.x0)
    # Evaluate x in original space
    x_orig = evalXOriginal(z, i)
    # Initialize/Reset evaluation flag
    z.err = 0
    # Increment function evaluation counter
    incrementFunctionCount(c)
    # Try functions evaluation
    # TODO: catch evaluation errors
    # c_orig = zeros(T, i.nE+i.nI)
    # try
    # Evaluate functions
    z.f = obj(i, x_orig)
    c_orig = cons(i, x_orig)
    # catch
    #     # Set evaluation flag, default values, and return
    #     z.err = 1
    #     z.f = NaN
    #     z.cE .= NaN
    #     z.cI .= NaN
    #     z.fu = NaN
    #     z.cEu .= NaN
    #     z.cIu .= NaN
    #     return
    # end
    # Set equality constraint values
    if i.nE > 0
        z.cE .= c_orig[i.I6] .- i.b6
    end
    # Set inequality constraint values
    if i.n3 > 0
        z.cI[1:i.n3] .= i.l3 .- z.x[1+i.n1:i.n1+i.n3]
    end
    if i.n4 > 0
        z.cI[1+i.n3:i.n3+i.n4] .= z.x[1+i.n1+i.n3:i.n1+i.n3+i.n4] .- i.u4
    end
    if i.n5 > 0
        z.cI[1+i.n3+i.n4:i.n3+i.n4+i.n5] .=
            i.l5 .- z.x[1+i.n1+i.n3+i.n4:i.n1+i.n3+i.n4+i.n5]
        z.cI[1+i.n3+i.n4+i.n5:i.n3+i.n4+i.n5+i.n5] .=
            z.x[1+i.n1+i.n3+i.n4:i.n1+i.n3+i.n4+i.n5] .- i.u5
    end
    if i.n7 > 0
        z.cI[1+i.n3+i.n4+i.n5+i.n5:i.n3+i.n4+i.n5+i.n5+i.n7] .= i.l7 .- c_orig[i.I7]
    end
    if i.n8 > 0
        z.cI[1+i.n3+i.n4+i.n5+i.n5+i.n7:i.n3+i.n4+i.n5+i.n5+i.n7+i.n8] .=
            c_orig[i.I8] .- i.u8
    end
    if i.n9 > 0
        z.cI[1+i.n3+i.n4+i.n5+i.n5+i.n7+i.n8:i.n3+i.n4+i.n5+i.n5+i.n7+i.n8+i.n9] .=
            i.l9 .- c_orig[i.I9]
        z.cI[1+i.n3+i.n4+i.n5+i.n5+i.n7+i.n8+i.n9:i.n3+i.n4+i.n5+i.n5+i.n7+i.n8+i.n9+i.n9] .=
            c_orig[i.I9] .- i.u9
    end
    # Store unscaled quantities
    z.fu = z.f
    if i.nE > 0
        z.cEu .= z.cE
    end
    if i.nI > 0
        z.cIu .= z.cI
    end
    # Scale quantities
    z.f *= z.fs
    if i.nE > 0
        z.cE .*= z.cEs
    end
    if i.nI > 0
        z.cI .*= z.cIs
    end
end

"""
    Gradient evaluator
    evalGradients(z, i, c)
"""
function evalGradients(z::PIPAL_iterate, i::PIPAL_input, c::PIPAL_counters)
    T = eltype(i.x0)
    # Evaluate x in original space
    x_orig = evalXOriginal(z, i)
    # Initialize/Reset evaluation flag
    z.err = 0
    # Increment gradient evaluation counter
    incrementGradientCount(c)
    # Try gradients evaluation
    # TODO: catch evaluation errors
    # g_orig = zeros(T, i.nV)
    # J_orig = spzeros(T, i.nE+i.nI, i.nV)
    # try
    # Evaluate gradients
    g_orig = grad(i, x_orig)
    J_orig = jac(i, x_orig)
    # catch
    #     # Set evaluation flag, default values, and return
    #     z.err = 1
    #     z.g .= NaN
    #     return
    # end
    # Set objective gradient
    z.g .= [g_orig[i.I1]; g_orig[i.I3]; g_orig[i.I4]; g_orig[i.I5]]
    # Set equality constraint Jacobian
    if i.nE > 0
        z.JE .=
            [J_orig[i.I6, i.I1] J_orig[i.I6, i.I3] J_orig[i.I6, i.I4] J_orig[i.I6, i.I5]]
    end
    # Initialize inequality constraint Jacobian
    if i.nI > 0
        z.JI .= zero(T)
    end
    # Set inequality constraint Jacobian
    if i.n3 > 0
        z.JI[1:i.n3, 1+i.n1:i.n1+i.n3] = sparse(-1.0I, i.n3, i.n3)
    end
    if i.n4 > 0
        z.JI[1+i.n3:i.n3+i.n4, 1+i.n1+i.n3:i.n1+i.n3+i.n4] = sparse(1.0I, i.n4, i.n4)
    end
    if i.n5 > 0
        z.JI[1+i.n3+i.n4:i.n3+i.n4+i.n5, 1+i.n1+i.n3+i.n4:i.n1+i.n3+i.n4+i.n5] =
            sparse(-1.0I, i.n5, i.n5)
        z.JI[1+i.n3+i.n4+i.n5:i.n3+i.n4+i.n5+i.n5, 1+i.n1+i.n3+i.n4:i.n1+i.n3+i.n4+i.n5] =
            sparse(1.0I, i.n5, i.n5)
    end
    if i.n7 > 0
        z.JI[1+i.n3+i.n4+i.n5+i.n5:i.n3+i.n4+i.n5+i.n5+i.n7, 1:i.n1+i.n3+i.n4+i.n5] =
            -[J_orig[i.I7, i.I1] J_orig[i.I7, i.I3] J_orig[i.I7, i.I4] J_orig[i.I7, i.I5]]
    end
    if i.n8 > 0
        z.JI[
            1+i.n3+i.n4+i.n5+i.n5+i.n7:i.n3+i.n4+i.n5+i.n5+i.n7+i.n8,
            1:i.n1+i.n3+i.n4+i.n5,
        ] = [J_orig[i.I8, i.I1] J_orig[i.I8, i.I3] J_orig[i.I8, i.I4] J_orig[i.I8, i.I5]]
    end
    if i.n9 > 0
        z.JI[
            1+i.n3+i.n4+i.n5+i.n5+i.n7+i.n8:i.n3+i.n4+i.n5+i.n5+i.n7+i.n8+i.n9,
            1:i.n1+i.n3+i.n4+i.n5,
        ] = -[J_orig[i.I9, i.I1] J_orig[i.I9, i.I3] J_orig[i.I9, i.I4] J_orig[i.I9, i.I5]]
        z.JI[
            1+i.n3+i.n4+i.n5+i.n5+i.n7+i.n8+i.n9:i.n3+i.n4+i.n5+i.n5+i.n7+i.n8+i.n9+i.n9,
            1:i.n1+i.n3+i.n4+i.n5,
        ] = [J_orig[i.I9, i.I1] J_orig[i.I9, i.I3] J_orig[i.I9, i.I4] J_orig[i.I9, i.I5]]
    end
    # Scale objective gradient
    z.g .*= z.fs
    # Scale constraint Jacobians
    if i.nE > 0
        z.JE = spdiagm(z.cEs) * z.JE
    end
    if i.nI > 0
        z.JI = spdiagm(z.cIs) * z.JI
    end
end

"""
    Hessian evaluator
    evalHessian(z, i, c)
"""
function evalHessian(z::PIPAL_iterate, i::PIPAL_input, c::PIPAL_counters)
    T = eltype(i.x0)
    # Evaluate x in original space
    x_orig = evalXOriginal(z, i)
    # Evaluate lambda in original space
    l_orig = evalLambdaOriginal(z, i)
    # Initialize/Reset evaluation flag
    z.err = 0
    # Increment Hessian evaluation counter
    incrementHessianCount(c)
    # Try Hessian evaluation
    H_orig = spzeros(T, i.nV, i.nV)
    try
        # Evaluate H_orig
        if i.nE + i.n7 + i.n8 + i.n9 == 0
            H_orig .= hess(i, x_orig)
        else
            H_orig .= hess(i, x_orig, l_orig)
        end
        # TODO: need whole matrix or upper/lower part only?
        # H_orig = H_orig + tril(H_orig, -1)'
    catch
        # Set evaluation flag, default values, and return
        z.err = 1
        return
    end
    # Set Hessian of the Lagrangian
    z.H = [
        H_orig[i.I1, i.I1] H_orig[i.I1, i.I3] H_orig[i.I1, i.I4] H_orig[i.I1, i.I5]
        H_orig[i.I3, i.I1] H_orig[i.I3, i.I3] H_orig[i.I3, i.I4] H_orig[i.I3, i.I5]
        H_orig[i.I4, i.I1] H_orig[i.I4, i.I3] H_orig[i.I4, i.I4] H_orig[i.I4, i.I5]
        H_orig[i.I5, i.I1] H_orig[i.I5, i.I3] H_orig[i.I5, i.I4] H_orig[i.I5, i.I5]
    ]
    # Rescale H
    z.H .*= z.rho * z.fs
end

"""
    KKT errors evaluator
    evalKKTErrors(z, i)
"""
function evalKKTErrors(z::PIPAL_iterate, i::PIPAL_input)
    # Loop to compute optimality errors
    z.kkt[1] = evalKKTError(z, i, 0.0, 0.0)
    z.kkt[2] = evalKKTError(z, i, z.rho, 0.0)
    z.kkt[3] = evalKKTError(z, i, z.rho, z.mu)
end

"""
    Newton right-hand side evaluator
    evalNewtonRhs(z, i)
"""
function evalNewtonRhs(z::PIPAL_iterate, i::PIPAL_input)
    T = eltype(i.x0)
    # Set gradient of objective
    z.b[1:i.nV] .= z.rho * z.g
    # Set gradient of Lagrangian for constraints
    if i.nE > 0
        z.b[1:i.nV] .+= z.JE' * z.lE
    end
    if i.nI > 0
        z.b[1:i.nV] .+= z.JI' * z.lI
    end
    # Set complementarity for constraint slacks
    if i.nE > 0
        z.b[1+i.nV:i.nV+2*i.nE] .=
            [1.0 .+ z.lE .- (z.mu ./ z.r1); 1.0 .- z.lE .- (z.mu ./ z.r2)]
    end
    if i.nI > 0
        z.b[1+i.nV+2*i.nE:i.nV+2*i.nE+2*i.nI] .=
            [z.lI .- (z.mu ./ z.s1); 1.0 .- z.lI .- (z.mu ./ z.s2)]
    end
    # Set penalty-interior-point constraint values
    if i.nE > 0
        z.b[1+i.nV+2*i.nE+2*i.nI:i.nV+3*i.nE+2*i.nI] .= z.cE + z.r1 - z.r2
    end
    if i.nI > 0
        z.b[1+i.nV+3*i.nE+2*i.nI:i.nV+3*i.nE+3*i.nI] .= z.cI + z.s1 - z.s2
    end
end

"""
    Termination checker
    checkTermination(z, p, i, c)
"""
function checkTermination(z::PIPAL_iterate, p::PIPAL_parameters, i, c::PIPAL_counters)
    # Initialize exitflag
    b = 0
    # Update termination based on optimality error of nonlinear optimization problem
    if z.kkt[2] ≤ p.opt_err_tol && z.v ≤ p.opt_err_tol
        b = 1
        return b
    end
    # Update termination based on optimality error of feasibility problem
    if z.kkt[1] ≤ p.opt_err_tol && z.v > p.opt_err_tol
        b = 2
        return b
    end
    # Update termination based on iteration count
    if c.k ≥ p.iter_max
        b = 3
        return b
    end
    # Update termination based on invalid bounds
    if i.vi > 0
        b = 4
        return b
    end
    # Update termination based on function evaluation error
    if z.err > 0
        b = 5
        return b
    end
    return b
end

"""
    Matrices evaluator
    evalMatrices(z, p, i, c)
"""
function evalMatrices(
    z::PIPAL_iterate,
    p::PIPAL_parameters,
    i::PIPAL_input,
    c::PIPAL_counters,
)
    # Evaluate Hessian and Newton matrices
    evalHessian(z, i, c)
    evalNewtonMatrix(z, p, i, c)
end

"""
    Dependent quantity evaluator
    evalDependent(z, p, i)
"""
function evalDependent(z::PIPAL_iterate, p::PIPAL_parameters, i::PIPAL_input)
    # Evaluate quantities dependent on penalty and interior-point parameters
    evalSlacks(z, p, i)
    evalMerit(z, i)
    evalKKTErrors(z, i)
end

"""
    Infeasibility evaluator
    evalInfeasibility(z, i)
"""
function evalInfeasibility(z::PIPAL_iterate, i::PIPAL_input)
    # Evaluate scaled and unscaled feasibility violations
    z.v = evalViolation(i, z.cE, z.cI) / max(1.0, z.v0)
    z.vu = evalViolation(i, z.cEu, z.cIu)
end

"""
    Merit evaluator
    evalMerit(z, i)
"""
function evalMerit(z::PIPAL_iterate, i::PIPAL_input)
    # Initialize merit for objective
    z.phi = z.rho * z.f
    # Update merit for slacks
    if i.nE > 0
        z.phi += sum([z.r1; z.r2]) - z.mu * sum(log.([z.r1; z.r2]))
    end
    if i.nI > 0
        z.phi += sum(z.s2) - z.mu * sum(log.([z.s1; z.s2]))
    end
end

"""
    Gets primal-dual point
    (x, y) = getSolution(z, i)
"""
function getSolution(z::PIPAL_iterate, i::PIPAL_input)
    x = evalXOriginal(z, i)
    y = evalLambdaOriginal(z, i)
    return (x, y)
end

"""
    Evaluator of x in original space
    x = evalXOriginal(z, i)
"""
function evalXOriginal(z::PIPAL_iterate, i::PIPAL_input)
    T = eltype(i.x0)
    # Initialize x in original space
    x = zeros(T, i.n0)
    # Evaluate x in original space
    x[i.I1] .= z.x[1:i.n1]
    x[i.I2] .= i.b2
    x[i.I3] .= z.x[1+i.n1:i.n1+i.n3]
    x[i.I4] .= z.x[1+i.n1+i.n3:i.n1+i.n3+i.n4]
    x[i.I5] .= z.x[1+i.n1+i.n3+i.n4:i.n1+i.n3+i.n4+i.n5]
    return x
end

"""
    Evaluator of lambda in original space
    y = evalLambdaOriginal(z, i)
"""
function evalLambdaOriginal(z::PIPAL_iterate, i::PIPAL_input)
    T = eltype(i.x0)
    # Initialize multipliers in original space
    y = zeros(T, i.nE + i.n7 + i.n8 + i.n9)
    # Scale equality constraint multipliers
    if i.nE > 0
        yE = z.lE .* (z.cEs ./ (z.rho * z.fs))
    end
    # Set equality constraint multipliers in original space
    if i.nE > 0
        y[i.I6] .= z.lE
    end
    # Scale inequality constraint multipliers
    if i.n7 + i.n8 + i.n9 > 0
        yI = z.lI .* (z.cIs ./ (z.rho * z.fs))
    end
    # Set inequality constraint multipliers in original space
    if i.n7 > 0
        y[i.I7] .= -yI[1+i.n3+i.n4+i.n5+i.n5:i.n3+i.n4+i.n5+i.n5+i.n7]
    end
    if i.n8 > 0
        y[i.I8] .= yI[1+i.n3+i.n4+i.n5+i.n5+i.n7:i.n3+i.n4+i.n5+i.n5+i.n7+i.n8]
    end
    if i.n9 > 0
        y[i.I9] .=
            yI[1+i.n3+i.n4+i.n5+i.n5+i.n7+i.n8+i.n9:i.n3+i.n4+i.n5+i.n5+i.n7+i.n8+i.n9+i.n9] .-
            yI[1+i.n3+i.n4+i.n5+i.n5+i.n7+i.n8:i.n3+i.n4+i.n5+i.n5+i.n7+i.n8+i.n9]
    end
    return y
end

"""
    Set interior-point parameter
    setMu(z, mu)
"""
function setMu(z::PIPAL_iterate, mu)
    z.mu = mu
end

"""
    Set primal variables
    setPrimals(z, i, x, r1, r2, s1, s2, lE, lI, f, cE, cI, phi)
"""
function setPrimals(
    z::PIPAL_iterate,
    i::PIPAL_input,
    x::Vector,
    r1::Vector,
    r2::Vector,
    s1::Vector,
    s2::Vector,
    lE::Vector,
    lI::Vector,
    f::Real,
    cE::Vector,
    cI::Vector,
    phi::Real,
)
    # Set primal variables
    z.x .= x
    z.f = f
    if i.nE > 0
        z.cE .= cE
        z.r1 .= r1
        z.r2 .= r2
        z.lE .= lE
    end
    if i.nI > 0
        z.cI .= cI
        z.s1 .= s1
        z.s2 .= s2
        z.lI .= lI
    end
    z.phi = phi
end

"""
    Set penalty parameter
    setRho(z, rho)
"""
function setRho(z::PIPAL_iterate, rho)
    z.rho = rho
end

"""
    Set last penalty parameter
    setRhoLast(z, rho)
"""
function setRhoLast(z::PIPAL_iterate, rho)
    z.rho_ = rho
end

"""
    Feasibility violation evaluator
    v = evalViolation(i, cE, cI)
"""
function evalViolation(i::PIPAL_input, cE, cI)
    # Initialize violation
    v = 0.0
    # Update violation for constraint values
    if i.nE > 0
        v += norm(cE, 1)
    end
    if i.nI > 0
        v += norm(max.(cI, 0.0), 1)
    end
    return v
end

"""
    Primal point updater
    updatePoint(z, i, d, a)
"""
function updatePoint(
    z::PIPAL_iterate,
    i::PIPAL_input,
    d::PIPAL_direction,
    a::PIPAL_acceptance,
)
    # Update primal and dual variables
    z.x .+= a.p .* d.x
    if i.nE > 0
        z.r1 .+= a.p .* d.r1
        z.r2 .+= a.p .* d.r2
    end
    if i.nI > 0
        z.s1 .+= a.p .* d.s1
        z.s2 .+= a.p .* d.s2
    end
    if i.nE > 0
        z.lE .+= a.d .* d.lE
    end
    if i.nI > 0
        z.lI .+= a.d .* d.lI
    end
end

"""
    Parameter updater
    updateParameters(z, p, i)
"""
function updateParameters(z::PIPAL_iterate, p::PIPAL_parameters, i::PIPAL_input)
    # Check for interior-point parameter update based on optimality error
    while (z.mu > p.mu_min) && (z.kkt[3] ≤ max(z.mu, p.opt_err_tol - z.mu))
        # Restrict interior-point parameter increase
        setMuMaxExpZero(p)
        # Update interior-point parameter
        if z.mu > p.mu_min
            # Decrease interior-point
            z.mu = max(p.mu_min, min(p.mu_factor * z.mu, z.mu^p.mu_factor_exp))
            # Evaluate penalty and interior-point parameter dependent quantities
            evalDependent(z, p, i)
        end
    end
    # Check for penalty parameter update based on optimality error
    if ((z.kkt[2] ≤ p.opt_err_tol) && (z.v > p.opt_err_tol)) ||
       (z.v > max(1.0, max(z.v_, p.infeas_max)))
        # Update penalty parameter
        if z.rho > p.rho_min
            # Decrease penalty parameter
            z.rho = max(p.rho_min, p.rho_factor * z.rho)
            # Evaluate penalty and interior-point parameter dependent quantities
            evalDependent(z, p, i)
        end
    end
end

"""
    Iterate updater
    updateIterate(z, p, i, d, a)
"""
function updateIterate(
    z::PIPAL_iterate,
    p::PIPAL_parameters,
    i::PIPAL_input,
    c::PIPAL_counters,
    d::PIPAL_direction,
    a::PIPAL_acceptance,
)
    # Update last quantities
    z.v_ = z.v
    z.cut_ = (a.p < a.p0)
    # Update iterate quantities
    updatePoint(z, i, d, a)
    evalInfeasibility(z, i)
    evalGradients(z, i, c)
    evalDependent(z, p, i)
    # Update last KKT errors
    z.kkt_[2:p.opt_err_mem] .= z.kkt_[1:p.opt_err_mem-1]
    z.kkt_[1] = z.kkt[2]
end

"""
    Slacks evaluator
    evalSlacks(z, p, i)
"""
function evalSlacks(z::PIPAL_iterate, p::PIPAL_parameters, i::PIPAL_input)
    # Check for equality constraints
    if i.nE > 0
        # Set slacks
        z.r1 .= (1 / 2) .* (z.mu .- z.cE .+ sqrt.(z.cE .^ 2 .+ z.mu^2))
        z.r2 .= (1 / 2) .* (z.mu .+ z.cE .+ sqrt.(z.cE .^ 2 .+ z.mu^2))
        # Adjust for numerical error
        z.r1 .= max.(z.r1, p.slack_min)
        z.r2 .= max.(z.r2, p.slack_min)
    end
    # Check for inequality constraints
    if i.nI > 0
        # Set slacks
        z.s1 .= (1 / 2) .* (2 * z.mu .- z.cI .+ sqrt.(z.cI .^ 2 .+ 4 * z.mu^2))
        z.s2 .= (1 / 2) .* (2 * z.mu .+ z.cI .+ sqrt.(z.cI .^ 2 .+ 4 * z.mu^2))
        # Adjust for numerical error
        z.s1 .= max.(z.s1, p.slack_min)
        z.s2 .= max.(z.s2, p.slack_min)
    end
end

"""
    KKT error evaluator
    e = evalKKTError(z, i, rho, mu)
"""
function evalKKTError(z::PIPAL_iterate, i::PIPAL_input, rho, mu)
    T = eltype(i.x0)
    # Initialize optimality vector
    kkt = zeros(T, i.nV + 2 * i.nE + 2 * i.nI)
    # Set gradient of penalty objective
    kkt[1:i.nV] .= rho .* z.g
    # Set gradient of Lagrangian for constraints
    if i.nE > 0
        kkt[1:i.nV] .+= z.JE' * z.lE
    end
    if i.nI > 0
        kkt[1:i.nV] .+= z.JI' * z.lI
    end
    # Set complementarity for constraint slacks
    if i.nE > 0
        kkt[1+i.nV:i.nV+2*i.nE] .=
            [z.r1 .* (1.0 .+ z.lE) .- mu; z.r2 .* (1.0 .- z.lE) .- mu]
    end
    if i.nI > 0
        kkt[1+i.nV+2*i.nE:i.nV+2*i.nE+2*i.nI] .=
            [z.s1 .* z.lI .- mu; z.s2 .* (1.0 .- z.lI) .- mu]
    end
    # Scale complementarity
    if rho > 0.0
        kkt .*= (1.0 / max(1.0, norm(rho * z.g, Inf)))
    end
    # Evaluate optimality error
    return norm(kkt, Inf)
end

"""
    Initializes Newton matrix
    initNewtonMatrix(z, i)
"""
function initNewtonMatrix(z::PIPAL_iterate{T}, i::PIPAL_input) where {T}
    # Allocate memory
    z.A .= spzeros(T, i.nA, i.nA) # nnz(A) = z.Hnnz + 5 * i.nE + 5 * i.nI + z.JEnnz + z.JInnz
    # Initialize interior-point Hessians
    z.A[1+i.nV:i.nV+2*i.nE, 1+i.nV:i.nV+2*i.nE] .= sparse(1.0I, 2 * i.nE, 2 * i.nE)
    z.A[1+i.nV+2*i.nE:i.nV+2*i.nE+2*i.nI, 1+i.nV+2*i.nE:i.nV+2*i.nE+2*i.nI] .=
        sparse(1.0I, 2 * i.nI, 2 * i.nI)
    # Check for equality constraints
    if i.nE > 0
        # Initialize constraint Jacobian
        z.A[1+i.nV+2*i.nE+2*i.nI:i.nV+3*i.nE+2*i.nI, 1+i.nV:i.nV+i.nE] .=
            sparse(1.0I, i.nE, i.nE)
        z.A[1+i.nV:i.nV+i.nE, 1+i.nV+2*i.nE+2*i.nI:i.nV+3*i.nE+2*i.nI] .=
            sparse(1.0I, i.nE, i.nE) # upper
        z.A[1+i.nV+2*i.nE+2*i.nI:i.nV+3*i.nE+2*i.nI, 1+i.nV+i.nE:i.nV+2*i.nE] .=
            sparse(-1.0I, i.nE, i.nE)
        z.A[1+i.nV+i.nE:i.nV+2*i.nE, 1+i.nV+2*i.nE+2*i.nI:i.nV+3*i.nE+2*i.nI] .=
            sparse(-1.0I, i.nE, i.nE) # upper
    end
    # Check for inequality constraints
    if i.nI > 0
        # Initialize constraint Jacobian
        z.A[1+i.nV+3*i.nE+2*i.nI:i.nV+3*i.nE+3*i.nI, 1+i.nV+2*i.nE:i.nV+2*i.nE+i.nI] .=
            sparse(1.0I, i.nI, i.nI)
        z.A[1+i.nV+2*i.nE:i.nV+2*i.nE+i.nI, 1+i.nV+3*i.nE+2*i.nI:i.nV+3*i.nE+3*i.nI] .=
            sparse(1.0I, i.nI, i.nI) # upper
        z.A[
            1+i.nV+3*i.nE+2*i.nI:i.nV+3*i.nE+3*i.nI,
            1+i.nV+2*i.nE+i.nI:i.nV+2*i.nE+2*i.nI,
        ] .= sparse(-1.0I, i.nI, i.nI)
        z.A[
            1+i.nV+2*i.nE+i.nI:i.nV+2*i.nE+2*i.nI,
            1+i.nV+3*i.nE+2*i.nI:i.nV+3*i.nE+3*i.nI,
        ] .= sparse(-1.0I, i.nI, i.nI) # upper
    end
end

"""
    Scalings evaluator
    evalScalings(z, p, i, c)
    Compute scaling factors for objective and constraints.

    [WB06] Wächter, Biegler (2006) On the implementation of an interior-point
           filter line-search algorithm for large-scale nonlinear programming.
"""
function evalScalings(
    z::PIPAL_iterate{T},
    p::PIPAL_parameters,
    i::PIPAL_input,
    c::PIPAL_counters,
) where {T}
    # Initialize scalings
    z.fs = one(T)
    z.cEs .= ones(T, i.nE)
    z.cIs .= ones(T, i.nI)
    # Evaluate gradients
    evalGradients(z, i, c)
    # Evaluate norm of objective gradient
    g_norm_inf = norm(z.g, Inf)
    # Scale down objective if norm of gradient is too large
    z.fs = p.grad_max / max(g_norm_inf, p.grad_max)
    # Loop through equality constraints
    for j = 1:i.nE
        # Evaluate norm of gradient of jth equality constraint
        JE_j_norm_inf = norm(z.JE[j, :], Inf)
        # Scale down equality constraint j if norm of gradient is too large
        z.cEs[j] = p.grad_max / max(JE_j_norm_inf, p.grad_max)
    end
    # Loop through inequality constraints
    for j = 1:i.nI
        # Evaluate norm of gradient of jth inequality constraint
        JI_j_norm_inf = norm(z.JI[j, :], Inf)
        # Scale down inequality constraint j if norm of gradient is too large
        z.cIs[j] = p.grad_max / max(JI_j_norm_inf, p.grad_max)
    end
end

"""
    Newton matrix evaluator
    evalNewtonMatrix(z, p, i, c)
    Set up Newton matrix, adjust inertia by regularizing the Hessian matrix, and
    factorize.

    [VS99] Vanderbei, Shanno (1999) An interior-point algorithm for nonconvex
           nonlinear programming.
"""
function evalNewtonMatrix(
    z::PIPAL_iterate,
    p::PIPAL_parameters,
    i::PIPAL_input,
    c::PIPAL_counters,
)
    # Check for equality constraints
    if i.nE > 0
        # Set diagonal terms
        for j = 1:i.nE
            z.A[i.nV+j, i.nV+j] = (1.0 + z.lE[j]) / z.r1[j]
            z.A[i.nV+i.nE+j, i.nV+i.nE+j] = (1.0 - z.lE[j]) / z.r2[j]
        end
        # Set constraint Jacobian
        z.A[1+i.nV+2*i.nE+2*i.nI:i.nV+3*i.nE+2*i.nI, 1:i.nV] .= z.JE
        z.A[1:i.nV, 1+i.nV+2*i.nE+2*i.nI:i.nV+3*i.nE+2*i.nI] .= z.JE' # upper
    end
    # Check for inequality constraints
    if i.nI > 0
        # Set diagonal terms
        for j = 1:i.nI
            z.A[i.nV+2*i.nE+j, i.nV+2*i.nE+j] = z.lI[j] / z.s1[j]
            z.A[i.nV+2*i.nE+i.nI+j, i.nV+2*i.nE+i.nI+j] = (1.0 - z.lI[j]) / z.s2[j]
        end
        # Set constraint Jacobian
        z.A[1+i.nV+3*i.nE+2*i.nI:i.nV+3*i.nE+3*i.nI, 1:i.nV] .= z.JI
        z.A[1:i.nV, 1+i.nV+3*i.nE+2*i.nI:i.nV+3*i.nE+3*i.nI] .= z.JI' # upper
    end
    # Set minimum potential shift
    min_shift = max(p.shift_min, p.shift_factor1 * z.shift)
    # Initialize Hessian modification
    if z.cut_
        z.shift = min(p.shift_max, min_shift / p.shift_factor2)
    else
        z.shift = 0.0
    end
    # Initialize inertia correction loop
    done = false
    z.shift22 = p.shift_min # ? originally 0.0
    # Loop until inertia is correct
    while !done && (z.shift < p.shift_max)
        # Set Hessian of Lagrangian
        z.A[1:i.nV, 1:i.nV] .= z.H + sparse(z.shift * 1.0I, i.nV, i.nV)
        # Set diagonal terms
        for j in 1:i.nE
            z.A[i.nV+2*i.nE+2*i.nI+j, i.nV+2*i.nE+2*i.nI+j] = -z.shift22
        end
        # Set diagonal terms
        for j in 1:i.nI
            z.A[i.nV+3*i.nE+2*i.nI+j, i.nV+3*i.nE+2*i.nI+j] = -z.shift22
        end
        # Set number of nonzeros in (upper triangle of) Newton matrix
        # ? z.Annz = nnz(tril(z.A))
        # Factor primal-dual matrix
        z.Af = ldl(z.A) # LDLFactorizations requires upper-triangle of A
        if factorized(z.Af)
            neig = sum(diag(z.Af.D) .< 0.0) # ?
        else
            neig = i.nA
            # z.shift22 = p.shift_min # ?
        end
        # Increment factorization counter
        incrementFactorizationCount(c)
        # Set number of nonnegative eigenvalues
        peig = i.nA - neig
        # Check inertia
        # ideally, peig = nV + 2 nE + 2 nI
        #          neig =        nE +   nI
        if peig < i.nV + 2 * i.nE + 2 * i.nI
            z.shift = max(min_shift, z.shift / p.shift_factor2)
        elseif (neig < i.nE + i.nI) && (z.shift22 == 0.0)
            z.shift22 = p.shift_min # ?
        else
            done = true
        end
    end
    # Update Hessian
    z.H += sparse(z.shift * 1.0I, i.nV, i.nV)
end
