"""
    constructor
    d = Direction(i)
"""
function Direction(i::PIPAL_input)
    T = eltype(i.x0)
    d = PIPAL_direction(
        zeros(T, i.nV), # x
        zero(T), # x_norm
        zero(T), # x_norm_
        zeros(T, i.nE), # r1
        zeros(T, i.nE), # r2
        zeros(T, i.nE), # lE
        zeros(T, i.nI), # s1
        zeros(T, i.nI), # s2
        zeros(T, i.nI), # lI
        zero(T), # l_norm
        zero(T), # lred0
        zero(T), # ltred0
        zero(T), # ltred
        zero(T), # qtred
        zero(T), # m
    )
    # Initialize last direction norm
    d.x_norm_ = Inf
    return d
end

"""
    Evaluate search direction quantities
    evalStep(d, p, i, c, z, a)
"""
function evalStep(
    d::PIPAL_direction,
    p::PIPAL_parameters,
    i::PIPAL_input,
    c::PIPAL_counters,
    z::PIPAL_iterate,
    a::PIPAL_acceptance,
)
    T = eltype(i.x0)
    # Reset maximum exponent for interior-point parameter increases
    resetMuMaxExp(p)
    # Update penalty-interior-point parameters based on KKT errors
    updateParameters(z, p, i)
    # Evaluate matrices
    evalMatrices(z, p, i, c)
    # Set last penalty parameter
    setRhoLast(z, z.rho)
    # Check for aggressive algorithm
    if p.algorithm == 1
        evalAggressive(d, p, i, c, z, a)
    end
    # Evaluate primal-dual right-hand side vector
    evalNewtonRhs(z, i)
    # Evaluate search direction
    evalNewtonStep(d, i, z)
    # Evaluate models
    evalModels(d, i, z)
    # Store last direction norm
    d.x_norm_ = d.x_norm
end

"""
    Aggressive de-tour
"""
function evalAggressive(
    d::PIPAL_direction,
    p::PIPAL_parameters,
    i::PIPAL_input,
    c::PIPAL_counters,
    z::PIPAL_iterate,
    a::PIPAL_acceptance,
)
    T = eltype(i.x0)
    # Check KKT memory for potential mu increase limit
    if z.kkt[2] > maximum(z.kkt_)
        setMuMaxExpZero(p)
    end
    # Store current penalty and interior-point parameters
    rho_curr = copy(z.rho)
    mu_curr = copy(z.mu)
    # Evaluate trial steps
    (d1, d2, d3) = evalTrialSteps(d, i, z)
    # Set trial interior-point parameter values
    Mu = p.mu_factor .^ (range(p.mu_trials - 1, step = -1, stop = 0) .- p.mu_max_exp)
    Mu .*= mu_curr
    Mu .= max.(p.mu_min, min.(Mu, p.mu_max))
    # Initialize feasibility direction data
    lred0_0_mu = zeros(T, p.mu_trials)
    # Loop through interior-point parameter values
    for j = 1:p.mu_trials
        # Set penalty and interior-point parameters
        setRho(z, 0.0)
        setMu(z, Mu[j])
        # Evaluate direction
        evalLinearCombination(
            d,
            i,
            d1,
            d2,
            d3,
            [
                (z.rho / rho_curr + z.mu / mu_curr - 1),
                (1 - z.mu / mu_curr),
                (1 - z.rho / rho_curr),
            ],
        )
        # Cut length
        d.x .*= min(d.x_norm_ / max(d.x_norm, 1.0), 1.0)
        # Run fraction-to-boundary
        fractionToBoundary(a, p, i, z, d)
        # Cut length
        evalTrialStepCut(d, i, a)
        # Evaluate models
        evalModels(d, i, z)
        # Set feasibility direction data
        lred0_0_mu[j] = d.lred0
    end
    # Initialize updating data
    ltred0_rho_mu = zeros(T, p.mu_trials)
    qtred_rho_mu = zeros(T, p.mu_trials)
    m_rho_mu = zeros(T, p.mu_trials)
    # Initialize check
    check = false
    # Loop through penalty parameter values
    for k = 1:p.rho_trials
        # Set penalty parameter
        setRho(z, max(p.rho_min, (p.rho_factor^(k - 1)) * rho_curr))
        # Set last penalty parameter
        if rho_curr > z.kkt[1]^2
            setRhoLast(z, z.rho)
        end
        # Loop through interior-point parameter values
        for j = 1:p.mu_trials
            # Set interior-point parameter
            setMu(z, Mu[j])
            # Evaluate direction
            evalLinearCombination(
                d,
                i,
                d1,
                d2,
                d3,
                [
                    (z.rho / rho_curr + z.mu / mu_curr - 1),
                    (1 - z.mu / mu_curr),
                    (1 - z.rho / rho_curr),
                ],
            )
            # Run fraction-to-boundary
            fractionToBoundary(a, p, i, z, d)
            # Cut steps
            evalTrialStepCut(d, i, a)
            # Evaluate models
            evalModels(d, i, z)
            # Set updating data
            ltred0_rho_mu[j] = d.ltred0
            qtred_rho_mu[j] = d.qtred
            m_rho_mu[j] = d.m
            # Check updating conditions for infeasible points
            if z.v > p.opt_err_tol && (
                ltred0_rho_mu[j] < p.update_con_1 * lred0_0_mu[j] ||
                qtred_rho_mu[j] < p.update_con_2 * lred0_0_mu[j] ||
                z.rho > z.kkt[1]^2
            )
                m_rho_mu[j] = Inf
            end
            # Check updating conditions for feasible points
            if z.v ≤ p.opt_err_tol && qtred_rho_mu[j] < 0.0
                m_rho_mu[j] = Inf
            end
        end
        # Find minimum m for current rho
        m_min = minimum(m_rho_mu)
        # Check for finite minimum
        if m_min < Inf
            # Loop through mu values
            for j = 1:p.mu_trials
                # Set mu
                mu = Mu[j]
                # Check condition
                if m_rho_mu[j] ≤ p.update_con_3 * m_min
                    setMu(z, mu)
                end
            end
            # Set condition check
            check = true
            # Break loop
            break
        end
    end
    # Check conditions
    if !check
        setRho(z, rho_curr)
        setMu(z, mu_curr)
    end
    # Evaluate merit
    evalMerit(z, i)
end

"""
    Evaluate model and model reductions
    evalModels(d, i, z)
"""
function evalModels(d::PIPAL_direction, i::PIPAL_input, z::PIPAL_iterate)
    # Evaluate reduction in linear model of penalty-interior-point objective for zero penalty parameter
    d.lred0 = 0.0
    if i.nE > 0
        d.lred0 += -sum([1.0 .- (z.mu ./ z.r1); 1.0 .- (z.mu ./ z.r2)] .* [d.r1; d.r2])
    end
    if i.nI > 0
        d.lred0 += -sum([-z.mu ./ z.s1; 1.0 .- (z.mu ./ z.s2)] .* [d.s1; d.s2])
    end
    # Evaluate remaining quantities only for nonzero penalty parameter
    if z.rho > 0
        T = eltype(i.x0)
        # Evaluate reduction in linear model of merit function for zero penalty parameter
        d.ltred0 = 0.0
        if i.nE > 0
            d.ltred0 +=
                -(1 / 2) * sum(
                    (
                        (1.0 .- (z.mu ./ z.r1)) .*
                        (-1.0 .+ z.cE ./ (sqrt.(z.cE .^ 2 .+ z.mu^2))) +
                        (1.0 .- z.mu ./ z.r2) .*
                        (1.0 .+ z.cE ./ (sqrt.(z.cE .^ 2 .+ z.mu^2)))
                    ) .* (z.JE * d.x),
                )
        end
        if i.nI > 0
            d.ltred0 +=
                -(1 / 2) * sum(
                    (
                        (-z.mu ./ z.s1) .*
                        ((z.cI ./ (sqrt.(z.cI .^ 2 .+ 4 * z.mu^2))) .- 1.0) +
                        (1.0 .- (z.mu ./ z.s2)) .*
                        (1.0 .+ (z.cI ./ (sqrt.(z.cI .^ 2 .+ 4 * z.mu^2))))
                    ) .* (z.JI * d.x),
                )
        end
        # Evaluate reduction in linear model of merit function
        d.ltred = -z.rho * z.g' * d.x + d.ltred0
        # Evaluate reduction in quadratic model of merit function
        d.qtred = d.ltred - (1 / 2) * (d.x' * (z.H * d.x))
        if i.nE > 0
            Jd = z.JE * d.x
            Dinv = z.r1 ./ (1.0 .+ z.lE) + z.r2 ./ (1.0 .- z.lE)
            d.qtred = d.qtred - (1 / 2) * Jd' * (Jd ./ Dinv)
        end
        if i.nI > 0
            Jd = z.JI * d.x
            Dinv = (z.s1 ./ z.lI) + z.s2 ./ (1.0 .- z.lI)
            d.qtred = d.qtred - (1 / 2) * Jd' * (Jd ./ Dinv)
        end
        # Initialize quality function vector
        vec = zeros(T, i.nV + 2 * i.nE + 2 * i.nI)
        # Set gradient of objective
        vec[1:i.nV] = z.rho * z.g
        # Set gradient of Lagrangian for constraints
        if i.nE > 0
            vec[1:i.nV] = vec[1:i.nV] + ((z.lE + d.lE)' * z.JE)'
        end
        if i.nI > 0
            vec[1:i.nV] = vec[1:i.nV] + ((z.lI + d.lI)' * z.JI)'
        end
        # Set complementarity for constraint slacks
        if i.nE > 0
            vec[1+i.nV:i.nV+2*i.nE] = [
                (z.r1 + d.r1) .* (1.0 .+ (z.lE + d.lE))
                (z.r2 + d.r2) .* (1.0 .- (z.lE + d.lE))
            ]
        end
        if i.nI > 0
            vec[1+i.nV+2*i.nE:i.nV+2*i.nE+2*i.nI] =
                [(z.s1 + d.s1) .* ((z.lI + d.lI)); (z.s2 + d.s2) .* (1.0 .- (z.lI + d.lI))]
        end
        # Evaluate quality function
        d.m = norm(vec, Inf)
    end
end

"""
    Evaluate linear combination of directions
    evalLinearCombination(d, i, d1, d2, d3, a)
"""
function evalLinearCombination(d::PIPAL_direction, i::PIPAL_input, d1, d2, d3, a)
    # Evaluate linear combinations
    d.x .= a[1] .* d1.x + a[2] .* d2.x + a[3] .* d3.x
    if i.nE > 0
        d.r1 = a[1] * d1.r1 + a[2] * d2.r1 + a[3] * d3.r1
        d.r2 = a[1] * d1.r2 + a[2] * d2.r2 + a[3] * d3.r2
        d.lE = a[1] * d1.lE + a[2] * d2.lE + a[3] * d3.lE
    end
    if i.nI > 0
        d.s1 = a[1] * d1.s1 + a[2] * d2.s1 + a[3] * d3.s1
        d.s2 = a[1] * d1.s2 + a[2] * d2.s2 + a[3] * d3.s2
        d.lI = a[1] * d1.lI + a[2] * d2.lI + a[3] * d3.lI
    end
    # Evaluate primal direction norm
    d.x_norm = norm(d.x)
    # Evaluate dual direction norm
    d.l_norm = norm([d.lE; d.lI])
end

"""
    Evaluate Newton step
    evalNewtonStep(d, i, z)
"""
function evalNewtonStep(d::PIPAL_direction, i::PIPAL_input, z::PIPAL_iterate)
    # Evaluate direction
    # A dir = - b
    dir = z.Af \ (-z.b)
    # Parse direction
    d.x .= dir[1:i.nV]
    if i.nE > 0
        d.r1 .= dir[1+i.nV:i.nV+i.nE]
        d.r2 .= dir[1+i.nV+i.nE:i.nV+i.nE+i.nE]
        d.lE .= dir[1+i.nV+i.nE+i.nE+i.nI+i.nI:i.nV+i.nE+i.nE+i.nI+i.nI+i.nE]
    end
    if i.nI > 0
        d.s1 .= dir[1+i.nV+i.nE+i.nE:i.nV+i.nE+i.nE+i.nI]
        d.s2 .= dir[1+i.nV+i.nE+i.nE+i.nI:i.nV+i.nE+i.nE+i.nI+i.nI]
        d.lI .= dir[1+i.nV+i.nE+i.nE+i.nI+i.nI+i.nE:i.nV+i.nE+i.nE+i.nI+i.nI+i.nE+i.nI]
    end
    # Evaluate primal direction norm
    d.x_norm = norm(d.x, 2)
    # Evaluate dual direction norm
    d.l_norm = norm([d.lE; d.lI], 2)
end

"""
    Evaluate and store trial step
    v = evalTrialStep(d, i)
"""
function evalTrialStep(d::PIPAL_direction, i::PIPAL_input)
    v = Direction(i)
    # Set direction components
    v.x .= d.x
    if i.nE > 0
        v.r1 .= d.r1
        v.r2 .= d.r2
        v.lE .= d.lE
    end
    if i.nI > 0
        v.s1 .= d.s1
        v.s2 .= d.s2
        v.lI .= d.lI
    end
    return v
end

"""
    Evaluate and store directions for parameter combinations
    (d1, d2, d3) = evalTrialSteps(d, i, z)
"""
function evalTrialSteps(d::PIPAL_direction, i::PIPAL_input, z::PIPAL_iterate)
    # Store current penalty and interior-point parameters
    rho_curr = z.rho
    mu_curr = z.mu
    # Evaluate direction for current penalty and interior-point parameters
    setRho(z, rho_curr)
    setMu(z, mu_curr)
    evalNewtonRhs(z, i)
    evalNewtonStep(d, i, z)
    d1 = evalTrialStep(d, i)
    # Evaluate direction for zero interior-point parameter
    setRho(z, rho_curr)
    setMu(z, 0.0)
    evalNewtonRhs(z, i)
    evalNewtonStep(d, i, z)
    d2 = evalTrialStep(d, i)
    # Evaluate direction for zero penalty parameter
    setRho(z, 0.0)
    setMu(z, mu_curr)
    evalNewtonRhs(z, i)
    evalNewtonStep(d, i, z)
    d3 = evalTrialStep(d, i)
    return (d1, d2, d3)
end

"""
    Evaluate trial step cut by fraction-to-boundary rule
    evalTrialStepCut(d, i, a)
"""
function evalTrialStepCut(d::PIPAL_direction, i::PIPAL_input, a::PIPAL_acceptance)
    # Set direction components
    d.x .*= a.p
    if i.nE > 0
        d.r1 .*= a.p
        d.r2 .*= a.p
        d.lE .*= a.d
    end
    if i.nI > 0
        d.s1 .*= a.p
        d.s2 .*= a.p
        d.lI .*= a.d
    end
end

"""
    Set search direction
    setDirection(d, i, dx, dr1, dr2, ds1, ds2, dlE, dlI, dx_norm, dl_norm)
"""
function setDirection(
    d::PIPAL_direction,
    i::PIPAL_input,
    dx,
    dr1,
    dr2,
    ds1,
    ds2,
    dlE,
    dlI,
    dx_norm,
    dl_norm,
)
    # Set primal variables
    d.x .= dx
    if i.nE > 0
        d.r1 = dr1
        d.r2 = dr2
        d.lE = dlE
    end
    if i.nI > 0
        d.s1 = ds1
        d.s2 = ds2
        d.lI = dlI
    end
    d.x_norm = dx_norm
    d.l_norm = dl_norm
end
