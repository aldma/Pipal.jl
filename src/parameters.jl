"""
    constructor
    p = Parameters(algorithm)
"""
function Parameters( ;
    algorithm::Int = 1,
    tol::Real = 1e-6,
    max_iter::Int = 3000,
    opt_err_mem::Int = 6,
    ls_factor::Real = 0.5,
    ls_thresh::Real = 1e-8,
    ls_frac::Real = 1e-2,
    pivot_thresh::Real = 0.5,
    slack_min::Real = 1e-20,
    shift_min::Real = 1e-12,
    shift_max::Real = 1e+8,
    rho_init::Real = 0.1,
    rho_min::Real = 1e-12,
    rho_factor::Real = 0.5,
    rho_trials::Int = 8,
    mu_init::Real = 0.1,
    mu_min::Real = 1e-12,
    mu_factor::Real = 0.1,
    mu_trials::Int = 4,
    mu_max::Real = 0.1,
)
    p = PIPAL_parameters(
        algorithm, # algorithm number
        0.0,
        tol, # optimality tolerance
        max_iter, # iteration limit
        0.0,
        0.0,
        0.0,
        0,
        opt_err_mem, # Optimality error history length
        ls_factor, # Line search reduction factor
        ls_thresh, # Line search threshold value
        ls_frac, # Line search fraction-to-boundary constant
        pivot_thresh, # Pivot threshold for LDL factorization
        slack_min, # Slack variable bound
        shift_min, # Hessian shift (nonzero) minimum value
        0.0,
        0.0,
        shift_max, # Hessian shift maximum value
        rho_init, # Penalty parameter initial value
        rho_min, # Penalty parameter minimum value
        rho_factor, # Penalty parameter reduction factor
        rho_trials, # Penalty parameter number of trial values per iteration
        mu_init, # Interior-point parameter initial value
        mu_min, # Interior-point parameter minimum value
        mu_factor, # Interior-point parameter reduction factor
        0.0,
        mu_trials, # Interior-point parameter number of trial values per iteration
        mu_max, # Interior-point parameter maximum value
        0.0,
        0.0,
        0.0,
        0.0,
    )
    p.mu_max_exp = 0.0   # Default interior-point parameter maximum exponent in increases
    p.rhs_bnd = 1e+18 # Max absolute value allowed for constraint right-hand side
    p.grad_max = 1e+2  # Gradient norm limit for scaling
    p.infeas_max = 1e+2  # Infeasibility limit for penalty parameter update
    p.nnz_max = 2e+4  # Max nonzeros in (upper triangle of) Newton matrix
    p.shift_factor1 = 5e-1  # Hessian shift update value (for decreases)
    p.shift_factor2 = 6e-1  # Hessian shift update value (for increases)
    p.mu_factor_exp = 1.5   # Interior-point parameter reduction exponent
    p.mu_max_exp0 = 0.0   # Interior-point parameter maximum exponent in increases (default)
    p.update_con_1 = 1e-2  # Steering rule constant 1
    p.update_con_2 = 1e-2  # Steering rule constant 2
    p.update_con_3 = 1e-2 + 1.0 # Adaptive interior-point rule constant
    return p
end

"""
    Reset interior-point parameter maximum exponent in increases to default
"""
function resetMuMaxExp(p::PIPAL_parameters)
    p.mu_max_exp = p.mu_max_exp0
end
"""
    Set interior-point parameter maximum exponent in increases to zero
"""
function setMuMaxExpZero(p::PIPAL_parameters)
    p.mu_max_exp = 0.0
end
