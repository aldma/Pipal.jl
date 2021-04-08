using NLPModels
using LDLFactorizations

"""
    parameters
"""
mutable struct PIPAL_parameters{T<:Real}
    algorithm::Int      # Default algorithm number
    mu_max_exp::T       # Default interior-point parameter maximum exponent in increases
    opt_err_tol::T      # Default optimality tolerance
    iter_max::Int       # Default iteration limit
    rhs_bnd::T          # Max absolute value allowed for constraint right-hand side
    grad_max::T         # Gradient norm limit for scaling
    infeas_max::T       # Infeasibility limit for penalty parameter update
    nnz_max::Int        # Max nonzeros in (upper triangle of) Newton matrix
    opt_err_mem::Int    # Optimality error history length
    ls_factor::T        # Line search reduction factor
    ls_thresh::T        # Line search threshold value
    ls_frac::T          # Line search fraction-to-boundary constant
    pivot_thresh::T     # Pivot threshold for LDL factorization
    slack_min::T        # Slack variable bound
    shift_min::T        # Hessian shift (nonzero) minimum value
    shift_factor1::T    # Hessian shift update value (for decreases)
    shift_factor2::T    # Hessian shift update value (for increases)
    shift_max::T        # Hessian shift maximum value
    rho_init::T         # Penalty parameter initial value
    rho_min::T          # Penalty parameter minimum value
    rho_factor::T       # Penalty parameter reduction factor
    rho_trials::Int     # Penalty parameter number of trial values per iteration
    mu_init::T          # Interior-point parameter initial value
    mu_min::T           # Interior-point parameter minimum value
    mu_factor::T        # Interior-point parameter reduction factor
    mu_factor_exp::T    # Interior-point parameter reduction exponent
    mu_trials::Int      # Interior-point parameter number of trial values per iteration
    mu_max::T           # Interior-point parameter maximum value
    mu_max_exp0::T      # Interior-point parameter maximum exponent in increases (default)
    update_con_1::T     # Steering rule constant 1
    update_con_2::T     # Steering rule constant 2
    update_con_3::T     # Adaptive interior-point rule constant
end

"""
    counters
"""
mutable struct PIPAL_counters
    f::Int # Function evaluation counter
    g::Int # Gradient evaluation counter
    H::Int # Hessian evaluation counter
    M::Int # Matrix factorization counter
    k::Int # Iteration counter
    t0::Real # Start time
end

"""
    input
"""
mutable struct PIPAL_input
    id::String # problem name
    n0::Int # number of original formulation variables
    I1::Vector{Int} # indices of free variables
    I2::Vector{Int} # indices of fixed variables
    I3::Vector{Int} # indices of lower bounded variables
    I4::Vector{Int} # indices of upper bounded variables
    I5::Vector{Int} # indices of lower and upper bounded variables
    I6::Vector{Int} # indices of equality constraints
    I7::Vector{Int} # indices of lower bounded constraints
    I8::Vector{Int} # indices of upper bounded constraints
    I9::Vector{Int} # indices of lower and upper bounded constraints
    b2::Vector # right-hand side of fixed variables
    l3::Vector # right-hand side of lower bounded variables
    u4::Vector # right-hand side of upper bounded variables
    l5::Vector # right-hand side of lower half of lower and upper bounded variables
    u5::Vector # right-hand side of upper half of lower and upper bounded variables
    b6::Vector # right-hand side of equality constraints
    l7::Vector # right-hand side of lower bounded constraints
    u8::Vector # right-hand side of upper bounded constraints
    l9::Vector # right-hand side of lower half of lower and upper bounded constraints
    u9::Vector # right-hand side of upper half of lower and upper bounded constraints
    n1::Int # number of free variables
    n2::Int # number of fixed variables
    n3::Int # number of lower bounded variables
    n4::Int # number of upper bounded variables
    n5::Int # number of lower and upper bounded variables
    n6::Int # number of equality constraints
    n7::Int # number of lower bounded constraints
    n8::Int # number of upper bounded constraints
    n9::Int # number of lower and upper bounded constraints
    nV::Int # number of variables
    nI::Int # number of inequality constraints
    nE::Int # number of equality constraints
    nA::Int # size of primal-dual matrix
    x0::Vector # initial point
    vi::Int # counter for invalid bounds
    nlp::AbstractNLPModel # NLP model
end

"""
    iterate
"""
mutable struct PIPAL_iterate{T<:Real,Tx<:AbstractVector{T}}
    x::Tx           # Primal point
    rho::T          # Penalty parameter value
    rho_::T         # Penalty parameter last value
    mu::T           # Interior-point parameter value
    f::T            # Objective function value (scaled)
    fu::T           # Objective function value (unscaled)
    g::Tx           # Objective gradient value
    r1::Tx          # Equality constraint slack value
    r2::Tx          # Equality constraint slack value
    cE::Tx          # Equality constraint value (scaled)
    JE::AbstractMatrix      # Equality constraint Jacobian value
    JEnnz::Int      # Equality constraint Jacobian nonzeros
    lE::Tx          # Equality constraint multipliers
    s1::Tx          # Inequality constraint slack value
    s2::Tx          # Inequality constraint slack value
    cI::Tx          # Inequality constraint value (scaled)
    JI::AbstractMatrix      # Inequality constraint Jacobian value
    JInnz::Int      # Inequality constraint Jacobian nonzeros
    lI::Tx          # Inequality constraint multipliers
    H::AbstractMatrix       # Hessian of Lagrangian
    Hnnz::Int       # Hessian of Lagrangian nonzeros
    v::T            # Feasibility violation measure value (scaled)
    vu::T           # Feasibility violation measure value (unscaled)
    v0::T           # Feasibility violation measure initial value
    phi::T          # Merit function value
    Af::LDLFactorizations.LDLFactorization # Newton matrix LDLᵀ factorization
    Annz::Int       # Newton matrix (upper triangle) nonzeros
    shift::T        # Hessian shift value
    b::Tx           # Newton right-hand side
    kkt::Tx         # KKT errors
    kkt_::Tx        # KKT errors last value
    err::Int        # Function evaluation error flag
    fs::T           # Objective scaling factor
    cEs::Tx         # Equality constraint scaling factors
    cEu::Tx         # Equality constraint value (unscaled)
    cIs::Tx         # Inequality constraint scaling factors
    cIu::Tx         # Inequality constraint value (unscaled)
    A::AbstractMatrix        # Newton matrix
    shift22::T      # Newton matrix (2,2)-block shift value
    v_::T           # Feasibility violation measure last value
    cut_::Bool      # Boolean value for last backtracking line search
end

"""
    direction
"""
mutable struct PIPAL_direction{T<:Real,Tx<:AbstractVector{T}}
    x::Tx       # Primal direction
    x_norm::T   # Primal direction norm value
    x_norm_::T  # Primal direction norm last value
    r1::Tx      # Equality constraint slack direction
    r2::Tx      # Equality constraint slack direction
    lE::Tx      # Equality constraint multiplier direction
    s1::Tx      # Inequality constraint slack direction
    s2::Tx      # Inequality constraint slack direction
    lI::Tx      # Inequality constraint multiplier direction
    l_norm::T   # Constraint multiplier direction norm
    lred0::T    # Penalty-interior-point linear model value for zero penalty parameter
    ltred0::T   # Penalty-interior-point linear model reduction value for zero penalty parameter
    ltred::T    # Penalty-interior-point linear model reduction value
    qtred::T    # Penalty-interior-point quadratic model reduction value
    m::T        # Quality function value
end

"""
    acceptance
"""
mutable struct PIPAL_acceptance{T<:Real}
    p0::T       # Fraction-to-the-boundary steplength
    p::T        # Primal steplength
    d::T        # Dual steplength
    s::Bool     # Bool for second-order correction
end
