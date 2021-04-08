"""
    constructor
    a = Acceptance()
"""
function Acceptance()
    return PIPAL_acceptance(0.0, 0.0, 0.0, false)
end

"""
    Line-search procedure
    lineSearch(a, p, i, c, z, d)
"""
function lineSearch(
    a::PIPAL_acceptance,
    p::PIPAL_parameters,
    i::PIPAL_input,
    c::PIPAL_counters,
    z::PIPAL_iterate,
    d::PIPAL_direction,
)
    # Check fraction-to-boundary rule
    fractionToBoundary(a, p, i, z, d)
    # Check for full step for trial penalty parameters
    b = fullStepCheck(a, p, i, c, z, d)
    # Run second-order correction
    a.s = false
    if b == 0
        b = secondOrderCorrection(a, p, i, c, z, d)
        if b == 2
            a.s = true
        end
    end
    # Run backtracking line search
    if b == 0
        backtracking(a, p, i, c, z, d)
    end
end

"""
    Fraction-to-boundary line search
    fractionToBoundary(a, p, i, z, d)
"""
function fractionToBoundary(
    a::PIPAL_acceptance,
    p::PIPAL_parameters,
    i::PIPAL_input,
    z::PIPAL_iterate,
    d::PIPAL_direction,
)
    # Initialize primal fraction-to-boundary
    a.p0 = 1.0
    # Update primal fraction-to-boundary for constraint slacks
    if i.nE > 0
        a.p0 = minimum(
            [
                a.p0
                (min(p.ls_frac, z.mu) - 1.0) .* (z.r1[d.r1.<0.0] ./ d.r1[d.r1.<0.0])
                (min(p.ls_frac, z.mu) - 1.0) .* (z.r2[d.r2.<0.0] ./ d.r2[d.r2.<0.0])
            ],
        )
    end
    if i.nI > 0
        a.p0 = minimum(
            [
                a.p0
                (min(p.ls_frac, z.mu) - 1.0) .* (z.s1[d.s1.<0.0] ./ d.s1[d.s1.<0.0])
                (min(p.ls_frac, z.mu) - 1.0) .* (z.s2[d.s2.<0.0] ./ d.s2[d.s2.<0.0])
            ],
        )
    end
    # Initialize primal steplength
    a.p = a.p0
    # Initialize dual fraction-to-boundary
    a.d = 1.0
    # Update dual fraction-to-boundary for constraint multipliers
    if i.nE > 0
        a.d = minimum(
            [
                a.d
                (min(p.ls_frac, z.mu) - 1.0) .*
                ((1.0 .+ z.lE[d.lE.<0.0]) ./ d.lE[d.lE.<0.0])
                (1.0 - min(p.ls_frac, z.mu)) .*
                ((1.0 .- z.lE[d.lE.>0.0]) ./ d.lE[d.lE.>0.0])
            ],
        )
    end
    if i.nI > 0
        a.d = minimum(
            [
                a.d
                (min(p.ls_frac, z.mu) - 1.0) .* (z.lI[d.lI.<0.0] ./ d.lI[d.lI.<0.0])
                (1.0 - min(p.ls_frac, z.mu)) .*
                ((1.0 .- z.lI[d.lI.>0.0]) ./ d.lI[d.lI.>0.0])
            ],
        )
    end
end


"""
    Full step search for trial penalty parameters
    fullStepCheck(a, p, i, c, z, d)
"""
function fullStepCheck(
    a::PIPAL_acceptance,
    p::PIPAL_parameters,
    i::PIPAL_input,
    c::PIPAL_counters,
    z::PIPAL_iterate,
    d::PIPAL_direction,
)
    # Initialize boolean
    b = 0
    # Set current and last penalty parameters
    rho = copy(z.rho)
    rho_temp = copy(z.rho_)
    # allocate temporary values
    x = similar(z.x)
    cE = similar(z.cE)
    r1 = similar(z.r1)
    r2 = similar(z.r2)
    lE = similar(z.lE)
    cI = similar(z.cI)
    s1 = similar(z.s1)
    s2 = similar(z.s2)
    lI = similar(z.lI)
    # Loop through last penalty parameters
    while rho < rho_temp
        # Set penalty parameter
        setRho(z, rho_temp)
        # Evaluate merit
        evalMerit(z, i)
        # Store current values
        x .= z.x
        f = copy(z.f)
        cE .= z.cE
        r1 .= z.r1
        r2 .= z.r2
        lE .= z.lE
        cI .= z.cI
        s1 .= z.s1
        s2 .= z.s2
        lI .= z.lI
        phi = copy(z.phi)
        # Set trial point
        updatePoint(z, i, d, a)
        evalFunctions(z, i, c)
        # Check for function evaluation error
        if z.err == 0
            # Set remaining trial values
            evalSlacks(z, p, i)
            evalMerit(z, i)
            # Check for nonlinear fraction-to-boundary violation
            ftb = 0.0
            if i.nE > 0
                ftb +=
                    sum(z.r1 .< min(p.ls_frac, z.mu) .* r1) +
                    sum(z.r2 .< min(p.ls_frac, z.mu) .* r2)
            end
            if i.nI > 0
                ftb +=
                    sum(z.s1 .< min(p.ls_frac, z.mu) .* s1) +
                    sum(z.s2 .< min(p.ls_frac, z.mu) .* s2)
            end
            # Check Armijo condition
            if (ftb == 0.0) && (z.phi - phi ≤ -p.ls_thresh * a.p * max(d.qtred, 0.0))
                # Reset variables, set boolean, and return
                setPrimals(z, i, x, r1, r2, s1, s2, lE, lI, z.f, z.cE, z.cI, z.phi)
                b = 1
                return b
            else
                # Reset variables
                setPrimals(z, i, x, r1, r2, s1, s2, lE, lI, f, cE, cI, phi)
            end
        else
            # Reset variables
            setPrimals(z, i, x, r1, r2, s1, s2, lE, lI, f, cE, cI, phi)
        end
        # Decrease rho
        rho_temp *= p.rho_factor
    end
    # Set rho
    setRho(z, rho)
    # Evaluate merit
    evalMerit(z, i)
    return b
end

"""
    Second-order Correction
    secondOrderCorrection(a, p, i, c, z, d)
"""
function secondOrderCorrection(
    a::PIPAL_acceptance,
    p::PIPAL_parameters,
    i::PIPAL_input,
    c::PIPAL_counters,
    z::PIPAL_iterate,
    d::PIPAL_direction,
)
    # Initialize flag
    b = 0
    # Store current iterate values
    x = copy(z.x)
    f = copy(z.f)
    cE = copy(z.cE)
    r1 = copy(z.r1)
    r2 = copy(z.r2)
    lE = copy(z.lE)
    cI = copy(z.cI)
    s1 = copy(z.s1)
    s2 = copy(z.s2)
    lI = copy(z.lI)
    phi = copy(z.phi)
    v = copy(z.v)
    # Set trial point
    updatePoint(z, i, d, a)
    evalFunctions(z, i, c)
    # Check for function evaluation error
    if z.err == 0
        # Set remaining trial values
        evalSlacks(z, p, i)
        evalMerit(z, i)
        # Check for nonlinear fraction-to-boundary violation
        ftb = 0.0
        if i.nE > 0
            ftb +=
                sum(z.r1 .< min(p.ls_frac, z.mu) .* r1) +
                sum(z.r2 .< min(p.ls_frac, z.mu) .* r2)
        end
        if i.nI > 0
            ftb +=
                sum(z.s1 .< min(p.ls_frac, z.mu) .* s1) +
                sum(z.s2 .< min(p.ls_frac, z.mu) .* s2)
        end
        # Check Armijo condition
        if ftb == 0.0 && z.phi - phi ≤ -p.ls_thresh * a.p * max(d.qtred, 0.0)
            # Reset variables, set flag, and return
            setPrimals(z, i, x, r1, r2, s1, s2, lE, lI, z.f, z.cE, z.cI, z.phi)
            b = 1
            return b
        elseif evalViolation(i, z.cE, z.cI) < v
            # Reset variables and return
            setPrimals(z, i, x, r1, r2, s1, s2, lE, lI, f, cE, cI, phi)
            return b
        else
            # Reset variables (but leave constraint values for second-order correction)
            setPrimals(z, i, x, r1, r2, s1, s2, lE, lI, f, z.cE, z.cI, phi)
        end
    else
        # Reset variables and return
        setPrimals(z, i, x, r1, r2, s1, s2, lE, lI, f, cE, cI, phi)
        return b
    end
    # Recompute slacks for second order correction
    evalSlacks(z, p, i)
    # Evaluate trial primal-dual right-hand side vector
    evalNewtonRhs(z, i)
    # Store current direction values
    dx = copy(d.x)
    dr1 = copy(d.r1)
    dr2 = copy(d.r2)
    dlE = copy(d.lE)
    ds1 = copy(d.s1)
    ds2 = copy(d.s2)
    dlI = copy(d.lI)
    dx_norm = copy(d.x_norm)
    dl_norm = copy(d.l_norm)
    # Evaluate search direction
    evalNewtonStep(d, i, z)
    # Set trial direction
    setDirection(
        d,
        i,
        a.p * dx + d.x,
        a.p * dr1 + d.r1,
        a.p * dr2 + d.r2,
        a.p * ds1 + d.s1,
        a.p * ds2 + d.s2,
        a.d * dlE + d.lE,
        a.d * dlI + d.lI,
        norm(a.p * dx + d.x),
        norm([a.d * dlE + d.lE; a.d * dlI + d.lI]),
    )
    # Set trial point
    updatePoint(z, i, d, a)
    evalFunctions(z, i, c)
    # Check for function evaluation error
    if z.err == 0
        # Set remaining trial values
        evalSlacks(z, p, i)
        evalMerit(z, i)
        # Check for nonlinear fraction-to-boundary violation
        ftb = 0.0
        if i.nE > 0
            ftb +=
                sum(z.r1 .< min(p.ls_frac, z.mu) .* r1) +
                sum(z.r2 .< min(p.ls_frac, z.mu) .* r2)
        end
        if i.nI > 0
            ftb +=
                sum(z.s1 .< min(p.ls_frac, z.mu) .* s1) +
                sum(z.s2 .< min(p.ls_frac, z.mu) .* s2)
        end
        # Check Armijo condition
        if ftb == 0.0 && z.phi - phi <= -p.ls_thresh * a.p * max(d.qtred, 0)
            # Reset variables, set flag, and return
            setPrimals(z, i, x, r1, r2, s1, s2, lE, lI, z.f, z.cE, z.cI, z.phi)
            b = 2
            return b
        else
            # Reset variables
            setPrimals(z, i, x, r1, r2, s1, s2, lE, lI, f, cE, cI, phi)
        end
    else
        # Reset variables
        setPrimals(z, i, x, r1, r2, s1, s2, lE, lI, f, cE, cI, phi)
    end
    # Reset direction
    setDirection(d, i, dx, dr1, dr2, ds1, ds2, dlE, dlI, dx_norm, dl_norm)
    # Reduce steplength
    a.p *= p.ls_factor
    return b
end

"""
    Backtracking line search
    backtracking(a, p, i, c, z, d)
"""
function backtracking(
    a::PIPAL_acceptance,
    p::PIPAL_parameters,
    i::PIPAL_input,
    c::PIPAL_counters,
    z::PIPAL_iterate,
    d::PIPAL_direction,
)
    T = eltype(i.x0)
    # Store current values
    x = copy(z.x)
    f = copy(z.f)
    cE = copy(z.cE)
    r1 = copy(z.r1)
    r2 = copy(z.r2)
    lE = copy(z.lE)
    cI = copy(z.cI)
    s1 = copy(z.s1)
    s2 = copy(z.s2)
    lI = copy(z.lI)
    phi = copy(z.phi)
    # Backtracking loop
    while a.p ≥ eps(T)
        # Set trial point
        updatePoint(z, i, d, a)
        evalFunctions(z, i, c)
        # Check for function evaluation error
        if z.err == 0
            # Set remaining trial values
            evalSlacks(z, p, i)
            evalMerit(z, i)
            # Check for nonlinear fraction-to-boundary violation
            ftb = 0.0
            if i.nE > 0
                ftb +=
                    sum(z.r1 .< min(p.ls_frac, z.mu) .* r1) +
                    sum(z.r2 .< min(p.ls_frac, z.mu) .* r2)
            end
            if i.nI > 0
                ftb +=
                    sum(z.s1 .< min(p.ls_frac, z.mu) .* s1) +
                    sum(z.s2 .< min(p.ls_frac, z.mu) .* s2)
            end
            # Check Armijo condition
            if ftb == 0.0 && z.phi - phi ≤ -p.ls_thresh * a.p * max(d.qtred, 0.0)
                # Reset variables and return
                setPrimals(z, i, x, r1, r2, s1, s2, lE, lI, z.f, z.cE, z.cI, z.phi)
                return
            else
                # Reset variables
                setPrimals(z, i, x, r1, r2, s1, s2, lE, lI, f, cE, cI, phi)
            end
        else
            # Reset variables
            setPrimals(z, i, x, r1, r2, s1, s2, lE, lI, f, cE, cI, phi)
        end
        # Reduce steplength
        a.p *= p.ls_factor
    end
end
