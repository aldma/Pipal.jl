import NLPModels: obj, cons, grad, jac, hess

"""
    constructor
    i = Input(p, nlp)
"""
function Input(p::PIPAL_parameters, nlp::AbstractNLPModel)
    # Set problem identity
    id = nlp.meta.name
    # Initialize data
    x = nlp.meta.x0
    bl = nlp.meta.lvar
    bu = nlp.meta.uvar
    y = nlp.meta.y0
    cl = nlp.meta.lcon
    cu = nlp.meta.ucon
    # Set number of original formulation variables
    n0 = nlp.meta.nvar
    # Find index sets
    I1 = findall((bl .≤ -p.rhs_bnd) .& (bu .≥ p.rhs_bnd))
    I2 = findall(bl .== bu)
    I3 = findall((bl .> -p.rhs_bnd) .& (bu .≥ p.rhs_bnd))
    I4 = findall((bl .≤ -p.rhs_bnd) .& (bu .< p.rhs_bnd))
    I5 = findall((bl .> -p.rhs_bnd) .& (bu .< p.rhs_bnd) .& (bl .!= bu))
    I6 = findall(cl .== cu)
    I7 = findall((cl .> -p.rhs_bnd) .& (cu .≥ p.rhs_bnd))
    I8 = findall((cl .≤ -p.rhs_bnd) .& (cu .< p.rhs_bnd))
    I9 = findall((cl .> -p.rhs_bnd) .& (cu .< p.rhs_bnd) .& (cl .!= cu))
    # Set right-hand side values
    b2 = bl[I2]
    l3 = bl[I3]
    u4 = bu[I4]
    l5 = bl[I5]
    u5 = bu[I5]
    b6 = cl[I6]
    l7 = cl[I7]
    u8 = cu[I8]
    l9 = cl[I9]
    u9 = cu[I9]
    # Set sizes of index sets
    n1 = length(I1)
    n2 = length(I2)
    n3 = length(I3)
    n4 = length(I4)
    n5 = length(I5)
    n6 = length(I6)
    n7 = length(I7)
    n8 = length(I8)
    n9 = length(I9)
    # Initialize number of invalid bounds
    vi = 0
    # Count invalid bounds
    if n2 > 0
        vi += sum(b2 .≤ -p.rhs_bnd)
        vi += sum(b2 .≥ p.rhs_bnd)
    end
    if n3 > 0
        vi += sum(l3 .≥ p.rhs_bnd)
    end
    if n4 > 0
        vi += sum(u4 .≤ -p.rhs_bnd)
    end
    if n5 > 0
        vi += sum(l5 .≥ p.rhs_bnd)
        vi += sum(u5 .≤ -p.rhs_bnd)
        vi += sum(l5 .> u5)
    end
    if n6 > 0
        vi += sum(b6 .≤ -p.rhs_bnd)
        vi += sum(b6 .≥ p.rhs_bnd)
    end
    if n7 > 0
        vi += sum(l7 .≥ p.rhs_bnd)
    end
    if n8 > 0
        vi += sum(u8 .≤ -p.rhs_bnd)
    end
    if n9 > 0
        vi += sum(l9 .≥ p.rhs_bnd)
        vi += sum(u9 .≤ -p.rhs_bnd)
        vi += sum(l9 .> u9)
    end
    # Set number of variables and constraints
    nV = n1 + n3 + n4 + n5
    nI = n3 + n4 + 2 * n5 + n7 + n8 + 2 * n9
    nE = n6
    # Set size of primal-dual matrix
    nA = nV + 3 * nE + 3 * nI
    # Set initial point
    x0 = [x[I1]; x[I3]; x[I4]; x[I5]]
    return PIPAL_input(
        id,
        n0,
        I1,
        I2,
        I3,
        I4,
        I5,
        I6,
        I7,
        I8,
        I9,
        b2,
        l3,
        u4,
        l5,
        u5,
        b6,
        l7,
        u8,
        l9,
        u9,
        n1,
        n2,
        n3,
        n4,
        n5,
        n6,
        n7,
        n8,
        n9,
        nV,
        nI,
        nE,
        nA,
        x0,
        vi,
        nlp,
    )
end

obj(i::PIPAL_input, x::Vector) = obj(i.nlp, x)
cons(i::PIPAL_input, x::Vector) = cons(i.nlp, x)
grad(i::PIPAL_input, x::Vector) = grad(i.nlp, x)
jac(i::PIPAL_input, x::Vector) = jac(i.nlp, x)
hess(i::PIPAL_input, x::Vector) = hess(i.nlp, x)
hess(i::PIPAL_input, x::Vector, y::Vector) = hess(i.nlp, x, y)
