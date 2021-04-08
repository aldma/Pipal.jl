# Pipal.jl

Pipal.jl provides a solver for nonlinear optimization problems of the form
```
	minimize    f(x)
	subject to  cL ≤ c(x) ≤ cU,
	            xL ≤   x  ≤ xU,
```
where `f` and `c` are assumed to be continuously differentiable in `Rⁿ`. Pipal.jl implements PIPAL, a penalty-interior-point algorithm for nonlinear opimization of potentially infeasible problems.

Pipal.jl is [JSO](https://github.com/JuliaSmoothOptimizers)-compliant, namely it requires [NLPModels.jl](https://github.com/JuliaSmoothOptimizers/NLPModels.jl) for defining the problem and uses [SolverCore.jl](https://github.com/JuliaSmoothOptimizers/SolverCore.jl) for the output. Given the NLPModel `nlp`, the solver can be consumed via the commands
```julia
julia> using Pipal
julia> out = pipal(nlp)
```
where the GenericExecutionStats `out` collects information about the solution.

This package is based on the original MATLAB code by Frank E. Curtis, in particular on PIPAL_1.2.zip. See [https://coral.ise.lehigh.edu/frankecurtis/software/](https://coral.ise.lehigh.edu/frankecurtis/software/). The original code was not accompanied by an open-source license. Frank E. Curtis has kindly provided his consent in writing to allow distribution of this Julia translation. See the `consent` folder.

Please cite this repository if you use Pipal.jl in your work: see `CITATION.bib`.

## References

Frank E. Curtis, *A Penalty-Interior-Point Algorithm for Nonlinear Constrained Optimization*, Mathematical Programming Computation 4, pages 181--209, 2012. DOI [10.1007/s12532-012-0041-4](https://doi.org/10.1007/s12532-012-0041-4).
