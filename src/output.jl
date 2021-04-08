"""
    Print break in output
"""
function printBreak(p::PIPAL_parameters{T}, c::PIPAL_counters) where {T}
    # Print break every 20 iterations
    if c.k % 20 == 0
        @info log_header([:iter, :objective, :primal_feas, :dual_feas, :rho, :mu],
                         [Int, T, T, T, T, T],
                         hdr_override=Dict(:rho =>"ρ", :mu => "μ"))
    end
end

"""
    Print iterate information
"""
function printIterate(c::PIPAL_counters, z::PIPAL_iterate)
    # Print iterate information
    @info log_row(Any[c.k, z.f, z.v, z.kkt[2], z.rho, z.mu])
end
