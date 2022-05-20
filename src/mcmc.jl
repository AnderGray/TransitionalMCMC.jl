###
#   Simple implmentation of the Metropolis Hasting algorithm in julia
#
#            Institute for Risk and Uncertainty, Uni of Liverpool
#
#                       Authors: Ander Gray, Adolphus Lye
#
#                       Email: Ander.Gray@liverpool.ac.uk,
#                              Adolphus.Lye@liverpool.ac.uk
#
#
#
#       W. K. Hastings (1970). Monte Carlo sampling methods using
#       Markov chains and their applications. Biometrika,
#       57(1), 97-109. doi:10.1093/biomet/57.1.97
#
###

###
#   Add description of function and inputs
###

function metropolis_hastings(
    Target::Function,
    Prop::Function,
    start::AbstractVector{<:Real},
    Nsamples::Integer,
    burnin::Integer=50,
    thin::Integer=3;
    islogged::Bool=true,
)
    dims = length(start)                                    # Dimensions of input/ prior

    PropRnd = mu -> rand(Prop(mu))                          # Generates a sample from the proposal given mean mu

    if dims > 1
        chain = zeros(Nsamples * thin + burnin, dims)
        chain[1, :] = start
    else
        chain = zeros(Nsamples * thin + burnin)
        chain[1] = start[1]
    end

    accRate = 0

    islogged ? evalDen = x -> exp(Target(x)) : evalDen = x -> Target(x)

    for i in 2:(Nsamples * thin + burnin)
        current = dims > 1 ? chain[i - 1, :] : chain[i - 1]
        next = PropRnd(current)              # Draw candidate

        targDen = evalDen(next)                  # Target Density at next sample
        targPrevious = evalDen(current)     # Target Density at current sample

        propDen = pdf(Prop(current), next)         # Proposal at next centred at current
        propPrevious = pdf(Prop(next), current)    # Propsoal at current centred at next

        α = targDen / targPrevious * propPrevious / propDen     # General formula for acceptance probability

        accepted = α >= rand()

        if accepted
            dims > 1 ? chain[i, :] = next : chain[i] = next
            accRate = accRate + 1
        else
            dims > 1 ? chain[i, :] = current : chain[i] = current
        end
    end
    accRate = accRate / (Nsamples * thin + burnin)

    if dims > 1
        chain = chain[(burnin + 1):thin:end, :]
    else
        chain = chain[(burnin + 1):thin:end]
    end

    return chain, accRate
end

function metropolis_hastings(
    Target::Function,
    Prop::Function,
    start::Real,
    Nsamples::Integer,
    burnin::Integer=50,
    thin::Integer=3;
    islogged::Bool=true,
)
    return metropolis_hastings(
        Target, Prop, [start], Nsamples, burnin, thin; islogged=islogged
    )
end

##
#   Symetric proposal and logged target
##
function metropolis_hastings_simple(
    Target::Function,
    PropRnd::Function,
    start::AbstractVector{<:Real},
    Nsamples::Integer,
    burnin::Integer=50,
    thin::Integer=3,
)
    dims = length(start) # Dimensions of input/ prior

    if dims > 1
        chain = zeros(dims, Nsamples * thin + burnin)
        chain[:, 1] = start
    else
        chain = zeros(Nsamples * thin + burnin)
        chain[1] = start[1]
    end

    accRate = 0

    for i in 2:(Nsamples * thin + burnin)
        current = dims > 1 ? chain[:, i - 1] : chain[i - 1]
        next = PropRnd(current)                 # Draw candidate

        targDen = Target(next)[1]                    # Target Density at next sample
        targPrevious = Target(current)[1]       # Target Density at current sample

        α = min(0, targDen - targPrevious)

        accepted = α >= log(rand())

        if accepted
            dims > 1 ? chain[:, i] = next : chain[i] = next
            accRate = accRate + 1
        else
            dims > 1 ? chain[:, i] = current : chain[i] = current
        end
    end

    accRate = accRate / (Nsamples * thin + burnin)

    if dims > 1
        chain = chain[:, (burnin + 1):thin:end]
    else
        chain = chain[(burnin + 1):thin:end]
    end

    return chain', accRate
end

function metropolis_hastings_simple(
    Target::Function,
    PropRnd::Function,
    start::Real,
    Nsamples::Integer,
    burnin::Integer=50,
    thin::Integer=3,
)
    return metropolis_hastings_simple(Target, PropRnd, [start], Nsamples, burnin, thin)
end
