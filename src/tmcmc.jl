###
#   An implementation of Transitional Markov Chain Monte Carlo in Julia
#
#            Institute for Risk and Uncertainty, Uni of Liverpool
#
#                       Authors: Ander Gray, Adolphus Lye
#
#                       Email: Ander.Gray@liverpool.ac.uk,
#                              Adolphus.Lye@liverpool.ac.uk
#
#
#   This Transitional MCMC algorithm is inspirted by OpenCOSSAN:
#                   https://github.com/cossan-working-group/OpenCossan
#
#
#   Algorithm originally proposed by:
#           J. Ching, and Y. Chen (2007). Transitional Markov Chain Monte Carlo method
#           for Bayesian model updating, Model class selection, and Model averaging.
#           Journal of Engineering Mechanics, 133(7), 816-832.
#           doi:10.1061/(asce)0733-9399(2007)133:7(816)
#
###

using TimerOutputs
const to = TimerOutput()

function tmcmc(
    log_fD_T::Function,
    log_fT::Function,
    sample_fT::Function,
    Nsamples::Integer,
    burnin::Integer=20,
    thin::Integer=3,
    beta2::Float64=0.01,
)
    j1 = 0                     # Iteration number
    βj = 0                     # Tempering parameter
    θ_j = sample_fT(Nsamples)  # Samples of prior
    Lp_j = zeros(Nsamples, 1)   # Log liklihood of first iteration

    Log_ev = 0                  # Log Evidence

    Ndims = size(θ_j, 2)         # Number of dimensions (input)

    covariance_method = LinearShrinkage(DiagonalUnitVariance(), :lw) # * DiagonalUnitVariance is always SPD!

    @timeit_debug to "Main while loop" while βj < 1
        j1 = j1 + 1

        @debug "Beginning iteration $j1"

        ###
        # Parallel evaluation of the likelihood
        ###

        @debug "Computing likelihood with $(nworkers()) workers..."

        Lp_j = pmap(log_fD_T, eachrow(θ_j))
        Lp_j = reduce(vcat, Lp_j)
        Lp_adjust = maximum(Lp_j)

        ###
        #   Computing new βj
        ###
        @debug "Computing β_j and weights..."

        @timeit_debug to "Compute β and weights " βj1, w_j = _beta_and_weights(
            βj, Lp_j .- Lp_adjust
        )

        @info "β_$(j1) = $(βj1)"

        @timeit_debug to "Log evidence" Log_ev =
            log(mean(w_j)) + (βj1 - βj) * Lp_adjust + Log_ev   # Log evidence in current iteration

        prop = mu -> proprnd(mu, Σ_j, log_fT) # Anonymous function for proposal
        target = x -> log_fD_T(x) .* βj1 .+ log_fT(x) # Anonymous function for transitional distribution

        # (Normalized) Weighted resampling of θj (indices with replacement)
        @timeit_debug to "Normalised weights" wn_j = w_j ./ sum(w_j) # normalize weights
        @timeit_debug to "Compute indices" indices = sample(
            1:Nsamples, Weights(wn_j), Nsamples; replace=true
        )
        @timeit_debug to "Update θ_j1" θ_j1 = θ_j[indices, :]

        # Estimate covariance matrix
        @timeit_debug to "Compute covariance" Σ_j = beta2 * cov(covariance_method, θ_j1)

        @debug "Markov chains with $(nworkers()) workers..."

        # pmap is a parallel map
        @timeit_debug to "Run markov chains" θ_j1 = pmap(
            x -> metropolis_hastings_simple(target, prop, x, 1, burnin, thin)[1],
            eachrow(θ_j1),
        )
        θ_j1 = reduce(vcat, θ_j1)

        βj = βj1
        θ_j = θ_j1
    end
    return θ_j, Log_ev
end

"""
    _beta_and_weights(β, likelihood)

Compute the next value for `β` and the nominal weights `w` using bisection.
"""
function _beta_and_weights(β::Real, adjusted_likelihood::AbstractVector{<:Real})
    low = β
    high = 2

    local x, w # Declare variables so they are visible outside the loop

    while (high - low) / ((high + low) / 2) > 1e-6 && high > eps()
        @show (high, low)
        x = (high + low) / 2
        w = exp.((x .- β) .* adjusted_likelihood)

        if std(w) / mean(w) > 1
            high = x
        else
            low = x
        end
    end

    return min(1, x), w
end

function proprnd(mu::AbstractVector, covMat::AbstractMatrix, prior::Function)
    samp = rand(MvNormal(mu, covMat), 1)
    while isinf(prior(samp))
        samp = rand(MvNormal(mu, covMat), 1)
    end
    return samp
end

function proprnd(mu::Real, σ::AbstractMatrix, prior::Function)
    σ = σ[1]
    samp = rand(Normal(mu, σ))
    while isinf(prior(samp))
        samp = rand(Normal(mu, σ))
    end
    return samp
end
