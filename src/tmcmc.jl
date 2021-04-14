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

function tmcmc(
    log_fD_T::Function,
    log_fT::Function,
    sample_fT::Function,
    Nsamples::Integer,
    burnin::Integer=20,
    thin::Integer=3,
    beta2::Float64=0.01)

    j1 = 0;                     # Iteration number
    βj = 0;                     # Tempering parameter
    θ_j = sample_fT(Nsamples);  # Samples of prior
    Lp_j = zeros(Nsamples, 1);   # Log liklihood of first iteration

    Log_ev = 0                  # Log Evidence

    Ndims = size(θ_j, 2)         # Number of dimensions (input)

    while βj < 1

        j1 = j1 + 1

        @debug "Beginnig iteration $j1"

        ###
        # Parallel evaluation of the likelihood
        ###

        @debug "Computing likelihood with $(nworkers()) workers..."

        ins = [θ_j[i,:] for i in 1:size(θ_j, 1)]
        Lp_j = pmap(log_fD_T, ins)
        Lp_j = reduce(vcat, Lp_j)

        ###
        #   Computing new βj
        #   Uses bisection method
        ###
        @debug "Computing β_j..."

        low_β = βj; hi_β = 2; Lp_adjust = maximum(Lp_j);
        x1 = (hi_β + low_β) / 2;

        while (hi_β - low_β) / ((hi_β + low_β) / 2) > 1e-6
            x1 = (hi_β + low_β) / 2;
            wj_test = exp.((x1 .- βj ) .* (Lp_j .- Lp_adjust));
            cov_w   = std(wj_test) / mean(wj_test);
            if cov_w > 1; hi_β = x1; else; low_β = x1; end
        end

        βj1 = min(1, x1)

        @info "β_$(j1) = $(βj1)"

        ###
        #   Computation of normalised weights
        ###
        @debug "Computing weights..."

        w_j = exp.((βj1 - βj) .* (Lp_j .- Lp_adjust))       # Nominal weights from likilhood and βjs

        Log_ev = log(mean(w_j)) + (βj1 - βj) * Lp_adjust + Log_ev   # Log evidence in current iteration

        # Normalised weights
        wn_j = w_j ./ sum(w_j);

        Th_wm = θ_j .* wn_j                 # Weighted mean of samples

        ###
        #   Calculation of COV matrix of proposal
        ###
        Σ_j    = zeros(Ndims, Ndims)

        for l = 1:Nsamples
            Σ_j = Σ_j + beta2 .* wn_j[l] .* (θ_j[l,:]' .- Th_wm)' * (θ_j[l,:]' .- Th_wm)
        end

        # Ensure that cov is symetric
        Σ_j = (Σ_j' + Σ_j) / 2

        prop = mu -> proprnd(mu, Σ_j, log_fT) # Anonymous function for proposal

        target = x -> log_fD_T(x) .* βj1 .+ log_fT(x) # Anonymous function for transitional distribution

        # Weighted resampling of θj (indecies with replacement)
        randIndex = sample(1:Nsamples, Weights(wn_j), Nsamples, replace=true)

        ins = [θ_j[randIndex[i], :] for i = 1:Nsamples]

        @debug "Markov chains with $(nworkers()) workers..."
        # pmap is a parallel map
        θ_j1 = pmap(x -> run_chains(target, prop, x, burnin, thin), ins)

        θ_j1 = reduce(vcat, θ_j1)

        βj = βj1
        θ_j = θ_j1
    end
    return θ_j, Log_ev
end


function proprnd(mu::AbstractVector, covMat::AbstractMatrix, prior::Function)
    samp = rand(MvNormal(mu, covMat), 1)
    while isinf(prior(samp))
        samp = rand(MvNormal(mu, covMat), 1)
    end
    return samp[:]
end

function run_chains(target::Function, prop::Function, θ_js::Vector{<:Real}, burnin::Integer, thin::Integer)
    samps, _  = metropolis_hastings_simple(target, prop, θ_js, 1, burnin, thin)
    return samps
end
