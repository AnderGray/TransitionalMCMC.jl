using Distributed, StatsBase


addprocs(4)
@everywhere begin

    using TransitionalMCMC, Distributions, LinearAlgebra
    Ndims = 10
    # Prior Bounds
    lb, ub  = -10, 10

    mean1 = ones(Ndims) * 5
    mean2 = ones(Ndims) * (-5)
    
    cov = 1* Matrix(I, Ndims, Ndims)

    # Prior Density and sampler
    priorDen(x) = sum(logpdf(Uniform.(lb, ub), x))
    priorRnd(Nsamples) = rand(Uniform(lb, ub), Nsamples, Ndims)

    # Log Likelihood
    logLik(x) = log.(pdf(MvNormal(mean1, cov), x) .+ pdf(MvNormal(mean2, cov), x))

end

Nsamples = 2000

@time samps, acc = tmcmc(logLik, priorDen, priorRnd, Nsamples, 5, 2)

#rmprocs(workers())