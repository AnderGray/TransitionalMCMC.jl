using Distributed, StatsBase, ClusterManagers



addprocs(SlurmManager(10))

@everywhere begin

    using TransitionalMCMC, Distributions, LinearAlgebra
    Ndims = parse(Int64, ARGS[1])
    
    # Prior Bounds
    lb, ub  = -9.5, 9.5

    mean1 = ones(Ndims) * 5
    mean2 = ones(Ndims) * (-5)
    
    cov = 1* Matrix(I, Ndims, Ndims)

    # Prior Density and sampler
    priorDen(x) = sum(logpdf(Uniform.(lb, ub), x),dims= 2)
    priorRnd(Nsamples) = rand(Uniform(lb, ub), Nsamples, Ndims)

    # Log Likelihood
    logLik(x) = log.(pdf(MvNormal(mean1, cov), x) .+ pdf(MvNormal(mean2, cov), x))

end

Nsamples = 1000

@time samps, acc = tmcmc(logLik, priorDen, priorRnd, Nsamples, 5, 2)

rmprocs(workers())