using Distributed, StatsBase, ClusterManagers

addprocs(SlurmManager(parse(Int64, ARGS[1])))
@everywhere begin

    using TransitionalMCMC, Distributions

    # Prior Bounds
    lb, ub  = -5, 5

    # Prior Density and sampler
    priorDen(x) = pdf(Uniform(lb, ub), x[1,:]) .* pdf(Uniform(lb, ub), x[2,:])
    priorRnd(Nsamples) = rand(Uniform(lb, ub), Nsamples, 2)

    # Log Likelihood
    function logLik(x)
        sleep(5)
        return -1 .* ((x[1,:].^2 .+ x[2,:] .- 11).^2 .+ (x[1,:] .+ x[2,:].^2 .- 7).^2)
    end

end

Nsamples = 2000

@time samps, acc = tmcmc(logLik, priorDen, priorRnd, Nsamples, 5, 2)

rmprocs(workers())