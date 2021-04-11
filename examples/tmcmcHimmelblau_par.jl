using Distributed, StatsBase, Distributions, PyPlot

addprocs(4; exeflags="--project")
@everywhere begin
    
    using TransitionalMCMC

    # Prior Bounds
    lb, ub  = -5, 5

    # Prior Density and sampler
    logprior(x) = logpdf(Uniform(lb, ub), x[1,:]) .+ logpdf(Uniform(lb, ub), x[2,:])
    priorRnd(Nsamples) = rand(Uniform(lb, ub), Nsamples, 2)

    # Log Likelihood
    function logLik(x)
        sleep(0.005)
        return -1 .* ((x[1,:].^2 .+ x[2,:] .- 11).^2 .+ (x[1,:] .+ x[2,:].^2 .- 7).^2)
    end

end

Nsamples = 200

samps, acc = tmcmc(logLik, logprior, priorRnd, Nsamples, 5, 2)

# plt.scatter(samps[:,1], samps[:,2])

rmprocs(workers())