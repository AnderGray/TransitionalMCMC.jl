# using PyPlot
using Distributed

addprocs(4; exeflags="--project")
@everywhere using TransitionalMCMC

# Prior Bounds
@everywhere lb, ub  = -5, 5

# Prior Density and sampler
@everywhere priorDen(x) = pdf(Uniform(lb, ub), x[1,:]) .* pdf(Uniform(lb, ub), x[2,:])
@everywhere priorRnd(Nsamples) = rand(Uniform(lb, ub), Nsamples, 2)

# Log Likelihood
@everywhere function logLik(x)
    sleep(0.005)
    return -1 .* ((x[1,:].^2 .+ x[2,:] .- 11).^2 .+ (x[1,:] .+ x[2,:].^2 .- 7).^2)
end

Nsamples = 200
Ndims = 2

samps, acc = tmcmc_par(logLik, priorDen, priorRnd, Nsamples, Ndims, 5, 2)

# plt.scatter(samps[:,1], samps[:,2])

#= Nsamples = 2000
beta2 = 0.01
fT(x) = priorDen(x)
sample_fT(x) = priorRnd(x)
log_fD_T(x) = logLik(x) =#

rmprocs(workers())