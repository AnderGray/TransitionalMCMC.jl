using PyPlot

include("mcmc.jl")
include("tmcmc.jl")


lb  = -15
ub  = 15

fT(x) = pdf(Uniform(lb,ub), x[1,:]) .* pdf(Uniform(lb,ub), x[2,:])
sample_fT(Nsamples) = rand(Uniform(lb,ub), Nsamples, 2)

function log_fD_T(x)
    return log.(pdf(MvNormal([0,0],[1 -0.5; -0.5 1]), x) + pdf(MvNormal([5,5],[1 0.5; 0.5 1]), x) + pdf(MvNormal([-5,5],[1 0.9; 0.9 1]), x))
end

Nsamples = 2000
burnin= 50
beta = 0.2
thin = 3

#samps, acc = tmcmc_par(log_fD_T, fT, sample_fT, Nsamples)
samps, acc =tmcmc(log_fD_T, fT, sample_fT, Nsamples)

plt.scatter(samps[:,1], samps[:,2])