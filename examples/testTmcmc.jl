using Distributed, StatsBase, Distributions, PyPlot
using TransitionalMCMC

lb  = -15
ub  = 15

fT(x) = logpdf(Uniform(lb,ub), x[1]) .+ logpdf(Uniform(lb,ub), x[2])
sample_fT(Nsamples) = rand(Uniform(lb,ub), Nsamples, 2)

log_fD_T(x) = log.(pdf(MvNormal([0,0],[1 -0.5; -0.5 1]), x) + pdf(MvNormal([5,5],[1 0.5; 0.5 1]), x) + pdf(MvNormal([-5,5],[1 0.9; 0.9 1]), x))

samps, acc =tmcmc(log_fD_T, fT, sample_fT, 2000)

plt.scatter(samps[:,1], samps[:,2])
