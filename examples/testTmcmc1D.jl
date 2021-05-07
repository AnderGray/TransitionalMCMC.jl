using Distributed, StatsBase, Distributions, PyPlot
using TransitionalMCMC

lb  = -15
ub  = 15

fT(x) = logpdf(Uniform(lb,ub), x[1])
sample_fT(Nsamples) = rand(Uniform(lb,ub), Nsamples, 1)

log_fD_T(x) = log(pdf(Normal(0, 1), x[1]) + pdf(Normal(5, 0.2), x[1]))

Nsamples = 2000
samps, acc = tmcmc(log_fD_T, fT, sample_fT, Nsamples)

plt.hist(samps,50)
