using Distributed, StatsBase, Distributions, PyPlot
using TransitionalMCMC

lb  = -15
ub  = 15

fT(x) = logpdf(Uniform(lb,ub), x)
sample_fT(Nsamples) = rand(Uniform(lb,ub), Nsamples, 1)

function log_fD_T(x)
    return log(pdf(Normal(0,1), x[1]) + pdf(Normal(5,0.2), x[1]))
end

Nsamples = 2000
samps, acc = tmcmc(log_fD_T, fT, sample_fT, Nsamples)

plt.hist(samps,50)