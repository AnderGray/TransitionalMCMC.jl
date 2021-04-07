using PyPlot

lb  = -15
ub  = 15

fT(x) = pdf(Uniform(lb,ub), x[1,:])
sample_fT(Nsamples) = rand(Uniform(lb,ub), Nsamples, 1)

function log_fD_T(x)
    return log.(pdf(MvNormal([0],1), x) + pdf(MvNormal([5],0.2), x))
end

Nsamples = 2000
samps, acc =tmcmc_par(log_fD_T, fT, sample_fT, Nsamples)

plt.hist(samps,50)
plt.scatter(samps[:,1], samps[:,2])