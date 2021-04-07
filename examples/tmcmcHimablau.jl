using PyPlot

# Prior Bounds
lb  = -5        
ub  = 5

# Prior Density and sampler
priorDen(x) = pdf(Uniform(lb,ub), x[1,:]) .* pdf(Uniform(lb,ub), x[2,:])
priorRnd(Nsamples) = rand(Uniform(lb,ub), Nsamples, 2)

# Log Likelihood
logLik(x) = -1 .* ((x[1,:].^2 .+ x[2,:] .- 11).^2 .+ (x[1,:] .+ x[2,:].^2 .- 7).^2)

samps, acc =tmcmc(logLik, priorDen, priorRnd, 2000)

plt.scatter(samps[:,1], samps[:,2])