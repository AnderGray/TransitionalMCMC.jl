include("mcmc.jl")
include("tmcmc.jl")


lb  = -15
ub  = 15

prior(x) = pdf(Uniform(lb,ub), x)
priorRnd(Nsamples) = rand(Uniform(lb,ub), Nsamples)

function target(x)
    return log.(pdf.(Normal(0,1),x))
end


samps, acc = tmcmc(target, prior, priorRnd, 200)