using Distributions




function MHsample(Target, Prop, start, Nsamples, burnin = 50, thin = 3; logged = true)

    dims = length(start)

    PropRnd = x -> rand(Prop(x))

    chain = zeros(Nsamples+burnin, dims)
    chain[1,:] = start
    accRate = 0

    logged ? evalDen =  x -> exp(Target(x)) : evalDen = x -> Target(x)

    for i = 2:(Nsamples+burnin )    

        next = PropRnd(chain[i-1,:])              # Draw candidate
        
        targDen = evalDen(next)    # Target Density
        targPrevious = evalDen(chain[i-1,:])     # Target Density

        propDen = pdf(Prop(chain[i-1,:]), next)
        propPrevious = pdf(Prop(next), chain[i-1,:])


        α = targDen/targPrevious * propPrevious/propDen     # General formula

        accepted = α >= rand()

        if accepted
            chain[i,:] = next
            accRate = accRate +1
        else
            chain[i,:] = chain[i-1, :]
        end

    end
    accRate = accRate/(Nsamples+burnin)
    return chain[burnin+1:end,:], accRate
end
