using Distributions




function MHsampleLog(Target, PropRnd, start, Nsamples, burnin = 50, thin = 3)

    dims = length(start)

    chain = zeros(Nsamples+burnin, dims)
    chain[1,:] = start
    accRate = 0

    for i = 2:(Nsamples+burnin )    

        next = PropRnd(chain[i-1,:])              # Draw candidate

        #propDen = Proposal(chain[end], next)    # Prop Density
        targDen = exp(Target(next))    # Target Density
        targPrevious = exp(Target(chain[i-1,:]))     # Target Density

        α = min(1,  targDen/targPrevious)

        accepted = α >= rand()


        if accepted
            #push!(chain, next)
            chain[i,:] = next
            accRate = accRate +1
        else
            #push!(chain, chain[end])
            chain[i,:] = chain[i-1, :]
        end

    end
    accRate = accRate/(Nsamples+burnin)
    return chain[burnin+1:end,:], accRate
end

function MHsample(Target, PropRnd, start, Nsamples, burnin = 50, thin = 3)

    dims = length(start)

    chain = zeros(Nsamples+burnin, dims)
    chain[1,:] = start
    accRate = 0

    for i = 2:(Nsamples+burnin )    

        next = PropRnd(chain[i-1,:])              # Draw candidate

        #propDen = Proposal(chain[end], next)    # Prop Density
        targDen = Target(next)    # Target Density
        targPrevious = Target(chain[i-1,:])     # Target Density

        α = min(1,  targDen/targPrevious)

        accepted = α >= rand()


        if accepted
            #push!(chain, next)
            chain[i,:] = next
            accRate = accRate +1
        else
            #push!(chain, chain[end])
            chain[i,:] = chain[i-1, :]
        end

    end
    accRate = accRate/(Nsamples+burnin)
    return chain[burnin+1:end,:], accRate
end