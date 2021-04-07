using Distributions


###
#   Add description of function and inputs
###

function MHsample(Target, Prop, start, Nsamples :: Integer, burnin :: Integer = 50, thin ::Integer = 3; islogged = true)

    dims = length(start)                                    # Dimensions of input/ prior

    PropRnd = mu -> rand(Prop(mu))                          # Generates a sample from the proposal given mean mu

    chain = zeros( Nsamples * thin + burnin, dims)
    chain[1,:] = start
    accRate = 0

    islogged ? evalDen =  x -> exp(Target(x)) : evalDen = x -> Target(x)

    for i = 2:( Nsamples*thin +burnin)    

        next = PropRnd(chain[i-1,:])              # Draw candidate
        
        targDen = evalDen(next)                  # Target Density at next sample
        targPrevious = evalDen(chain[i-1,:])     # Target Density at current sample

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
    accRate = accRate/(Nsamples*thin+burnin)
    return chain[burnin+1:thin:end,:], accRate
end
