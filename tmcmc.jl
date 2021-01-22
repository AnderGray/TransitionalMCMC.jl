using Distributions
using Optim
using StatsBase   


function tmcmc(LogLik, prior, priorRnd, Nsamples, burnin = 50, LastBurnin = burnin, thin = 3, beta = 0.2)

    Bj = 0;     # Temp
    j  = 0;     # Iter

    priorSamps = priorRnd(Nsamples);

    thetaj = priorSamps;
    
    acceptanceRates = []

    while Bj < 1
        
        j = j + 1
        println("Beginning iteration $j")

        LogLikj = LogLik(thetaj)            # Array of log Lik vals of Bj
        
        #LogLikj = zeros(Nsamples)

        #=
        for i =1:Nsamples
            LogLikj[i] = LogLik(thetaj[i,:])
        end
        =#
        println("Computing Bj")
        Bj1 = computeBj(LogLikj, Bj)
        println("B_$j = $Bj1")
        # Define target 
        target(x) = LogLik(x) * Bj1 .+ log.(prior(x))

        println("Computing weights")
        # Computing proposal dist
        Liks = exp.(LogLikj .* (Bj1 - Bj) )
        wTheta = Liks./sum(Liks)            # Normalised weights

        
        mWj = sum(wTheta .* thetaj)         # Samples 
        cov = beta^2 * sum(wTheta .* (thetaj .- mWj)' * (thetaj .-mWj))
        
        prop(mu) = proprnd(mu, cov, prior)

        # Better to sample index
        resamps = sample(thetaj, Weights(wTheta[:]), Nsamples, replace=true)
        
        thetaj1 = deepcopy(resamps)

        acc = zeros(Nsamples)

        
        println("Markov chains...")
        for i = 1:Nsamples

            this, that = MHsampleLog(target, prop, resamps[i,:], 1, 20)

            thetaj1[i] = this[1]
            acc[i] = that


        end

        meanAcc = mean(acc)
        push!(acceptanceRates, meanAcc)
        Bj = Bj1
        thetaj = thetaj1
    end

    
    return thetaj, acceptanceRates
end

function computeBj(logLiks, pj)

    threshold = 1

    wj(e) = exp.(abs.(e) .* logLiks)
    fmin(e) = std(wj(e)) - (threshold * mean(wj(e))) + eps()
    
    
    e = abs(optimize(fmin,0,1).minimum)

    return min(1, pj + e)

end

function proprnd(mu, cov, prior)

    samp = rand(MvNormal(mu, cov),1)
    while prior(samp) == 0
        samp = rand(MvNormal(mu, cov),1)
    end
    return samp

end

#=
function computeBj( logLiks, pj)

    threshold = 1

    wj(e) = exp.(abs.(e) .* logLiks)
    fmin(e) = std(wj(e)) - threshold * mean(wj(e)) + nextfloat(0.0)
    
    
    e = abs(optimize(fmin,[0.0], Newton()).minimum)

    return min(1, pj + e)

end
=#