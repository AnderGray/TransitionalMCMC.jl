using Distributions
using StatsBase   



function tmcmc(log_fD_T, fT, sample_fT, Nsamples, beta2 = 0.01, burnin= 20, thin=3)

    j1 = 0;                     # Iteration number
    βj = 0;                     # Tempering parameter
    θ_j = sample_fT(Nsamples);  # Samples of prior
    Lp_j = zeros(Nsamples,1);   # Log liklihood of first iteration

    Log_ev = 0                  # Log Evidence

    Ndims = size(θ_j,2)         # Number of dimensions (input)
    

    while βj < 1

        j1 = j1 + 1
        println()
        println("Beginning iteration $j1")

        ###
        # Compute likelihood values (to be parallelised)
        ###
        print("Computing likelihood of samples....")
        Lp_j = log_fD_T(θ_j')                         
        println("Done!")

        ###
        #   Computing new βj
        #   Uses bisection method
        ###
        println("Computing Bj")

        low_β = βj; hi_β = 2; Lp_adjust = maximum(Lp_j);
        x1 = (hi_β + low_β)/2;

        while (hi_β - low_β)/((hi_β + low_β)/2) > 1e-6

            x1 = (hi_β + low_β)/2;
            wj_test = exp.((x1 .- βj ) .* (Lp_j .- Lp_adjust));
            cov_w   = std(wj_test)/mean(wj_test);
            if cov_w > 1; hi_β = x1; else; low_β = x1; end
        end
        
        βj1 = min(1,x1)
        println("B_$(j1) = $(βj1)")

        ###
        #   Computation of normalised weights
        ###
        println("Computing weights")

        w_j = exp.((βj1 - βj) .* (Lp_j .- Lp_adjust))       # Nominal weights from liklihood and βjs

        Log_ev = log(mean(w_j)) + (βj1 - βj) * Lp_adjust + Log_ev   # Log evidence in current iteration
        
        # Normalised weights
        wn_j = w_j ./sum(w_j);

        Th_wm = θ_j .* wn_j                 # Weighted mean of samples

        ###
        #   Calculation of COV matrix of proposal
        ###
        SIGMA_j = zeros(Ndims, Ndims)
        
        for l = 1:Nsamples
            SIGMA_j = SIGMA_j + beta2 .* wn_j[l] .* (θ_j[l,:]' .- Th_wm)' * (θ_j[l,:]' .- Th_wm)
        end
        
        # Ensure that cov is symetric
        SIGMA_j = (SIGMA_j' + SIGMA_j)/2

        prop = mu -> proprnd(mu, SIGMA_j, fT)           # Anonymous function for proposal

        target = x -> log_fD_T(x) .* βj1 .+ log.(fT(x)) # Anonymous function for transitional distribution

        # Weighted resampling of θj (indecies with replacement)
        randIndex = sample(1:Nsamples, Weights(wn_j), Nsamples, replace=true)

        θ_j1 = zeros(Nsamples, Ndims)
        α = zeros(Nsamples)                 # acceptance rates

        print("Markov chains...")
        for i = 1:Nsamples

            this, that = MHsampleSimple(target, prop, θ_j[randIndex[i], :], 1, burnin, thin)
            θ_j1[i,:] = this
            α[i] = that
        end
        println("Done!")
        meanα = mean(α)

        #println("Mean α = $(meanα)")
        
        βj = βj1
        θ_j = θ_j1
    end
    return θ_j, Log_ev
end



function proprnd(mu, covMat, prior)

    samp = rand(MvNormal(mu, covMat), 1)
    while iszero(prior(samp))
        samp = rand(MvNormal(mu, covMat),1)
    end
    return samp[:]
end



function tmcmc_old(LogLik, prior, priorRnd, Nsamples, burnin = 50, LastBurnin = burnin, thin = 3, beta = 0.2)

    Bj = 0;     # Temp
    j  = 0;     # Iter

    priorSamps = priorRnd(Nsamples);

    thetaj = priorSamps;
    
    Ndims = size(thetaj,1)

    acceptanceRates = []

    while Bj < 1
        
        j = j + 1
        println("Beginning iteration $j")

        LogLikj = LogLik(thetaj)            # Array of log Lik vals of Bj
        #=
        LogLikj = zeros(Nsamples)
        for i =1:Nsamples
            LogLikj[i] = LogLik(thetaj[i,:])
        end
        =#
        priln("Computing Bj")nt
        Bj1 = computeBj2(LogLikj, Bj)
        println("B_$j = $Bj1")
        # Define target 
        target(x) = LogLik(x) .* Bj1 .+ log.(prior(x))

        println("Computing weights")
        # Computing proposal dist
        Liks = exp.(LogLikj .* (Bj1 - Bj) )
        wTheta = Liks./sum(Liks)            # Normalised weights

        if Ndims >1; 
            mWj = sum(wTheta' .* thetaj, dims = 2)         # Samples 
        else
            mWj = sum(wTheta .* thetaj)         # Samples 
        end

        sums = zeros(Ndims, Ndims)
        for i = 1:Nsamples
            sums = wTheta[i] .* (thetaj[:,i] .- mWj) * (thetaj[:,i] .-mWj)' + sums
        end
        covMat = beta^2 .* sums
        
        prop = mu -> proprnd(mu, covMat, prior)

        # Better to sample index
        #resamps = sample(thetaj, Weights(wTheta[:]), Nsamples, replace=true)

        randIndex = sample(1:Nsamples, Weights(wTheta), Nsamples, replace=true)

        thetaj1 = zeros(Ndims, Nsamples)

        acc = zeros(Nsamples)

        
        println("Markov chains...")
        for i = 1:Nsamples

            this, that = MHsampleSimple(target, prop, thetaj[:, randIndex[i]], 1, 20,thin)
            thetaj1[:,i] = this
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

function computeBj2(logLiks, pj)

    low_alpha = pj; up_alpha = 2; Lp_adjust = maximum(logLiks);

    x1 = (up_alpha + low_alpha)/2;
    while (up_alpha - low_alpha)/((up_alpha + low_alpha)/2) > 1e-6
        x1 = (up_alpha + low_alpha)/2;

        wj_test = exp.((x1 .- pj) .* (logLiks .-Lp_adjust));    

        cov_w   = std(wj_test)/mean(wj_test);
        if cov_w > 1; up_alpha = x1; else; low_alpha = x1; end
    end

    return min(1,x1);

end
