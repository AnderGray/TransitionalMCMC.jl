@testset "TransitionalMCMC" begin

    disable_logging(Logging.Info)

    @testset "1D" begin
        Random.seed!(123456)
        lb, ub  = -15, 15

        μ = [0 5]
        σ = [1 0.2]
        w = [0.7 0.3]

        fT(x) = logpdf(Uniform(lb, ub), x[1])
        sample_fT(Nsamples) = rand(Uniform(lb, ub), Nsamples, 1)

        log_fD_T(x) = log(w[1] * pdf(Normal(μ[1], σ[1]), x[1]) + w[2] * pdf(Normal(μ[2], σ[2]), x[1]))

        Nsamples = 1000
        X, _ = tmcmc(log_fD_T, fT, sample_fT, Nsamples)

        m = sum(w .* μ)
        s = sqrt(sum(w .* (σ.^2 + μ.^2 .- m^2)))

        h0 = ExactOneSampleKSTest(vec(X), Normal(m, s))

        @test pvalue(h0) < 1e-4
    end

    @testset "2D" begin
        Random.seed!(123456)
        lb  = -15
        ub  = 15

        w = [0.6 0.4]
        μ = [0 5; 0 5]

        Σ1 = [1 -0.5; -0.5 1]
        Σ2 = [1 0.5; 0.5 1]
        fT(x) = logpdf(Uniform(lb, ub), x[1]) .+ logpdf(Uniform(lb, ub), x[2])
        sample_fT(Nsamples) = rand(Uniform(lb, ub), Nsamples, 2)

        log_fD_T(x) = log.(w[1] * pdf(MvNormal(μ[:, 1], Σ1), x) + w[2] * pdf(MvNormal(μ[:, 2], Σ2), x))

        Nsamples = 2000
        X, _ = tmcmc(log_fD_T, fT, sample_fT, Nsamples)

        m = sum(w .* μ, dims=2) |> vec
        C = w[1] * Σ1 + w[2] * Σ2
        C += w[1] * (μ[:, 1] - m) * (μ[:, 1] - m)'
        C += w[2] * (μ[:, 2] - m) * (μ[:, 2] - m)'

        h0 = OneSampleHotellingT2Test(X, m)
        @test pvalue(h0) < 1e-4

        Y = rand(MvNormal(m, C), Nsamples)'
        h0 = EqualCovHotellingT2Test(X, Y)

        @test pvalue(h0) < 1e-3
    end

    @testset "Himmelblau" begin
        Random.seed!(123456)
        lb, ub  = -5, 5

        # Prior Density and sampler
        logprior(x) = logpdf(Uniform(lb, ub), x[1]) + logpdf(Uniform(lb, ub), x[2])
        priorRnd(Nsamples) = rand(Uniform(lb, ub), Nsamples, 2)

        # Log Likelihood
        logLik(x) = -1 * ((x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2)

        samps, acc = tmcmc(logLik, logprior, priorRnd, 2000)

        μ = mean(samps, dims=1)
        σ = std(samps, dims=1)
        corrs = cor(samps)
        @test vec(μ) ≈ [  0.6448241433718321; 0.5207538113617672] atol = 0.3
        @test vec(σ) ≈ [3.09665;  2.40973] atol = 0.2
        @test corrs ≈ [1.0 -0.0661209; -0.0661209 1.0] atol = 0.3
    end


    @testset "Himmelblau parallel" begin
        @everywhere Random.seed!(123456)
        addprocs(2; exeflags="--project")
        @everywhere begin

            using TransitionalMCMC, Distributions

            # Prior Bounds
            lb, ub  = -5, 5

            # Prior Density and sampler
            logprior(x) = logpdf(Uniform(lb, ub), x[1]) + logpdf(Uniform(lb, ub), x[2])
            priorRnd(Nsamples) = rand(Uniform(lb, ub), Nsamples, 2)

            # Log Likelihood
            logLik(x) = -1 * ((x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7).^2)

        end

        Nsamples = 2000

        samps, acc = tmcmc(logLik, logprior, priorRnd, Nsamples)

        μ = mean(samps, dims=1)
        σ = std(samps, dims=1)
        corrs = cor(samps)
        @test vec(μ) ≈ [  1.0273;  0.355209] atol = 0.4
        @test vec(σ) ≈ [3.09665;  2.40973] atol = 0.2
        @test corrs ≈ [1.0 -0.0661209; -0.0661209 1.0] atol = 0.3

        rmprocs(workers())
    end

    Random.seed!()
end
