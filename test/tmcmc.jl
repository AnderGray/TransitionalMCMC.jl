@testset "TransitionalMCMC" begin
    disable_logging(Logging.Info)

    @testset "1D" begin
        Random.seed!(123456)
        lb, ub = -15, 15

        μ = [0 5]
        σ = [1 0.2]
        w = [0.7 0.3]

        fT(x) = logpdf(Uniform(lb, ub), x)
        sample_fT(Nsamples) = rand(Uniform(lb, ub), Nsamples)

        function log_fD_T(x)
            return log(
                w[1] * pdf(Normal(μ[1], σ[1]), x[1]) + w[2] * pdf(Normal(μ[2], σ[2]), x[1])
            )
        end

        Nsamples = 1000
        X, _ = tmcmc(log_fD_T, fT, sample_fT, Nsamples)

        m = sum(w .* μ)
        s = sqrt(sum(w .* (σ .^ 2 + μ .^ 2 .- m^2)))

        h0 = ExactOneSampleKSTest(vec(X), Normal(m, s))

        @test pvalue(h0) < 1e-4
    end

    @testset "2D" begin
        Random.seed!(1234)

        lb, ub = -15, 15

        w = [0.6, 0.4]
        μ = [0 5; 0 5]

        Σ1 = [1 -0.5; -0.5 1]
        Σ2 = [1 0.5; 0.5 1]

        # Compute mean and cov of resulting gaussian mixture
        m = vec(sum(w .* μ; dims=2))
        C = w[1] * Σ1 + w[2] * Σ2
        C += w[1] * (μ[:, 1] - m) * (μ[:, 1] - m)'
        C += w[2] * (μ[:, 2] - m) * (μ[:, 2] - m)'

        Nsamples = 10000
        Y = rand(MvNormal(m, C), Nsamples)'

        fT(x) = logpdf(Uniform(lb, ub), x[1]) .+ logpdf(Uniform(lb, ub), x[2])
        sample_fT(Nsamples) = rand(Uniform(lb, ub), Nsamples, 2)

        function log_fD_T(x)
            return log.(
                w[1] * pdf(MvNormal(μ[:, 1], Σ1), x) + w[2] * pdf(MvNormal(μ[:, 2], Σ2), x)
            )
        end

        X, _ = tmcmc(log_fD_T, fT, sample_fT, Nsamples)

        h0 = OneSampleHotellingT2Test(X, m)
        @test pvalue(h0) < 0.05

        h0 = BartlettTest(X, Y)
        @test pvalue(h0) < 0.05
    end

    @testset "Himmelblau" begin
        Random.seed!(123456)
        lb, ub = -5, 5

        # Prior Density and sampler
        logprior(x) = logpdf(Uniform(lb, ub), x[1]) + logpdf(Uniform(lb, ub), x[2])
        priorRnd(Nsamples) = rand(Uniform(lb, ub), Nsamples, 2)

        # Log Likelihood
        logLik(x) = -1 * ((x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2)

        samps, acc = tmcmc(logLik, logprior, priorRnd, 20000)

        μ = vec(mean(samps; dims=1))
        σ = vec(std(samps; dims=1))
        corrs = cor(samps)

        # Reference values obtained using 1e7 samples
        @test [0.7, 0.2] < μ < [1, 0.4] # [0.832681 0.286942]
        @test [2.8, 2.1] < σ < [3.4, 2.8] # [3.16236 2.45603]
        @test corrs ≈ [1.0 0; 0 1.0] atol = 0.1 # independent
    end

    @testset "Himmelblau parallel" begin
        addprocs(2; exeflags="--project")
        @everywhere begin
            using TransitionalMCMC, Distributions, Random

            Random.seed!(123456)

            # Prior Bounds
            lb, ub = -5, 5

            # Prior Density and sampler
            logprior(x) = logpdf(Uniform(lb, ub), x[1]) + logpdf(Uniform(lb, ub), x[2])
            priorRnd(Nsamples) = rand(Uniform(lb, ub), Nsamples, 2)

            # Log Likelihood
            logLik(x) = -1 * ((x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7) .^ 2)
        end

        Nsamples = 20000

        samps, acc = tmcmc(logLik, logprior, priorRnd, Nsamples)

        μ = vec(mean(samps; dims=1))
        σ = vec(std(samps; dims=1))
        corrs = cor(samps)

        # Reference values obtained using 1e7 samples
        @test [0.7, 0.2] < μ < [1, 0.4] # [0.832681 0.286942]
        @test [2.8, 2.1] < σ < [3.4, 2.8] # [3.16236 2.45603]
        @test corrs ≈ [1.0 0; 0 1.0] atol = 0.1 # independent

        rmprocs(workers())
    end

    Random.seed!()
end
