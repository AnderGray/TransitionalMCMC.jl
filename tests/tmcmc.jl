@testset "TransitionalMCMC" begin

    disable_logging(Logging.Info)

    @testset "1D" begin
        lb  = -15
        ub  = 15

        fT(x) = logpdf(Uniform(lb, ub), x[1])
        sample_fT(Nsamples) = rand(Uniform(lb, ub), Nsamples, 1)

        log_fD_T(x) = log(pdf(Normal(0, 1), x[1]) + pdf(Normal(5, 0.2), x[1]))

        Nsamples = 2000
        samps, acc = tmcmc(log_fD_T, fT, sample_fT, Nsamples)
        @test mean(samps) ≈ 2.5 atol = 0.2
        @test std(samps) ≈ 2.6 atol = 0.2
    end

    @testset "2D" begin

        lb  = -15
        ub  = 15

        fT(x) = logpdf(Uniform(lb, ub), x[1]) .+ logpdf(Uniform(lb, ub), x[2])
        sample_fT(Nsamples) = rand(Uniform(lb, ub), Nsamples, 2)

        log_fD_T(x) = log.(pdf(MvNormal([0,0], [1 -0.5; -0.5 1]), x) + pdf(MvNormal([5,5], [1 0.5; 0.5 1]), x) + pdf(MvNormal([-5,5], [1 0.9; 0.9 1]), x))

        samps, acc = tmcmc(log_fD_T, fT, sample_fT, 2000)

        μ = mean(samps, dims = 1)
        σ = std(samps, dims = 1)
        corrs = cor(samps)
        @test vec(μ) ≈ [ -0.04904654270772346; 3.324840746169731] atol = 0.3
        @test vec(σ) ≈ [4.1931;  2.56974] atol = 0.2
        @test corrs ≈ [ 1.0 0.019783; 0.019783  1.0] atol = 0.2
    end

    @testset "Himmelblau" begin

        lb, ub  = -5, 5

        # Prior Density and sampler
        logprior(x) = logpdf(Uniform(lb, ub), x[1]) + logpdf(Uniform(lb, ub), x[2])
        priorRnd(Nsamples) = rand(Uniform(lb, ub), Nsamples, 2)

        # Log Likelihood
        logLik(x) = -1 * ((x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2)

        samps, acc = tmcmc(logLik, logprior, priorRnd, 2000)

        μ = mean(samps, dims = 1)
        σ = std(samps, dims = 1)
        corrs = cor(samps)
        @test vec(μ) ≈ [  1.0273;  0.355209] atol = 0.3
        @test vec(σ) ≈ [3.09665;  2.40973] atol = 0.2
        @test corrs ≈ [1.0 -0.0661209; -0.0661209 1.0] atol = 0.3
    end


    @testset "Himmelblau parallel" begin

        addprocs(2; exeflags = "--project")
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

        μ = mean(samps, dims = 1)
        σ = std(samps, dims = 1)
        corrs = cor(samps)
        @test vec(μ) ≈ [  1.0273;  0.355209] atol = 0.4
        @test vec(σ) ≈ [3.09665;  2.40973] atol = 0.2
        @test corrs ≈ [1.0 -0.0661209; -0.0661209 1.0] atol = 0.3

        rmprocs(workers())
    end

end
