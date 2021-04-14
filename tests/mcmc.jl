@testset "Metropolis Hastings general" begin

    @testset "1D" begin
        prop(mu) = MvNormal(mu, 3)

        target(x) = log(pdf(MvNormal([2], 1), x))


        samps, acc = metropolis_hastings(target, prop, [2], 2000, 200)

        @test mean(samps) ≈ 2 atol = 0.1
        @test std(samps) ≈ 1 atol = 0.1
    end

    @testset "3D" begin
        prop(mu) = MvNormal(mu, [3 0 0; 0 3 0; 0 0 3])

        target(x) = log(pdf(MvNormal([-2, 2, 0], [1 0.5 0.5; 0.5 1 0.5; 0.5 0.5 1]), x))

        samps, acc = metropolis_hastings(target, prop, [-2, 2, 0], 2000, 200)

        μ = mean(samps, dims = 1)
        σ = std(samps, dims = 1)
        corrs = cor(samps)
        @test vec(μ) ≈ [-2; 2; 0] atol = 0.2
        @test vec(σ) ≈ [1; 1; 1] atol = 0.1
        @test corrs ≈ [1 0.5 0.5; 0.5 1 0.5; 0.5 0.5 1] atol = 0.2
    end

    @testset "No logged target" begin
        prop(mu) = MvNormal(mu, [3 0 0; 0 3 0; 0 0 3])

        target(x) = pdf(MvNormal([-2, 2, 0], [1 0.5 0.5; 0.5 1 0.5; 0.5 0.5 1]), x)

        samps, acc = metropolis_hastings(target, prop, [-2, 2, 0], 2000, 200, islogged = false)

        μ = mean(samps, dims = 1)
        σ = std(samps, dims = 1)
        corrs = cor(samps)
        @test vec(μ) ≈ [-2; 2; 0] atol = 0.2
        @test vec(σ) ≈ [1; 1; 1] atol = 0.1
        @test corrs ≈ [1 0.5 0.5; 0.5 1 0.5; 0.5 0.5 1] atol = 0.2
    end

end


@testset "Metropolis Hastings simple" begin

    @testset "1D" begin

        proprnd(mu) = rand(MvNormal(mu, 3))

        target(x) = log(pdf(MvNormal([2], 1), x))

        samps, acc = metropolis_hastings_simple(target, proprnd, [2], 2000, 200)

        @test mean(samps) ≈ 2 atol = 0.1
        @test std(samps) ≈ 1 atol = 0.1

    end

    @testset "3D" begin

        proprnd(mu) = rand(MvNormal(mu, [3 0 0; 0 3 0; 0 0 3]))

        target(x) = log(pdf(MvNormal([-2, 2, 0], [1 0.5 0.5; 0.5 1 0.5; 0.5 0.5 1]), x))

        samps, acc = metropolis_hastings_simple(target, proprnd, [-2,2, 0], 2000, 200)

        μ = mean(samps, dims = 1)
        σ = std(samps, dims = 1)
        corrs = cor(samps)
        @test vec(μ) ≈ [-2; 2; 0] atol = 0.2
        @test vec(σ) ≈ [1; 1; 1] atol = 0.1
        @test corrs ≈ [1 0.5 0.5; 0.5 1 0.5; 0.5 0.5 1] atol = 0.2

    end

end
