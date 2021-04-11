# TransitionalMCMC.jl


Implementation of Transitional Markov Chain Monte Carlo (TMCMC) in Julia. This implementation is heavily inspired by the implemntation of TMCMC in [OpenCOSSAN](https://github.com/cossan-working-group/OpenCossan).

The TMCMC algorithm can be used to sample from un-normalised probability density function (i.e. posterior distributions in Bayesian Updating). The TMCMC algorithm overcomes some of the issues with Metropolis Hastings:

* Can efficiently sample multimodal distributions
* Works well in high dimensions (within reason)
* Computes the evidence
* Proposal distribution selected by algorithm
* Easy to parallelise

Instead of atempting to directly sample from the posterior, TMCMC samples from easy-to-sample "transitional" distributions. Defined by:

<img src="https://imgur.com/5p4APND.png" data-canonical-src="https://imgur.com/5p4APND.png" width="300" />

where 0 <= B<sub>j</sub> <= 1, is iterated in the algorithm starting from B<sub>j</sub> = 0 (prior) to B<sub>j</sub> = 1 (posterior).

## Installation

This is not yet a registered Julia package. However this package may be installed using the Julia package manager:

```Julia
julia> ]
pkg> add https://github.com/AnderGray/TransitionalMCMC.jl
```

## Usage

Sampling [Himmelblau's Function](https://en.wikipedia.org/wiki/Himmelblau%27s_function):

```Julia
using StatsBase, Distributions, PyPlot
using TransitionalMCMC

# Prior Bounds
lb  = -5        
ub  = 5

# Prior log Density and sampler
priorDen(x) = logpdf(Uniform(lb,ub), x[1,:]) .* logpdf(Uniform(lb,ub), x[2,:])
priorRnd(Nsamples) = rand(Uniform(lb,ub), Nsamples, 2)

# Log Likelihood
logLik(x) = -1 .* ((x[1,:].^2 .+ x[2,:] .- 11).^2 .+ (x[1,:] .+ x[2,:].^2 .- 7).^2)

samps, Log_ev = tmcmc(logLik, priorDen, priorRnd, 2000)

plt.scatter(samps[:,1], samps[:,2])
```



<img src="https://imgur.com/ySv4BzI.png" data-canonical-src="https://imgur.com/ySv4BzI.png" width="1500" />

### For parallel excution

```Julia
using Distributed, StatsBase, Distributions, PyPlot

addprocs(4; exeflags="--project")
@everywhere begin
    using TransitionalMCMC

    # Prior Bounds
    lb, ub  = -5, 5

    # Prior log Density and sampler
    priorDen(x) = logpdf(Uniform(lb, ub), x[1,:]) .* logpdf(Uniform(lb, ub), x[2,:])
    priorRnd(Nsamples) = rand(Uniform(lb, ub), Nsamples, 2)

    # Log Likelihood
    function logLik(x)
        return -1 .* ((x[1,:].^2 .+ x[2,:] .- 11).^2 .+ (x[1,:] .+ x[2,:].^2 .- 7).^2)
    end

end

Nsamples = 2000

samps, Log_ev = tmcmc(logLik, priorDen, priorRnd, Nsamples, 5, 2)
```
### Benchmarks

Found in [/slurm_benchmarks](https://github.com/AnderGray/TransitionalMCMC.jl/tree/main/slurm_benchmark)

Testing scalability of `tmcmcHimmelblau.jl` with different model evaluations times

<p float="left">
  <img src="https://imgur.com/Q8mZWM1.png" data-canonical-src="https://imgur.com/Q8mZWM1.png" width="400" />
  <img src="https://imgur.com/PALKfor.png" data-canonical-src="https://imgur.com/PALKfor.png" width="400" />
</p>

Testing slowdown and iteration number for various dimensions. Target is a mixture of 2 Gaussians in N dimensions, with means located at [-5, -5 , ...] and [5, 5, ...]

<p float="left">
  <img src="https://imgur.com/O5nwSSR.png" data-canonical-src="https://imgur.com/Q8mZWM1.png" width="400" />
  <img src="https://imgur.com/fcOxklJ.png" data-canonical-src="https://imgur.com/PALKfor.png" width="400" />
</p>


## Todo
* Plotting functions
* Storing samples across iterations

## Bibiography

* J. Ching, and Y. Chen (2007). [Transitional Markov Chain Monte Carlo method for Bayesian model updating, Model class selection, and Model averaging.](https://ascelibrary.org/doi/pdf/10.1061/(ASCE)0733-9399(2007)133%3A7(816)?casa_token=mGf_dvFGtYcAAAAA%3AvPklSPi0HXqUX9VabgqN5xILx6e8cH973IUbkgCKkRjooKku7__DhKk3yuYqzyTSIXBluhaEes2MXg&) Journal of Engineering Mechanics, 133(7), 816-832. doi:10.1061/(asce)0733-9399(2007)133:7(816)
* A. Lye, A. Cicirello, and E. Patelli (2021). [Sampling methods for solving Bayesian model Updating problems: A tutorial.](https://livrepository.liverpool.ac.uk/3115734/) Mechanical Systems and Signal Processing, 159, 107760. doi:10.1016/j.ymssp.2021.107760
* E. Patelli, M. Broggi, M. D. Angelis, and M. Beer (2014). [OpenCossan: An efficient open tool for dealing with epistemic AND Aleatory Uncertainties. Vulnerability, Uncertainty, and Risk.](https://www.researchgate.net/publication/263732354_OpenCossan_An_Efficient_Open_Tool_for_Dealing_with_Epistemic_and_Aleatory_Uncertainties) doi:10.1061/9780784413609.258
