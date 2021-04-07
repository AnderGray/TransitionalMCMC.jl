# TransitionalMCMC.jl


Implementation of Transitional Markov Chain Monte Carlo (TMCMC) in Julia. This implementation is heavily inspired by the implemntation of TMCMC in [OpenCOSSAN](https://github.com/cossan-working-group/OpenCossan)

The TMCMC alorgithm can be used to sample from un-normalised probability density function (i.e. posterior distributions in Bayesian Updating). The TMCMC alorgithm overcomes some of the issues with Metropolis Hastings:

* Can efficiently sample multimodal distributions
* Works well in high dimensions (within reason)
* Compute the evidence
* Proposal distribution selected by algorithm
* Easy to parallelise

Instead of atempting to directly sample from the posterior, TMCMC samples from easy-to-sample "transitional" distributions. Defined by:



## Installation

This is not yet a registered julia package. However this package may be installed using the Julia package manager:

```Julia
julia> ]
pkg> add https://github.com/AnderGray/TransitionalMCMC.jl
```

## Bibiography

* J. Ching, and Y. Chen (2007). [Transitional Markov Chain Monte Carlo method for Bayesian model updating, Model class selection, and Model averaging.](https://ascelibrary.org/doi/pdf/10.1061/(ASCE)0733-9399(2007)133%3A7(816)?casa_token=mGf_dvFGtYcAAAAA%3AvPklSPi0HXqUX9VabgqN5xILx6e8cH973IUbkgCKkRjooKku7__DhKk3yuYqzyTSIXBluhaEes2MXg&) Journal of Engineering Mechanics, 133(7), 816-832. doi:10.1061/(asce)0733-9399(2007)133:7(816)
* A. Lye, A. Cicirello, and E. Patelli (2021). [Sampling methods for solving Bayesian model Updating problems: A tutorial. Mechanical Systems and Signal Processing](https://livrepository.liverpool.ac.uk/3115734/), 159, 107760. doi:10.1016/j.ymssp.2021.107760
* E. Patelli, M. Broggi, M. D. Angelis, and M. Beer (2014). [OpenCossan: An efficient open tool for dealing with epistemic AND Aleatory Uncertainties. Vulnerability, Uncertainty, and Risk.](https://www.researchgate.net/publication/263732354_OpenCossan_An_Efficient_Open_Tool_for_Dealing_with_Epistemic_and_Aleatory_Uncertainties) doi:10.1061/9780784413609.258
