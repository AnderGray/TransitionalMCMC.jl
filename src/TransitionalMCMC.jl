###
#   An implmentation of Transitional Markov Chain Monte Carlo in Julia
#
#            Institute for Risk and Uncertainty, Uni of Liverpool
#
#                       Authors: Ander Gray, Adolphus Lye
#
#                       Email: Ander.Gray@liverpool.ac.uk, 
#                              Adolphus.Lye@liverpool.ac.uk
#
#
#   This Transitional MCMC algorithm is inspirted by OpenCOSSAN: 
#                   https://github.com/cossan-working-group/OpenCossan
#
#
#   Algorithm originally proposed by: 
#           J. Ching, and Y. Chen (2007). Transitional Markov Chain Monte Carlo method 
#           for Bayesian model updating, Model class selection, and Model averaging. 
#           Journal of Engineering Mechanics, 133(7), 816-832. 
#           doi:10.1061/(asce)0733-9399(2007)133:7(816)
#
###
module TransitionalMCMC

using Reexport
@reexport using Distributed

@everywhere using Distributions
@reexport using StatsBase   
@reexport using Distributions


export 
    tmcmc, MHsample, MHsampleSimple, tmcmc_par

@everywhere include("tmcmc.jl")
@everywhere include("tmcmc_par.jl")
@everywhere include("mcmc.jl")
end # module
