module SpinModels

using LinearAlgebra
using NNlib

export SpinModel, Pairwise, SRBM, RBM
export paramnames, getparams, veccopy, zerosum!
include("types.jl")

export energy, ∂x_energy, EnergyBuffer
include("energy.jl")

export ratiomatch, RatioMatchBuffer
include("losses/ratiomatch.jl")

export SpinSampler, acceptrate, currentstate, batchsize, spinstates, spinsnum
export MetropolisHastings, metropolishastings!
export GibbsWithGradients, gibbswithgradients!
include("samplers/SpinSamplers.jl")
using .SpinSamplers


end # module
