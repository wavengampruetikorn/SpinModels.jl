module SpinModels

using LinearAlgebra, Statistics
using NNlib

export SpinModel, Pairwise, SRBM, RBM
export paramnames, getparams, veccopy, zerosum!
include("types.jl")

export energy, ∂x_energy, EnergyBuffer, ΔE₁, ΔEnergyBuffer
include("energy.jl")

export ratiomatch, RatioMatchBuffer
include("losses/ratiomatch.jl")

export SpinSampler, acceptrate, currentstate, batchsize, spinstates, spinsnum
export MetropolisHastings, metropolishastings!
export GibbsWithGradients, gibbswithgradients!
export ZanellaMH, zanella_mh!
export EnergyDescent, energydescent!
using .SpinSamplers


end # module
