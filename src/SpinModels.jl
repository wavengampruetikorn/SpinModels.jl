module SpinModels

using LinearAlgebra
using NNlib

export SpinModel, Pairwise, SRBM, RBM
export paramnames, getparams, veccopy
include("types.jl")

export energy, ∂x_energy, EnergyBuffer
include("energy.jl")

export ratiomatch, RatioMatchBuffer
include("losses/ratiomatch.jl")

end # module
