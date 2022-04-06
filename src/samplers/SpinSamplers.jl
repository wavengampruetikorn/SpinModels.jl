module SpinSamplers
using Random, NNlib
include("categoricalsamplers.jl")


export SpinSampler, 
    acceptrate, currentstate, batchsize, spinstates, spinsnum
abstract type SpinSampler{T3} end
acceptrate(x::SpinSampler) = x.α
batchsize(x::SpinSampler) = x.M
spinstates(x::SpinSampler) = x.q
spinsnum(x::SpinSampler) = x.N

export MetropolisHastings, metropolishastings!
include("metropolishastings.jl")

export GibbsWithGradients, gibbswithgradients!
include("gibbswithgradients.jl")

export ZanellaMH, zanella_mh!
include("zanella_mh.jl")

export EnergyDescent, energydescent!
include("energydescent.jl")

end
