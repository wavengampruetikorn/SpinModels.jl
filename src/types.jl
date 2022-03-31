# type definitions
"Abstract type containing model parameters"
abstract type SpinModel{T<:AbstractArray} end

# generic methods for SpinModel
"Return iterators of symbols of parameter names in model"
paramnames(θ::SpinModel) = Iterators.filter(s -> isa(getproperty(θ, s), AbstractArray), propertynames(θ))

"Randomize model parameters"
function random! end

"Return tuple of arrays containing model parameters"
getparams(θ::SpinModel) = (getfield(θ, s) for s ∈ paramnames(θ))


Base.length(θ::SpinModel) = sum(length(getfield(θ,s)) for s ∈ paramnames(θ))
Base.eltype(θ::SpinModel) = eltype(first(getparams(θ)))
function Base.copyto!(v::AbstractArray, θ::SpinModel)
    k = 0
    for s ∈ paramnames(θ)
        x = getfield(θ,s)
        l = length(x)
        copyto!(@view(v[(1:l).+k]), x)
        k+=l
    end
    return v
end
function Base.copyto!(θ::SpinModel, v::AbstractArray)
    k = 0
    for s ∈ paramnames(θ)
        x = getfield(θ,s)
        l = length(x)
        copyto!(x, @view(v[(1:l).+k]))
        k+=l
    end
    return θ
end
function Base.copyto!(θ′::T, θ::T) where T<:SpinModel
    for s ∈ paramnames(θ)
        copyto!(getfield(θ′,s), getfield(θ,s))
    end
    return θ′
end
"""
    copy(θ::SpinModel) = copyto!(similar(θ), θ)
"""
Base.copy(θ::SpinModel) = copyto!(similar(θ), θ)

"""
    rmul!(θ::SpinModel, b::Number)

Multiply the parameters in `θ` by a scalar `b`
"""
function LinearAlgebra.rmul!(θ::SpinModel, b::Number)
    for s ∈ paramnames(θ)
        rmul!(getfield(θ,s), b)
    end
    return θ
end

"""
    sum(x::AbstractVector{T}) where T<:SpinModel

Return a model with the energy function equal to the sum of energy functions of models in the input vector
"""
function Base.sum(x::AbstractVector{T}) where T<:SpinModel
    hasfield(T, :h) && (h = sum(ϕ.h for ϕ ∈ x))
    hasfield(T, :J) && (J = sum(ϕ.J for ϕ ∈ x))
    hasfield(T, :W) && (W = reduce(vcat, (ϕ.W for ϕ ∈ x)))
    hasfield(T, :b) && (b = reduce(vcat, (ϕ.b for ϕ ∈ x)))
    T <: Pairwise && return Pairwise(h, J)
    T <: SRBM && return SRBM(h, J, W, b)
    T <: RBM && return RBM(h, W, b)
end


"Copy all model parameters to a new vector"
veccopy(θ::SpinModel) = copyto!(similar(first(getparams(θ)), length(θ)), θ)

"Convert model parameters to zero-sum gauge"
function zerosum!(θ::T) where T<:SpinModel
    hasfield(T, :h) && (θ.h .-= mean(θ.h, dims = 1))
    if hasfield(T, :J)
        J = θ.J
        J_μ̄iνj = mean(J, dims = 1)
        J_μiν̄j = mean(J, dims = 3)
        J_μ̄iν̄j = mean(J, dims = (1,3))
        @. J += - J_μ̄iνj - J_μiν̄j + J_μ̄iν̄j
        hasfield(T, :h) && (θ.h .+= 2sum(J_μiν̄j .- J_μ̄iν̄j, dims = 4))
    end
    if hasfield(T, :W)
        (; W, b) = θ
        W_aμ̄i = mean(W, dims = 2)
        W.-= W_aμ̄i
        b.+= sum(W_aμ̄i, dims = 3)
    end
    return θ
end

"Set all self-couplings in pairwise interactions to zero"
dropselfcoupling!(θ::SpinModel) = (hasproperty(θ,:J) && dropselfcoupling!(θ.J))
function dropselfcoupling!(J::AbstractArray{T,4}) where T<:Number
    for i ∈ axes(J,2)
        @. J[:,i,:,i] = zero(T)
    end
    return J
end

"Symmetrize pairwise coupling"
symmetrizecoupling!(θ::SpinModel) = (hasproperty(θ,:J) && symmetrizecoupling!(θ.J))
function symmetrizecoupling!(J::AbstractArray{<:Real,4})
    J_mat = reshape(J,size(J,1)*size(J,2),:)
    J_mat.= J_mat .+ transpose(J_mat)
    J_mat.*= 1//2
    return J
end


function random_J!(J::AbstractArray{<:Number,4}, scale=0.1)
    N = size(J,2)
    @. J = randn() * scale * 1/√(N * (N-1))
    symmetrizecoupling!(J)      # J[μi,νj] = # J[νj,μi]
    dropselfcoupling!(J)
end
random_h!(h::AbstractMatrix{<:Number}, scale=0.1) = (@. h = randn() * scale * 1/√size(h,2))
random_W!(W::AbstractArray{<:Number,3}, scale=0.1) = (
    @. W = randn() * scale * 1/√(size(W,1) * size(W,3))
)
random_b!(b::AbstractVector{<:Number}, scale=0.1) = (@. b = randn() * scale * 1/√length(b))



"""
    Pairwise(; N, q, similarto = Float32[]) -> θ::Pairwise

Create a container for pairwise model parameters `J`. The diagonal elements plat the role of bias fileds.

Parameters are arrays of type similar to `similarto`. 
"""
struct Pairwise{H,T} <: SpinModel{H}
    h::H
    J::T    # J[μ,i,ν,j]
    q::Int
    N::Int
    function Pairwise(; N::T, q::T, similarto = Float32[]) where T<:Int
        h = similar(similarto, q,N)
        random_h!(h, 0.01)
        J = similar(similarto, q,N,q,N)
        random_J!(J, 0.1)
        return new{typeof(h),typeof(J)}(h, J, q, N)
    end
    function Pairwise(h::AbstractMatrix{T}, J::AbstractArray{T,4}) where T
        if !(size(h) == size(J)[1:2] == size(J)[3:4])
            return error("Incompatible arrays: size(h) = $(size(h)), size(J) = $(size(J))")
        end
        return new{typeof(h),typeof(J)}(h, J, size(h,1), size(h,2))
    end
    Pairwise(J::AbstractArray{T,4}, h::AbstractMatrix{T}) where T = Pairwise(h,J)
end
Base.similar(θ::Pairwise) = Pairwise(q = θ.q, N = θ.N, similarto = θ.h)
random!(θ::Pairwise, scale = 0.1) = (random_h!(θ.h, scale); random_J!(θ.J, scale); θ)

"""
    SRBM(; N, P, q, similarto=zeros(0)) -> θ::SRBM

Create a container for sRBM parameters (`J`, `h`, `W` & `b`).

Parameters are arrays of type similar to `similarto`.
"""
struct SRBM{H,TJ,TW,Tb} <: SpinModel{H}
    h::H
    J::TJ               # J[i,μ,j,ν]
    W::TW               # W[a,i,μ]
    b::Tb               # b[a]
    q::Int
    N::Int
    P::Int
    function SRBM(; N::T, P::T, q::T, similarto = Float32[]) where T<:Int
        θ = Pairwise(; N=N, q=q, similarto=similarto)
        iszero(P) && return θ
        (; h, J) = θ
        W = similar(J, P,q,N)
        random_W!(W, 0.1)
        b = similar(J, P)
        random_b!(b, 0.01)
        return new{typeof(h),typeof(J),typeof(W),typeof(b)}(h, J, W, b, q, N, P)
    end
    function SRBM(
            h::AbstractMatrix{T}, J::AbstractArray{T,4}, 
            W::AbstractArray{T,3}, b::AbstractVector{T}
        ) where T
        if !(size(h) == size(J)[1:2] == size(J)[3:4] == size(W)[2:3])
            return error("Incompatible arrays: size(h) = $(size(h)), size(J) = $(size(J)), size(W) = $(size(W))")
        end
        if !(size(W,1) == size(b,1))
            return error("Incompatible arrays: size(W) = $(size(W)), size(b) = $(size(b))")
        end
        P, q, N = size(W)
        return new{typeof(h),typeof(J),typeof(W),typeof(b)}(h, J, W, b, q, N, P)
    end
    function SRBM(a1::AbstractArray, a2::AbstractArray, a3::AbstractArray, a4::AbstractArray)
        params = (a1,a2,a3,a4)
        h = params[findfirst(x -> ndims(x)==2, params)]
        J = params[findfirst(x -> ndims(x)==4, params)]
        W = params[findfirst(x -> ndims(x)==3, params)]
        b = params[findfirst(x -> ndims(x)==1, params)]
        return SRBM(h,J,W,b)
    end
end
Base.similar(θ::SRBM) = SRBM(P = θ.P, q = θ.q, N = θ.N, similarto = θ.h)
random!(θ::SRBM, scale = 0.1) = (
    random_h!(θ.h, scale); random_J!(θ.J, scale); 
    random_W!(θ.W, scale); random_b!(θ.b, scale); 
    θ
)


"""
    RBM(; N, P, q, similarto=zeros(0)) -> θ::RBM

Create a container for RBM parameters (`h`, `W` & `b`).

Parameters are arrays of type similar to `similarto`.
"""
struct RBM{H,TW,Tb} <: SpinModel{H}
    h::H               # h[i,μ]
    W::TW               # W[a,i,μ]
    b::Tb               # b[a]
    q::Int
    N::Int
    P::Int
    function RBM(; N::T, P::T, q::T, similarto = Float32[]) where T<:Int
        h = random_h!(similar(similarto, q,N), 0.01)
        W = random_W!(similar(h, P,q,N), 0.1)
        b = random_b!(similar(h, P), 0.01)
        return new{typeof(h),typeof(W),typeof(b)}(h, W, b, q, N, P)
    end
    function RBM(h::AbstractMatrix{T}, W::AbstractArray{T,3}, b::AbstractVector{T}) where T
        if !(size(h) == size(W)[2:3])
            return error("Incompatible arrays: size(h) = $(size(h)), size(W) = $(size(W))")
        end
        if !(size(W,1) == size(b,1))
            return error("Incompatible arrays: size(W) = $(size(W)), size(b) = $(size(b))")
        end
        P, q, N = size(W)
        return new{typeof(h),typeof(W),typeof(b)}(h, W, b, q, N, P)
    end
    function RBM(a1::AbstractArray, a2::AbstractArray, a3::AbstractArray)
        params = (a1,a2,a3)
        h = params[findfirst(x -> ndims(x)==2, params)]
        W = params[findfirst(x -> ndims(x)==3, params)]
        b = params[findfirst(x -> ndims(x)==1, params)]
        return RBM(h,W,b)
    end
end
Base.similar(θ::RBM) = RBM(P = θ.P, q = θ.q, N = θ.N, similarto = θ.W)
random!(θ::RBM, scale = 0.1) = (
    random_h!(θ.h, scale); random_W!(θ.W, scale); random_b!(θ.b, scale); θ
)