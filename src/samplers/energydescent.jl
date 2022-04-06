# follow the steepest path to minima

struct EnergyDescent{T3} <: SpinSampler{T3}
    A::T3           # q × N × M
    A₁::T3          # 1 × N × M
    α::T3           # 1 × 1 × M
    q::Int          # number of spin states
    N::Int          # number of spins
    M::Int          # sample size
    function EnergyDescent(X::AbstractArray{<:AbstractFloat,3})
        q, N, M = size(X)
        _eltype = promote_type(eltype(X), Float32)
        A = similar(X, _eltype, q, N, M)
        A₁= similar(X, _eltype, 1, N, M)
        α = similar(X, _eltype, 1, 1, M)
        return new{typeof(A)}(A, A₁, α, q, N, M)
    end
end

(f::EnergyDescent)(X, Δenergy) = energydescent!(X, Δenergy, f)

function energydescent!(X::AbstractArray{T,3}, Δenergy, buffer = EnergyDescent(X)) where T
    (; A, A₁, α) = buffer
    _onehalf = convert(T, 1//2)

    ΔE = Δenergy(X)                 # E(xᵠᵏ)-E(x)
    δE, I = findmin(ΔE, dims = 1:2)

    fill!(A, zero(T))               # initialize
    fill!(@view(A[I]), one(T))      # onehot encode flipped spin and proposed spin state
    sum!(A₁, A)                     # onehot encode site of flipped spin
    @. A = X + (A - A₁ * X)         # proposed state
    @. A = A > _onehalf             # ensure the entries are in (0,1)

    @. α = δE < 0                   # flip to new state only if energy decreases

    @. X = (1-α) * X + α * A        # update X
    @. X = X > _onehalf             # ensure the entries are in (0,1)
    return X
end

