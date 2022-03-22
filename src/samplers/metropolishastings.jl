struct MetropolisHastings{T3, V} <: SpinSampler{T3}
    A::T3           # q × N × M
    A₁::T3          # 1 × N × M
    α::T3           # 1 × 1 × M
    I::V            # M element vector
    batchshift::V   # M element vector
    q::Int          # number of spin states
    N::Int          # number of spins
    M::Int          # sample size
    function MetropolisHastings(X::AbstractArray{<:Real,3})
        q, N, M = size(X)
        _eltype = promote_type(eltype(X), Float32)
        A = similar(X, _eltype, q, N, M)
        A₁= similar(X, _eltype, 1, N, M)
        α = similar(X, _eltype, 1, 1, M)
        I = similar(X, Int, M)
        batchshift = oftype(I, range(0; step = q*N, length = M))
        return new{typeof(A), typeof(I)}(
            A, A₁, α, I, batchshift, q, N, M
        )
    end
end
(f::MetropolisHastings)(X, energy) = metropolishastings!(X, energy, f)

function metropolishastings!(X::AbstractArray{T,3}, energy, buffer = MetropolisHastings(X)) where T
    (; A, A₁, α, I, batchshift, q, N, M) = buffer
    onehalf = convert(T, 1//2)
    E = energy(X)
    α.= reshape(E, 1,1,M)

    unitrange_rand!(I, 1:q*N)   # flipped spin and proposed spin state for each sample
    I .+= batchshift            # shift index with sample index

    fill!(A, zero(T))           # initialize
    fill!(@view(A[I]), one(T))  # onehot encode flipped spin and proposed spin state
    sum!(A₁, A)                 # onehot encode site of flipped spin
    @. A = X + (A - A₁ * X)     # proposed state
    @. A = A > onehalf          # ensure the entries are in (0,1)

    Ẽ = energy(A)
    α.-= reshape(Ẽ, 1,1,M)      # α = E(x) - E(x̃)
    @. α = exp(-relu(-α))       # α = min{ 1, p(x′)/p(x) }
    @. α = α ≥ rand()           # accept proposal

    @. X = (1-α) * X + α * A    # update X
    @. X = X > onehalf          # ensure the entries are in (0,1)
    return X
end

