struct GibbsWithGradients{T3, V, B3, CS} <: SpinSampler{T3}
    A::T3           # q × N × M
    A₁::T3          # 1 × N × M
    α::T3           # 1 × 1 × M
    I::V            # M element vector
    batchshift::V   # M element vector
    𝕡::T3           # q × N × M
    𝕀::B3           # q × N × M with Bool entries
    q::Int          # number of spin states
    N::Int          # number of spins
    M::Int          # sample size
    catsampler::CS
    function GibbsWithGradients(X::AbstractArray{<:AbstractFloat,3})
        q, N, M = size(X)
        _eltype = promote_type(eltype(X), Float32)
        A = similar(X, _eltype, q, N, M)
        A₁= similar(X, _eltype, 1, N, M)
        α = similar(X, _eltype, 1, 1, M)
        I = similar(X, Int, M)
        batchshift = oftype(I, range(0; step = q*N, length = M))
        𝕡 = similar(X, _eltype, q, N, M)
        𝕀 = similar(X, Bool, q, N, M)
        catsampler = InverseCDF(𝕡, q*N, M)
        return new{typeof(A), typeof(I), typeof(𝕀), typeof(catsampler)}(
            A, A₁, α, I, batchshift, 𝕡, 𝕀, q, N, M, catsampler
        )
    end
end

(f::GibbsWithGradients)(X, energy) = gibbswithgradients!(X, energy, f)

function gibbswithgradients!(X::AbstractArray{T,3}, energy, buffer = MetropolisHastings(X)) where T
    (; A, A₁, α, I, batchshift, 𝕡, 𝕀, q, N, M, catsampler) = buffer
    _onehalf = convert(T, 1//2)
    E, ∇E = energy(X)

    _gwg_proposal!(𝕡, X, ∇E, A₁)
    unitrange_rand!(I, 1:q*N, 𝕡, catsampler)    
                                    # flipped spin and proposed spin state for each sample
    I .+= batchshift                # shift index with sample index

    α .= reshape(E, 1,1,M)
    @view(α[:]) .-= log.(@view(𝕡[I]))   # α = E(x) - ln(π(x′|x))

    fill!(A, zero(T))               # initialize
    fill!(@view(A[I]), one(T))      # onehot encode flipped spin and proposed spin state
    sum!(A₁, A)                     # onehot encode site of flipped spin
    @. A = X + (A - A₁ * X)         # proposed state
    @. A = A > _onehalf              # ensure the entries are in (0,1)
    @. 𝕀 = A₁ == X == 1             # onehot encode flipped spin and original spin state

    Ẽ, ∇Ẽ = energy(A)
    _gwg_proposal!(𝕡, A, ∇Ẽ, A₁)
    α .-= reshape(Ẽ, 1,1,M)         # α = E(x) - ln(π(x′|x)) - E(x′)
    
    @view(α[:]) .+= log.(𝕡[𝕀])      # α = E(x) - ln(π(x′|x)) - E(x′) + ln(π(x|x′))
    
    @. α = exp(-relu(-α))       # α = min{ 1, p(x′)π(x|x′) / p(x)π(x′|x) }
    @. α = α ≥ rand()           # accept proposal

    @. X = (1-α) * X + α * A    # update X
    @. X = X > _onehalf          # ensure the entries are in (0,1)
    return X
end



function _gwg_proposal!(𝕡::AbstractArray{T,3}, X, ∇E, A₁) where T
    _typemax = typemax(T)
    _onehalf = convert(T, 1//2)
    @. 𝕡 = X * ∇E
    sum!(A₁, 𝕡)
    @. 𝕡 = -_onehalf * (∇E - A₁)
    @. 𝕡-= (X > _onehalf) * _typemax # disallow transition to current state
    softmax!(𝕡, dims = 1:2)
end