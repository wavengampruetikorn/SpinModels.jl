# Giacomo Zanella (2020) Informed Proposals for Local MCMC in Discrete Spaces
# Journal of the American Statistical Association

struct ZanellaMH{T3, V, CS} <: SpinSampler{T3}
    A::T3           # q × N × M
    A₁::T3          # 1 × N × M
    A₁₂::T3         # 1 × 1 × M
    α::T3           # 1 × 1 × M
    I::V            # M element vector
    batchshift::V   # M element vector
    𝕡::T3           # q × N × M
    q::Int          # number of spin states
    N::Int          # number of spins
    M::Int          # sample size
    catsampler::CS
    function ZanellaMH(X::AbstractArray{<:AbstractFloat,3})
        q, N, M = size(X)
        _eltype = promote_type(eltype(X), Float32)
        A  = similar(X, _eltype, q, N, M)
        A₁ = similar(X, _eltype, 1, N, M)
        A₁₂= similar(X, _eltype, 1, 1, M)
        α  = similar(X, _eltype, 1, 1, M)
        I  = similar(X, Int, M)
        batchshift = oftype(I, range(0; step = q*N, length = M))
        𝕡 = similar(X, _eltype, q, N, M)
        catsampler = InverseCDF(𝕡, q*N, M)
        return new{typeof(A), typeof(I), typeof(catsampler)}(
            A, A₁, A₁₂, α, I, batchshift, 𝕡, q, N, M, catsampler
        )
    end
end

(f::ZanellaMH)(X, Δenergy) = zanella_mh!(X, Δenergy, f)

function zanella_mh!(X::AbstractArray{T,3}, Δenergy, buffer = ZanellaMH(X)) where T
    (; A, A₁, A₁₂, α, I, batchshift, 𝕡, q, N, catsampler) = buffer
    _onehalf = convert(T, 1//2)

    ΔE = Δenergy(X)                 # E(xᵠᵏ)-E(x)
    _zanella_proposal!!(𝕡, A₁₂, ΔE) # proposal π(xᵠᵏ|x) and partition function Z(x)
    unitrange_rand!(I, 1:q*N, 𝕡, catsampler)    
                                    # flipped spin and proposed spin state for each sample
    I .+= batchshift                # shift index with sample index

    α .= A₁₂                        # α = Z(x) = ∑ᵩₖ exp(-1//2 * [E(xᵠᵏ)-E(x)])

    fill!(A, zero(T))               # initialize
    fill!(@view(A[I]), one(T))      # onehot encode flipped spin and proposed spin state
    sum!(A₁, A)                     # onehot encode site of flipped spin
    @. A = X + (A - A₁ * X)         # proposed state
    @. A = A > _onehalf             # ensure the entries are in (0,1)

    ΔẼ = Δenergy(A)                 # E(x̃ᵠᵏ)-E(x̃)
    _zanella_proposal!!(𝕡, A₁₂, ΔẼ, onlypartionfunc=true) 
                                    # Z(x̃)
    α ./= A₁₂                       # α = Z(x)/Z(x̃)

    @. α = min(α, 1)                # α = min(1, Z(x)/Z(x̃))
    @. α = α ≥ rand()               # accept proposal

    @. X = (1-α) * X + α * A        # update X
    @. X = X > _onehalf             # ensure the entries are in (0,1)
    return X
end



function _zanella_proposal!!(
        𝕡::AbstractArray{T,3}, A₁₂::AbstractArray{T,3}, ΔE::AbstractArray{T,3}; 
        onlypartionfunc = false
    ) where T<:Real
    _typemax = typemax(T)
    _onehalf = convert(T, 1//2)
    @. 𝕡 = -_onehalf * ΔE               # log probability of q(zᵠᵏ|z)
    @. 𝕡-= (X > _onehalf) * _typemax    # disallow transition to current state
    sum!(exp, A₁₂, 𝕡)                   # partition function
    onlypartionfunc || softmax!(𝕡, dims = 1:2)  # normalized proposal q(zᵠᵏ|z)
end
