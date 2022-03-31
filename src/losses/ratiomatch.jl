struct RatioMatchBuffer{T3,M,T4}
    ΔE::T3      # q × N × M
    A::T3       # q × N × M
    A₁::T3      # 1 × N × M     (for J & h)
    A₁₂::T3     # 1 × 1 × M     (for W)
    K::M        # qN × qN       (for J)
    B::M        # P × M         (for W)
    B₁::M       # 1 × M         (for W)
    B̃::T4       # P × q × N × M (for W)
    B̃₁::T4      # 1 × q × N × M (for W)
    B̃₂::T4      # P × 1 × N × M (for W)
    B̃₄::T4      # P × q × N × 1 (for W)
    function RatioMatchBuffer(θ::SpinModel, M::Int)
        (;q, N) = θ
        has_h, has_J, has_W = (hasproperty(θ, s) for s ∈ (:h, :J, :W))
        X = has_h ? θ.h : has_J ? θ.J : θ.W
        ΔE, A = ntuple(i -> similar(X, q,N,M), 2)
        A₁ = (has_h | has_J) ? similar(X, 1,N,M) : similar(X, 0,0,0)
        A₁₂= has_W ? similar(X, 1,1,M) : similar(X, 0,0,0)
        K  = has_J ? similar(X, q*N,q*N) : similar(X, 0,0)
        B  = has_W ? similar(X, θ.P,M) : similar(X, 0,0)
        B₁ = has_W ? similar(X, 1,M) : similar(X, 0,0)
        B̃  = has_W ? similar(X, θ.P,q,N,M) : similar(X, 0,0,0,0)
        B̃₁ = has_W ? similar(X, 1,q,N,M) : similar(X, 0,0,0,0)
        B̃₂ = has_W ? similar(X, θ.P,1,N,M) : similar(X, 0,0,0,0)
        B̃₄ = has_W ? similar(X, θ.P,q,N,1) : similar(X, 0,0,0,0)
        return new{typeof(A), typeof(K), typeof(B̃)}(
            ΔE, A, A₁, A₁₂, K, B, B₁, B̃, B̃₁, B̃₂, B̃₄
        )
    end
    RatioMatchBuffer(z::AbstractArray{<:Any,3}, θ::SpinModel) = RatioMatchBuffer(θ, size(z,3))
end

"""
    ratiomatch(z, θ::SpinModel)

Return ratio matching loss (Hyvärinen, Comput Stat Data Anal 2007)

Computed as, for a system of `N` `q`-state spins,

    mean(abs2(1 - sigmoid( E(zᵠᵏ) - E(z) )) for ϕ ∈ 1:q, k ∈ 1:N, z ∈ Data) - 1//4

where `zᵠᵏ` is the same as `z` but with spin `k` in state `ϕ`. 
This loss vanishes for unstructured models, in which `E(zᵠᵏ) - E(z) = 0` for all `ϕ, k, x`.
"""
ratiomatch(z, θ::SpinModel; energyscale = 1) = ratiomatch(z, θ, RatioMatchBuffer(z, θ); energyscale = energyscale)
ratiomatch(z, θ::SpinModel, buffer; energyscale = 1) = ratiomatch(z, θ, θ, buffer, false, energyscale)
ratiomatch(z, θ::T, θ̄::T) where T<:SpinModel = ratiomatch(z, θ, θ̄, RatioMatchBuffer(z, θ))
function ratiomatch(z, θ::T, θ̄::T, buffer, gradient = true, energyscale = 1) where T<:SpinModel
    (; ΔE, A, A₁, A₁₂, K, B, B₁, B̃, B̃₁, B̃₂, B̃₄) = buffer
    fill!(ΔE, zero(eltype(ΔE))) # initialize ΔE
    q, N, M = size(z)

    ## First we compute:    ΔE[ϕ,k,m] = E(zᵠᵏ) - E(z)
    if hasproperty(θ, :J)
        J = θ.J
        K .= reshape(J, q*N, q*N)
        K.+= transpose(reshape(J, q*N, q*N))
        mul!(reshape(ΔE, :,M), K, reshape(z, :,M), -1, 1)
        @. A = ΔE * z
        ΔE .-= sum!(A₁, A)
    end
    
    if hasproperty(θ, :h)
        h = θ.h
        @. A = h * z
        sum!(A₁, A)
        @. ΔE += -(h - A₁)
    end

    if hasproperty(θ, :W) && hasproperty(θ, :b)
        (;W, b) = θ
        mul!(B, reshape(W, :, q*N), reshape(z, :, M))   # B[a,m] = ∑ᵨₖ W[a,ρ,k] * z[ρ,k,m]
        B.+= b                                          # B[a,m] = b[a] + ∑ᵨₖ W[a,ρ,k] * z[ρ,k,m]
        B̃ .= W .* reshape(z, 1,q,N,M)   # B̃[a,ρ,k,m] = W[a,ρ,k] * z[ρ,k,m]
        sum!(B̃₂, B̃)                     # B̃₂[a,.,k,m] = ∑ᵨ W[a,ρ,k] * z[ρ,k,m]
        @. B̃ = W - B̃₂                   # B̃[a,σ,k,m] = W[a,σ,k] - ∑ᵨ W[a,ρ,k] * z[ρ,k,m]
        B̃.+= reshape(B, :,1,1,M)        # B̃[a,σ,k,m] = B[a,m] + W[a,σ,k] - ∑ᵨ W[a,ρ,k] * z[ρ,k,m]
        ΔE.+= reshape(sum!(logcosh, B₁, B), 1,1,M)
        ΔE.-= reshape(sum!(logcosh, B̃₁, B̃), q,N,M)
    end

    isone(energyscale) || rmul!(ΔE, convert(eltype(ΔE), energyscale))

    ## Loss ∝ ∑ᵩₖₘ (1-sigmoid(E(zᵠᵏ)-E(z)))² - ∑ᵩₖₘ (zᵠᵏ==z) * (1/4)
    ## We subtract off the terms for which zᵠᵏ==z 
    f = ΔE                      # non-allocating alias
    @. f = sigmoid(f)           # f[σ,k,m] = sigmoid(ΔE[σ,k,m])
    L = sum(x -> abs2(1-x), f)
    # subtract loss from zᵠᵏ=z (N terms per sample) for which ΔE = 0 and (1-σ(ΔE))² = 1/4
    L-= 1//4 * N * M
    # take the average over M samples and N*(q-1) neighboring states per sample
    L*= 1//(M * N * (q-1))
    # multiply by 4 so that unstructured model (ΔE = 0 for all zᵠᵏ, z) gives L = 1
    L*= 4
    gradient || return L

    #### Gradient calculations start here
    @. f = abs2(1-f) * 2f       # f[σ,k,m] = (1-sigmoid(ΔE[σ,k,m]))² * 2sigmoid(ΔE[σ,k,m])
    A.= z .* sum!(A₁, f)        # A[σ,k,m] = z[σ,k,m] * ∑ᵨ f[ρ,k,m]
    A.= f .- A                  # A[σ,k,m] = f[σ,k,m] - z[σ,k,m] * ∑ᵨ f[ρ,k,m]

    if hasproperty(θ̄, :h)
        h̄ = θ̄.h
        sum!(h̄, A)
        h̄ .*= 4//(M * N * (q-1))
    end

    if hasproperty(θ̄, :J)
        J̄ = θ̄.J
        Ĵ = reshape(J̄, q*N, q*N)
        mul!(Ĵ, reshape(z,:,M), transpose(reshape(A,:,M)))
        Ĵ .+= transpose(Ĵ)
        dropselfcoupling!(J̄)    # donot update self coupling
        J̄ .*= 4//(M * N * (q-1))
    end

    if hasproperty(θ̄, :W) && hasproperty(θ̄, :b)
        W̄, b̄ = θ̄.W, θ̄.b
        @. B = tanh(B)  # B[a,m] = tanh(b[a] + ∑ᵨₖ W[a,ρ,k]*z[ρ,k,m])
        @. B̃ = tanh(B̃)  # B̃[a,σ,k,m] = tanh(B[a,m] + W[a,σ,k] - ∑ᵨ W[a,ρ,k]*z[ρ,k,m])
        
        sum!(A₁₂, f)
        mul!(b̄, B, vec(A₁₂), -1, 0)
        mul!(b̄, reshape(B̃,:,q*N*M), vec(f), 1, 1)
        b̄ .*= 4//(M * N * (q-1))
        
        Bf̄ = B  # non-allocating alias
        Bf̄ .*= reshape(A₁₂, :,M)        # Bf̄[a,m] = B[a,m] * ∑ᵨₖ f[ρ,k,m]
        B̃f = B̃  # non-allocating alias
        B̃f .*= reshape(f, 1,q,N,M)      # B̃f[a,σ,k,m] = B̃[a,σ,k,m] * f[σ,k,m]
        sum!(W̄, B̃f)
        mul!(reshape(W̄, :,q*N), Bf̄, transpose(reshape(z, q*N,M)), -1, 1)

        B̃xf = B  # non-allocating alias
        sum!(reshape(B̃xf, :,1,1,M), B̃f) # B̃xf[a,m] = ∑ᵨₖ B̃[a,ρ,k,m] * f[ρ,k,m]
        mul!(reshape(W̄, :,q*N), B̃xf, transpose(reshape(z, q*N,M)), 1, 1)

        B̃ .= .-sum!(B̃₂, B̃f) .* reshape(z,1,q,N,M)

        W̄ .+= sum!(B̃₄, B̃)
        W̄ .*= 4//(M * N * (q-1))
    end

    return L
end