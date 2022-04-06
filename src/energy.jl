energy(z::AbstractMatrix, θ) = energy(z, θ, EnergyBuffer(θ, 1))
energy(z::AbstractArray{<:Any,3}, θ) = energy(z, θ, EnergyBuffer(θ, size(z, 3)))
function energy(z, θ::SpinModel, buffer)
    (; E, A, A₁₂, B, B₁) = buffer
    (; q, N) = θ
    qN = q*N

    # E[m] = h[μ,i] * Z[μ,i,m]
    if hasproperty(θ, :h)
        mul!(vec(E), transpose(reshape(z, qN, :)), vec(θ.h))
    else
        fill!(E, zero(eltype(E)))
    end
    
    # E[m]+= Z[μ,i,m] J[μ,i,ν,j] Z[ν,j,m]
    if hasproperty(θ, :J)
        mul!(reshape(A, qN,:), reshape(θ.J, qN,qN), reshape(z, qN,:))
        A .*= z
        sum!(A₁₂, A)
        E .+= vec(A₁₂)
    end
    
    # E[m]+= logcosh(B[a,m])  &  B[a,m] = W[a,μ,i] Z[μ,i,m] + b[a]
    if hasproperty(θ, :W)
        mul!(B, reshape(θ.W, :, qN), reshape(z, qN, :))
        @. B = logcosh(θ.b + B)
        sum!(B₁, B)
        rmul!(B₁, θ.rbm_wt)
        E .+= vec(B₁)
    end

    return rmul!(E, -1)
end


∂x_energy(z::AbstractMatrix, θ) = ∂x_energy(z, θ, EnergyBuffer(θ, 1))
∂x_energy(z::AbstractArray{<:Any,3}, θ) = ∂x_energy(z, θ, EnergyBuffer(θ, size(z, 3)))
function ∂x_energy(z, θ::SpinModel, buffer)
    (; A, B, K) = buffer
    (; q, N) = θ
    qN = q*N
    ∇E = A         # non-allocating alias
    
    if hasproperty(θ, :h)
        @. ∇E = θ.h    # ∇E[μim] = h[μi]
    else
        fill!(∇E, zero(eltype(∇E)))
    end

    # ∇E[μim]+= (J[μiνj] + J[νjμi]) * Z[νjm]
    if hasproperty(θ, :J)
        K .= reshape(θ.J, qN,qN) .+ transpose(reshape(θ.J, qN,qN))
        mul!(reshape(∇E, qN, :), K, reshape(z, qN, :), 1, 1)
    end

    # ∇E[μim]+= W[aμi] * tanh(W[aνj]*Z[νjm] + b[a])
    if hasproperty(θ, :W)
        mul!(B, reshape(θ.W, :, qN), reshape(z, qN, :))
        @. B = tanh_fast(B + θ.b)
        mul!(reshape(∇E, qN, :), transpose(reshape(θ.W, :, qN)), B, θ.rbm_wt, 1)
    end

    return rmul!(∇E, -1)
end


struct EnergyBuffer{V,T3,M}
    E::V
    A::T3
    A₁₂::T3
    B::M
    B₁::M
    K::M
    function EnergyBuffer(θ::SpinModel{T}, M::Int) where T
        (; q, N) = θ
        has_h, has_J, has_W = (hasproperty(θ, s) for s ∈ (:h, :J, :W))
        X = has_h ? θ.h : has_J ? θ.J : θ.W
        E = similar(X, M)
        A = similar(X, q, N, M)
        A₁₂ = similar(X, 1, 1, M)
        B = has_W ? similar(X, θ.P, M) : similar(X, 0, 0)
        B₁ = has_W ? similar(X, 1, M) : similar(X, 0, 0)
        K = has_J ? similar(X, q*N, q*N) : similar(X, 0, 0)
        return new{typeof(E),typeof(A),typeof(B)}(E, A, A₁₂, B, B₁, K)
    end
    EnergyBuffer(z::AbstractArray{<:Any,3}, θ::SpinModel) = EnergyBuffer(θ, size(z,3))
end



struct ΔEnergyBuffer{T3,M,T4}
    ΔE::T3      # q × N × M
    A::T3       # q × N × M
    A₁::T3      # 1 × N × M     (for J & h)
    K::M        # qN × qN       (for J)
    B::M        # P × M         (for W)
    B₁::M       # 1 × M         (for W)
    B̃::T4       # P × q × N × M (for W)
    B̃₁::T4      # 1 × q × N × M (for W)
    B̃₂::T4      # P × 1 × N × M (for W)
    function ΔEnergyBuffer(θ::SpinModel, M::Int)
        (;q, N) = θ
        has_h, has_J, has_W = (hasproperty(θ, s) for s ∈ (:h, :J, :W))
        X = has_h ? θ.h : has_J ? θ.J : θ.W
        ΔE, A = ntuple(i -> similar(X, q,N,M), 2)
        A₁ = (has_h | has_J) ? similar(X, 1,N,M) : similar(X, 0,0,0)
        K  = has_J ? similar(X, q*N,q*N) : similar(X, 0,0)
        B  = has_W ? similar(X, θ.P,M) : similar(X, 0,0)
        B₁ = has_W ? similar(X, 1,M) : similar(X, 0,0)
        B̃  = has_W ? similar(X, θ.P,q,N,M) : similar(X, 0,0,0,0)
        B̃₁ = has_W ? similar(X, 1,q,N,M) : similar(X, 0,0,0,0)
        B̃₂ = has_W ? similar(X, θ.P,1,N,M) : similar(X, 0,0,0,0)
        return new{typeof(A), typeof(K), typeof(B̃)}(
            ΔE, A, A₁, K, B, B₁, B̃, B̃₁, B̃₂
        )
    end
    ΔEnergyBuffer(z::AbstractArray{<:Any,3}, θ::SpinModel) = ΔEnergyBuffer(θ, size(z,3))
end

# Compute energy changes from one spin flip:    ΔE[ϕ,k,m] = E(zᵠᵏ) - E(z)
ΔE₁(z, θ::SpinModel) = ΔE₁(z, θ, ΔEnergyBuffer(z, θ))
function ΔE₁(z, θ::SpinModel, buffer)
    (; ΔE, A, A₁, K, B, B₁, B̃, B̃₁, B̃₂) = buffer
    fill!(ΔE, zero(eltype(ΔE))) # initialize ΔE
    q, N, M = size(z)
    
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
        ΔE.+= θ.rbm_wt .* reshape(sum!(logcosh, B₁, B), 1,1,M)
        ΔE.-= θ.rbm_wt .* reshape(sum!(logcosh, B̃₁, B̃), q,N,M)
    end

    return ΔE
end