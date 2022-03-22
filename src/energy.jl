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
        mul!(reshape(∇E, qN, :), transpose(reshape(θ.W, :, qN)), B, 1, 1)
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