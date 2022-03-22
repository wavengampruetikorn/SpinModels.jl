# Ratio matching loss and pullbacks

Load required packages and define data structure 
```julia
using LinearAlgebra
using NNlib, Zygote, BenchmarkTools, CUDA
using SpinModels: SRBM, ratiomatch, RatioMatchBuffer

function randdata(P = 50, q = 21, N = 59, M = 256)
    z = Float32.(1:q .== rand(1:q, 1, N, M))
    θ = SRBM(N = N, P = P, q = q, similarto = z)
    @. θ.J = randn()
    @. θ.h = randn()
    if hasproperty(θ, :W)
        @. θ.W = randn()
        @. θ.b = randn()
    end
    if CUDA.functional()
        cu_z = CuArray(z)
        cu_θ = SRBM(CuArray(θ.J), CuArray(θ.h), CuArray(θ.W), CuArray(θ.b))
        return z, θ, cu_z, cu_θ
    else
        return z, θ, nothing, nothing
    end
end
```




We start from the changes in energy due to flipping spin `k` into new state `ϕ`: `ΔE[ϕ,k,m] = E(zᵠᵏ) - E(z)`
```julia
ΔEʰ(z, h) = .-(h .- sum(h .* z; dims = 1))
function ΔEᴶ(z, J)
    q, N, M = size(z)
    K = reshape(J, q*N, q*N) .+ transpose(reshape(J, q*N, q*N))
    ΔE = .-K * reshape(z, :, M)
    return reshape(ΔE, q,N,M) .- sum(reshape(ΔE, q,N,M) .* z, dims = 1)
end
function ΔEᵂ(z, W, b)
    q, N, M = size(z)
    B = b .+ (reshape(W, :, q*N) * reshape(z, :, M))    # B[a,m]
    E = .-sum(logcosh, B, dims = 1)
    B̃ = reshape(B, :,1,1,M) .+ W .- sum(W .* reshape(z, 1,q,N,M), dims = 2)
    Ẽ = .-sum(logcosh, B̃, dims = 1)
    return reshape(Ẽ, q,N,M) .- reshape(E, 1,1,M)
end
```




The score matching loss is proportional to `∑ᵩₖₘ(1-sigmoid(E(zᵠᵏ)-E(z)))²` with constant shift and scaling defined below
```julia
ℓ(z, θ::SRBM) = ℓ(z, θ.J, θ.h, θ.W, θ.b)
function ℓ(z, J, h, W, b)
    q, N, M = size(z)
    ΔE = ΔEʰ(z, h) .+ ΔEᴶ(z, J) .+ ΔEᵂ(z, W,b)
    L = sum(x -> abs2(1-sigmoid(x)), ΔE)
    # subtract contribution from zᵠᵏ = z (N terms per sample) for which ΔE = 0 and (1-σ(ΔE))² = 1/4
    L-= 1//4 * N * M
    # take the average over M samples and N*(q-1) neighboring states per sample
    L*= 1//(M * N * (q-1))
    # multiply by 4 so that unstructured model (ΔE = 0 for all zᵠᵏ and z) gives L = 1
    L*= 4
    return L
end
```




Get some data and a model with random parameters
```julia
z, θ, cu_z, cu_θ = randdata()
```




Check if the energies from energy function above agree with `SpinModels.ratiomatch()`
```julia
julia> ℓ(z, θ) ≈ ratiomatch(z, θ)
true

julia> ℓ(z, θ) - ratiomatch(z, θ)
2.3841858f-7
```




Take the gradients of the loss using automatic differentiation on the explicit score matching loss (defined above)
```julia
∂ℓ(z, θ) = gradient((x...) -> ℓ(z, x...), θ.J, θ.h, θ.W, θ.b)
J̄, h̄, W̄, b̄ = ∂ℓ(z, θ)
```



Compute the gradients using `SpinModels.ratiomatch()`
```julia
θ̄ = similar(θ);
ratiomatch(z, θ, θ̄);
```



Check if the gradients from the two methods agree
```julia
julia> (J̄, h̄, W̄, b̄) .≈ (θ̄.J, θ̄.h, θ̄.W, θ̄.b)
(true, true, true, true)

julia> [norm(x .- y, Inf) for (x, y) ∈ zip((J̄, h̄, W̄, b̄), (θ̄.J, θ̄.h, θ̄.W, θ̄.b))]'
1×4 adjoint(::Vector{Float32}) with eltype Float32:
 1.1205f-9  6.04814f-10  7.03585f-9  4.97919f-7
```



## GPU test


Check the correcness of GPU codes 
```julia
julia> CUDA.functional() && (ℓ(z, θ) ≈ ℓ(cu_z, cu_θ))
true

julia> CUDA.functional() && (ℓ(z, θ) - ℓ(cu_z, cu_θ))
-2.3841858f-7

julia> CUDA.functional() && (ℓ(z, θ) ≈ ratiomatch(cu_z, cu_θ))
true

julia> CUDA.functional() && (ℓ(z, θ) - ratiomatch(cu_z, cu_θ))
0.0f0
```



Compute the gradients
```julia
cu_θ̄ = similar(cu_θ)
if CUDA.functional()
    (cu_J̄, cu_h̄, cu_W̄, cu_b̄) = ∂ℓ(cu_z, cu_θ)       # Explicit code
    ratiomatch(cu_z, cu_θ, cu_θ̄)                    # From `SpinModels`
end
```




Check if the gradients on GPU agree with those on CPU
```julia
julia> CUDA.functional() && ((J̄, h̄, W̄, b̄) .≈ collect.((cu_J̄, cu_h̄, cu_W̄, cu_b̄)))
(true, true, true, true)

julia> CUDA.functional() && [norm(x .- y, Inf) for (x, y) ∈ zip((J̄, h̄, W̄, b̄), collect.((cu_J̄, cu_h̄, cu_W̄, cu_b̄)))]'
1×4 adjoint(::Vector{Float32}) with eltype Float32:
 2.88128f-9  1.85537f-9  6.03904f-9  1.57488f-8

julia> CUDA.functional() && ((J̄, h̄, W̄, b̄) .≈ collect.((cu_θ̄.J, cu_θ̄.h, cu_θ̄.W, cu_θ̄.b)))
(true, true, true, true)

julia> CUDA.functional() && [norm(x .- y, Inf) for (x, y) ∈ zip((J̄, h̄, W̄, b̄), collect.((cu_θ̄.J, cu_θ̄.h, cu_θ̄.W, cu_θ̄.b)))]'
1×4 adjoint(::Vector{Float32}) with eltype Float32:
 2.86673f-9  1.9154f-9  6.37374f-9  3.17814f-8
```



## Benchmarks

Create buffers for pre-allocated versions of the code
```julia
buffer = RatioMatchBuffer(z, θ)
cu_buffer = CUDA.functional() ? RatioMatchBuffer(cu_z, cu_θ) : nothing
```




### Loss function
On CPU
```julia
julia> @btime ℓ($z, $θ);                                               # Explicit code (defined above)
  517.669 ms (72 allocations: 146.70 MiB)

julia> @btime ratiomatch($z, $θ);                                      # From `SpinModels`
  498.698 ms (44 allocations: 73.21 MiB)

julia> @btime ratiomatch($z, $θ, $buffer);                             # From `SpinModels` (pre-allocated)
  473.608 ms (20 allocations: 1.00 KiB)
```


On GPU
```julia
julia> CUDA.functional() && @btime CUDA.@sync ℓ(cu_z, cu_θ);                  # Explicit code (defined above)
  6.141 ms (701 allocations: 47.84 KiB)

julia> CUDA.functional() && @btime CUDA.@sync ratiomatch(cu_z, cu_θ);         # From `SpinModels`
  6.852 ms (617 allocations: 41.25 KiB)

julia> CUDA.functional() && @btime CUDA.@sync ratiomatch(cu_z, cu_θ, cu_buffer);# From `SpinModels` (pre-allocated)
  6.643 ms (562 allocations: 39.41 KiB)
```



### Gradients

On CPU
```julia
julia> @btime ∂ℓ($z, $θ);                                              # Explicit code (defined above)
  3.795 s (464 allocations: 3.31 GiB)

julia> @btime ratiomatch($z, $θ, $θ̄);                                  # From `SpinModels`
  808.252 ms (74 allocations: 79.07 MiB)

julia> @btime ratiomatch($z, $θ, $θ̄, $buffer);                         # From `SpinModels` (pre-allocated)
  758.372 ms (50 allocations: 5.86 MiB)
```


On GPU
```julia
julia> CUDA.functional() && @btime CUDA.@sync ∂ℓ(cu_z, cu_θ);                 # Explicit code (defined above)
  20.061 ms (2622 allocations: 174.06 KiB)

julia> CUDA.functional() && @btime CUDA.@sync ratiomatch(cu_z, cu_θ, cu_θ̄);   # From `SpinModels`
  12.684 ms (1165 allocations: 78.28 KiB)

julia> CUDA.functional() && @btime CUDA.@sync ratiomatch(cu_z, cu_θ, cu_θ̄, cu_buffer);# From `SpinModels` (pre-allocated)
  12.398 ms (1110 allocations: 76.44 KiB)
```




## System information

```julia
julia> using InteractiveUtils; versioninfo()
Julia Version 1.7.1
Commit ac5cc99908 (2021-12-22 19:35 UTC)
Platform Info:
  OS: Linux (x86_64-pc-linux-gnu)
  CPU: Intel(R) Xeon(R) Platinum 8259CL CPU @ 2.50GHz
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-12.0.1 (ORCJIT, cascadelake)
```


Thread information
```julia
julia> Sys.CPU_THREADS
4

julia> BLAS.get_num_threads()
4
```


GPU information
```julia
julia> CUDA.functional() && run(`nvidia-smi`);
Tue Mar 22 08:22:19 2022       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 450.142.00   Driver Version: 450.142.00   CUDA Version: 11.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla T4            On   | 00000000:00:1E.0 Off |                    0 |
| N/A   50C    P0    72W /  70W |   2624MiB / 15109MiB |     91%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A     20752      C   julia                            2621MiB |
+-----------------------------------------------------------------------------+
```
