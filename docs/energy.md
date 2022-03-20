# Spin model energy

Load required packages and define data structure 
```julia
using LinearAlgebra
using NNlib, BenchmarkTools
using SpinModels: SRBM, energy, EnergyBuffer

function randdata(P = 50, q = 21, N = 59, M = 256)
    z = Float32.(1:q .== rand(1:q, 1, N, M))
    θ = SRBM(N = N, P = P, q = q, similarto = z)
    @. θ.J = randn()
    @. θ.h = randn()
    @. θ.W = randn()
    @. θ.b = randn()
    return z, θ
end
```




We first write down the energy function which will be compared to `SpinModels.energy()`
```julia
ε(z, θ::SRBM) = ((; J, h, W, b) = θ; ε(z, J, h, W, b))
function ε(z::AbstractMatrix, J, h, W, b)
    E = dot(vec(z), vec(h))
    E+= transpose(vec(z)) * reshape(J, length(z), length(z)) * vec(z)
    E+= sum(logcosh, b .+ (reshape(W, :, length(z)) * vec(z)))
    return -E
end
```



For batched data we define another method
```julia
ε(z::AbstractArray{<:Any, 3}, θ::SRBM) = [ε(xₘ, θ) for xₘ ∈ eachslice(z; dims = 3)]
```




Get some data and a model with random parameters
```julia
z, θ = randdata()
```




Check if the energies from energy function above agree with `SpinModels.energy()`
```julia
julia> ε(z, θ) ≈ energy(z, θ)
true

julia> norm(ε(z, θ) .- energy(z, θ), Inf)
9.1552734f-5
```



## GPU test

```julia
using CUDA
gpu(x) = CuArray(x)
```




Copy data and model to GPU
```julia
cu_z = CUDA.functional() ? gpu(z) : nothing
cu_θ = CUDA.functional() ? SRBM(gpu(θ.J), gpu(θ.h), gpu(θ.W), gpu(θ.b)) : nothing
```




Check the correcness of GPU codes
```julia
julia> CUDA.functional() && (ε(z, θ) ≈ ε(cu_z, cu_θ))
true

julia> CUDA.functional() && norm(ε(z, θ) .- ε(cu_z, cu_θ), Inf)
0.00012207031f0
```

```julia
julia> CUDA.functional() && (ε(z, θ) ≈ collect(energy(cu_z, cu_θ)))
true

julia> CUDA.functional() && norm(ε(z, θ) .- collect(energy(cu_z, cu_θ)), Inf)
9.1552734f-5
```



## Benchmarks

Create buffers for pre-allocated versions of the code
```julia
buffer = EnergyBuffer(z, θ)
cu_buffer = CUDA.functional() ? EnergyBuffer(cu_z, cu_θ) : nothing
```




On CPU
```julia
julia> @btime ε($z, $θ);                                               # Explicit code (defined above)
  22.349 ms (2310 allocations: 1.44 MiB)

julia> @btime energy($z, $θ);                                          # From `SpinModels`
  5.827 ms (27 allocations: 7.12 MiB)

julia> @btime energy($z, $θ, $buffer);                                 # From `SpinModels` (pre-allocated)
  5.722 ms (18 allocations: 816 bytes)
```



On GPU
```julia
julia> CUDA.functional() && CUDA.@time ε(cu_z, cu_θ);                  # Explicit code (defined above)
  0.046551 seconds (28.95 k CPU allocations: 1.266 MiB) (1.02 k GPU allocations: 1.309 MiB, 12.40% memmgmt time)

julia> CUDA.functional() && CUDA.@time energy(cu_z, cu_θ);             # From `SpinModels`
  0.000629 seconds (247 CPU allocations: 12.781 KiB) (6 GPU allocations: 7.118 MiB, 8.74% memmgmt time)

julia> CUDA.functional() && CUDA.@time energy(cu_z, cu_θ, cu_buffer);  # From `SpinModels` (pre-allocated)
  0.000565 seconds (217 CPU allocations: 11.812 KiB)
```



## System information


```julia
julia> using InteractiveUtils; versioninfo()
Julia Version 1.7.1
Commit ac5cc99908 (2021-12-22 19:35 UTC)
Platform Info:
  OS: Linux (x86_64-pc-linux-gnu)
  CPU: Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-12.0.1 (ORCJIT, broadwell)
```


GPU information
```julia
julia> CUDA.functional() && run(`nvidia-smi`);
Sun Mar 20 05:55:01 2022       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 450.142.00   Driver Version: 450.142.00   CUDA Version: 11.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla V100-SXM2...  On   | 00000000:00:1E.0 Off |                    0 |
| N/A   35C    P0    40W / 300W |    619MiB / 16160MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      2502      C   julia                             617MiB |
+-----------------------------------------------------------------------------+
```
