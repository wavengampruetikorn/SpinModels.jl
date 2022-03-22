# Spin model energy

Load required packages and define data structure 
```julia
using LinearAlgebra
using NNlib, BenchmarkTools, CUDA
using SpinModels: SRBM, energy, EnergyBuffer

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
z, θ, cu_z, cu_θ = randdata()
```




Check if the energies from energy function above agree with `SpinModels.energy()`
```julia
julia> ε(z, θ) ≈ energy(z, θ)
true

julia> norm(ε(z, θ) .- energy(z, θ), Inf)
9.1552734f-5
```



## GPU test

Check the correcness of GPU codes
```julia
julia> CUDA.functional() && (ε(z, θ) ≈ ε(cu_z, cu_θ))
true

julia> CUDA.functional() && norm(ε(z, θ) .- ε(cu_z, cu_θ), Inf)
9.1552734f-5
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
  37.507 ms (2310 allocations: 1.44 MiB)

julia> @btime energy($z, $θ);                                          # From `SpinModels`
  5.703 ms (27 allocations: 7.12 MiB)

julia> @btime energy($z, $θ, $buffer);                                 # From `SpinModels` (pre-allocated)
  6.098 ms (18 allocations: 816 bytes)
```



On GPU
```julia
julia> CUDA.functional() && @btime CUDA.@sync ε(cu_z, cu_θ);                  # Explicit code (defined above)
  32.235 ms (28944 allocations: 1.27 MiB)

julia> CUDA.functional() && @btime CUDA.@sync energy(cu_z, cu_θ);             # From `SpinModels`
  255.372 μs (250 allocations: 12.83 KiB)

julia> CUDA.functional() && @btime CUDA.@sync energy(cu_z, cu_θ, cu_buffer);  # From `SpinModels` (pre-allocated)
  233.307 μs (220 allocations: 11.86 KiB)
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
Tue Mar 22 08:16:46 2022       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 450.142.00   Driver Version: 450.142.00   CUDA Version: 11.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla T4            On   | 00000000:00:1E.0 Off |                    0 |
| N/A   43C    P0    36W /  70W |    480MiB / 15109MiB |     80%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A     20752      C   julia                             477MiB |
+-----------------------------------------------------------------------------+
```
