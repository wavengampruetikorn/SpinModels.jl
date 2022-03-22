# Samplers

Load required packages
```julia
using LinearAlgebra, Random, Printf
using NNlib, Zygote, BenchmarkTools, CUDA
using Flux: onecold
using StatsBase: kldivergence
using SpinModels
```




## Set-up

```julia
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




True probability mass distribution
```julia
function true_pmf(q, N, θ)
    q^N > 10^7 && error("too many states!")
    ℙ = Vector{Float32}(undef, q^N)
    x = zeros(q, N)
    inds = Vector{CartesianIndex{2}}(undef, N)
    for (n,s) ∈ enumerate(Iterators.product(ntuple(i -> 1:q, N)...))
        inds .= CartesianIndex.(s, 1:N)
        x[inds] .= 1
        ℙ[n] = -energy(x, θ)[1]
        n == onehot2ind(x)[1] || error("incorrect `onehot2ind()`!")
        x[inds] .= 0
    end
    return softmax!(ℙ)
end
```



Define a function for converting onehot encoded states to the state indices in the true probability vector
```julia
function onehot2ind(x)
    q, N = size(x)[1:2]
    y = collect(onecold(x, 1:q))
    return 1 .+ transpose(y.-1) * (q.^(0:N-1))
end
```



Empirical probability mass distribution
```julia
function empr_pmf(x::AbstractArray{<:Any,3})
    q, N, M = size(x)
    q^N > 10^7 && error("too many states!")
    𝕡 = zeros(q^N)
    for n ∈ onehot2ind(x)
        𝕡[n]+=1
    end
    𝕡 .*= 1//M
    return 𝕡
end
```




Interfact to the samplers in `SpinModels`
```julia
function draw(sampler::SpinSampler, energy, z₀; showevery = 50, steps = 500, true_pmf)
    z = copy(z₀)
    flips = 0.0
    for i ∈ 0:steps
        i > 0 && sampler(z, energy)
        flips += iszero(i) ? 0 : mean(acceptrate(sampler))
        iszero(i) && @printf "step   rate  <flips>  KL[q|p]\n"
        iszero(i) && @printf "-----  ----  -------  -------\n"
        if iszero(i%showevery)
            accrate = iszero(i) ? 0 : mean(acceptrate(sampler))
            kl = kldivergence(empr_pmf(z), true_pmf, 2)
            @printf "%5d  %4.2f  %7.1f  %7.3f\n" i accrate flips kl
        end
    end
end
```




## initialize data and model

Get data, model and construct true probability vector
```julia
P, q, N, M = 20, 4, 5, 2048 # 20 hidden units, 5 4-state spins and sample size 2048
z, θ, cu_z, cu_θ = randdata(P, q, N, M)
ℙ = true_pmf(q, N, θ)
```




## Matropolis-Hastings method

```julia
function init_mh(z, θ)
    energybuffer = EnergyBuffer(z, θ)
    E = x -> energy(x, θ, energybuffer)
    mh = MetropolisHastings(z)
    return  mh, E
end
```


```julia
julia> draw(init_mh(z, θ)..., z; true_pmf = ℙ)                                 # on CPU
step   rate  <flips>  KL[q|p]
-----  ----  -------  -------
    0  0.00      0.0   29.348
   50  0.34     20.0    5.118
  100  0.29     35.2    3.182
  150  0.27     49.5    2.496
  200  0.27     63.5    2.209
  250  0.26     77.1    2.014
  300  0.28     90.7    1.931
  350  0.28    104.2    1.834
  400  0.26    117.8    1.801
  450  0.28    131.5    1.758
  500  0.26    144.9    1.727

julia> CUDA.functional() && draw(init_mh(cu_z, cu_θ)..., cu_z; true_pmf = ℙ)   # on GPU
step   rate  <flips>  KL[q|p]
-----  ----  -------  -------
    0  0.00      0.0   29.348
   50  0.35     21.4    6.168
  100  0.31     38.0    3.958
  150  0.32     53.5    3.137
  200  0.29     68.6    2.749
  250  0.30     83.5    2.540
  300  0.29     98.2    2.357
  350  0.30    113.0    2.207
  400  0.30    127.6    2.142
  450  0.30    142.3    2.070
  500  0.27    156.9    1.983
```



## Gibbs with gradients

```julia
function init_gwg(z, θ)
    energybuffer = EnergyBuffer(z, θ)
    ∂E = x -> (
        E  = energy(x, θ, energybuffer); 
        ∇E = ∂x_energy(x, θ, energybuffer); 
        (E, ∇E)
    )
    gwg = GibbsWithGradients(z)
    return  gwg, ∂E
end
```


```julia
julia> draw(init_gwg(z, θ)..., z; true_pmf = ℙ)                                # on CPU
step   rate  <flips>  KL[q|p]
-----  ----  -------  -------
    0  0.00      0.0   29.348
   50  0.28     17.1    3.180
  100  0.23     29.4    2.393
  150  0.22     40.9    2.078
  200  0.21     52.1    1.933
  250  0.22     63.3    1.890
  300  0.21     74.2    1.801
  350  0.22     84.9    1.786
  400  0.20     95.7    1.780
  450  0.21    106.5    1.749
  500  0.22    117.1    1.727

julia> CUDA.functional() && draw(init_gwg(cu_z, cu_θ)..., cu_z; true_pmf = ℙ)  # on GPU
step   rate  <flips>  KL[q|p]
-----  ----  -------  -------
    0  0.00      0.0   29.348
   50  0.27     16.5    3.933
  100  0.23     28.5    3.219
  150  0.22     39.7    2.887
  200  0.21     50.5    2.784
  250  0.22     61.4    2.704
  300  0.21     71.9    2.634
  350  0.21     82.5    2.546
  400  0.21     92.9    2.505
  450  0.20    103.2    2.467
  500  0.20    113.4    2.433
```



## Benchmarks

### Metropolis-Hastings
```julia
function benchmark_mh(M; P = 20, q = 21, N = 59, gpu = false)
    z, θ, cu_z, cu_θ = randdata(P, q, N, M)
    if gpu && CUDA.functional()
        mh, E = init_mh(cu_z, cu_θ)
        @btime CUDA.@sync $mh($cu_z, $E)
    else
        mh, E = init_mh(z, θ)
        @btime $mh($z, $E)
    end
    (gpu & !CUDA.functional()) && @printf "No GPU"
    return nothing
end
```




Test at different numbers of parallel MC chains
```julia
julia> benchmark_mh(100)
  7.671 ms (42 allocations: 1.86 KiB)

julia> benchmark_mh(500)
  28.618 ms (42 allocations: 1.86 KiB)

julia> benchmark_mh(1000)
  51.419 ms (42 allocations: 1.86 KiB)

julia> benchmark_mh(3000)
  157.067 ms (42 allocations: 1.86 KiB)
```

```julia
julia> benchmark_mh(100,  gpu = true)
  500.975 μs (803 allocations: 59.27 KiB)

julia> benchmark_mh(500,  gpu = true)
  1.020 ms (803 allocations: 62.45 KiB)

julia> benchmark_mh(1000, gpu = true)
  1.947 ms (833 allocations: 66.62 KiB)

julia> benchmark_mh(3000, gpu = true)
  5.382 ms (862 allocations: 83.67 KiB)
```



### Gibbs with gradients
```julia
function benchmark_gwg(M; P = 20, q = 21, N = 59, gpu = false)
    z, θ, cu_z, cu_θ = randdata(P, q, N, M)
    if gpu && CUDA.functional()
        gwg, ∂E = init_gwg(cu_z, cu_θ)
        @btime CUDA.@sync $gwg($cu_z, $∂E)
    else
        gwg, ∂E = init_gwg(z, θ)
        @btime $gwg($z, $∂E)
    end
    (gpu & !CUDA.functional()) && @printf "No GPU"
    return nothing
end
```




Test at different numbers of parallel MC chains
```julia
julia> benchmark_gwg(100)
  30.385 ms (133 allocations: 7.67 KiB)

julia> benchmark_gwg(500)
  105.053 ms (133 allocations: 15.56 KiB)

julia> benchmark_gwg(1000)
  197.108 ms (133 allocations: 25.56 KiB)

julia> benchmark_gwg(3000)
  487.360 ms (133 allocations: 64.62 KiB)
```

```julia
julia> benchmark_gwg(100,  gpu = true)
  1.396 ms (1942 allocations: 130.02 KiB)

julia> benchmark_gwg(500,  gpu = true)
  3.250 ms (1947 allocations: 136.47 KiB)

julia> benchmark_gwg(1000, gpu = true)
  5.401 ms (2123 allocations: 151.08 KiB)

julia> benchmark_gwg(3000, gpu = true)
  14.617 ms (2180 allocations: 185.22 KiB)
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
Tue Mar 22 08:26:10 2022       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 450.142.00   Driver Version: 450.142.00   CUDA Version: 11.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla T4            On   | 00000000:00:1E.0 Off |                    0 |
| N/A   56C    P0    36W /  70W |    576MiB / 15109MiB |     95%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A     20752      C   julia                             573MiB |
+-----------------------------------------------------------------------------+
```
