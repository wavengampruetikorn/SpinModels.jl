"""
    unitrange_rand!(A, r::UnitRange)

Populate the array `A` with values picked randomly from `r`.

    unitrange_rand!(A, r::UnitRange, 𝕡, [method = InverseCDF(𝕡)])

Populate the array `A` with values picked randomly from `r` with weights given by the distribution `𝕡`. 
If `length(𝕡) == length(r) * length(A)`, different weight vectors are used for each entry  of `A`.
"""
function unitrange_rand!(A, r::UnitRange)
    r1 = first(r)   # label of first class
    n = length(r)   # number of classes
    @. A = r1 + ((n * rand()) ÷ 1)
    return A
end
unitrange_rand!(A::Array, r::UnitRange) = rand!(A, r)


function unitrange_rand!(A, r::UnitRange, 𝕡::AbstractArray, method = InverseCDF(𝕡, length(r), length(A)))
    if length(𝕡) == length(r)
        isone(sum(𝕡)) || return error("`sum(𝕡) = $(sum(𝕡)) ≠ 1`")
    elseif length(𝕡) == length(r) * length(A)
        sum!(method.C₁, reshape(𝕡, :, length(A)))
        all(≈(1), method.C₁) || return error("`𝕡` is not normalized")
    else
        msg = "Require `length(𝕡) == length(r)` or `length(𝕡) == length(r) * length(A)`."
        msg*= "Get `(length(𝕡), length(r), length(A)) = $((length(𝕡), length(r), length(A)))`."
        return error(msg)
    end

    method(A, 𝕡)
    A .+= first(r)
    return A
end


struct InverseCDF{T}
    C::T        # qN × M
    C₁::T       #  1 × M
    C₂::T       # qN × 1
    qN::Int
    function InverseCDF(𝕡::AbstractArray, qN::Integer, M::Integer)
        _eltype = promote_type(eltype(𝕡), Float32)
        C = similar(𝕡, _eltype, qN, M)
        C₁= similar(𝕡, _eltype,  1, M)
        C₂= similar(𝕡, _eltype, qN, 1)
        return new{typeof(C)}(C, C₁, C₂, qN)
    end
end
function (f::InverseCDF)(I::AbstractArray, 𝕡)
    (; C, C₁, C₂, qN) = f
    if length(𝕡) == qN
        cumsum!(C₂, reshape(𝕡, qN, 1); dims = 1)
        C₂ ./= maximum(C₂)  # make sure CDF sum to one!
        rand!(C₁)           # one uniform random varaible per sample
        @. C = C₂ ≤ C₁
    else
        cumsum!(C, reshape(𝕡, qN, :); dims = 1)
        maximum!(C₁, C)
        C ./= C₁            # make sure CDF sum to one!
        rand!(C₁)           # one uniform random varaible per sample
        @. C = C ≤ C₁
    end
    sum!(C₁, C)
    return copyto!(I, C₁)      # index of new spin state for each sample
end