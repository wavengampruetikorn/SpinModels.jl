"""
    unitrange_rand!(A, r::UnitRange)

Populate the array `A` with values picked randomly from `r`.

    unitrange_rand!(A, r::UnitRange, ๐ก, [method = InverseCDF(๐ก)])

Populate the array `A` with values picked randomly from `r` with weights given by the distribution `๐ก`. 
If `length(๐ก) == length(r) * length(A)`, different weight vectors are used for each entry  of `A`.
"""
function unitrange_rand!(A, r::UnitRange)
    r1 = first(r)   # label of first class
    n = length(r)   # number of classes
    @. A = r1 + ((n * rand()) รท 1)
    return A
end
unitrange_rand!(A::Array, r::UnitRange) = rand!(A, r)


function unitrange_rand!(A, r::UnitRange, ๐ก::AbstractArray, method = InverseCDF(๐ก, length(r), length(A)))
    if length(๐ก) == length(r)
        isone(sum(๐ก)) || return error("`sum(๐ก) = $(sum(๐ก)) โ  1`")
    elseif length(๐ก) == length(r) * length(A)
        sum!(method.Cโ, reshape(๐ก, :, length(A)))
        all(โ(1), method.Cโ) || return error("`๐ก` is not normalized")
    else
        msg = "Require `length(๐ก) == length(r)` or `length(๐ก) == length(r) * length(A)`."
        msg*= "Get `(length(๐ก), length(r), length(A)) = $((length(๐ก), length(r), length(A)))`."
        return error(msg)
    end

    method(A, ๐ก)
    A .+= first(r)
    return A
end


struct InverseCDF{T}
    C::T        # qN ร M
    Cโ::T       #  1 ร M
    Cโ::T       # qN ร 1
    qN::Int
    function InverseCDF(๐ก::AbstractArray, qN::Integer, M::Integer)
        _eltype = promote_type(eltype(๐ก), Float32)
        C = similar(๐ก, _eltype, qN, M)
        Cโ= similar(๐ก, _eltype,  1, M)
        Cโ= similar(๐ก, _eltype, qN, 1)
        return new{typeof(C)}(C, Cโ, Cโ, qN)
    end
end
function (f::InverseCDF)(I::AbstractArray, ๐ก)
    (; C, Cโ, Cโ, qN) = f
    if length(๐ก) == qN
        cumsum!(Cโ, reshape(๐ก, qN, 1); dims = 1)
        Cโ ./= maximum(Cโ)  # make sure CDF sum to one!
        rand!(Cโ)           # one uniform random varaible per sample
        @. C = Cโ โค Cโ
    else
        cumsum!(C, reshape(๐ก, qN, :); dims = 1)
        maximum!(Cโ, C)
        C ./= Cโ            # make sure CDF sum to one!
        rand!(Cโ)           # one uniform random varaible per sample
        @. C = C โค Cโ
    end
    sum!(Cโ, C)
    return copyto!(I, Cโ)      # index of new spin state for each sample
end