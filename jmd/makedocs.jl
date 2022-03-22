using Weave

docs_path = splitpath(@__DIR__)
while docs_path[end] ≠ "SpinModels"
    pop!(docs_path)
end
iszero(length(docs_path)) && error("Cannot locate `SpinModels` dir!")
docs_path = joinpath(push!(docs_path, "docs"))

jmddir(x...) = joinpath(@__DIR__, x...)

weave(jmddir("energy.jmd"), out_path = docs_path, doctype = "github");
weave(jmddir("ratiomatch.jmd"), out_path = docs_path, doctype = "github");
weave(jmddir("samplers.jmd"), out_path = docs_path, doctype = "github");
