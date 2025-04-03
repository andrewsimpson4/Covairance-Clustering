

using Pkg
using LinearAlgebra
using OrderedCollections
using Base.Threads
using Distances
using Clustering
using StatsBase
using Distributions
using BenchmarkTools
using SpecialFunctions


struct WishartMixture
    V::Array
    Pi::Vector
end

struct LatentCovairanceData
    X::AbstractMatrix
    class::AbstractVector
    mu::AbstractMatrix
    cov::AbstractArray
    ID::AbstractVector
    n::AbstractVector
    a::AbstractFloat
    b::AbstractFloat
end

struct ScatterMatrix
    S::AbstractArray
    n::AbstractVector
end

struct WishartMixture_Fitted
    mix::WishartMixture
    mix_REML::WishartMixture
    S::ScatterMatrix
    X::AbstractMatrix
    class::AbstractVector
    mu::AbstractMatrix
    ID::AbstractVector
    log_likelihood::AbstractFloat
    imputed_cov::Any
    imputed_cov_REML::Any
    z::AbstractMatrix
    class_map::Any
end

include("./Sampling.jl")
include("./Initalization.jl")
include("./Likelihoods.jl")





function wishart_reduced(S::AbstractMatrix, V::AbstractMatrix, n::AbstractFloat)
    det(V)^(-n / 2) * exp(-1 / 2 * sum(diag(inv(V) * S)))
end

function wishart_reduced_fast(
    S::AbstractMatrix,
    V_det::AbstractFloat,
    V_inv::AbstractMatrix, n::Int)

    V_det^(-n / 2) * exp(-1 / 2 * sum(diag(V_inv * S)))
end

function wishart_reduced_computational(
    S::AbstractMatrix,
    V_det_k::AbstractFloat,
    V_inv_k::AbstractMatrix,
    V_det::AbstractVector,
    V_inv::AbstractArray,
    Pi_k::AbstractFloat,
    Pi::AbstractVector, n::Int)

    res = 0
    for k_prime in 1:size(V_inv, 3)
        res += Pi[k_prime] / Pi_k *
               exp((-n / 2) * log(V_det[k_prime]) +
                   -1 / 2 * sum(diag(V_inv[:, :, k_prime] * S)) -
                   (-n / 2) * log(V_det_k) - -1 / 2 * sum(diag(V_inv_k * S)))
    end
    return 1 / res
end

function W_E_step(S::ScatterMatrix, mix::WishartMixture; computational=false)
    B = size(S.S, 3)
    K = length(mix.Pi)
    z = zeros(B, K)
    V_det = zeros(K)
    V_inv = zeros(size(mix.V))
    for k in 1:K
        V_det[k] = det(mix.V[:, :, k])
        V_inv[:, :, k] = inv(mix.V[:, :, k])
    end
    for i in 1:B
        for k in 1:K
            if !computational
                z[i, k] = mix.Pi[k] * wishart_reduced_fast(
                    S.S[:, :, i],
                    V_det[k],
                    V_inv[:, :, k],
                    S.n[i])
            else
                z[i, k] = mix.Pi[k] * wishart_reduced_computational(
                    S.S[:, :, i],
                    V_det[k],
                    V_inv[:, :, k],
                    V_det,
                    V_inv,
                    mix.Pi[k],
                    mix.Pi,
                    S.n[i])
            end
        end
    end
    z ./= sum(z, dims=2)
    return z
end




function W_M_step(S::ScatterMatrix, z::Matrix, is_diag::Bool)
    B = size(S.S, 3)
    p = size(S.S, 2)
    K = size(z, 2)
    V = zeros(p, p, K)
    Pi = [sum(z[:, k]) / B for k in 1:K]
    for k in 1:K
        for i in 1:B
            V[:, :, k] += z[i, k] * S.S[:, :, i]
        end
        V[:, :, k] /= sum(z[:, k] .* (S.n))
        if is_diag
            V[:, :, k] = Diagonal(diag(V[:, :, k]))
        end
    end
    return WishartMixture(V, Pi)
end


function W_em_base(
    X::AbstractMatrix,
    class_any::AbstractVector,
    K::Int;
    M=100,
    mix=nothing,
    n=nothing,
    z=nothing,
    computational=false,
    is_diag=false,
    initalization=wishart_mixture_initialization_hierarchical,
    args...)

    classes = unique(class_any)
    class_map = Dict([(i => findall(i .== classes)[1]) for i in classes])
    class = [class_map[d] for d in class_any]

    S = get_S(X, class)
    ID = nothing

    if isnothing(mix) && isnothing(z)
        init = initalization(X, class, K, is_diag; M=M, args)
        if isnothing(init)
            @warn "Failed to initialize.. returning nothing"
            return nothing
        end
        mix = init
    end
    log_new = 0
    log_old = -Inf
    steps = 0
    params_sum_new = 0
    params_sum_old = -1
    if !isnothing(z)
        mix = W_M_step(S, z, is_diag)
    end
    while abs(params_sum_new - params_sum_old) / abs(params_sum_new) > 1e-6
        z = W_E_step(S, mix; computational=computational)
        ID = argmax.(eachrow(z))
        mix = W_M_step(S, z, is_diag)

        for k in 1:K
            error_found = false
            try
                cholesky(mix.V[:, :, k])
            catch e
                if computational == false
                    @warn "Degenerate model.. try computational=true.. returning nothing"
                    error_found = true
                else
                    @warn "Degenerate model.. returning nothing"
                    error_found = true
                end
            end
            if error_found
                return nothing
            end
        end
        log_old = log_new
        params_sum_old = params_sum_new
        params_sum_new = sum(abs, mix.V) + sum(abs, mix.Pi)

        steps += 1
    end
    # println(steps)
    mix_REML = WishartMixture(corrected_V(mix.V, z, S.n), mix.Pi)
    cov = W_imputed_cov(S, mix, z)
    cov_REML = W_imputed_cov(S, mix_REML, z)
    log_new = W_normal_likelihood(X, class, mix)
    # println(log_new)
    return WishartMixture_Fitted(mix, mix_REML, S, X, class, get_mu(X, class),
        ID, log_new, cov, cov_REML, z, class_map)

end

function corrected_V(V::AbstractArray, z::AbstractMatrix, n::AbstractVector)
    K = size(V, 3)
    V_new = zeros(size(V))
    for k in 1:K
        correction = sum(z[:, k] .* n) / sum(z[:, k] .* (n .- 1))
        V_new[:, :, k] = V[:, :, k] * correction
    end
    return V_new
end

function predict(
    em_results::WishartMixture_Fitted,
    X::AbstractMatrix;
    est_type="MLE",
    top=1)

    mix = est_type == "MLE" ? em_results.mix : em_results.mix_REML

    B = size(em_results.mu, 1)
    n = size(X, 1)
    K = length(mix.Pi)
    z = zeros(n, B)

    for b in 1:B
        for k in 1:K
            N = MvNormal(em_results.mu[b, :], mix.V[:, :, k])
            z[:,b] += pdf(N, transpose(X)) * em_results.z[b,k]
        end
    end

    map_inv = Dict(value => key for (key, value) in em_results.class_map)

    indx = ([[string(map_inv[y]) for y in x[1:top]] for x in sortperm.(eachrow(z), rev=true)])

    return mapreduce(permutedims, vcat, indx)

    return [string(map_inv[x]) for x in argmax.(eachrow(z))]
end

function predict_old(
    em_results::WishartMixture_Fitted,
    X::AbstractMatrix;
    est_type="MLE",
    bayes_type="true_bayes")

    mix = est_type == "MLE" ? em_results.mix : em_results.mix_REML

    B = size(em_results.mu, 1)
    n = size(X, 1)
    K = length(mix.Pi)
    z = zeros(n, B)

    if bayes_type == "true_bayes"
        class_wise_like = zeros(B)
        for b in 1:B
            for k in 1:K
                N = MvNormal(em_results.mu[b, :], mix.V[:, :, k])
                class_wise_like[b] += mix.Pi[k] * exp(sum(logpdf(N, transpose(em_results.X[em_results.class.==b, :]))))
            end
        end
        for b in 1:B
            for i in 1:n
                for k in 1:K
                    N = MvNormal(em_results.mu[b, :], mix.V[:, :, k])
                    z[i,b] += mix.Pi[k] * pdf(N, (X[i,:])) * exp(sum(logpdf(N,  transpose(em_results.X[em_results.class.==b, :]))))
                end
                z[i,b] = log(z[i,b]) + (sum(log.(class_wise_like[1:end .!= b])))
            end
        end
    end



    if bayes_type == "soft"
        cov = est_type == "MLE" ? em_results.imputed_cov : em_results.imputed_cov_REML
        for b in 1:B
            N = MvNormal(em_results.mu[b, :], cov[1][:, :, b])
            z[:, b] = logpdf(N, transpose(X))
        end
    end

    if bayes_type == "hard"
        cov = est_type == "MLE" ? em_results.imputed_cov : em_results.imputed_cov_REML
        for b in 1:B
            N = MvNormal(em_results.mu[b, :], cov[2][:, :, b])
            z[:, b] = logpdf(N, transpose(X))
        end
    end

    if bayes_type == "mixture"
        for b in 1:B
            for k in 1:K
                N = MvNormal(em_results.mu[b, :], mix.V[:, :, k])
                z[:, b] += mix.Pi[k] * pdf(N, transpose(X))
            end
        end
    end

    map_inv = Dict(value => key for (key, value) in em_results.class_map)

    return [string(map_inv[x]) for x in argmax.(eachrow(z))]
end

function predict(
    dict::Dict,
    X::AbstractMatrix;
    est_type="MLE",
    top=1)

    mix = TypeDictToWishart(dict)
    return predict(mix, X; est_type, top)
end
function predict(
    dict::OrderedDict,
    X::AbstractMatrix;
    est_type="MLE",
    top=1)

    mix = TypeDictToWishart(dict)
    return predict(mix, X; est_type, top)
end
function TypeWishartToDict(em::WishartMixture_Fitted)
    dict = Dict(
        "V" => em.mix.V,
        "V_REML" => em.mix_REML.V,
        "Pi" => em.mix.Pi,
        "S" => Dict("S" => em.S.S, "n" => em.S.n),
        "X" => em.X,
        "class" => em.class,
        "mu" => em.mu,
        "ID" => em.ID,
        "log_likelihood" => em.log_likelihood,
        "cov" => em.imputed_cov,
        "cov_REML" => em.imputed_cov_REML,
        "z" => em.z,
        "class_map" => em.class_map)
    return dict
end

function TypeDictToWishart(dict::Dict)
    mix = WishartMixture(dict["V"], dict["Pi"])
    mix_REML = WishartMixture(dict["V_REML"], dict["Pi"])
    S = ScatterMatrix(dict["S"]["S"], dict["S"]["n"])
    mix = WishartMixture_Fitted(
        mix,
        mix_REML,
        S, dict["X"],
        dict["class"],
        dict["mu"],
        dict["ID"],
        dict["log_likelihood"],
        dict["cov"],
        dict["cov_REML"],
        dict["z"],
        dict["class_map"])
    return mix
end

function TypeDictToWishart(dict::OrderedDict)
    mix = WishartMixture(dict[:V], dict[:Pi])
    mix_REML = WishartMixture(dict[:V_REML], dict[:Pi])
    S = ScatterMatrix(dict[:S][:S], dict[:S][:n])
    mix = WishartMixture_Fitted(
        mix,
        mix_REML,
        S,
        dict[:X],
        dict[:class],
        dict[:mu],
        dict[:ID],
        dict[:log_likelihood],
        dict[:cov],
        dict[:cov_REML],
        dict[:z],
        dict[:class_map])

    return mix
end

function W_em(
    X::AbstractMatrix,
    class::AbstractVector;
    K=[Inf],
    return_dict=true,
    M=100,
    mix=nothing,
    n=nothing,
    z=nothing,
    computational=false,
    is_diag=false,
    initalization="hierarchical",
    args...)

    models = []
    bics = [Inf]
    classes = unique(class)
    p = size(X, 2)
    n = size(X, 1)
    B = length(classes)

    if initalization == "hierarchical"
        initalization_func=wishart_mixture_initialization_hierarchical
    else
        initalization_func=wishart_mixture_initialization_random
    end
    if isinf(K[1])
        k = 1
        while true
            model = W_em_base(
                X,
                class,
                k;
                M,
                mix,
                n,
                z,
                computational,
                is_diag,
                initalization=initalization_func,
                args...)
            if isnothing(model)
                if length(bics) > 1
                    if return_dict
                        return (TypeWishartToDict(models[end]), bics[2:end])
                    end
                    return (models[end], bics[2:end])
                else
                    return nothing
                end
            end
            new_bic = W_bic(model.log_likelihood, k, n, p, B; args...)
            push!(bics, new_bic)
            push!(models, model)
            if length(bics) > 1
                if (new_bic > bics[end-1])
                    if return_dict
                        return (TypeWishartToDict(models[end-1]), bics[2:end])
                    end
                    return (models[end-1], bics[2:end])
                    break
                end
            else
                return nothing
            end
            k = k + 1
        end

    else
        for k in K
            model = W_em_base(
                    X,
                    class,
                    k;
                    return_dict,
                    M,
                    mix,
                    n,
                    z,
                    computational,
                    is_diag,
                    initalization=initalization_func,
                    args...)
            if isnothing(model)
                break
            end
            push!(bics, W_bic(model.log_likelihood, k, n, p, B; args...))
            push!(models, model)
        end
        if length(bics) > 1
            if return_dict
                return (TypeWishartToDict(models[argmin(bics[2:end])]), bics[2:end])
            end
            return (models[argmin(bics[2:end])], bics[2:end])
        else
            return nothing
        end
    end
end

