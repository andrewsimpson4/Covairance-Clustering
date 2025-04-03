

function wishart_mixture_initialization_hierarchical(X::AbstractMatrix,
    class::AbstractVector, K::Int, is_diag::Bool; M=100, args...)

    S = get_S(X, class)

    B = size(S.S, 3)
    p = size(S.S, 2)
    best_lik = -Inf
    best_pars = nothing
    S_vec = nothing
    if !is_diag
        S_vec = zeros(p*p, B)
        for i in 1:B
            S_tmp = S.S[: ,: ,i] / S.n[i]
            eig_val = eigvals(S_tmp)
            eig_vec = eigvecs(S_tmp)
            S_vec[:, i] = vec(eig_vec * diagm(sqrt.(abs.(eig_val))) * transpose(eig_vec))
        end
    end

    if is_diag
        S_vec = zeros(p, B)
        for i in 1:B
            S_tmp = S.S[: ,: ,i] / S.n[i]
            eig_val = eigvals(S_tmp)
            eig_vec = eigvecs(S_tmp)
            S_vec[:, i] = diag(eig_vec * diagm(sqrt.(abs.(eig_val))) * transpose(eig_vec))
        end
    end

    d = pairwise(Euclidean(), S_vec, dims=2)
    hc = hclust(d, linkage=:ward)
    IDs = cutree(hc, k=K)
    #return IDs

    bad_clust = false
    V = zeros(p, p, K)

    Pi = zeros(K)

    for k in 1:K
        W = S.S[:, :, IDs .== k]
        df_sub = S.n[IDs .== k]
        for i in 1:size(W, 3)
            V[:, :, k] += W[:, :, i]
        end
        V[:, :, k] /= sum(S.n[IDs .== k])
        Pi[k] = sum(S.n[IDs .== k])
        if is_diag
            V[:, :, k] = Diagonal(diag(V[:, :, k]))
        end
        if Pi[k] <= p
             return nothing
        end
        if isnothing(try
            cholesky(V[:, :, k])
        catch e
            nothing
        end)
            @warn "Degenerate model in initalization.. returning nothing"
            return nothing
        end
    end


    Pi = Pi / sum(Pi)


    return WishartMixture(V, Pi)
end

function wishart_mixture_initialization_random(X::AbstractMatrix,
    class::AbstractVector, K::Int, is_diag::Bool; M=100, args...)

    S = get_S(X, class)
    B = size(S.S, 3)
    p = size(S.S, 2)
    best_lik = -Inf
    best_pars = nothing

    S_vec = nothing

    if !is_diag
        S_vec = zeros(p*p, B)
        for i in 1:B
            S_tmp = S.S[: ,: ,i] / S.n[i]
            eig_val = eigvals(S_tmp)
            eig_vec = eigvecs(S_tmp)
            S_vec[:, i] = vec(eig_vec * diagm(sqrt.(abs.(eig_val))) * transpose(eig_vec))
        end
    end

    if is_diag
        S_vec = zeros(p, B)
        for i in 1:B
            S_tmp = S.S[: ,: ,i] / S.n[i]
            eig_val = eigvals(S_tmp)
            eig_vec = eigvecs(S_tmp)
            S_vec[:, i] = diag(eig_vec * diagm((sqrt.(abs.(eig_val)))) * transpose(eig_vec))
        end
    end

    d = pairwise(Euclidean(), S_vec, dims=2)

    for m in 1:M
        seeds = rand(1:B, K)

        D = d[:,seeds]

        IDs = argmax.(eachrow(D))

        bad_clust = false
        V = zeros(p, p, K)

        Pi = zeros(K)

        for k in 1:K
            W = S.S[:, :, IDs .== k]
            df_sub = S.n[IDs .== k]
            for i in 1:size(W, 3)
                V[:, :, k] += W[:, :, i]
            end
            V[:, :, k] /= sum(S.n[IDs .== k])
            Pi[k] = sum(S.n[IDs .== k])
            if is_diag
                V[:, :, k] = diagm(diag(V[:, :, k]))
            end
            if Pi[k] <= p + 1
                bad_clust = true
            end

            try
                cholesky(V[:, :, k])
            catch e
                bad_clust = true
            end
        end

        if !bad_clust
            Pi = counts(IDs) / sum(counts(IDs))
            mix = WishartMixture(V, Pi)
            lik = W_normal_likelihood(X, class, mix)
            if lik > best_lik
                best_lik = lik
                best_pars = mix
            end
        end
    end
    return best_pars
end
