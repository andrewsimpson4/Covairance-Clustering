

function generate_random_wishart_mixture(K::Int, p::Int)
    cov = zeros(p, p, K)
    Pi = zeros(4)
    for k in 1:K
        R = rand(Uniform(-1,1), p,p)
        W = R' * R
        cov[:,:,k] = W

        Pi = rand(10:1000, K)
        Pi = Pi / sum(Pi)
    end

    return WishartMixture(cov, Pi)
end

function uniform_sampling_closed_set(mix::WishartMixture, B::Int, n::Int, a::Int, b::Int)
    p = size(mix.V, 1)
    mu = zeros(B, p)
    cov = zeros(p, p, B)
    X = Matrix{Float64}(undef, 0, p)
    ID = zeros(Int, B)
    class = Int[]

    if length(n) == 1
        n = fill(n[1], B)
    elseif length(n) == 2
        n = rand(n[1]:n[2], B)
    end

    for i in 1:B
        mu[i, :] = rand(Uniform(a, b), p)
        k = wsample(1:length(mix.Pi), mix.Pi, 1)
        ID[i] = k[1]
        cov[:, :, i] = mix.V[:, :, k]
        X = vcat(X, transpose(rand(MvNormal(mu[i, :], cov[:, :, i]), n[i])))
        class = vcat(class, fill(i, n[i]))
    end

    return LatentCovairanceData(X, class, mu, cov, ID, n, a, b)
end


function get_S(X::AbstractMatrix, class::AbstractVector)
    classes = unique(class)
    B = length(classes)
    p = size(X, 2)
    S = zeros(p, p, B)
    n = zeros(Int, B)

    for i in 1:B
        W = X[class .== classes[i], :]
        n[i] = size(W, 1)
        S[:, :, i] = (n[i] - 1) * cov(W)
    end

    return ScatterMatrix(S, n)
end


function W_imputed_cov(S::ScatterMatrix, mix::WishartMixture, z::AbstractMatrix)
    B = size(S.S, 3)
    p = size(S.S, 2)
    K = length(mix.Pi)

    cov_soft = zeros(p, p, B)
    cov_hard = zeros(p, p, B)

    for i in 1:B
        for k in 1:K
            cov_soft[:,:,i] += z[i,k] * mix.V[:,:,k]
        end
        k_hard = findmax(z[i,:])[2]
        cov_hard[:,:,i] = mix.V[:,:,k_hard]
    end

    return (soft = cov_soft, hard = cov_hard)
end


