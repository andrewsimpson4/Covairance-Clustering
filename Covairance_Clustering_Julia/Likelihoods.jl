

function gam(p, x)
    return pi^(p * (p - 1) / 4) * prod([(gamma(x + ((1 - j) / 2))) for j in 1:p])
end

function lgam(p, x)
    return (p * (p - 1) / 4) * log(pi) + sum([(lgamma(x + ((1 - j) / 2))) for j in 1:p])
end

function digam(p, x)
    return sum([(digamma(x + (1 - i) / 2)) for i in 1:p])
end

function W_bic(log_lik::AbstractFloat, K::Int, n::Int, p::Int, C::Int; is_diag=false, args...)
    # Pi.     C mu's with p parameters    K pxp matricies
    M = (K - 1) + C * p + (K * p * (p + 1) / 2)

    if is_diag
        M = (K - 1) + C * p + (K * p)
    end

    return M * log(n) - 2 * log_lik
end


function W_normal_likelihood(X::AbstractMatrix, class::AbstractVector, mix::WishartMixture)
    classes = unique(class)
    mu = get_mu(X, class)
    B = size(mu, 1)
    K = size(mix.V, 3)

    res::BigFloat = 0

    for b in 1:B
        res_k = 0

        X_b = transpose(X[class.==classes[b], :])

        for k in 1:K

            N = MvNormal(mu[b, :], mix.V[:, :, k])

            res_k += mix.Pi[k] * exp(BigFloat(sum(logpdf(N, X_b))))

        end

        res += log(res_k)
    end

    return Float64(res)
end

function wishart(S, V, n)
    p = size(V, 1)

    t1 = 2^(n * p / 2) * det(V)^(n / 2) * exp(lgam(p, n / 2))
    t2 = det(S)^((n - p - 1) / 2) * exp(-1 / 2 * sum(diag(inv(V) * S)))

    return t2 / t1
end

function wishart_log(S, V, n)
    p = size(V, 1)

    t1 = (n * p / 2) * log(2) + (n / 2) * log(det(V)) + lgam(p, n / 2)
    t2 = ((n - p - 1) / 2) * log(det(S)) + (-1 / 2 * sum(diag(inv(V) * S)))

    return t2 - t1
end

function singular_wishart(S, V, n)
    p = size(V, 1)

    eigs = eigen(S).values
    L = [e < 1e-15 ? 1 : e for e in eigs]

    t1 = 2^(n * p / 2) * det(V)^(n / 2) * exp(lgam(n, n / 2))
    t2 = pi^((-p * n + n^2) / 2) * prod(L)^((n - p - 1) / 2) *
         exp(-1 / 2 * sum(diag(inv(V) * S)))

    return t2 / t1
end

function singular_wishart_log(S, V, n)
    p = size(V, 1)

    eigs = eigen(S).values
    L = [e < 1e-15 ? 1 : e for e in eigs]

    t1 = (n * p / 2) * log(2) + (n / 2) * log(det(V)) + lgam(n, n / 2)
    t2 = ((-p * n + n^2) / 2) * log(pi) + ((n - p - 1) / 2) *
                                          log(prod(L)) + (-1 / 2 * sum(diag(inv(V) * S)))

    return t2 - t1
end



function get_mu(X::AbstractMatrix, class::AbstractVector)
    classes = unique(class)
    B = length(classes)
    p = size(X, 2)
    mu = zeros(B, p)
    n = zeros(Int, B)
    for i in 1:B
        W = X[class.==classes[i], :]
        mu[i, :] = mean(W, dims=1)[:]
    end
    return mu
end

# @time get_mu(uu.X, uu.class)

# @time W_normal_likelihood(uu.X, uu.class, yy)






