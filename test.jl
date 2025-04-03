
include("./module.jl")

using .Covairance_Clustering

p = 5 # dimensions
K = 3
n = 1000 # Classes
n_i = 5 # Samples within each class
a = 0 # lower bound for sampling mean of class on hypercube
b = 5 # upper bound for sampling mean of class on hypercube

mix = generate_random_wishart_mixture(K, p) # Generate Mixture Model
X = uniform_sampling_closed_set(mix, n, n_i, a, b) # Sample from model
em = W_em(X.X, X.class, K=[3]; M=100, initalization="hierarchical", return_dict=false) # Fit model



