
module Covairance_Clustering
    include("./EM.jl")
    export  W_em, predict, generate_random_wishart_mixture, uniform_sampling_closed_set
end

