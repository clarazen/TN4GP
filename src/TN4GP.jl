module TN4GP

using Plots, BigMat, LinearAlgebra, SparseArrays

export gensynthdata, covSE, gengriddata, fullGP, # basics
       colectofbasisfunc, ALS_krtt, getCovUnscTrans, khr2mat, getU,
       initialTT


include("basics/gensynthdata.jl")
include("bayesian ALS weights/collectofbasisfunc.jl")
include("bayesian ALS weights/TT_UTcov.jl")
include("bayesian ALS weights/ALS_krtt.jl")
include("bayesian ALS weights/BayALS_krtt.jl")

end
