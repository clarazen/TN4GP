function ALS_krtt(y::Vector,kr::Vector{Matrix},rnks::Vector{Int},maxiter::Int,invΛ::Matrix,σ_n::Float64,ortho::Bool)

    D   = size(kr,1)
    Md  = size(kr[1],2)

    tt0 = initialTT(D,Md,rnks,true)
    tt  = tt0;

    mean  = Vector{Vector}(undef,D)
    cova  = Vector{Matrix}(undef,D)
    res   = zeros(maxiter,2D-2)
    swipe = [collect(D:-1:2)...,collect(1:D-1)...];
    Dir   = Int.([-ones(1,D-1)...,ones(1,D-1)...]);
    for iter = 1:maxiter
        for k = 1:2D-2
            d           = swipe[k];
            ttm         = getU(tt,d)   # works       
            U           = krtimesttm(kr,transpose(ttm)) # works
            W           = mpo2mat(ttm)
            tmp         = U*U' + σ_n*W'*invΛ*W
            mean[d]     = tmp\(U*y)
            tt[d]       = reshape(mean[d],size(tt[d]))
            cova[d]     = inv(tmp)
            tt,cova[d]  = shiftMPTnorm(tt,cova[d],d,Dir[k]) 
            res[iter,k] = norm(y - khr2mat(kr)*mps2vec(tt))/norm(y)
        end
    end
    return tt,mean,cova,res
end




function ALS_krtt(y::Vector,kr::Vector{Matrix},rnks::Vector{Int},maxiter::Int,invΛ::Matrix,σ_n::Float64)
    # computes components of tt from y = kr*tt with ALS:
    # tt is first split into the to be updates component and the rest W, 
    # then W and kr are multiplied with each other giving U
    # then y = U*tt[d] is solved for tt[d]

    D     = size(kr,1)
    Md    = size(kr[1],2)
##########################################################################
    cores = Vector{Array{Float64,3}}(undef,D)
    for i = 1:D # creating initial tensor train
        tmp      = rand(rnks[i]*Md, rnks[i+1])
        cores[i] = reshape(Matrix(tmp),(rnks[i], Md, rnks[i+1]))
    end
    tt0 = MPT(cores)
###########################################################################
    tt       = tt0
    mean     = Vector{Vector}(undef,D)
    cova     = Vector{Matrix}(undef,D)
    res      = zeros(maxiter,D)
    for iter = 1:maxiter
        for d = 1:D
            ttm     = getU(tt,d)   # works       
            U       = krtimesttm(kr,transpose(ttm)) # works
            W       = mpo2mat(ttm)
            tmp     = U*U' + σ_n*W'*invΛ*W
            mean[d] = tmp\(U*y)
            tt[d]   = reshape(mean[d],size(tt[d]))
            cova[d] = tmp\Matrix(I,size(tmp))
            res[iter,d] = norm(y - khr2mat(kr)*mps2vec(tt))/norm(y)
        end
    end
    return tt,mean,cova,res
end
