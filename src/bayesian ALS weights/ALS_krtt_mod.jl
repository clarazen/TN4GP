function ALS_krtt(y::Vector,kr::Vector{Matrix},rnks::Vector{Int},maxiter,σ_n::Float64)
    # computes components of tt from y = kr*tt with ALS:
    # tt is first split into the to be updates component and the rest U, 
    # then U and kr are multiplied with each other giving Ũ
    # then y = Ũ*tt[d] is solved for tt[d]
    # version with orthogonalization (dunno if it is correct)
    D     = size(kr,1)
    Md    = size(kr[1],2)
    N     = length(y)
##########################################################################
    cores = Vector{Array{Float64,3}}(undef,D);
    for i = 1:D-1 # creating site-D canonical initial tensor train
        tmp = qr(rand(rnks[i]*Md, rnks[i+1]));
        cores[i] = reshape(Matrix(tmp.Q),(rnks[i], Md, rnks[i+1]));
    end
    cores[D] = reshape(rand(rnks[D]*Md),(rnks[D], Md, 1));
    tt0 = MPT(cores,D);
###########################################################################
    tt    = tt0
    mean  = Vector{Vector}(undef,D)
    cova  = Vector{Matrix}(undef,D)
    noco     = zeros(maxiter,2D)
    res   = zeros(maxiter,2D)
    swipe = [collect(D:-1:2)..., collect(1:D-1)...];
    Dir   = Int.([-ones(1,D-1)...,ones(1,D-1)...]);
    for iter = 1:maxiter
        for k = 1:2D-2
            d       = swipe[k];
            ttm     = getU(tt,d)   # works       
            U       = krtimesttm(kr,transpose(ttm)) # works
            tmp     = U*U';
            tmp     = tmp + σ_n*Matrix(I,size(tmp));
            mean[d] = tmp\(U*y)
            tt[d]   = reshape(mean[d],size(tt0[d]))
            cova[d] = inv(tmp)
            shiftMPTnorm(tt,d,Dir[k])    # not shifting for covariance yet       
            res[iter,k] = norm(y - kr2mat(Φ)*mps2vec(tt))/norm(y)
            noco[iter,k] = norm(cova[d])
        end
    end
    return tt,mean,cova,res,noco
end

#=
function ALS_krtt(y::Vector,kr::Vector{Matrix},rnks::Vector{Int},maxiter,σ_n::Float64)
    # Lambda is absorbed into Phi, simplifying the prior
    # this seems to be numerically more stable
    # without orthogonalization
        D     = size(kr,1)
        Md    = size(kr[1],2)
        N     = length(y)
    ##########################################################################
        cores = Vector{Array{Float64,3}}(undef,D)
        for i = 1:D # creating initial tensor train
            tmp = rand(rnks[i]*Md, rnks[i+1])
            cores[i] = reshape(tmp,(rnks[i], Md, rnks[i+1]))
        end
        tt0      = MPT(cores)
    ###########################################################################
        tt       = tt0
        mean     = Vector{Vector}(undef,D)
        cova     = Vector{Matrix}(undef,D)
        res      = zeros(maxiter,D)
        noco     = zeros(maxiter,D)
        for iter = 1:maxiter
            for d = 1:D
                ttm     = getU(tt,d)   # works       
                W       = mpo2mat(ttm) # avoid this in future
                U       = krtimesttm(kr,transpose(ttm)) # works
                tmp     = U*U' + σ_n*W'*W
                mean[d] = tmp\(U*y)
                tt[d]   = reshape(mean[d],size(tt0[d]))
                cova[d] = inv(tmp)
                noco[iter,d] = norm(cova[d])
                res[iter,d] = norm(y - kr2mat(Φ)*mps2vec(tt))/norm(y)
            end
        end
        return tt,mean,cova,res,noco
    end
=#