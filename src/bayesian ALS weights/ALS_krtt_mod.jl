function ALS_krtt_mod(y::Vector,kr::Vector{Matrix},rnks::Vector{Int},maxiter,λ::Float64,σ_n::Float64)
    # computes components of tt from y = kr*tt with ALS:
    # tt is first split into the to be updates component and the rest U, 
    # then U and kr are multiplied with each other giving Ũ
    # then y = Ũ*tt[d] is solved for tt[d]
    # version with orthogonalization 
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
    res   = zeros(maxiter,2D-2)
    swipe = [collect(D:-1:2)..., collect(1:D-1)...];
    Dir   = Int.([-ones(1,D-1)...,ones(1,D-1)...]);
    for iter = 1:maxiter
        for k = 1:2D-2
            d           = swipe[k];
            ttm         = getU(tt,d)   # works       
            U           = krtimesttm(kr,transpose(ttm)) # works
            tmp         = U*U';
            tmp         = tmp + λ*Matrix(I,size(tmp));
            mean[d]     = tmp\(U*y)
            tt[d]       = reshape(mean[d],size(tt0[d]))
            tt          = shiftMPTnorm(tt,d,Dir[k])        
            res[iter,k] = norm(y - khr2mat(kr)*mps2vec(tt))/norm(y)
        end
    end
    
    return tt,res
end

function ALS_(y::Vector,kr::Vector{Matrix},rnks::Vector{Int},maxiter,λ::Float64,σ_n::Float64)
    # computes components of tt from y = kr*tt with ALS:
    # tt is first split into the to be updates component and the rest U, 
    # then U and kr are multiplied with each other giving Ũ
    # then y = Ũ*tt[d] is solved for tt[d]
    # version with orthogonalization 
    D     = size(kr,1)
    Md    = size(kr[1],2)
    N     = length(y)
##########################################################################
    cores       = Vector{Array{Float64,3}}(undef,D);
    for i = 1:D-1 # creating site-D canonical initial tensor train
        tmp = qr(rand(rnks[i]*Md, rnks[i+1]));
        cores[i] = reshape(Matrix(tmp.Q),(rnks[i], Md, rnks[i+1]));
    end
    cores[D]    = reshape(rand(rnks[D]*Md),(rnks[D], Md, 1));
    tt0         = MPT(cores,D);
    ttm         = getU(tt0,D)
    U           = krtimesttm(kr,transpose(ttm))
###########################################################################
    mean  = Vector{Vector}(undef,D)
    res   = zeros(maxiter,2D-2)
    swipe = [collect(D:-1:2)..., collect(1:D-1)...];
    Dir   = Int.([-ones(1,D-1)...,ones(1,D-1)...]);
    for iter = 1:maxiter
        for k = 1:2D-2
            d           = swipe[k];
            tmp         = U*U';
            tmp         = tmp + λ*Matrix(I,size(tmp));
            mean[d]     = tmp\(U*y)
            tt[d]       = reshape(mean[d],size(tt0[d]))
            tt          = shiftMPTnorm(tt,d,Dir[k])        
            res[iter,k] = norm(y - khr2mat(kr)*mps2vec(tt))/norm(y)

            
        end
    end
    
    return tt,res
end


function khr2ttm(W::Vector{Matrix})
    D      = size(W,1)
    N      = size(W[1],1)
    M      = size(W[1],2)
    coresW = Vector{Array{Float64,4}}(undef,D)
    
    coresW[1] = reshape(W[1]',(1,1,M,N))
    for d = 2:D
        coresW[d] = zeros(N,1,M,N)
        for m = 1:M
            coresW[d][:,1,m,:] = Diagonal(W[d][:,m])
        end
    end
    coresW[D] = permutedims(coresW[D],[1,4,3,2])
    
    return MPT(coresW)
end

function khr2ttm(W::Vector{SparseMatrixCSC})
    D      = size(W,1)
    N      = size(W[1],1)
    M      = size(W[1],2)
    coresW = Vector{Array{Float64,4}}(undef,D)
    
    coresW[1] = reshape(W[1]',(1,1,M,N))
    for d = 2:D
        coresW[d] = zeros(N,1,M,N)
        for m = 1:M
            coresW[d][:,1,m,:] = Diagonal(W[d][:,m])
        end
    end
    coresW[D] = permutedims(coresW[D],[1,4,3,2])
    
    return MPT(coresW)
end