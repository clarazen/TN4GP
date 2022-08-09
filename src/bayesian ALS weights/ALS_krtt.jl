function ALS_krtt(y::Vector,kr::Vector{Matrix},rnks::Vector{Int},maxiter)
    # computes components of tt from y = kr*tt with ALS:
    # tt is first split into the to be updates component and the rest W, 
    # then W and kr are multiplied with each other giving U
    # then y = U*tt[d] is solved for tt[d]

    D     = size(kr,1)
    Md    = size(kr[1],2)
##########################################################################
    cores = Vector{Array{Float64,3}}(undef,D)
    for i = 1:D # creating initial tensor train
        tmp = qr(rand(rnks[i]*Md, rnks[i+1]))
        cores[i] = reshape(Matrix(tmp),(rnks[i], Md, rnks[i+1]))
    end
    tt0 = MPT(cores)
###########################################################################
    tt = tt0;
    for iter = 1:maxiter
        for d = 1:D
            ttm     = getU(tt,d)   # works       
            U       = krtimesttm(kr,transpose(ttm)) # works
            tt[d]   = reshape((U*U')\(U*y),size(tt0[d]))
        end
    end
    return tt
end

function ALS_krtt(y::Vector,kr::Vector{Matrix},rnks::Vector{Int},maxiter,ortho::Bool)
    # computes components of tt from y = kr*tt with ALS:
    # tt is first split into the to be updates component and the rest U, 
    # then U and kr are multiplied with each other giving Ũ
    # then y = Ũ*tt[d] is solved for tt[d]
    # version with orthogonalization (dunno if it is correct)
    D     = size(kr,1)
    Md    = size(kr[1],2)
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
    res   = zeros(maxiter,2D)
    swipe = [collect(D:-1:2)..., collect(1:D-1)...];
    Dir   = Int.([-ones(1,D-1)...,ones(1,D-1)...]);
    for iter = 1:maxiter
        for k = 1:2D-2
            d       = swipe[k];
            ttm     = getU(tt,d)   # works       
            U       = krtimesttm(kr,transpose(ttm)) # works
            tmp     = U*U';
            tt[d]   = reshape(tmp\(U*y),size(tt0[d]))
            shiftMPTnorm(tt,d,Dir[k]) 
        end
    end
    return tt
end

# functions    
function getU(tt::MPT{3},d::Int)
    D           = order(tt)
    middlesizes = size(tt,true)
    M           = middlesizes[d]
    newms       = zeros(2,D)
    newms[1,:]  = middlesizes
    newms[2,:]  = ones(D)
    ttm         = mps2mpo(tt,Int.(newms))
    ttm[d]      = reshape(Matrix(I,(M,M)),(1,M,M,1))
    if d>1
        ttm[d-1] = permutedims(ttm[d-1],(1,2,4,3))
    end
    if d<D
        ttm[d+1] = permutedims(ttm[d+1],(3,2,1,4))
    end
    return ttm
end

function khr2mat(Φ::Vector{Matrix})
    # computes the row-wise Khatri-Rao product for given set of matrices
    Φ_mat = ones(size(Φ[1],1),1)
    for d = size(Φ,1):-1:1
        Φ_mat = KhatriRao(Φ_mat,Φ[d],1)
    end    
    return Φ_mat
end

function kr2ttm(A::Vector{Matrix})
    # transforms a set of matrices into rank-1-TTm
    D    = size(A,1)
    Attm = Vector{Array{Float64,4}}(undef,D)
    for d = 1:D
        Attm[d] = reshape(A[d],(1,size(A[d],1),size(A[d],2),1))
    end
    return MPT(Attm)
end


function initialTT(D::Int,Md::Int,rnks::Vector,b::Bool)
    cores = Vector{Array{Float64,3}}(undef,D)
    for i = 1:D-1 # creating initial site-D tensor train
        tmp      = qr(rand(rnks[i]*Md, rnks[i+1])).Q
        cores[i] = reshape(Matrix(tmp),(rnks[i], Md, rnks[i+1]))
    end
    cores[D] = randn(rnks[D], Md, rnks[D+1])
    return MPT(cores,D)
end

function initialTT(D::Int,Md::Int,rnks::Vector)
    cores = Vector{Array{Float64,3}}(undef,D)
    for i = 1:D # creating initial tensor train
        cores[i] = rand(rnks[i], Md, rnks[i+1])
    end
    return MPT(cores)
end