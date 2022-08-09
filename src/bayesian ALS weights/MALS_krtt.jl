function ALS_krtt(y::Vector,kr::Vector{Matrix},rnks::Vector{Int},maxiter::Int,ϵ::Float64)
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
        for d = 1:D-1
            ttm     = getU2(tt,d)   # works       
            U       = krtimesttm(kr,transpose(ttm)) # works
            tmp     = svd(reshape((U*U')\(U*y),(size(tt[d],1)*size(tt[d],2),size(tt[d+1],1)*size(tt[d+1],2))))
            R       = length(tmp.S);
            sv2     = cumsum(reverse(tmp.S).^2);
            tr      = Int(findfirst(sv2 .> ϵ^2))-1;
            if tr > 0
                R = length(tmp.S) - tr;
                err2 += sv2[tr];
            end

            tt[d]   = reshape(tmp.U[:,1:R],(size(tt[d],1),size(tt[d],2),R))
            tt[d+1] = reshape(Diagonal(tmp.S[1:R])*tmp.Vt[1:R,:],(R,size(tt[d+1],2),size(tt[d],3)))
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
function getU2(tt::MPT{3},d::Int) # works
    D           = order(tt)
    middlesizes = size(tt,true)
    M1          = middlesizes[d]
    M2          = middlesizes[d+1]
    newms       = zeros(2,D)
    newms[1,:]  = middlesizes
    newms[2,:]  = ones(D)
    tmp         = mps2mpo(tt,Int.(newms))
    ttm1        = reshape(Matrix(I,(M1,M1)),(1,M1,M1,1))
    ttm2        = reshape(Matrix(I,(M2,M2)),(1,M2,M2,1))
    if d == 1
        ttm = MPT([ttm1,ttm2,tmp[3:D]...])
    elseif d == D-1
        ttm = MPT([tmp[1:d-1]...,ttm1,ttm2])
    else
        ttm = MPT([tmp[1:d-1]...,ttm1,ttm2,tmp[d+2:D]...])
    end

    if d>1
        ttm[d-1] = permutedims(ttm[d-1],(1,2,4,3))
    end
    if d<D-1
        ttm[d+2] = permutedims(ttm[d+2],(3,2,1,4))
    end
    return ttm
end

