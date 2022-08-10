function MALS_krtt(y::Vector,kr::Vector{Matrix},rnks::Vector{Int},maxiter::Int,ϵ::Float64)
    # computes components of tt from y = kr*tt with modified ALS:
    # tt is first split into the to be updated components and the rest W, 
    # then W and kr are multiplied with each other giving U
    # then y = U*tt[d,d+1] is solved for tt[d,d+1]
    # then t[d,d+1] is divided with trunacted SVD

    D     = size(kr,1)
    Md    = size(kr[1],2)
##########################################################################
    cores = Vector{Array{Float64,3}}(undef,D);
    for i = 1:D-1 # creating site-D canonical initial tensor train
        tmp      = qr(rand(rnks[i]*Md, rnks[i+1]));
        cores[i] = reshape(Matrix(tmp.Q),(rnks[i], Md, rnks[i+1]));
    end
    cores[D] = reshape(rand(rnks[D]*Md),(rnks[D], Md, 1));
    tt0 = MPT(cores,D);
###########################################################################
    tt = tt0;
    R_all = zeros(maxiter,2D-2)
    res   = zeros(maxiter,2D)
    swipe = [collect(D:-1:2)..., collect(1:D-1)...];
    Dir   = Int.([-ones(1,D-1)...,ones(1,D-1)...]);
    for iter = 1:maxiter
        for k = 1:2D-2
            d = swipe[k];
            
            if Dir[k] < 0 # from right to left
                ttm = getU2(tt,d-1,d) 
                U   = krtimesttm(kr,transpose(ttm))
                lsq = Matrix(((U*U')\(U*y))')
                tmp = svd(reshape(lsq,(size(tt[d-1],1)*Md,Md*size(tt[d],3))))
            else # from left to right
                ttm = getU2(tt,d,d+1) 
                U   = krtimesttm(kr,transpose(ttm))
                lsq = (U*U')\(U*y)
                tmp = svd(reshape(lsq,(size(tt[d],1)*Md,Md*size(tt[d+1],3))))
            end

            ### truncated SVD
            R       = length(tmp.S);
            sv2     = cumsum(reverse(tmp.S).^2);
            tr      = Int(findfirst(sv2 .> ϵ^2))-1;
            if tr > 0
                R = length(tmp.S) - tr;
                err2 += sv2[tr];
            end
            ###
            
            # updates of current (orthogonalized) and next TT-core
            if Dir[k] < 0 # from right to left
                tt[d-1]  = reshape(tmp.U[:,1:R]*Diagonal(tmp.S[1:R]), (size(tt[d-1],1),Md,R) )
                tt[d]    = reshape(tmp.Vt[1:R,:],(R,Md,size(tt[d],3)))
            else # from left to right
                tt[d]    = reshape(tmp.U[:,1:R],(size(tt[d],1),Md,R))
                tt[d+1]  = reshape(Diagonal(tmp.S[1:R])*tmp.Vt[1:R,:],(R,Md,size(tt[d+1],3)))
            end
            R_all[iter,k] = R
            
            # if we want to impose the rank instead:
            #tt[d]   = reshape(tmp.U[:,1:rnks[d+1]],(size(tt[d],1),size(tt[d],2),rnks[d+1]))
            #tt[d+1] = reshape(Diagonal(tmp.S[1:rnks[d+1]])*tmp.Vt[1:rnks[d+1],:],(rnks[d+1],size(tt[d+1],2),size(tt[d+1],3)))
        end
    end
    return tt,R_all
end

# functions    
function getU2(tt::MPT{3},d1::Int,d2::Int) # works
    D           = order(tt)
    middlesizes = size(tt,true)
    M1          = middlesizes[d1]
    M2          = middlesizes[d2]
    newms       = zeros(2,D)
    newms[1,:]  = middlesizes
    newms[2,:]  = ones(D)
    tmp         = mps2mpo(tt,Int.(newms))
    ttm1        = reshape(Matrix(I,(M1,M1)),(1,M1,M1,1))
    ttm2        = reshape(Matrix(I,(M2,M2)),(1,M2,M2,1))
    if d1 == 1
        ttm = MPT([ttm1,ttm2,tmp[3:D]...])
    elseif d1 == D-1
        ttm = MPT([tmp[1:d1-1]...,ttm1,ttm2])
    else
        ttm = MPT([tmp[1:d1-1]...,ttm1,ttm2,tmp[d1+2:D]...])
    end

    if d1>1
        ttm[d1-1] = permutedims(ttm[d1-1],(1,2,4,3))
    end
    if d1<D-1
        ttm[d1+2] = permutedims(ttm[d1+2],(3,2,1,4))
    end
    return ttm
end

