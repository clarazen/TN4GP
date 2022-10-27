function gensynthdata(N::Int64,D::Int64,hyp::Vector)
    σ_n   = sqrt(hyp[3]);
    X     = zeros(N,D);
    jitter = sqrt(eps(1.))
    for d = 1:D
        for n = 1:N
            X[n,d] = rand(1)[1].* 2 .-1
        end
    end
    K      = covSE(X,X,hyp);
    f      = Matrix(cholesky(K+jitter*Matrix(I,size(K))))*randn(N);
    y      = f + σ_n*randn(size(f,1));
    return X, y, f, K 
end

function gensynthdata(N::Int64,D::Int64,hyp::Vector,M::Int)
    σ_n   = sqrt(hyp[3]);
    X     = zeros(N,D);
    jitter = sqrt(eps(1.))
    for d = 1:D
        for n = 1:N
            X[n,d] = rand(1)[1].* 2 .-1
        end
    end
    K      = covSE(X,X[1:M,:],hyp);
    Kuu    = covSE(X[1:M,:],X[1:M,:],hyp);
    f      = K*inv(cholesky(Kuu+sqrt(eps(1.)*Matrix(I,M,M))).L)*randn(M);
    y      = f + σ_n*randn(size(f,1));
    return X, y, f, K 
end

function gensynthdata(p::Int64,hyp::Vector,tr::Float64)
    σ_n    = sqrt(hyp[3]);
    jitter = sqrt(eps(1.))
    X      = zeros(p);
    for i = 1:p
        X[i] = rand(1)[1].* 2 .-1
    end
    X = sort(X)
    K = covSE(X,X,hyp);
    f = Matrix(cholesky(K+jitter*Matrix(I,size(K))))*randn(p);
    y = f + σ_n*randn(size(f,1));
    return X, y, f, K 
end

function gensynthdata(p::Int64,hyp::Vector{Any})
    σ_n    = sqrt(hyp[3]);
    jitter = sqrt(eps(1.))
    X      = zeros(p,1);
    for i = 1:p
        X[i] = rand(1)[1].* 2 .-1
    end
    X[:,1] = sort(X[:,1])
    K      = covSE(X,X,hyp);
    f      = Matrix(cholesky(K+jitter*Matrix(I,size(K))))*randn(p);
    y      = f + σ_n*randn(size(f,1));
    return X, y, f, K 
end

function gengriddata(Md::Int,D::Int,min::Vector,max::Vector,m::Bool)
    coord = Vector{Vector}(undef,D)
    X     = zeros(Md^D,D)
    for d = 1:D
        coord[d] = range(min[d],max[d],length=Md) # coordinate in dth dimension
    end
    Coord = Tuple(coord)
    if m == true
        i=1;
        for d = 1:D
            X[:,i] = getindex.(Iterators.product(Coord...), d)[:]
            i = i+1
        end
        return X,coord
    else
        return coord
    end
end

function gengriddata(Md::Vector,D::Int,min::Vector,max::Vector,m::Bool)
    coord = Vector{Vector}(undef,D)
    X     = zeros(prod(Md),D)
    for d = 1:D
        coord[d] = range(min[d],max[d],length=Md[d]) # coordinate in dth dimension
    end
    Coord = Tuple(coord)
    if m == true
        i=1;
        for d = D:-1:1
            X[:,i] = getindex.(Iterators.product(Coord...), d)[:]
            i = i+1
        end
        return X,coord
    else
        return coord
    end
end

function covSE(Xp::Matrix{Float64},Xq::Matrix{Float64},hyp::Vector)
    ℓ     = hyp[1];
    σ_f   = hyp[2];
    D     = size(Xp,2)

    K = zeros(size(Xp,1),size(Xq,1))
    for i = 1:size(Xp,1)
        for j = 1:size(Xq,1)
            sum = 0
            for d = 1:D
                sum = sum + (Xp[i,d]-Xq[j,d])^2/2ℓ[d]
            end
            K[i,j] = σ_f * exp(-sum)
        end
    end
    return K
end

function covSE(Xp::Vector{Float64},Xq::Vector{Float64},hyp::Vector{Any})
    ℓ     = hyp[1];
    σ_f   = hyp[2];
    D     = size(Xp,2)

    K = zeros(size(Xp,1),size(Xq,1))
    for i = 1:size(Xp,1)
        for j = 1:size(Xq,1)
            sum = 0
            for d = 1:D
                sum = sum + (Xp[i,d]-Xq[j,d])^2/2ℓ[d]
            end
            K[i,j] = σ_f * exp(-sum/2)
        end
    end
    return K
end

function covSE(xp::Vector,xq::Vector,hyp::Vector,tr::Float64)
    # sqared exponential kernel (truncated)
    ℓ     = hyp[1];
    σ_f   = hyp[2];
    K     = spzeros(size(xp,1),size(xq,1))
    for i = 1:size(xp,1)
        for j = 1:size(xq,1)
            exparg = norm(xp[i,:]-xq[j,:]);
            if σ_f * exp(-(exparg^2/2ℓ)) > tr
                K[i,j] = σ_f * exp(-(exparg^2/2ℓ))
            end
        end
    end
    return K
end

function covSE(xp::Matrix,xq::Matrix,hyp::Vector{Any},tr::Float64)
    # sqared exponential kernel (truncated) and length scale for each dimension
    ℓ     = hyp[1];
    σ_f   = hyp[2];
    K     = spzeros(size(xp,1),size(xq,1))
    D     = size(xp,2)
    for i = 1:size(xp,1)
        for j = 1:i
            sum = 0
            for d = 1:D
                sum = sum + (xp[i,d]-xq[j,d])^2/2ℓ[d]
            end
            val = σ_f * exp(-sum)
            if val > tr
                K[i,j] = val
                K[j,i] = val
            end
        end
    end
    return K
end

######### full GP with plot function

function fullGP(K::Matrix,X::Vector,Xstar::Vector,y::Vector,hyp::Vector,plot::Bool)
# one-dimensional
    σ_n   = hyp[3];

    L     = cholesky(K+(σ_n+sqrt(eps(1.0)))*Matrix(I,p,p)).L;
    Ks    = covSE(Xstar,X,hyp);
    Kss   = covSE(Xstar,Xstar,hyp);
    α     = L'\(L\y);
    mstar = Ks*α;
    v     = L\Ks';
    Pstar = Kss - v'*v;
    std   = 2*sqrt.(diag(Pstar));

    if plot == true
        scatter(X,f)
        scatter!(X,y)
        plot!(Xstar, mstar, ribbon = (std, std))
    else
        return mstar, std 
    end
end



function fullGP(K::Matrix,X::Matrix,Xstar::Matrix,y::Vector,hyp::Vector{Any},plot::Bool)
    # two-dimensional
    σ_n   = hyp[3];
    N     = size(X,1)
    D     = size(X,2)
    
    L     = cholesky(K+(σ_n+sqrt(eps(1.0)))*Matrix(I,N,N)).L;
    Ks    = covSE(Xstar,X,hyp);
    Kss   = covSE(Xstar,Xstar,hyp);
    α     = L'\(L\y);
    mstar = Ks*α;
    v     = L\Ks';
    Pstar = Kss - v'*v;
    std   = 2*sqrt.(diag(Pstar));
    
    if plot == true
        if D == 1
            scatter(X,y)
            plot!(Xstar, mstar, ribbon = (std, std))
        elseif D ==2
        #@pgf Plot(
        #    {only_marks, scatter, scatter_src = "explicit"},
        #    Table(
        #        {x = "x", y = "y", meta = "col"}, 
        #        x = Xstar[:,1], y = Xstar[:,2], col = mstar
        #    )
        #)
        surface(Xstar[:,1],Xstar[:,2],mstar)
        end
    else
        return mstar, std 
    end
end

