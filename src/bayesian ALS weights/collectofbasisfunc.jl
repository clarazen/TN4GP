function colectofbasisfunc(M::Vector,X::Matrix{Float64},ℓ::Vector,σ_f::Float64,L::Vector)
 # computes Φ,S,Λ such that Φ*S*S*Φ' approx K and Λ contains inserse eigenvalues
    N = size(X,1)
    D = size(X,2)
    Φ = Vector{Matrix}(undef,D)
    
    Φ  = Vector{Matrix}(undef,D);
    Λ  = Vector{Matrix}(undef,D);
    S  = Vector{Matrix}(undef,D);
    for d = 1:D
        w    = collect(1:M[d])';
        tmp  = σ_f^(1/D)*sqrt(2π*ℓ[d]) .* exp.(- ℓ[d]/2 .* ((π.*w')./(2L[d])).^2 )
        Λ[d] = Diagonal(1 ./ tmp)
        S[d] = Diagonal(sqrt.(tmp));
        Φ[d] = (1/sqrt(L[d])) .*sinpi.(  ((X[:,d].+L[d])./2L[d]).*w);
    end

    return Φ,kr2ttm(Λ),kr2ttm(S)
end

function colectofbasisfunc(M::Vector,X::Matrix{Float64},ℓ::Vector,σ_f::Float64,L::Vector,R::Int)
       # computes M^D basis functions and keeps the R leading ones
       N = size(X,1)
       D = size(X,2)
       Λ = 1;
       for d = D:-1:1
           w    = collect(1:M[d])';
           tmp  = σ_f^(1/D)*sqrt(2π*ℓ[d]) .* exp.(- ℓ[d]/2 .* ((π.*w')./(2L[d])).^2 )
           Λ    = kron(Λ,spdiag(tmp))
       end
       p  = sortperm(diag(Λ))[1:R] # first R values of sorting permutation
       ΛR = Λ[p,p] # first R eigenvalues
       ind = CartesianIndices([1:M,1:M,1:M])[p]
       tmp = 1;
       for r = 1:R
            for d = 1:D
                tmp = kron(tmp,(1/sqrt(L[d])) .*sinpi.(  ((X[:,d].+L[d])./2L[d]).*ind[d]));
            end
            Φ[:,r] = tmp
       end

       return Φ,ΛR
   end

function colectofbasisfunc(Φ::Vector{Matrix},S::MPT{4},trunc::Float64)
    
    Φmat = khr2mat(Φ)
    ev   = diag(mpo2mat(S)).^2
    perm = sortperm(ev,rev=true)
    ev   = sort(ev,rev=true)
    indx = findfirst(ev.<trunc)

    Φmat = Φmat[:,perm[1:indx]]
    Λ    = diagm(1 ./ ev[1:indx])
    
    return Φmat,Λ
end

function colectofbasisfunc(M::Vector,X::Matrix{Float64},ℓ::Vector,σ_f::Float64,L::Vector,bool::Bool)
# computes Φ_, such that Φ_*Φ_' approx K
    D = size(X,2)

    Φ_ = Vector{Matrix}(undef,D);
    S  = Vector{Matrix}(undef,D);
    for d = 1:D
        w     = collect(1:M[d])';
        S     =  σ_f^(1/D)*sqrt(2π*ℓ[d]) .* exp.(- ℓ[d]/2 .* ((π.*w')./(2L[d])).^2 )
        Φ_[d] = (1/sqrt(L[d])) .*sinpi.(  ((X[:,d].+L[d])./2L[d]).*w).*sqrt.(S)';
    end

    return Φ_
end
