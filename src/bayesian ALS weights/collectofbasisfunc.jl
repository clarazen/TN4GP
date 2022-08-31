function colectofbasisfunc(M::Vector,X::Matrix{Float64},ℓ::Vector,σ_f::Float64,L::Vector)
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