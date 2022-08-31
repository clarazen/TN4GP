function hypopt(hyp0::Vector)
    res     = optimize(logmarglik,hyp0,LBFGS())
    hypopt  = Optim.minimizer(res);
    ℓ_opt  = exp.(hypopt[1:D])
    σ_f_opt = exp(hypopt[D+1])
    σ_n_opt = exp(hypopt[D+2])

    return [ℓ_opt,σ_f_opt,σ_n_opt]
end


function logmarglik_(hyp::Vector,X,y)
    # computes the values of the log marginal likelihood

    D    = size(X,2)
    K    = covSE(X,X,[exp.(hyp[1:D]),exp(hyp[D+1]),exp(hyp[D+2])])
    id   = Matrix(I,size(X,1),size(X,1));
    Ky   = K + exp(hyp[3])*id
    
    F    = cholesky(Hermitian(Ky + sqrt(eps(1.))*id));
    L    = Matrix(F.L);
    α    = L'\(L\y);

    return 1/2 * (y'*α + log(det(Ky)) + size(X,1)*log(2π))
end

logmarglik(hyp::Vector) = logmarglik_(hyp,X,y)

#=
function dlogmarglik_!(hyp::Vector,dlml::Vector,X,y)
    # computes derivatives of log marginal likelihood for given hyperparamters
    D    = size(X,2)
    K    = covSE(X,X,[exp(hyp[1])*ones(D),exp(hyp[2]),exp(hyp[3])])
    id   = Matrix(I,size(X,1),size(X,1));
    Ky   = K + exp(hyp[3])*id
    
    F    = cholesky(Hermitian(Ky + sqrt(eps(1.))*id));
    L    = Matrix(F.L);
    α    = L'\(L\y);

    # derivatives of kernel with respect to theta_i, where hyp_i = exp(theta_i)
    dKd1 = K.*[norm(X[i,:]-X[j,:])^2/(2exp(hyp[1][1])) for i=1:size(X,1), j=1:size(X,1)]
    dKd2 = K
    dKd3 = exp(hyp[3])*id

    # derivatives of log marginal likelihood 
    dlml[1] = -0.5*tr( (α*α' - Ky\id)*dKd1)
    dlml[2] = -0.5*tr( (α*α' - Ky\id)*dKd2)
    dlml[3] = -0.5*tr( (α*α' - Ky\id)*dKd3)

end

dlogmarglik!(hyp::Vector,dmlm::Vector) = dlogmarglik_!(hyp,dmlm,X,y)
=#