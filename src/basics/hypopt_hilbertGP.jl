function hypopt(hyp0::Vector)
    res     = optimize(logmarglik,hyp0,LBFGS())
    hypopt  = Optim.minimizer(res);
    ℓ_opt  = exp.(hypopt[1:D])
    σ_f_opt = exp(hypopt[D+1])
    σ_n_opt = exp(hypopt[D+2])

    return [ℓ_opt,σ_f_opt,σ_n_opt]
end


function logmarglik_(hyp::Vector,Φ,Λ,y)
    # computes the values of the log marginal likelihood

    N     = size(y,1)
    D     = size(Φ,1)
    M     = size(Φ[1],2)
    Φmat  = kh2mat(Φ)
    Z     = Φmat'*Φmat + exp(hyp[D+2])*inv(Λ)
    term1 = (N-M)* log(exp(hyp[D+2])) + log(det(Z)) + sum(log.(diag(Λ)))
    term2 = 1/exp(hyp[D+2])*(y'*y - )

    return 1/2 * (term1 + term2 + size(X,1)*log(2π))
end

logmarglik(hyp::Vector) = logmarglik_(hyp,Φ,Λ,y)