function getCovUnscTrans(tt::MPT,Pi::Vector{Matrix})
    M     = length(tt)[2][1]
    elems = prod(size(tt,true))
    Put   = zeros(elems*elems)
    D     = order(tt)

    # parameters
    κ = 3-M; β = 2; α = .001;
    λ = α^2*(M+κ) - M;

    w = 1/(2*(M+λ))
    for d = 1:D    
        numelOfCore = length(tt[d])
        dimsOfCore  = size(tt[d])
        QQT = outerprod(tt,tt)
        bbt = zeros(numelOfCore*numelOfCore)
        PiC = cholesky(Symmetric(Pi[d]))#,Val(true),check=false)
        for i = 1:numelOfCore
            b   = sqrt(M+λ)*PiC.L[:,i]
            bbt = bbt + vec(b*b')
        end
        bbt    = reshape(bbt, (dimsOfCore...,dimsOfCore...))
        bbt    = permutedims(bbt,[1 4 2 5 3 6])
        bbt    = reshape(bbt,size(QQT[d]))
        QQT[d] = bbt
        #for future: don't return the reconstructed version!!
        #add and round in non-tt-structure if necessary!!
        temp = vec(mpo2mat(QQT))
        Put = Put + 2*w*temp
    end
    return Put
end