# test krtimesttm: for ALS
    Φ    = Vector{Matrix}(undef,4)
    Φ[1] = randn(100,4)
    Φ[2] = randn(100,4)
    Φ[3] = randn(100,4)
    Φ[4] = randn(100,4)
    Φmat    = zeros(100,64*4)
    for n = 1:100
        Φmat[n,:] = kron(kron(kron(Φ[4][n,:],Φ[3][n,:]),Φ[2][n,:]),Φ[1][n,:])
    end
    norm(Φmat-khr2mat(Φ))/norm(Φmat)

    w       = rand(256,1);
    mps,err = MPT_SVD(w,[4 4 4 4],0.0);
    mps_    = transpose(getU(mps,4))
    test    = mps[4][:]'*krtimesttm(Φ,mps_)
    ref     = Φmat*w
    
    norm(test-ref')/norm(ref')