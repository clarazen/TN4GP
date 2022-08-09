N           = 1000;    # number of data points
D           = 3;       # dimensions
M           = 20       # number of basis functions per dimension
Nstar       = 10;      # number of test points per dimension
hyp         = [.1*ones(D), 1.0, 0.01];
X,y,f,K     = gensynthdata(N,D,hyp);
Xstar,coord = gengriddata(Nstar,D,-1*ones(D),1*ones(D),true);
mstar,Pstar = fullGP(K,X,Xstar,y,hyp,false);

boundsMin = minimum(X,dims=1);
boundsMax = maximum(X,dims=1);
L         = ((boundsMax.-boundsMin)./2)[1,:]  + 2*sqrt.(.1*ones(D));

# ALS with prior
Φ,invΛ,S          = colectofbasisfunc(M*ones(D),X,hyp[1],hyp[2],L);
Φstar,Λstar,Sstar = colectofbasisfunc(M*ones(D),Xstar,hyp[1],hyp[2],L);
rnks              = Int.([1, 8*ones(D-1,1)..., 1]);
maxiter           = 4;

##########################################
tmp  = khr2mat(Φ)'*khr2mat(Φ) + hyp[3]*mpo2mat(invΛ);
tt_o = initialTT(D,M,rnks,true)      
shiftMPTnorm(tt_o,3,-1)
W_o  = mpo2mat(getU(tt_o,1))
inv1 = inv(W_o'*tmp*W_o)
tt_n = initialTT(D,M,rnks)      
W_n  = mpo2mat(getU(tt_n,1))
inv2 = inv(W_n'*tmp*W_n)
##########################################


# normal ALS
wtt = ALS_krtt(y,Φ,rnks,maxiter,true) # works
norm(y-khr2mat(Φ)*mps2vec(wtt))/norm(y)

@run ALS_krtt(y,Φ,rnks,maxiter,mpo2mat(invΛ),hyp[3],true); 

# with ortho
wtt,mean,covw,res = ALS_krtt(y,Φ,rnks,maxiter,mpo2mat(invΛ),hyp[3],true); 
m_tt              = khr2mat(Φstar)*mps2vec(wtt);
PUT               = reshape(getCovUnscTrans(wtt,covw),(M^D,M^D));
P_tt              = 2*sqrt.(diag(hyp[3]*khr2mat(Φstar)*PUT*khr2mat(Φstar)'));
norm(mstar-m_tt)/norm(mstar)
norm(Pstar-P_tt)/norm(Pstar)


# without ortho
wtt,w,covw = ALS_krtt(y,Φ,rnks,maxiter,mpo2mat(invΛ),hyp[3]); 
m_tt       = khr2mat(Φstar)*mps2vec(wtt);
PUT        = reshape(getCovUnscTrans(wtt,covw),(M^D,M^D));
P_tt       = 2*sqrt.(diag(hyp[3]*khr2mat(Φstar)*PUT*khr2mat(Φstar)'));
norm(mstar-m_tt)/norm(mstar)
norm(Pstar-P_tt)/norm(Pstar)

@run ALS_krtt(y,Φ,rnks,maxiter,mpo2mat(invΛ),hyp[3])


# Hilbert-GP (reduced rank - rr)
tmp   = khr2mat(Φ)'*khr2mat(Φ) + hyp[3]*mpo2mat(invΛ);
w_rr  = tmp\(khr2mat(Φ)'*y);
m_rr  = khr2mat(Φstar) * w_rr;
P_rr  = hyp[3]*khr2mat(Φstar)*inv(tmp) * khr2mat(Φstar)';
P_rr  = 2*sqrt.(diag(P_rr));
norm(mstar-m_rr)/norm(mstar)
norm(Pstar-P_rr)/norm(Pstar)