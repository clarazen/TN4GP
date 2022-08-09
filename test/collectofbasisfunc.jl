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
ΦS                = krtimesttm(Φ,S,1e-15)
ΦSstar            = krtimesttm(Φstar,Sstar,1e-15)
norm(mpo2mat(ΦS)'*mpo2mat(ΦS)-K)/norm(K)
norm(mpo2mat(ΦSstar)'*mpo2mat(ΦS)-covSE(Xstar,X,hyp))/norm(covSE(Xstar,X,hyp))