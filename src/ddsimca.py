import numpy as np
from scipy.stats import chi2
class DDSIMCA:
    def __init__(s,k=3,alpha=0.05): s.k=k; s.alpha=alpha
    def fit(s,A):
        s.mu=A.mean(0); Y=A-s.mu
        U,S,Vt=np.linalg.svd(Y,full_matrices=False)
        s.P=Vt[:s.k].T; T=Y@s.P; s.lam=(T**2).sum(0)/(len(A)-1)
        h,v=s._hv(A)
        s.h0,s.v0=h.mean(),v.mean()
        s.Nh=max(1,round(2*s.h0**2/h.var(ddof=1))); s.Nv=max(1,round(2*s.v0**2/v.var(ddof=1)))
        s.crit=chi2.ppf(1-s.alpha,s.Nh+s.Nv); return s
    def _hv(s,A):
        Y=A-s.mu; T=Y@s.P; h=((T**2)/s.lam).sum(1); E=Y-T@s.P.T; v=(E**2).sum(1); return h,v
    def f(s,A):
        h,v=s._hv(A); return s.Nh*h/s.h0+s.Nv*v/s.v0
    def accept(s,A): return s.f(A)<s.crit
