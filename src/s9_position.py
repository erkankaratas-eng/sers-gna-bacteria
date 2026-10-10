"""Removal of the acquisition-order-related direction(s) estimated from within-strain spectral gradients."""
from common import *
from scipy.spatial.distance import pdist, squareform
A=DATA['minmax']; BLOCKS=[[1,2,3,4],[5,6,7,8,9],[10,11,12,13]]
def pos_dirs(Ab,lab,ids,k):
    """orthonormal directions spanned by within-strain regression slopes of spectrum on acquisition order"""
    S=[]
    for s in np.unique(lab):
        m=lab==s
        if m.sum()<3: continue
        x=ids[m]-ids[m].mean(); Y=Ab[m]-Ab[m].mean(0); S.append(x@Y/(x@x))
    U,sv,Vt=np.linalg.svd(np.array(S),full_matrices=False)
    return Vt[:k]
def proj(Ab,V): return Ab-(Ab@V.T)@V
def gower(D):
    N=len(D); H=np.eye(N)-1/N; return H@(-0.5*D**2)@H
def ss(G,g):
    N=len(g); P=np.zeros((N,N))
    for l in np.unique(g):
        i=g==l; P[np.ix_(i,i)]=1/i.sum()
    return np.trace(P@G)
rows=[]
for k in [0,1,2,3]:
    # variance removed and acquisition-order correlation remaining (all data, descriptive)
    V=pos_dirs(A,strain,sid,k) if k else np.zeros((0,A.shape[1]))
    Ac=proj(A,V) if k else A
    Wi=np.concatenate([Ac[strain==s]-Ac[strain==s].mean(0) for s in np.unique(strain)])
    Wo=np.concatenate([A[strain==s]-A[strain==s].mean(0) for s in np.unique(strain)])
    removed=1-(Wi**2).sum()/(Wo**2).sum()
    G=gower(squareform(pdist(Ac))); St=np.trace(G); r2s=(ss(G,strain)-ss(G,species))/St; r2sp=ss(G,species)/St
    # strain classification, acquisition-order-blocked CV; directions estimated from training folds only
    corr=0
    for sp in SPP:
        m=species==sp; Am=A[m]; y=strain[m]; ids=sid[m]
        for b in BLOCKS:
            t=np.isin(ids,b)
            if k:
                Vt=pos_dirs(A[~np.isin(sid,b)],strain[~np.isin(sid,b)],sid[~np.isin(sid,b)],k)
                tr,te=proj(Am[~t],Vt),proj(Am[t],Vt)
            else: tr,te=Am[~t],Am[t]
            corr+=(pipe('PCA-LDA',3).fit(tr,y[~t]).predict(te)==y[t]).sum()
    # species leave-strain-out; directions from training strains only
    acc=[]
    for held in itertools.product(*[SB[sp] for sp in SPP]):
        te=np.isin(strain,held)
        if k:
            Vt=pos_dirs(A[~te],strain[~te],sid[~te],k); tr,ts=proj(A[~te],Vt),proj(A[te],Vt)
        else: tr,ts=A[~te],A[te]
        acc.append((pipe('PCA-LDA',5).fit(tr,species[~te]).predict(ts)==species[te]).mean())
    rows.append([k,f'{100*removed:.0f}%',f'{100*r2s:.0f}%',f'{100*r2sp:.0f}%',fmt(corr,117),f'{100*np.mean(acc):.0f}%'])
df=pd.DataFrame(rows,columns=['Acquisition-order directions removed','Within-strain variance removed','Strain within species (% SS)','Species (% SS)','Strain classification, acquisition-order-blocked CV','Species of unseen strain (27 combinations)'])
pd.to_pickle(df,os.path.join(INTER,'s9.pkl')); print(df.to_string(index=False))
