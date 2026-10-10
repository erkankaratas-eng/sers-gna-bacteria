"""Cross-conformal prediction sets (target coverage 90%) with calibration at spectrum or strain level."""
from common import *
A=DATA['minmax']; ALPHA=0.10; BLOCKS=[[1,2,3,4],[5,6,7,8,9],[10,11,12,13]]
def scores_inner(Atr,ytr,groups,k,classes):
    """nonconformity 1 - p(true class), obtained out-of-fold over inner groups"""
    s=np.zeros(len(ytr))
    for g in np.unique(groups):
        u=groups==g; mdl=pipe('PCA-LDA',k).fit(Atr[~u],ytr[~u])
        P=mdl.predict_proba(Atr[u]); ci=[list(mdl.classes_).index(c) for c in ytr[u]]
        s[u]=1-P[np.arange(u.sum()),ci]
    return s
def qhat(s):
    n=len(s); return np.quantile(s,min(1,np.ceil((n+1)*(1-ALPHA))/n),method='higher')
def predict_sets(Atr,ytr,Ate,k,q):
    mdl=pipe('PCA-LDA',k).fit(Atr,ytr); P=mdl.predict_proba(Ate)
    return [set(mdl.classes_[1-P[i]<=q]) for i in range(len(Ate))]
def summarize(sets,ytrue,name,cal):
    cov=np.mean([y in S for S,y in zip(sets,ytrue)]); size=np.mean([len(S) for S in sets])
    single=np.mean([len(S)==1 for S in sets]); sc=np.mean([S=={y} for S,y in zip(sets,ytrue)]); empty=np.mean([len(S)==0 for S in sets])
    return [name,cal,f'{100*cov:.0f}%',f'{size:.2f}',f'{100*single:.0f}%',f'{100*sc:.0f}%',f'{100*empty:.0f}%']
rows=[]
# (a) strain within species: outer acquisition-order-blocked; calibration by inner acquisition-order blocks
for sp in SPP:
    m=species==sp; Am=A[m]; y=strain[m]; ids=sid[m]; sets=[None]*m.sum()
    for b in BLOCKS:
        t=np.isin(ids,b); grp=np.array([[i for i,bb in enumerate(BLOCKS) if v in bb][0] for v in ids[~t]])
        q=qhat(scores_inner(Am[~t],y[~t],grp,3,SB[sp]))
        for i,S in zip(np.where(t)[0],predict_sets(Am[~t],y[~t],Am[t],3,q)): sets[i]=S
    rows.append(summarize(sets,y,f'Strain within {sp}','acquisition-order blocks'))
# (b) species of withheld strains: calibration at spectrum level (random inner folds) vs at strain level (inner leave-one-strain-out)
for cal in ['spectra (random inner folds)','strains (inner leave-one-strain-out)']:
    sets=[]; ys=[]
    for held in itertools.product(*[SB[sp] for sp in SPP]):
        te=np.isin(strain,held); Atr,ytr=A[~te],species[~te]
        grp=(np.random.default_rng(0).permutation(len(ytr))%4) if cal.startswith('spectra') else strain[~te]
        q=qhat(scores_inner(Atr,ytr,grp,5,SPP)); sets+=predict_sets(Atr,ytr,A[te],5,q); ys+=list(species[te])
    rows.append(summarize(sets,ys,'Species of withheld strains',cal))
df=pd.DataFrame(rows,columns=['Task','Calibration level','Coverage (target 90%)','Mean set size','Single-label sets','Correct single label','Empty sets'])
pd.to_pickle(df,os.path.join(INTER,'s12.pkl')); print(df.to_string(index=False))
