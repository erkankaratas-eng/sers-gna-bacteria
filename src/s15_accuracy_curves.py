"""Accuracy as a function of (a) validation design, (b) number of PCs, (c) number of training spectra per strain."""
from common import *
from sklearn.model_selection import RepeatedStratifiedKFold
rng=np.random.default_rng(0); A=DATA['minmax']; BL=[[1,2,3,4],[5,6,7,8,9],[10,11,12,13]]
COMB=list(itertools.product(*[SB[sp] for sp in SPP]))
def strain_blocked(k):
    c=0
    for sp in SPP:
        m=species==sp; Am=A[m]; y=strain[m]; ids=sid[m]
        for b in BL:
            t=np.isin(ids,b); c+=(pipe('PCA-LDA',k).fit(Am[~t],y[~t]).predict(Am[t])==y[t]).sum()
    return c/117
def species_lso(k,n=None):
    acc=[]
    for held in COMB:
        te=np.isin(strain,held); tr=np.where(~te)[0]
        if n is not None:
            tr=np.concatenate([rng.choice(np.where(strain==s)[0],n,replace=False) for s in np.unique(strain[tr])])
        acc.append((pipe('PCA-LDA',k).fit(A[tr],species[tr]).predict(A[te])==species[te]).mean())
    return np.array(acc)
# (b) number of PCs
KS=list(range(1,13)); pcs=[]
for k in KS:
    a=species_lso(k); pcs.append([k,strain_blocked(k),a.mean(),np.percentile(a,25),np.percentile(a,75)])
pcs=pd.DataFrame(pcs,columns=['PCs','strain_blocked','species_lso_mean','species_lso_q25','species_lso_q75'])
print(pcs.round(3).to_string(index=False))
# (c) training spectra per strain
NS=[2,3,4,6,8,11]; lc=[]
for n in NS:
    st=[]
    for r in range(100):
        c=0; tot=0
        for sp in SPP:
            m=np.where(species==sp)[0]; tr=np.concatenate([rng.choice(m[strain[m]==s],n,replace=False) for s in SB[sp]]); te=np.setdiff1d(m,tr)
            k=min(3,len(tr)-3); c+=(pipe('PCA-LDA',k).fit(A[tr],strain[tr]).predict(A[te])==strain[te]).sum(); tot+=len(te)
        st.append(c/tot)
    sp_=np.concatenate([species_lso(min(5,6*n-3),n) for _ in range(5)])
    lc.append([n,np.mean(st),np.percentile(st,25),np.percentile(st,75),sp_.mean(),np.percentile(sp_,25),np.percentile(sp_,75)])
lc=pd.DataFrame(lc,columns=['n_per_strain','strain_mean','strain_q25','strain_q75','species_mean','species_q25','species_q75'])
print(lc.round(3).to_string(index=False))
pd.to_pickle({'pcs':pcs,'lc':lc},os.path.join(INTER,'s15.pkl'))
