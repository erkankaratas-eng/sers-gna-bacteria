"""Bootstrap stability of the PC1/PC2 loadings (spectra resampled within strains, Procrustes-aligned)."""
from common import *
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.linalg import orthogonal_procrustes
rng=np.random.default_rng(0); A=DATA['minmax']; NB=1000
out={}; rows=[]
for sp in SPP:
    m=species==sp; Am=A[m]; lab=strain[m]
    ref=PCA(2).fit(Am).components_            # 2 x p
    B=np.zeros((NB,2,Am.shape[1]))
    for b in range(NB):
        idx=np.concatenate([rng.choice(np.where(lab==s)[0],13,replace=True) for s in SB[sp]])
        L=PCA(2).fit(Am[idx]).components_
        R,_=orthogonal_procrustes(L.T,ref.T)     # rotate bootstrap loadings onto reference
        B[b]=(L.T@R).T
    lo,hi=np.percentile(B,2.5,axis=0),np.percentile(B,97.5,axis=0); se=B.std(0)
    out[sp]=(ref,lo,hi,se)
    for k in range(2):
        L=ref[k]; pk,_=find_peaks(np.abs(L),distance=10); top=sorted(pk[np.argsort(-np.abs(L[pk]))[:10]])
        for t in top:
            rows.append([sp,f'PC{k+1}',round(W[t]),round(float(L[t]),3),round(float(L[t]/se[k,t]),1),'yes' if (lo[k,t]>0 or hi[k,t]<0) else 'no'])
        stable=np.mean((lo[k]>0)|(hi[k]<0))
        print(sp,f'PC{k+1}: {100*stable:.0f}% of wavenumbers with CI excluding 0')
df=pd.DataFrame(rows,columns=['Species','PC','Raman shift (cm-1)','Loading','Bootstrap ratio','95% CI excludes 0'])
pd.to_pickle({'peaks':df,'frac':{sp:[float(np.mean((out[sp][1][k]>0)|(out[sp][2][k]<0))) for k in range(2)] for sp in SPP}},os.path.join(INTER,'s10.pkl'))
print(df.groupby(['Species','PC'])['95% CI excludes 0'].apply(lambda s:(s=='yes').sum()))
fig,ax=plt.subplots(3,2,figsize=(11,8),sharex=True)
for i,sp in enumerate(SPP):
    ref,lo,hi,se=out[sp]
    for k in range(2):
        a=ax[i,k]; a.fill_between(W,lo[k],hi[k],color='#9ecae1',lw=0,label='95% bootstrap CI'); a.plot(W,ref[k],'k',lw=.7,label='loading')
        a.axhline(0,color='grey',lw=.5); a.set_title(f'{sp} – PC{k+1}',loc='left',fontsize=9,style='italic')
        if i==2: a.set_xlabel('Raman shift (cm$^{-1}$)')
        if k==0: a.set_ylabel('Loading')
ax[0,0].legend(fontsize=7,frameon=False)
plt.tight_layout(); plt.savefig(os.path.join(FIG,'FigS3_bootstrap_loadings.png'),dpi=300)
