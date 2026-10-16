"""Spectral window analysis: where in the spectrum is strain and species information located?"""
from common import *
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
A=DATA['minmax']; BLOCKS=[[1,2,3,4],[5,6,7,8,9],[10,11,12,13]]
WIDTH,STEP=100,50
starts=np.arange(W[0],W[-1]-WIDTH+1e-9,STEP)
def gower(D):
    N=len(D); H=np.eye(N)-1/N; return H@(-0.5*D**2)@H
def ss(G,g):
    N=len(g); P=np.zeros((N,N))
    for l in np.unique(g):
        i=g==l; P[np.ix_(i,i)]=1/i.sum()
    return np.trace(P@G)
rows=[]
for s0 in starts:
    cols=(W>=s0)&(W<s0+WIDTH); Aw=A[:,cols]
    G=gower(squareform(pdist(Aw))); St=np.trace(G); Ssp=ss(G,species); Sst=ss(G,strain)
    # strain classification within species, acquisition-order-blocked CV
    corr=0
    for sp in SPP:
        m=species==sp; Am=Aw[m]; y=strain[m]; ids=sid[m]
        for b in BLOCKS:
            t=np.isin(ids,b); corr+=(pipe('PCA-LDA',3).fit(Am[~t],y[~t]).predict(Am[t])==y[t]).sum()
    # species leave-strain-out, 27 combinations
    acc=[]
    for held in itertools.product(*[SB[sp] for sp in SPP]):
        te=np.isin(strain,held); acc.append((pipe('PCA-LDA',3).fit(Aw[~te],species[~te]).predict(Aw[te])==species[te]).mean())
    rows.append([s0+WIDTH/2,round(s0),round(s0+WIDTH),Ssp/St,(Sst-Ssp)/St,corr/117,np.mean(acc),np.max(acc)])
df=pd.DataFrame(rows,columns=['centre','from','to','R2_species','R2_strain_within','strain_acc_blocked','species_LSO_mean','species_LSO_best'])
pd.to_pickle(df,os.path.join(INTER,'s8.pkl'))
full_strain=None
print(df.round(3).to_string(index=False))
# figure
fig,ax=plt.subplots(3,1,figsize=(8,8.5),sharex=True,gridspec_kw={'height_ratios':[1,1,1]})
for sp,c in zip(SPP,['#0072B2','#009E73','#D55E00']):
    ax[0].plot(W,A[species==sp].mean(0),color=c,lw=1,label=sp)
ax[0].set_ylabel('Mean intensity'); ax[0].legend(fontsize=8,frameon=False,prop={'style':'italic','size':8}); ax[0].set_title('a) Mean spectra per species',loc='left',fontsize=9)
ax[1].plot(df.centre,100*df.R2_strain_within,'o-',color='k',ms=3,label='strain within species')
ax[1].plot(df.centre,100*df.R2_species,'s-',color='#CC79A7',ms=3,label='species')
ax[1].set_ylabel('Explained variance (%)'); ax[1].legend(fontsize=8,frameon=False); ax[1].set_title('b) PERMANOVA variance components per 100 cm$^{-1}$ window',loc='left',fontsize=9)
ax[2].plot(df.centre,100*df.strain_acc_blocked,'o-',color='k',ms=3,label='strain within species (acquisition-order-blocked CV)')
ax[2].plot(df.centre,100*df.species_LSO_mean,'s-',color='#CC79A7',ms=3,label='species of unseen strain (27 combinations)')
ax[2].axhline(100/3,color='grey',ls='--',lw=.8); ax[2].text(W[0]+5,35,'chance',fontsize=7,color='grey')
ax[2].set_ylabel('Accuracy (%)'); ax[2].set_xlabel('Raman shift (cm$^{-1}$)'); ax[2].legend(fontsize=8,frameon=False,loc='center right'); ax[2].set_ylim(0,105)
ax[2].set_title('c) PCA-LDA (3 PCs) on each window',loc='left',fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(FIG,'FigS8_spectral_windows.png'),dpi=300)
