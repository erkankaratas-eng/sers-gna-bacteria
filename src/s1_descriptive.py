from common import *
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.signal import find_peaks
from scipy.stats import chi2
from sklearn.metrics import adjusted_rand_score
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9})
A=DATA['minmax']; rng=np.random.default_rng(0)
# ---- Figure 2: mean +- SD spectra (vertically offset)
OFF=1.15
fig,ax=plt.subplots(3,1,figsize=(7,9.6),sharex=True)
for a,sp in zip(ax,SPP):
    for j,(s,c) in enumerate(zip(SB[sp],COL[sp])):
        m=strain==s; mu=A[m].mean(0); sd=A[m].std(0,ddof=1); off=j*OFF
        a.fill_between(W,mu-sd+off,mu+sd+off,color=c,alpha=.25,lw=0); a.plot(W,mu+off,color=c,lw=1)
        a.text(W[-1]+15,mu[-1]+off+0.05,s,color=c,fontsize=8,fontweight='bold',va='bottom',ha='left')
    a.set_yticks([]); a.set_ylabel('Intensity (a.u.)'); a.set_ylim(-0.1,2*OFF+1.15); a.set_xlim(W[0]-10,W[-1]+150)
    a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
    a.text(-0.06,1.0,'abc'[SPP.index(sp)],transform=a.transAxes,fontsize=12,fontweight='bold',va='top')
    a.set_title(sp,loc='left',fontsize=10,style='italic')
ax[-1].set_xlabel('Raman shift (cm$^{-1}$)'); ax[-1].set_xticks([250,500,750,1000,1250,1500,1750,2000])
plt.tight_layout(); plt.savefig(os.path.join(FIG,'Fig2_mean_SD_spectra.png'),dpi=600); plt.savefig(os.path.join(FIG,'Fig2_mean_SD_spectra.pdf')); plt.close()
# ---- PCA + loadings + HCA per species
def conf_ellipse(ax,Z,c):
    mu=Z.mean(0); C=np.cov(Z.T); v,U=np.linalg.eigh(C); r=np.sqrt(chi2.ppf(.95,2))
    ang=np.degrees(np.arctan2(U[1,1],U[0,1])); ax.add_patch(Ellipse(mu,2*r*np.sqrt(v[1]),2*r*np.sqrt(v[0]),angle=ang,fc='none',ec=c,lw=1,ls='--'))
pca_rows=[]; load_rows=[]; hca_rows=[]
for fi,sp in zip([3,4,5],SPP):
    m=species==sp; Am=A[m]; lab=strain[m]
    p=PCA(5).fit(Am); Z=p.transform(Am); ev=p.explained_variance_ratio_
    pca_rows.append([sp]+[f'{100*e:.1f}' for e in ev[:3]]+[f'{100*ev[:2].sum():.1f}'])
    link=linkage(Am,'ward'); cl=fcluster(link,3,'maxclust'); ari=adjusted_rand_score(lab,cl)
    # subsampling stability
    pure=0; aris=[]
    for _ in range(1000):
        idx=np.concatenate([rng.choice(np.where(lab==s)[0],10,replace=False) for s in SB[sp]])
        c2=fcluster(linkage(Am[idx],'ward'),3,'maxclust'); a_=adjusted_rand_score(lab[idx],c2); aris.append(a_); pure+=a_==1
    hca_rows.append([sp,f'{ari:.2f}',f'{100*pure/1000:.1f}%',f'{np.median(aris):.2f} [{np.percentile(aris,2.5):.2f}–{np.percentile(aris,97.5):.2f}]'])
    fig=plt.figure(figsize=(13,4.2)); g=fig.add_gridspec(1,3,width_ratios=[1,1.3,1.3])
    a=fig.add_subplot(g[0])
    for s,c in zip(SB[sp],COL[sp]):
        i=lab==s; a.scatter(Z[i,0],Z[i,1],color=c,s=24,label=s,edgecolor='k',lw=.3,marker=['o','s','^'][SB[sp].index(s)]); conf_ellipse(a,Z[i,:2],c)
    a.axhline(0,color='grey',lw=.5); a.axvline(0,color='grey',lw=.5)
    a.set_xlabel(f'PC1 ({100*ev[0]:.0f}%)'); a.set_ylabel(f'PC2 ({100*ev[1]:.0f}%)'); a.legend(fontsize=8,frameon=False); a.set_title('a) PCA scores',loc='left')
    a=fig.add_subplot(g[1])
    for k,ls in [(0,'-'),(1,'-')]:
        L=p.components_[k]; a.plot(W,L+(0 if k==0 else -0.15),lw=.9,color=['k','#7d3c98'][k],label=f'PC{k+1}')
        pk,_=find_peaks(np.abs(L),distance=10); top=pk[np.argsort(-np.abs(L[pk]))[:10]]
        for t in sorted(top):
            load_rows.append([sp,f'PC{k+1}',round(W[t]),round(float(L[t]),3)])
    a.set_yticks([]); a.set_xlabel('Raman shift (cm$^{-1}$)'); a.set_ylabel('Loading (offset)'); a.legend(fontsize=8,frameon=False); a.set_title('b) PCA loadings',loc='left')
    a=fig.add_subplot(g[2])
    dendrogram(link,labels=[f'{l}' for l in lab],ax=a,color_threshold=(link[-2,2]+link[-3,2])/2,above_threshold_color='grey',leaf_font_size=5,link_color_func=None)
    cmap=dict(zip(SB[sp],COL[sp]))
    for t in a.get_xmajorticklabels(): t.set_color(cmap[t.get_text()])
    a.set_ylabel('Ward distance'); a.set_title('c) HCA (Ward, Euclidean)',loc='left')
    fig.suptitle(f'{sp}',style='italic'); plt.tight_layout(); plt.savefig(os.path.join(FIG,f'Fig{fi}_PCA_HCA_{sp.replace(". ","_")}.png'),dpi=300); plt.close()
TABLES['PCA_variance']=pd.DataFrame(pca_rows,columns=['Species','PC1 (%)','PC2 (%)','PC3 (%)','PC1+PC2 (%)'])
TABLES['PCA_loading_peaks']=pd.DataFrame(load_rows,columns=['Species','PC','Raman shift (cm-1)','Loading'])
TABLES['HCA']=pd.DataFrame(hca_rows,columns=['Species','ARI (3-cluster cut, all spectra)','Perfect strain recovery in 1000 subsamples (10/13 per strain)','ARI median [95% range]'])
pd.to_pickle(TABLES,os.path.join(INTER,'s1.pkl'))
for k,v in TABLES.items(): print('##',k); print(v.head(12).to_string(index=False))
