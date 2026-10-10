from common import *
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False})
d=pd.read_pickle(os.path.join(INTER,'s15.pkl')); pcs,lc=d['pcs'],d['lc']
null=pd.read_pickle(os.path.join(INTER,'s5.pkl'))['_null'][('minmax','PCA-LDA')][1]

# ---- values of panel a, computed here (previously typed in by hand)
from sklearn.model_selection import RepeatedStratifiedKFold
_A=DATA['minmax']; _BL=[[1,2,3,4],[5,6,7,8,9],[10,11,12,13]]
def _rep(Am,y,k):
    cv=RepeatedStratifiedKFold(n_splits=4,n_repeats=50,random_state=1)
    return np.mean([(pipe('PCA-LDA',k).fit(Am[tr],y[tr]).predict(Am[te])==y[te]).mean() for tr,te in cv.split(Am,y)])
ST_RAND=100*np.mean([_rep(_A[species==sp],strain[species==sp],3) for sp in SPP])     # strain, random CV (mean of the three species)
_c=0
for sp in SPP:
    m=species==sp
    for b in _BL:
        t=m&np.isin(sid,b); _c+=(pipe('PCA-LDA',3).fit(_A[m&~t],strain[m&~t]).predict(_A[t])==strain[t]).sum()
ST_BLOCK=100*_c/len(_A)                                                               # strain, acquisition-order-blocked CV
SP_RAND=100*_rep(_A,species,5)                                                        # species, random CV (5 PCs)
_p=np.empty(len(_A),object)
for b in _BL:
    t=np.isin(sid,b); _p[t]=pipe('PCA-LDA',5).fit(_A[~t],species[~t]).predict(_A[t])
SP_BLOCK=100*(_p==species).mean()                                                     # species, acquisition-order-blocked CV
SP_LSO=100*np.mean(pd.read_pickle(os.path.join(INTER,'s5.pkl'))['_null'][('minmax','PCA-LDA')][0])  # species, strains withheld
print('Fig 10a values: strain random %.1f, strain blocked %.1f, species random %.1f, species blocked %.1f, species LSO %.1f'%(ST_RAND,ST_BLOCK,SP_RAND,SP_BLOCK,SP_LSO))
CS,CP,GR='#1f4e79','#D55E00','#8c8c8c'
def chance(a,x0):
    a.axhline(100/3,color=GR,ls=':',lw=1); a.text(x0,35.5,'chance',color=GR,fontsize=6.5)
def lab(a,s,t):
    if s: a.text(-0.24,1.1,s,transform=a.transAxes,fontsize=11,fontweight='bold')
    a.set_title(t,loc='left',fontsize=8.2,fontweight='bold')
# Figure 10: validation design
fig,a=plt.subplots(figsize=(2.6,3.4)); x=[0,1,2]
lo,hi=np.percentile(null,[2.5,97.5]); a.fill_between([1.75,2.25],100*lo,100*hi,color='#d9d9d9',lw=0)
a.text(2,100*hi+2,'random\ngroupings',ha='center',fontsize=6,color='#666')
a.plot([0,1],[ST_RAND,ST_BLOCK],'-o',color=CS,lw=1.8,ms=5)
a.plot(x,[SP_RAND,SP_BLOCK,SP_LSO],'-s',color=CP,lw=1.8,ms=5)
for xi,v in zip([0,1],[ST_RAND,ST_BLOCK]): a.text(xi,v+4,f'{v:.0f}%',ha='center',color=CS,fontsize=7,fontweight='bold')
for xi,v in zip(x,[SP_RAND,SP_BLOCK,SP_LSO]): a.text(xi,v-9,f'{v:.0f}%',ha='center',color=CP,fontsize=7,fontweight='bold',bbox=dict(boxstyle='round,pad=0.1',fc='white',ec='none',alpha=.85) if xi==2 else None)
chance(a,-0.3); a.set_xticks(x); a.set_xticklabels(['Random\nCV','Acq.-order\nblocked CV','Strain\nwithheld'],fontsize=7)
a.set_xlim(-0.35,2.45); a.set_ylim(0,105); a.set_ylabel('Correct assignments (%)'); lab(a,'','Validation design')
for ext in ['png','pdf']: fig.savefig(os.path.join(FIG,f'Fig10_validation_design.{ext}'),dpi=600 if ext=='png' else None,bbox_inches='tight')
plt.close(fig)
# Figure SI-9: model complexity and number of training spectra
fig,ax=plt.subplots(1,2,figsize=(5.4,3.2),gridspec_kw={'wspace':0.3})
a=ax[0]
a.plot(pcs.PCs,100*pcs.strain_blocked,'-o',color=CS,lw=1.8,ms=3.5)
a.fill_between(pcs.PCs,100*pcs.species_lso_q25,100*pcs.species_lso_q75,color=CP,alpha=.15,lw=0)
a.plot(pcs.PCs,100*pcs.species_lso_mean,'-s',color=CP,lw=1.8,ms=3.5)
chance(a,9.3); a.set_xlabel('Number of principal components'); a.set_ylim(0,105); a.set_xticks([1,3,5,7,9,11]); lab(a,'','Model complexity')
a=ax[1]
a.fill_between(lc.n_per_strain,100*lc.strain_q25,100*lc.strain_q75,color=CS,alpha=.15,lw=0)
a.plot(lc.n_per_strain,100*lc.strain_mean,'-o',color=CS,lw=1.8,ms=3.5,label='Strain within species')
a.fill_between(lc.n_per_strain,100*lc.species_q25,100*lc.species_q75,color=CP,alpha=.15,lw=0)
a.plot(lc.n_per_strain,100*lc.species_mean,'-s',color=CP,lw=1.8,ms=3.5,label='Species')
chance(a,8.3); a.set_xlabel('Training spectra per strain'); a.set_ylim(0,105); a.set_xticks(list(lc.n_per_strain)); lab(a,'','Number of spectra')
fig.legend(*ax[1].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(0.5,-0.1),fontsize=7.5)
for ext in ['png','pdf']: fig.savefig(os.path.join(FIG,f'FigSI9_model_complexity_training_size.{ext}'),dpi=600 if ext=='png' else None,bbox_inches='tight')
