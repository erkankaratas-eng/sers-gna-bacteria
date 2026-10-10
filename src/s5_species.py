from common import *
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold
def pc(x,n): return f'{x}/{n} ({100*x/n:.0f}%)'
plt.rcParams.update({'font.size':9}); T={}
S9=[s for sp in SPP for s in SB[sp]]
parts=[]
for g1 in itertools.combinations(S9,3):
    if S9[0] not in g1: continue
    rest=[s for s in S9 if s not in g1]
    for g2 in itertools.combinations(rest,3):
        if rest[0] not in g2: continue
        parts.append((g1,g2,tuple(s for s in rest if s not in g2)))
true=tuple(tuple(SB[sp]) for sp in SPP)
def lso(A,groups,kind,k):
    lab=np.zeros(len(strain),int)
    for j,g in enumerate(groups): lab[np.isin(strain,g)]=j
    accs=[]; per=np.zeros((3,2))
    for held in itertools.product(*groups):
        te=np.isin(strain,held); pr=pipe(kind,k).fit(A[~te],lab[~te]).predict(A[te])
        accs.append((pr==lab[te]).mean())
        for j in range(3): i=lab[te]==j; per[j]+=[(pr[i]==j).sum(),i.sum()]
    return np.array(accs),per
rows=[]; NULL={}
for key in ['minmax','snv']:
    A=DATA[key]
    for kind in ['PCA-LDA','SVM']:
        k=5
        acc,per=lso(A,true,kind,k)
        cache=os.path.join(INTER,f'null_{key}_{kind}.npy')   # cached: this step is the slowest part
        if os.path.exists(cache): null=np.load(cache)
        else:
            null=np.array([lso(A,g,kind,k)[0].mean() for g in parts]); np.save(cache,null)
        NULL[(key,kind)]=(acc,null)
        p=np.mean(null>=acc.mean()-1e-12)
        # spectrum-level random CV (same strains in train/test) for contrast
        lab=species; cv=RepeatedStratifiedKFold(n_splits=4,n_repeats=20,random_state=0)
        rc=np.mean([(pipe(kind,k).fit(A[tr],lab[tr]).predict(A[te])==lab[te]).mean() for tr,te in cv.split(A,lab)])
        rows.append([key,kind,f'{100*rc:.0f}%',f'{100*acc.mean():.0f}% (range {100*acc.min():.0f}–{100*acc.max():.0f}%)']+[pc(int(per[j,0]),int(per[j,1])) for j in range(3)]+[f'{100*null.mean():.0f}%',f'{p:.2f}'])
T['Species_LSO']=pd.DataFrame(rows,columns=['Preprocessing','Model (5 PCs)','Random CV (strains shared)','Leave-strain-out, 27 combinations','S. aureus','P. aeruginosa','E. coli','Null mean (280 random strain groupings)','p (exact)'])
# species DD-SIMCA LSO
rows=[]
for key in ['minmax','snv']:
    A=DATA[key]
    for k in [2,3]:
        for sp in SPP:
            acc=0; n=0; rej=0; nr=0
            for held in SB[sp]:
                M=DDSIMCA(k).fit(A[(species==sp)&(strain!=held)])
                acc+=M.accept(A[strain==held]).sum(); n+=13; rej+=(~M.accept(A[species!=sp])).sum(); nr+=78
            rows.append([key,k,sp,fmt(acc,n),pc(rej,nr)])
T['Species_SIMCA_LSO']=pd.DataFrame(rows,columns=['Preprocessing','PCs','Species model','Unseen strain of the species accepted','Other-species spectra rejected'])
pd.to_pickle({**T,'_null':NULL},os.path.join(INTER,'s5.pkl'))
# figure
fig,ax=plt.subplots(1,2,figsize=(11,4))
acc,null=NULL[('minmax','PCA-LDA')]
ax[0].hist(100*null,bins=25,color='#cccccc',edgecolor='k'); ax[0].axvline(100*acc.mean(),color='r',lw=2,label=f'true species grouping ({100*acc.mean():.0f}%)')
ax[0].axvline(100/3,color='k',ls='--',label='chance (33%)'); ax[0].set_xlabel('Mean leave-strain-out accuracy (%)'); ax[0].set_ylabel('Number of strain groupings')
ax[0].set_title('a) True vs. all 280 possible strain groupings (PCA-LDA)',loc='left'); ax[0].legend(frameon=False,fontsize=8)
combos=list(itertools.product(*true)); order=np.argsort(acc)
ax[1].bar(range(27),100*acc[order],color='#5d6d7e'); ax[1].axhline(100/3,color='k',ls='--')
ax[1].set_xticks(range(27)); ax[1].set_xticklabels(['/'.join(combos[i]) for i in order],rotation=90,fontsize=5.5)
ax[1].set_ylabel('Accuracy (%)'); ax[1].set_title('b) Accuracy for each withheld-strain combination',loc='left')
plt.tight_layout(); plt.savefig(os.path.join(FIG,'FigSI7_species_leave_strain_out.png'),dpi=300); plt.close()
for k,v in T.items(): print('##',k); print(v.to_string(index=False))
