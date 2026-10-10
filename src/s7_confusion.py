from common import *
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
A=DATA['minmax']; BLOCKS=[[1,2,3,4],[5,6,7,8,9],[10,11,12,13]]
def blocked(Am,y,ids,k):
    pred=np.empty(len(y),object)
    for b in BLOCKS:
        t=np.isin(ids,b); pred[t]=pipe('PCA-LDA',k).fit(Am[~t],y[~t]).predict(Am[t])
    return pred
res={}
for sp in SPP:
    m=species==sp; p=blocked(A[m],strain[m],sid[m],3); res[sp]=confusion_matrix(strain[m],p,labels=SB[sp])
S9=[s for sp in SPP for s in SB[sp]]
p9=blocked(A,strain,sid,8); res['9']=confusion_matrix(strain,p9,labels=S9)
# species leave-strain-out pooled confusion (27 combos, PCA 5 + LDA)
cmL=np.zeros((3,3),int)
for held in itertools.product(*[SB[sp] for sp in SPP]):
    te=np.isin(strain,held); pr=pipe('PCA-LDA',5).fit(A[~te],species[~te]).predict(A[te])
    cmL+=confusion_matrix(species[te],pr,labels=SPP)
res['LSO']=cmL
# Gram status: leave-one-strain-out (9 folds), exact null over 84 choices of 3 'positive' strains
gram=np.where(species=='S. aureus','Gram+','Gram−')
def loso(lab):
    pr=np.empty(len(lab),object)
    for s in S9:
        t=strain==s; pr[t]=pipe('PCA-LDA',5).fit(A[~t],lab[~t]).predict(A[t])
    return pr
pg=loso(gram); acc=(pg==gram).mean(); res['Gram']=confusion_matrix(gram,pg,labels=['Gram+','Gram−'])
bal=lambda y,p: np.mean([(p[y==c]==c).mean() for c in np.unique(y)])
obs=bal(gram,pg); null=[]
for g in itertools.combinations(S9,3):
    lab=np.where(np.isin(strain,g),'Gram+','Gram−'); null.append(bal(lab,loso(lab)))
null=np.array(null); pval=np.mean(null>=obs-1e-12)
print(f'Gram LOSO: acc {acc:.2f}, balanced {obs:.2f}, null mean {null.mean():.2f}, p={pval:.2f}')
print(res['Gram']); print(res['LSO']); print(res['9'].trace(),'/117')
pd.to_pickle({'cm':res,'gram':(acc,obs,null.mean(),pval)},os.path.join(INTER,'s7.pkl'))
# ---- figure
IT=lambda t: t.replace('S. aureus',r'$\it{S.\ aureus}$').replace('P. aeruginosa',r'$\it{P.\ aeruginosa}$').replace('E. coli',r'$\it{E.\ coli}$')
def draw(ax,cm,labs,title,rot=0,fs=8):
    labs=[IT(l) for l in labs]; title=IT(title)
    pct=cm/cm.sum(1,keepdims=True)
    ax.imshow(pct,cmap='Blues',vmin=0,vmax=1)
    for i in range(len(labs)):
        for j in range(len(labs)):
            ax.text(j,i,f'{cm[i,j]}',ha='center',va='center',fontsize=fs,color='white' if pct[i,j]>.6 else 'black')
    ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs,rotation=rot,fontsize=fs); ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs,fontsize=fs)
    ax.set_xlabel('Predicted',fontsize=8); ax.set_ylabel('True',fontsize=8); ax.set_title(title,fontsize=9,loc='left')
fig=plt.figure(figsize=(13,8.6)); g=fig.add_gridspec(2,4,height_ratios=[1,1.35])
for k,sp in enumerate(SPP):
    a=fig.add_subplot(g[0,k]); acc_=np.trace(res[sp]); draw(a,res[sp],SB[sp],f'{"abc"[k]}) {sp}: {acc_}/39',fs=8)
a=fig.add_subplot(g[0,3]); draw(a,res['Gram'],['Gram+','Gram−'],f'd) Gram status, unseen strain\n   ({100*obs:.0f}% balanced; null {100*null.mean():.0f}%, p = {pval:.2f})')
a=fig.add_subplot(g[1,0:2]); draw(a,res['9'],S9,f'e) Nine strains: {res["9"].trace()}/117',rot=45,fs=7)
a=fig.add_subplot(g[1,2:4]); draw(a,res['LSO'],['S. aureus','P. aeruginosa','E. coli'],f'f) Species, unseen strains (27 combinations pooled): {100*np.trace(cmL)/cmL.sum():.0f}%')
fig.text(0.01,0.99,'Panels a–c, e: PCA-LDA, acquisition-order-blocked cross-validation (acquisition ranks 1–4, 5–9, 10–13 withheld). Panels d, f: strain withheld from training. Colour = row percentage; numbers = counts.',fontsize=8,va='top')
plt.tight_layout(rect=(0,0,1,0.97)); plt.savefig(os.path.join(FIG,'FigSI2_confusion_matrices.png'),dpi=300)
