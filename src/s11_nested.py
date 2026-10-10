"""Nested cross-validation: number of PCs selected in an inner loop."""
from common import *
from sklearn.model_selection import StratifiedKFold
from collections import Counter
A=DATA['minmax']; BLOCKS=[[1,2,3,4],[5,6,7,8,9],[10,11,12,13]]; KS=list(range(1,11))
rows=[]; sel={}
# (a) strain within species: outer acquisition-order-blocked, inner acquisition-order-blocked on training acquisition blocks
for sp in SPP:
    m=species==sp; Am=A[m]; y=strain[m]; ids=sid[m]; corr=0; ks=[]
    for b in BLOCKS:
        t=np.isin(ids,b); Atr,ytr,itr=Am[~t],y[~t],ids[~t]
        inner=[bb for bb in BLOCKS if bb is not b]
        score={k:0 for k in KS}
        for ib in inner:
            u=np.isin(itr,ib)
            for k in KS: score[k]+=(pipe('PCA-LDA',k).fit(Atr[~u],ytr[~u]).predict(Atr[u])==ytr[u]).sum()
        kb=max(KS,key=lambda k:(score[k],-k)); ks.append(kb)
        corr+=(pipe('PCA-LDA',kb).fit(Atr,ytr).predict(Am[t])==y[t]).sum()
    rows.append([f'Strain within {sp}','acquisition-order-blocked (3 folds)',fmt(corr,39),', '.join(map(str,ks))]); sel[sp]=ks
# (b) species, leave-strain-out (27 outer combos), inner leave-strain-out over the 8 combos of training strains
accs=[]; ks=[]
for held in itertools.product(*[SB[sp] for sp in SPP]):
    te=np.isin(strain,held); trs=[[s for s in SB[sp] if s not in held] for sp in SPP]
    score={k:[] for k in KS}
    for ih in itertools.product(*trs):
        u=np.isin(strain,ih); tr=~te&~u
        for k in KS: score[k].append((pipe('PCA-LDA',k).fit(A[tr],species[tr]).predict(A[u])==species[u]).mean())
    kb=max(KS,key=lambda k:(np.mean(score[k]),-k)); ks.append(kb)
    accs.append((pipe('PCA-LDA',kb).fit(A[~te],species[~te]).predict(A[te])==species[te]).mean())
c=Counter(ks)
rows.append(['Species of withheld strains','leave-strain-out (27 combinations)',f'{100*np.mean(accs):.0f}% (range {100*min(accs):.0f}–{100*max(accs):.0f}%)',', '.join(f'{k}×{v}' for k,v in sorted(c.items()))])
df=pd.DataFrame(rows,columns=['Task','Outer validation','Accuracy (PCs selected in inner loop)','Selected number of PCs per outer fold'])
pd.to_pickle(df,os.path.join(INTER,'s11.pkl')); print(df.to_string(index=False))
