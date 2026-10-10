from common import *
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix
rng=np.random.default_rng(0); T={}
A=DATA['minmax']
BLOCKS=[[1,2,3,4],[5,6,7,8,9],[10,11,12,13]]
rows=[]; cms={}
def rep_cv(Am,y,kind,k,rep=50):
    cv=RepeatedStratifiedKFold(n_splits=4,n_repeats=rep,random_state=1); acc=[]
    for tr,te in cv.split(Am,y):
        acc.append((pipe(kind,k).fit(Am[tr],y[tr]).predict(Am[te])==y[te]).mean())
    return np.mean(acc)
for sp in SPP:
    m=species==sp; Am=A[m]; y=strain[m]; ids=sid[m]; te0=TEST[m]
    for kind in ['PCA-LDA','SVM']:
        k=3
        # fixed split
        pr=pipe(kind,k).fit(Am[~te0],y[~te0]).predict(Am[te0]); fx=(pr==y[te0]).sum()
        # acquisition-order-blocked CV (acquisition ranks 1-4, 5-9, 10-13)
        pred=np.empty(len(y),object)
        for b in BLOCKS:
            t=np.isin(ids,b); pred[t]=pipe(kind,k).fit(Am[~t],y[~t]).predict(Am[t])
        bk=(pred==y).sum(); cms[(sp,kind)]=pd.DataFrame(confusion_matrix(y,pred,labels=SB[sp]),index=SB[sp],columns=SB[sp])
        # repeated random CV + permutation
        a=rep_cv(Am,y,kind,k); null=[rep_cv(Am,rng.permutation(y),kind,k,rep=3) for _ in range(199)]
        p=(1+np.sum(np.array(null)>=a))/200
        rows.append([sp,kind,f'{fx}/6',fmt(bk,39),f'{100*a:.0f}% (permutation null {100*np.mean(null):.0f}%, p = {p:.3f})'])
T['Strain_classification']=pd.DataFrame(rows,columns=['Species','Model (3 PCs, PCA fitted in each fold)','Fixed test split (2/strain)','Acquisition-order-blocked CV (3 blocks)','Repeated 4-fold CV (50x)'])
T['Strain_confusion_blocked']=cms
pd.to_pickle(T,os.path.join(INTER,'s3.pkl'))
print(T['Strain_classification'].to_string(index=False))
for k,v in cms.items():
    if k[1]=='PCA-LDA': print(k); print(v)
