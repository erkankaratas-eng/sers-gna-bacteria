"""Summary of the species posteriors of withheld strains shown in Figure 9b (PCA-LDA, 5 PCs).
For each spectrum, the posterior probabilities are averaged over the nine leave-strain-out
combinations in which its strain is withheld; the table reports means over the 13 spectra of each
strain and over the 39 spectra of each species."""
from common import *
A=DATA['minmax']; P=np.zeros((len(A),3)); n=np.zeros(len(A))
for held in itertools.product(*[SB[sp] for sp in SPP]):
    te=np.isin(strain,held); m=pipe('PCA-LDA',5).fit(A[~te],species[~te])
    P[te]+=m.predict_proba(A[te])[:,[list(m.classes_).index(sp) for sp in SPP]]; n[te]+=1
P/=n[:,None]; rows=[]
for sp in SPP:
    j=SPP.index(sp)
    for s in SB[sp]+['all']:
        i=(strain==s) if s!='all' else (species==sp)
        rows.append([sp,s if s!='all' else 'all three strains',int(i.sum())]+[round(v,2) for v in P[i].mean(0)]+[round(P[i][:,j].mean(),2),f'{int((P[i].argmax(1)==j).sum())}/{int(i.sum())}'])
df=pd.DataFrame(rows,columns=['Species','Withheld strain','Spectra','Mean P(S. aureus)','Mean P(P. aeruginosa)','Mean P(E. coli)','Mean P(correct species)','Highest posterior for correct species'])
df.to_csv(os.path.join(TAB,'withheld_strain_posteriors.csv'),index=False); print(df.to_string(index=False))
