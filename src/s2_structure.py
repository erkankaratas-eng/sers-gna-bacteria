from common import *
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
rng=np.random.default_rng(0)
T={}
# --- drift
rows=[]
for sp in SPP:
    for s in SB[sp]:
        m=strain==s; Z=PCA(3).fit(DATA['minmax'][m]); sc=Z.transform(DATA['minmax'][m])
        r=[spearmanr(sid[m],sc[:,k]) for k in range(3)]
        best=int(np.argmax([abs(x.statistic) for x in r]))
        rows.append([sp,s]+[f'{x.statistic:+.2f} ({x.pvalue:.3f})' for x in r]+[f'PC{best+1}: {100*Z.explained_variance_ratio_[best]:.0f}% of within-strain variance'])
T['Drift']=pd.DataFrame(rows,columns=['Species','Strain','PC1 rho (p)','PC2 rho (p)','PC3 rho (p)','Most acquisition-order-related PC'])
# --- PERMANOVA
def ss_parts(G,groups):
    N=len(groups); P=np.zeros((N,N))
    for l in np.unique(groups):
        i=groups==l; P[np.ix_(i,i)]=1/i.sum()
    return np.trace(P@G)
def gower(D):
    N=len(D); H=np.eye(N)-1/N; return H@(-0.5*D**2)@H
S9=[s for sp in SPP for s in SB[sp]]
parts=[]
for g1 in itertools.combinations(S9,3):
    if S9[0] not in g1: continue
    rest=[s for s in S9 if s not in g1]
    for g2 in itertools.combinations(rest,3):
        if rest[0] not in g2: continue
        parts.append((g1,g2,tuple(s for s in rest if s not in g2)))
rows=[]
for key in ['minmax','snv']:
    A=DATA[key]; G=gower(squareform(pdist(A))); SSt=np.trace(G)
    SSsp=ss_parts(G,species); SSst=ss_parts(G,strain)
    # species tested at strain level: pseudo-F = (SSsp/2)/((SSst-SSsp)/6) ; exact over 280 partitions
    Fobs=(SSsp/2)/((SSst-SSsp)/6); Fn=[]
    for g in parts:
        lab=np.empty(len(strain),object)
        for k,grp in enumerate(g): lab[np.isin(strain,grp)]=k
        s_=ss_parts(G,lab); Fn.append((s_/2)/((SSst-s_)/6))
    p_sp=np.mean(np.array(Fn)>=Fobs-1e-12)
    # strain within species, permute spectra within species
    Fs=((SSst-SSsp)/6)/((SSt-SSst)/(117-9)); c=0
    for _ in range(999):
        perm=strain.copy()
        for sp in SPP:
            i=np.where(species==sp)[0]; perm[i]=strain[rng.permutation(i)]
        s2=ss_parts(G,perm); c+=((s2-SSsp)/6)/((SSt-s2)/108)>=Fs
    rows.append([key,f'{100*SSsp/SSt:.1f}%',f'{Fobs:.2f}',f'{p_sp:.3f} (exact, 280 strain partitions)',f'{100*(SSst-SSsp)/SSt:.1f}%',f'{Fs:.1f}',f'{(c+1)/1000:.3f}',f'{100*(SSt-SSst)/SSt:.1f}%'])
T['PERMANOVA']=pd.DataFrame(rows,columns=['Preprocessing','Species (% SS)','Species pseudo-F (strains as units)','Species p','Strain within species (% SS)','Strain pseudo-F','Strain p','Within-strain residual (% SS)'])
pd.to_pickle(T,os.path.join(INTER,'s2.pkl'))
for k,v in T.items(): print('##',k); print(v.to_string(index=False))
