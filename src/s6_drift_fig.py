from common import *
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
fig,ax=plt.subplots(1,3,figsize=(12,3.6),sharey=False)
for a,sp in zip(ax,SPP):
    for s,c in zip(SB[sp],COL[sp]):
        m=strain==s; sc=PCA(3).fit_transform(DATA['minmax'][m]); r=[spearmanr(sid[m],sc[:,k]).statistic for k in range(3)]
        k=int(np.argmax(np.abs(r))); z=sc[:,k]*np.sign(r[k]); z=(z-z.mean())/z.std(); o=np.argsort(sid[m])
        a.plot(sid[m][o],z[o],'o-',color=c,ms=4,label=f'{s} (PC{k+1}, ρ = {abs(r[k]):.2f})')
    a.set_title(sp,style='italic'); a.set_xlabel('Spectrum ID'); a.set_ylabel('Standardized within-strain PC score'); a.legend(fontsize=7,frameon=False)
plt.tight_layout(); plt.savefig(os.path.join(FIG,'FigS1_within_strain_trend.png'),dpi=300)
