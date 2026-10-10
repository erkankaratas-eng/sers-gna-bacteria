"""Figure 9: distance matrix of all spectra and species posteriors for withheld strains."""
from common import *
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8})
A=DATA['minmax']; S9=[s for sp in SPP for s in SB[sp]]
order=np.concatenate([np.where(strain==s)[0][np.argsort(sid[strain==s])] for s in S9])
D=squareform(pdist(A[order])); P=pd.read_pickle(os.path.join(INTER,'s18.pkl'))['lso_post']
SPC={'S. aureus':'#2171b5','P. aeruginosa':'#238b45','E. coli':'#d94801'}; spof={s:sp for sp in SPP for s in SB[sp]}
def ylab(ax,lines_v=False):
    ax.set_yticks([13*k+6 for k in range(9)]); ax.set_yticklabels(S9,fontsize=6.5)
    for t in ax.get_yticklabels(): t.set_color(SPC[spof[t.get_text()]]); t.set_fontweight('bold')
fig=plt.figure(figsize=(7.4,4.1)); gs=fig.add_gridspec(1,2,width_ratios=[1.35,0.75],wspace=0.55)
ax=fig.add_subplot(gs[0]); im=ax.imshow(D,cmap='magma_r',interpolation='nearest')
for k in range(1,9): ax.axhline(13*k-.5,color='white',lw=.4); ax.axvline(13*k-.5,color='white',lw=.4)
for k in range(1,3): ax.axhline(39*k-.5,color='white',lw=1.8); ax.axvline(39*k-.5,color='white',lw=1.8)
ylab(ax); ax.set_xticks([13*k+6 for k in range(9)]); ax.set_xticklabels(S9,rotation=90,fontsize=6.5)
for t in ax.get_xticklabels(): t.set_color(SPC[spof[t.get_text()]]); t.set_fontweight('bold')
cb=plt.colorbar(im,ax=ax,fraction=.046,pad=.03); cb.set_label('Distance between spectra',fontsize=7); cb.ax.tick_params(labelsize=6)
ax.set_title('Distances between all 117 spectra',loc='left',fontsize=8.5,fontweight='bold')
ax.text(-0.27,1.05,'a',transform=ax.transAxes,fontsize=12,fontweight='bold')
ax=fig.add_subplot(gs[1]); im=ax.imshow(P,aspect='auto',cmap='Greys',vmin=0,vmax=1,interpolation='nearest'); ylab(ax)
for k in range(1,9): ax.axhline(13*k-.5,color='#bbbbbb',lw=.4)
for k in range(1,3): ax.axhline(39*k-.5,color='k',lw=1.2)
ax.set_xticks(range(3)); ax.set_xticklabels(SPP,rotation=90,fontsize=7,style='italic')
for t,sp in zip(ax.get_xticklabels(),SPP): t.set_color(SPC[sp])
cb=plt.colorbar(im,ax=ax,fraction=.09,pad=.06); cb.set_label('Posterior probability',fontsize=7); cb.ax.tick_params(labelsize=6)
ax.set_xlabel('Predicted species',fontsize=7.5); ax.set_title('Species assigned when\nthe strain is withheld',loc='left',fontsize=8.5,fontweight='bold')
ax.text(-0.62,1.05,'b',transform=ax.transAxes,fontsize=12,fontweight='bold')
for ext,kw in [('png',dict(dpi=600)),('pdf',{})]: plt.savefig(os.path.join(FIG,f'Fig9_distance_matrix_and_withheld_strains.{ext}'),bbox_inches='tight',**kw)
