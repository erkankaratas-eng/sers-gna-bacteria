"""Spectrum-level heatmaps ordered by species, strain and acquisition order."""
from common import *
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8})
A=DATA['minmax']; S9=[s for sp in SPP for s in SB[sp]]
order=np.concatenate([np.where(strain==s)[0][np.argsort(sid[strain==s])] for s in S9])
Ao=A[order]; D=squareform(pdist(Ao))
SPC={'S. aureus':'#2171b5','P. aeruginosa':'#238b45','E. coli':'#d94801'}
spof={s:sp for sp in SPP for s in SB[sp]}
fig=plt.figure(figsize=(6.2,4.6)); ax=fig.add_subplot(111)
im=ax.imshow(Ao,aspect='auto',cmap='viridis',extent=[W[0],W[-1],len(Ao)-0.5,-0.5],interpolation='nearest')
for k in range(1,9): ax.axhline(13*k-.5,color='white',lw=.4)
for k in range(1,3): ax.axhline(39*k-.5,color='white',lw=1.6)
ax.set_yticks([13*k+6 for k in range(9)]); ax.set_yticklabels(S9,fontsize=7)
for t in ax.get_yticklabels(): t.set_color(SPC[spof[t.get_text()]]); t.set_fontweight('bold')
ax.set_xlabel('Raman shift (cm$^{-1}$)'); cb=plt.colorbar(im,ax=ax,fraction=.04,pad=.02); cb.set_label('Normalized intensity',fontsize=7); cb.ax.tick_params(labelsize=6)
h=[plt.Line2D([],[],ls='',marker='s',ms=7,color=SPC[sp],label=sp) for sp in SPP]
fig.legend(handles=h,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(0.5,-0.06),prop={'style':'italic','size':7.5})
for ext in ['png','pdf']: plt.savefig(os.path.join(FIG,f'FigSI5_spectra_heatmap.{ext}'),dpi=600 if ext=='png' else None,bbox_inches='tight')
