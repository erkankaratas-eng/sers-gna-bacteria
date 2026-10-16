"""Alternative summary figure: a different strain of the same species is as different as another species."""
from common import *
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False})
A=DATA['minmax']; S9=[s for sp in SPP for s in SB[sp]]
SPC={'S. aureus':'#2171b5','P. aeruginosa':'#238b45','E. coli':'#d94801'}
spof={s:sp for sp in SPP for s in SB[sp]}
D=squareform(pdist(A)); iu=np.triu_indices(len(A),1)
same_st=(strain[:,None]==strain[None,:])[iu]; same_sp=(species[:,None]==species[None,:])[iu]; d=D[iu]
groups=[d[same_st],d[same_sp&~same_st],d[~same_sp]]
labels=['same\nstrain','same species,\nother strain','other\nspecies']
fig=plt.figure(figsize=(7.8,3.6)); gs=fig.add_gridspec(1,3,width_ratios=[1.15,1.25,0.85],wspace=0.6)
# a) distance distributions
ax=fig.add_subplot(gs[0])
cols=['#9e9e9e','#3b6ea8','#c65d1e']
P=[0,1.45,2.9]
vp=ax.violinplot(groups,positions=P,showextrema=False,widths=1.05)
for b,c in zip(vp['bodies'],cols): b.set_facecolor(c); b.set_alpha(.55); b.set_edgecolor('none')
med=[np.median(g) for g in groups]
for i,(g,m) in enumerate(zip(groups,med)):
    i=P[i]
    q1,q3=np.percentile(g,[25,75]); ax.plot([i,i],[q1,q3],color='k',lw=3,solid_capstyle='butt'); ax.plot(i,m,'o',color='white',ms=4,mec='k',zorder=3)
    ax.text(i+0.16,m,f'{m:.1f}',ha='left',va='center',fontsize=7.5,fontweight='bold',color='k',bbox=dict(boxstyle='round,pad=0.12',fc='white',ec='none',alpha=.8),zorder=4)
ax.set_xticks(P); ax.set_xticklabels(labels,fontsize=6.6); ax.set_ylabel('Distance between two spectra')
ax.set_title('Another strain of the same species\nis as different as another species',loc='left',fontsize=8.3,fontweight='bold')
ax.text(-0.33,1.13,'a',transform=ax.transAxes,fontsize=12,fontweight='bold'); ax.set_ylim(0,max(np.percentile(g,99) for g in groups)+1.6)
# b) strain-by-strain mean distance
ax=fig.add_subplot(gs[1])
M=np.array([[cdist(A[strain==a],A[strain==b]).mean() for b in S9] for a in S9])
im=ax.imshow(M,cmap='magma_r')
ax.set_xticks(range(9)); ax.set_yticks(range(9)); ax.set_xticklabels(S9,rotation=90,fontsize=6.3); ax.set_yticklabels(S9,fontsize=6.3)
for t in ax.get_xticklabels()+ax.get_yticklabels(): t.set_color(SPC[spof[t.get_text()]]); t.set_fontweight('bold')
for k in range(3): ax.add_patch(Rectangle((3*k-.5,3*k-.5),3,3,fill=False,ec='white',lw=2.2))
cb=plt.colorbar(im,ax=ax,fraction=.046,pad=.04); cb.set_label('Mean distance',fontsize=6.5); cb.ax.tick_params(labelsize=6)
ax.set_title('No species blocks\n(white squares = same species)',loc='left',fontsize=8.3,fontweight='bold')
ax.spines[:].set_visible(False); ax.text(-0.36,1.13,'b',transform=ax.transAxes,fontsize=12,fontweight='bold')
# c) dendrogram of strain means
ax=fig.add_subplot(gs[2])
L=linkage(np.array([A[strain==s].mean(0) for s in S9]),'average')
dendrogram(L,labels=S9,ax=ax,orientation='right',color_threshold=0,above_threshold_color='#555',leaf_font_size=7)
for t in ax.get_yticklabels(): t.set_color(SPC[spof[t.get_text()]]); t.set_fontweight('bold')
ax.set_xlabel('Distance (strain means)'); ax.spines['left'].set_visible(False)
ax.set_title('Clustering the strains\nmixes the species',loc='left',fontsize=8.3,fontweight='bold')
ax.text(-0.42,1.13,'c',transform=ax.transAxes,fontsize=12,fontweight='bold')
h=[plt.Line2D([],[],ls='',marker='s',ms=7,color=SPC[sp],label=sp) for sp in SPP]
fig.legend(handles=h,loc='lower center',ncol=3,frameon=False,fontsize=7.5,bbox_to_anchor=(0.5,-0.08),prop={'style':'italic','size':7.5})
for ext in ['png','pdf','svg']: plt.savefig(os.path.join(FIG,f'FigS4_distances.{ext}'),dpi=600 if ext=='png' else None,bbox_inches='tight')
print([round(m,2) for m in med],[len(g) for g in groups])
