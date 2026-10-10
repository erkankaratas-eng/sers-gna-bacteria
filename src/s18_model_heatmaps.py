"""Model-based heatmaps: DD-SIMCA distances of every spectrum to every strain model, and species posteriors for withheld strains."""
from common import *
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8})
A=DATA['minmax']; S9=[s for sp in SPP for s in SB[sp]]
order=np.concatenate([np.where(strain==s)[0][np.argsort(sid[strain==s])] for s in S9])
SPC={'S. aureus':'#2171b5','P. aeruginosa':'#238b45','E. coli':'#d94801'}; spof={s:sp for sp in SPP for s in SB[sp]}
# (a) DD-SIMCA: log10(f/fcrit) of each spectrum to each strain model (3 PCs, 11 training spectra)
M=np.zeros((117,9))
for j,s in enumerate(S9):
    mdl=DDSIMCA(3).fit(A[(strain==s)&~TEST]); M[:,j]=np.log10(mdl.f(A)/mdl.crit)
M=M[order]
# (b) species posteriors when the strain is withheld (PCA-LDA, 5 PCs), averaged over the 9 combinations containing that strain
P=np.zeros((117,3)); n=np.zeros(117)
for held in itertools.product(*[SB[sp] for sp in SPP]):
    te=np.isin(strain,held); mdl=pipe('PCA-LDA',5).fit(A[~te],species[~te])
    pr=mdl.predict_proba(A[te]); cols=[list(mdl.classes_).index(sp) for sp in SPP]
    P[te]+=pr[:,cols]; n[te]+=1
P=(P/n[:,None])[order]
pd.to_pickle({'simca_log_f':M,'lso_post':P,'order':order},os.path.join(INTER,'s18.pkl'))
fig=plt.figure(figsize=(5.2,5.4)); ax=fig.add_subplot(111); lim=2.5
def ylab(ax):
    ax.set_yticks([13*k+6 for k in range(9)]); ax.set_yticklabels(S9,fontsize=6.5)
    for t in ax.get_yticklabels(): t.set_color(SPC[spof[t.get_text()]]); t.set_fontweight('bold')
    for k in range(1,9): ax.axhline(13*k-.5,color='white',lw=.5)
    for k in range(1,3): ax.axhline(39*k-.5,color='k',lw=1.2)
im=ax.imshow(np.clip(M,-1,lim),aspect='auto',cmap='RdBu_r',norm=TwoSlopeNorm(0,-1,lim),interpolation='nearest')
ylab(ax); ax.set_xticks(range(9)); ax.set_xticklabels(S9,rotation=90,fontsize=6.5)
for t in ax.get_xticklabels(): t.set_color(SPC[spof[t.get_text()]]); t.set_fontweight('bold')
for k in range(1,3): ax.axvline(3*k-.5,color='k',lw=1.2)
tr=~TEST[order]; ax.scatter(np.repeat(-0.75,117)[~tr],np.arange(117)[~tr],marker='>',s=8,color='k',clip_on=False)
cb=plt.colorbar(im,ax=ax,fraction=.05,pad=.03); cb.set_label('log$_{10}$(f / f$_{crit}$)   (< 0: accepted)',fontsize=7); cb.ax.tick_params(labelsize=6)
ax.set_xlabel('Strain-specific SIMCA model')
for ext in ['png','pdf']: plt.savefig(os.path.join(FIG,f'FigSI6_simca_distance_heatmap.{ext}'),dpi=600 if ext=='png' else None,bbox_inches='tight')
# summary numbers
acc=(M<0); own=np.array([S9.index(strain[i]) for i in order])
print('own-model acceptance (train / test):', acc[np.arange(117),own][tr].mean().round(2), acc[np.arange(117),own][~tr].mean().round(2))
sp_true=np.array([SPP.index(species[i]) for i in order]); print('mean posterior for true species by species:',[P[sp_true==k,k].mean().round(2) for k in range(3)])
print('mean posterior rows SA->(SA,PA,EC):',P[sp_true==0].mean(0).round(2),' EC->',P[sp_true==2].mean(0).round(2),' PA->',P[sp_true==1].mean(0).round(2))
