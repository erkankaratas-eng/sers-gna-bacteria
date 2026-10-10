from common import *
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':8})
A=DATA['minmax']; T={}; rng=np.random.default_rng(0)
K=3
rows=[]; sens=[]; members=[]
models={}
for sp in SPP:
    for s in SB[sp]:
        models[s]=DDSIMCA(K).fit(A[(strain==s)&~TEST])
for sp in SPP:
    for s in SB[sp]:
        M=models[s]
        own=M.accept(A[(strain==s)&TEST]).sum()
        others=[o for o in SB[sp] if o!=s]
        oth=[M.accept(A[strain==o]).sum() for o in others]
        osp=M.accept(A[species!=sp]).sum()
        rows.append([sp,s,f'{own}/2',f'{oth[0]}/13 ({others[0]}), {oth[1]}/13 ({others[1]})',f'{osp}/78',f'{M.Nh}/{M.Nv}'])
    # membership table of the 6 test spectra
    for s in SB[sp]:
        for idx in np.where((strain==s)&TEST)[0]:
            members.append([sp,f'{s} (file {RAWFILE[idx]}, acquisition rank {sid[idx]})']+['*' if models[t].accept(A[[idx]])[0] else '' for t in SB[sp]])
T['SIMCA_strain_models']=pd.DataFrame(rows,columns=['Species','Class model','Own test spectra accepted','Other strains of same species accepted (all 13 spectra each)','Other-species spectra accepted','DoF Nh/Nv'])
T['SIMCA_test_membership']=pd.DataFrame(members,columns=['Species','Test spectrum','Model 1','Model 2','Model 3'])
# repeated random 11/2 splits and k sensitivity
rows=[]
for k in [2,3,4]:
    own=0; n=0; fa=0; nfa=0
    for r in range(100):
        tr=np.zeros(len(strain),bool)
        for s in np.unique(strain):
            i=np.where(strain==s)[0]; tr[rng.choice(i,11,replace=False)]=True
        for sp in SPP:
            for s in SB[sp]:
                M=DDSIMCA(k).fit(A[(strain==s)&tr]); te=(strain==s)&~tr
                own+=M.accept(A[te]).sum(); n+=te.sum()
                o=(species==sp)&(strain!=s); fa+=M.accept(A[o]).sum(); nfa+=o.sum()
    rows.append([k,f'{100*own/n:.0f}%',f'{100*fa/nfa:.1f}%'])
T['SIMCA_random_splits']=pd.DataFrame(rows,columns=['PCs per class model','Own-strain test spectra accepted (100 random 11/2 splits)','Other-strain (same species) spectra accepted'])
# cross-species challenge (Table 1 redo)
rows=[]
for sp in SPP:
    for ch in SPP:
        if ch==sp: continue
        m=species==ch; acc=np.any([models[s].accept(A[m]) for s in SB[sp]],axis=0)
        rows.append([sp,ch,39,f'{(~acc).sum()} ({100*(~acc).mean():.0f}%)',f'{acc.sum()} ({100*acc.mean():.0f}%)'])
T['Cross_species_challenge']=pd.DataFrame(rows,columns=['Strain-specific SIMCA models','Challenge species','n','Rejected by all three models','Accepted by >=1 model'])
# ---- figures
def ap(M,Ab):
    h,v=M._hv(Ab); return np.log1p(h/M.h0), np.log1p(v/M.v0)
for fi,sp in zip([6,7,8],SPP):
    fig,ax=plt.subplots(2,3,figsize=(12,7.6))
    pairs=list(itertools.combinations(SB[sp],2))
    for a,(s1,s2) in zip(ax[0],pairs):
        M1,M2=models[s1],models[s2]
        def pt(mask,**kw):
            a.scatter(M1.f(A[mask])/M1.crit,M2.f(A[mask])/M2.crit,**kw)
        third=[s for s in SB[sp] if s not in (s1,s2)][0]
        cl=dict(zip(SB[sp],COL[sp]))
        pt((strain==s1)&~TEST,color=cl[s1],s=18,label=f'{s1} training'); pt((strain==s2)&~TEST,color=cl[s2],s=18,label=f'{s2} training')
        pt((strain==s1)&TEST,color=cl[s1],s=60,marker='*',edgecolor='k',lw=.5,label=f'{s1} test'); pt((strain==s2)&TEST,color=cl[s2],s=60,marker='*',edgecolor='k',lw=.5,label=f'{s2} test')
        pt(strain==third,color=cl[third],s=22,marker='s',edgecolor='k',lw=.3,label=f'{third} (all spectra)')
        pt(species!=sp,color='lightgrey',s=10,marker='x',label='other species')
        a.set_xscale('log'); a.set_yscale('log'); lo,hi=0.05,max(a.get_xlim()[1],a.get_ylim()[1]); a.set_xlim(lo,hi); a.set_ylim(lo,hi); a.set_aspect('equal')
        a.axvline(1,color='r',lw=.8); a.axhline(1,color='r',lw=.8)
        a.set_xlabel(f'Distance to {s1} model (f/f$_{{crit}}$)'); a.set_ylabel(f'Distance to {s2} model (f/f$_{{crit}}$)'); a.legend(fontsize=6,frameon=False,loc='upper left')
    for a,s in zip(ax[1],SB[sp]):
        M=models[s]; cl=dict(zip(SB[sp],COL[sp]))
        x,y=ap(M,A[species!=sp]); a.scatter(x,y,color='lightgrey',marker='x',s=10,label='other species')
        for o in SB[sp]:
            if o==s: continue
            x,y=ap(M,A[strain==o]); a.scatter(x,y,color=cl[o],marker='s',s=20,edgecolor='k',lw=.3,label=f'{o} (all)')
        x,y=ap(M,A[(strain==s)&~TEST]); a.scatter(x,y,color=cl[s],s=18,label=f'{s} training')
        x,y=ap(M,A[(strain==s)&TEST]); a.scatter(x,y,color=cl[s],s=70,marker='*',edgecolor='k',lw=.5,label=f'{s} test')
        hh=np.linspace(0,M.crit/M.Nh,200); vv=(M.crit-M.Nh*hh)/M.Nv
        a.plot(np.log1p(hh),np.log1p(np.clip(vv,0,None)),'r-',lw=1)
        a.set_xlabel('log(1 + h/h$_0$)  [score distance]'); a.set_ylabel('log(1 + v/v$_0$)  [orthogonal distance]'); a.set_title(f'{s} class model',fontsize=9)
        a.legend(fontsize=6,frameon=False,loc='upper right')
    fig.suptitle(f'{sp}: DD-SIMCA (3 PCs, α = 0.05). Top: Coomans plots (equal log axes). Bottom: acceptance plots.',style='italic')
    plt.tight_layout(); plt.savefig(os.path.join(FIG,f'Fig{fi}_SIMCA_{sp.replace(". ","_")}.png'),dpi=300); plt.close()
pd.to_pickle(T,os.path.join(INTER,'s4.pkl'))
for k,v in T.items(): print('##',k); print(v.to_string(index=False))
