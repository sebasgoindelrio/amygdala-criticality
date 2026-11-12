import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
P=[("Qwen2.5-0.5B (5 seeds)",{"Random":[8,8,10,10,8],"Forgotten":[11,13,11,10,10],"Judge":[15,22,19,24,18]}),
   ("Llama-3.2-1B (3 seeds)",{"Random":[12,10,13],"Judge":[24,23,19]})]
ink="#3b3b38"; mut="#8a8984"
fig,axs=plt.subplots(1,2,figsize=(7.2,3.2),dpi=200,sharey=True,gridspec_kw=dict(width_ratios=[3,2]))
for ax,(t,d) in zip(axs,P):
    xs=list(range(len(d)))
    for x,(k,v) in zip(xs,d.items()):
        seen=Counter(); jx=[]
        for val in v: jx.append(x+0.07*(seen[val]-(v.count(val)-1)/2)); seen[val]+=1
        ax.scatter(jx,v,s=34,color="#2a78d6" if k=="Judge" else mut,edgecolor="white",linewidth=1.5,zorder=3)
        m=sum(v)/len(v); ax.plot([x-0.22,x+0.22],[m,m],color=ink,lw=2,zorder=2); ax.text(x+0.27,m,f"{m:.1f}",va="center",fontsize=8,color=ink)
    ax.axhline(30,ls=(0,(3,3)),lw=1,color=mut)
    ax.set_xticks(xs,list(d)); ax.set_xlim(-0.5,len(d)-0.4); ax.set_ylim(0,38); ax.set_title(t,fontsize=9,color=ink)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    for sp in ["left","bottom"]: ax.spines[sp].set_color("#bdbcb6")
    ax.tick_params(colors=ink,labelsize=8); ax.yaxis.grid(True,color="#ecebe7",lw=0.8); ax.set_axisbelow(True)
axs[1].text(len(P[1][1])-0.45,30.6,"streaming ceiling (30)",ha="right",fontsize=7,color=mut)
axs[0].set_ylabel("Critical facts retained (of 38)",fontsize=9,color=ink)
plt.tight_layout(); plt.savefig("../paper/figures/e21b_retention.png"); plt.savefig("../paper/figures/e21b_retention.pdf")
