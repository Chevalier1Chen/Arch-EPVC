from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT=Path(r"C:\Users\DELL\Documents\论文\internal_two_route_rebuild"); OUT=ROOT/"ablation"; OUT.mkdir(exist_ok=True)
r1=pd.read_csv(OUT/"route1"/"route1_ablation_metrics.csv")
r2=pd.read_csv(OUT/"route2"/"route2_ablation_metrics.csv"); r2=r2[r2.model.ne("LSTM")]
lstm=pd.read_csv(ROOT/"route2_lstm"/"internal_test_metrics.csv"); mp={"Hourly energy":"Energy","Hourly PV generation":"PV","Hourly carbon emission":"Carbon"}; lstm["target"]=lstm.target.map(mp);lstm["model"]="LSTM (proposed)";lstm["route"]="Route II"
r2=pd.concat([r2,lstm[["route","model","target","R2","RMSE","MAE","n"]]],ignore_index=True)
r1.loc[r1.model.eq("3D-CNN + TabTransformer"),"model"]="3D-CNN + TabTransformer (proposed)"
truth1=pd.read_csv(ROOT/"route1"/"internal_test_predictions.csv"); mean1={k:truth1[f"true_{k}"].mean() for k in ["EUI","Epv","CEI"]}
a=np.load(ROOT/"route2_lstm"/"internal_test_predictions_8760x3.npz"); mean2=dict(zip(["Energy","PV","Carbon"],a["true"].mean((0,1))))
r1["NMAE_pct"]=[100*x/mean1[t] for x,t in zip(r1.MAE,r1.target)]; r2["NMAE_pct"]=[100*x/mean2[t] for x,t in zip(r2.MAE,r2.target)]
allm=pd.concat([r1,r2],ignore_index=True); allm.to_csv(OUT/"dual_route_ablation_metrics.csv",index=False,encoding="utf-8-sig")
summary=allm.groupby(["route","model"],as_index=False).agg(mean_R2=("R2","mean"),mean_NMAE_pct=("NMAE_pct","mean"));summary.to_csv(OUT/"dual_route_ablation_summary.csv",index=False,encoding="utf-8-sig")

plt.rcParams.update({"font.family":"Times New Roman","font.size":9,"axes.linewidth":.8,"xtick.direction":"in","ytick.direction":"in","svg.fonttype":"none"})
colors=["#9ECAE1","#6BAED6","#3182BD","#08519C"]
fig,axes=plt.subplots(2,2,figsize=(10.8,7.2),constrained_layout=False)

def grouped(ax,data,models,targets,metric,title,letter,ylabel,ylim=None):
    x=np.arange(len(targets));w=.18; allvals=[]
    for i,m in enumerate(models):
        vals=[float(data[(data.model==m)&(data.target==t)][metric].iloc[0]) for t in targets]
        allvals.extend(vals)
        bars=ax.bar(x+(i-1.5)*w,vals,w,label=m.replace(" (proposed)","*"),color=colors[i],edgecolor="#1f1f1f" if "proposed" in m else "none",linewidth=1.1 if "proposed" in m else 0)
        for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,b.get_height()+(.012 if metric=="R2" else .35),f"{v:.3f}" if metric=="R2" else f"{v:.1f}",ha="center",va="bottom",fontsize=7.2,rotation=90)
    ax.set_xticks(x);ax.set_xticklabels(targets);ax.set_ylabel(ylabel);ax.set_title(title,fontweight="bold",pad=7);ax.text(-.12,1.05,letter,transform=ax.transAxes,fontweight="bold",fontsize=11)
    ax.grid(axis="y",color="#d9d9d9",lw=.5,alpha=.75);ax.set_axisbelow(True)
    if ylim:ax.set_ylim(*ylim)
    else:ax.set_ylim(0,max(allvals)*1.16)

m1=["TabTransformer only","3D-CNN only","3D-CNN + MLP","3D-CNN + TabTransformer (proposed)"]
m2=["Hourly MLP","GRU","Transformer","LSTM (proposed)"]
grouped(axes[0,0],r1,m1,["EUI","Epv","CEI"],"R2","Route I · Annual prediction","(a)","$R^2$",(0,1.04))
grouped(axes[0,1],r2,m2,["Energy","PV","Carbon"],"R2","Route II · Hourly prediction","(b)","$R^2$",(0,1.04))
grouped(axes[1,0],r1,m1,["EUI","Epv","CEI"],"NMAE_pct","Route I · Normalized absolute error","(c)","NMAE (%)")
grouped(axes[1,1],r2,m2,["Energy","PV","Carbon"],"NMAE_pct","Route II · Normalized absolute error","(d)","NMAE (%)")
handles,labels=axes[0,0].get_legend_handles_labels(); fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.28,.92),ncol=2,frameon=False)
handles,labels=axes[0,1].get_legend_handles_labels(); fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.76,.92),ncol=2,frameon=False)
fig.suptitle("Dual-route Ablation and Architecture Comparison",fontsize=12,fontweight="bold",y=.985)
fig.text(.5,.018,"* Proposed architecture. NMAE = MAE / mean observed value × 100%; lower is better.",ha="center",fontsize=8)
fig.subplots_adjust(left=.075,right=.985,bottom=.09,top=.78,wspace=.18,hspace=.32)
fig.savefig(OUT/"Figure_Dual_Route_Ablation_Bar.png",dpi=600,bbox_inches="tight");fig.savefig(OUT/"Figure_Dual_Route_Ablation_Bar.svg",bbox_inches="tight");plt.close(fig)
print(summary.to_string(index=False));print(OUT)
