from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

ROOT=Path(r"C:\Users\DELL\Documents\论文\internal_two_route_rebuild")
OUT=ROOT/"figures"; OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({"font.family":"Times New Roman","font.size":9,"axes.linewidth":.8,"xtick.direction":"in","ytick.direction":"in","xtick.major.width":.8,"ytick.major.width":.8,"svg.fonttype":"none"})


def metrics(t,p): return r2_score(t,p),mean_squared_error(t,p)**.5,mean_absolute_error(t,p)


def panel(ax,t,p,title,xlabel,letter,log=False):
    r2,rmse,mae=metrics(t,p); both=np.r_[t,p]; lo=max(0,float(np.nanmin(both))); hi=float(np.nanmax(both)); pad=(hi-lo)*.035 or 1
    ax.scatter(t,p,s=7 if len(t)<1000 else 2.2,c="#2878B5",alpha=.38 if len(t)<1000 else .13,edgecolors="none",rasterized=True)
    ax.plot([lo,hi],[lo,hi],color="#D95319",lw=1.2,ls="--",label="1:1 line")
    if log:
        ax.set_xlim(0,hi+pad); ax.set_ylim(0,hi+pad)
        ax.set_xscale("symlog",linthresh=1); ax.set_yscale("symlog",linthresh=1)
    else:
        ax.set_xlim(lo-pad,hi+pad); ax.set_ylim(lo-pad,hi+pad)
    ax.set_title(title,pad=7,fontweight="bold"); ax.set_xlabel("Actual "+xlabel); ax.set_ylabel("Predicted "+xlabel)
    ax.text(.04,.96,f"$R^2$ = {r2:.3f}\nRMSE = {rmse:,.2f}\nMAE = {mae:,.2f}\nn = {len(t):,}",transform=ax.transAxes,va="top",ha="left",bbox=dict(facecolor="white",edgecolor="none",alpha=.82,pad=2.5))
    ax.text(-.17,1.08,letter,transform=ax.transAxes,fontweight="bold",fontsize=11)
    ax.grid(True,color="#d9d9d9",lw=.45,alpha=.65); ax.set_axisbelow(True)
    for s in ax.spines.values(): s.set_color("#333333")


def route1():
    d=pd.read_csv(ROOT/"route1"/"internal_test_predictions.csv")
    fig,axes=plt.subplots(1,3,figsize=(10.2,3.35),constrained_layout=True)
    specs=[("EUI","Annual EUI","EUI (kWh m$^{-2}$ yr$^{-1}$)"),("Epv","Annual PV generation","Epv (kWh yr$^{-1}$)"),("CEI","Annual carbon-emission intensity","CEI (kgCO$_2$ m$^{-2}$ yr$^{-1}$)")]
    for k,(name,title,label) in enumerate(specs): panel(axes[k],d[f"true_{name}"].to_numpy(),d[f"pred_{name}"].to_numpy(),title,label,f"({chr(97+k)})")
    fig.suptitle("Route I: 3D-CNN–TabTransformer Annual Prediction (Internal Test Set)",fontsize=11,fontweight="bold")
    fig.savefig(OUT/"Figure_Route_I_Internal_Test_Scatter.png",dpi=600,bbox_inches="tight"); fig.savefig(OUT/"Figure_Route_I_Internal_Test_Scatter.svg",bbox_inches="tight"); plt.close(fig)
    d.to_csv(OUT/"Route_I_Internal_Test_Predictions.csv",index=False,encoding="utf-8-sig")


def route2():
    a=np.load(ROOT/"route2_lstm"/"internal_test_predictions_8760x3.npz"); t=a["true"].reshape(-1,3); p=a["prediction"].reshape(-1,3)
    rng=np.random.default_rng(42); n=len(t); random_idx=rng.choice(n,size=min(90000,n),replace=False)
    # Retain extreme observations so that the displayed domain represents the full test set.
    extreme=np.unique(np.concatenate([np.argpartition(t[:,j],-300)[-300:] for j in range(3)])); idx=np.unique(np.r_[random_idx,extreme])
    ts,ps=t[idx],p[idx]
    fig,axes=plt.subplots(1,3,figsize=(10.2,3.35),constrained_layout=True)
    specs=[("Hourly energy","Energy (kWh h$^{-1}$)"),("Hourly PV generation","PV generation (kWh h$^{-1}$)"),("Hourly carbon emission","Carbon emission (kgCO$_2$ h$^{-1}$)")]
    for j,(title,label) in enumerate(specs):
        panel(axes[j],ts[:,j],ps[:,j],title,label,f"({chr(97+j)})",log=True)
        r2,rmse,mae=metrics(t[:,j],p[:,j]); axes[j].texts[-2].set_text(f"$R^2$ = {r2:.3f}\nRMSE = {rmse:,.2f}\nMAE = {mae:,.2f}\nn = {len(t):,}")
    fig.suptitle("Route II: Static Multimodal Fusion + LSTM Hourly Prediction (Internal Test Set)",fontsize=11,fontweight="bold")
    fig.text(.5,-.015,f"For visual clarity, {len(idx):,} reproducibly sampled points are displayed; all reported metrics use the complete 4,415,040-point test set.",ha="center",fontsize=8)
    fig.savefig(OUT/"Figure_Route_II_Internal_Test_Scatter.png",dpi=600,bbox_inches="tight"); fig.savefig(OUT/"Figure_Route_II_Internal_Test_Scatter.svg",bbox_inches="tight"); plt.close(fig)
    out=pd.DataFrame({"actual_energy":ts[:,0],"pred_energy":ps[:,0],"actual_pv":ts[:,1],"pred_pv":ps[:,1],"actual_carbon":ts[:,2],"pred_carbon":ps[:,2]})
    out.to_csv(OUT/"Route_II_Internal_Test_Display_Sample.csv",index=False,encoding="utf-8-sig")


if __name__=="__main__": route1(); route2(); print(OUT)
