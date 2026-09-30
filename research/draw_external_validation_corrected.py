from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT=Path(__file__).resolve().parent
BASE=ROOT/"leakage_free_retraining"
OUT=BASE/"external_validation_corrected"
OUT.mkdir(parents=True,exist_ok=True)

plt.rcParams.update({"font.family":"Times New Roman","font.size":10,"axes.labelsize":11,"xtick.labelsize":9,"ytick.labelsize":9})

def panel(ax,y,p,label,unit):
    y=np.asarray(y,float);p=np.asarray(p,float);lo=min(y.min(),p.min());hi=max(y.max(),p.max());pad=.05*(hi-lo)
    ax.scatter(y,p,s=18,c="#74add1",alpha=.72,edgecolors="none")
    ax.plot([lo-pad,hi+pad],[lo-pad,hi+pad],"--",c="#e6550d",lw=1.4)
    ax.set_xlim(lo-pad,hi+pad);ax.set_ylim(lo-pad,hi+pad)
    ax.set_xlabel(f"Measured {label} ({unit})");ax.set_ylabel(f"Simulated {label} ({unit})")
    r2=r2_score(y,p);rmse=mean_squared_error(y,p)**.5;mae=mean_absolute_error(y,p)
    ax.text(.035,.965,f"$R^2$ = {r2:.3f}\nRMSE = {rmse:,.2f}\nMAE = {mae:,.2f}\nn = {len(y)}",transform=ax.transAxes,va="top")
    ax.grid(True,color="#dddddd",lw=.55,alpha=.7)
    return {"target":label,"n":len(y),"R2":r2,"RMSE":rmse,"MAE":mae}

def save(fig,stem):
    fig.savefig(OUT/f"{stem}.png",dpi=600,bbox_inches="tight")
    fig.savefig(OUT/f"{stem}.svg",bbox_inches="tight")
    fig.savefig(OUT/f"{stem}.tif",dpi=600,bbox_inches="tight",pil_kwargs={"compression":"tiff_lzw"})
    plt.close(fig)

def main():
    raw=pd.read_csv(BASE/"external_61_true_model_predictions.csv")
    measured_eui=raw["Observed EUI"].to_numpy(float);simulated_eui=raw["Predicted EUI"].to_numpy(float)
    measured_cei=measured_eui*.5942;simulated_cei=simulated_eui*.5942
    fig,axes=plt.subplots(1,2,figsize=(8.3,3.65))
    rows=[panel(axes[0],measured_eui,simulated_eui,"EUI","kWh/m²·year"),panel(axes[1],measured_cei,simulated_cei,"CEI","kgCO₂/m²·year")]
    fig.suptitle("External Validation of the Building-Energy Simulation Dataset (61 School Buildings)",fontsize=12.5,fontweight="bold")
    fig.tight_layout(rect=(0,0,1,.93));save(fig,"External_simulation_validation_EUI_CEI")
    pd.DataFrame(rows).to_csv(OUT/"External_simulation_validation_metrics.csv",index=False)
    result=raw[["Validation ID","Number","Project","City_code"]].copy()
    result["Measured EUI"]=measured_eui;result["Simulated EUI"]=simulated_eui
    result["Measured CEI"]=measured_cei;result["Simulated CEI"]=simulated_cei
    result.to_csv(OUT/"External_simulation_validation_predictions.csv",index=False,encoding="utf-8-sig")

    adapted=pd.read_csv(BASE/"external_domain_adaptation"/"cross_fitted_predictions_61.csv")
    fig,axes=plt.subplots(1,2,figsize=(8.3,3.65))
    adapted_rows=[panel(axes[0],adapted["Observed EUI"],adapted["Domain-adapted predicted EUI"],"EUI","kWh/m²·year"),
                  panel(axes[1],adapted["Observed CEI"],adapted["Domain-adapted predicted CEI"],"CEI","kgCO₂/m²·year")]
    for ax in axes:
        ax.set_ylabel(ax.get_ylabel().replace("Simulated","Domain-adapted predicted"))
    fig.suptitle("Cross-Fitted Domain Adaptation of the Surrogate Model (61 School Buildings)",fontsize=12.5,fontweight="bold")
    fig.tight_layout(rect=(0,0,1,.93));save(fig,"External_surrogate_domain_adaptation_EUI_CEI")
    pd.DataFrame(adapted_rows).to_csv(OUT/"External_surrogate_domain_adaptation_metrics.csv",index=False)
    print(pd.DataFrame(rows).to_string(index=False));print(pd.DataFrame(adapted_rows).to_string(index=False))

if __name__=="__main__":main()
