from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT=Path(__file__).resolve().parent
BASE=ROOT/"leakage_free_retraining"
R1=BASE/"route1_building_split"
OUT=BASE/"improved_internal_test"
OUT.mkdir(parents=True,exist_ok=True)

def metric(y,p):return r2_score(y,p),mean_squared_error(y,p)**.5,mean_absolute_error(y,p)

def main():
    neural=pd.read_csv(R1/"internal_test_predictions.csv")
    epv=pd.read_csv(R1/"epv_yield_head"/"test_predictions.csv")
    data=neural.drop(columns=["pred_Epv"]).merge(epv[["row_index","pred_Epv"]],on="row_index",how="left",validate="one_to_one")
    data.to_csv(OUT/"Route_I_improved_test_predictions.csv",index=False,encoding="utf-8-sig")
    rows=[]
    for name in ("EUI","Epv","CEI"):
        r2,rmse,mae=metric(data[f"true_{name}"],data[f"pred_{name}"])
        rows.append({"route":"Route I improved","target":name,"n":len(data),"R2":r2,"RMSE":rmse,"MAE":mae})
    metrics=pd.DataFrame(rows);metrics.to_csv(OUT/"Route_I_improved_test_metrics.csv",index=False)

    plt.rcParams.update({"font.family":"Times New Roman","font.size":10,"axes.labelsize":11})
    fig,axes=plt.subplots(1,3,figsize=(12.2,3.65))
    specs=[("EUI","kWh/m²·year"),("Epv","kWh/year"),("CEI","kgCO₂/m²·year")]
    for ax,(name,unit),row in zip(axes,specs,rows):
        y=data[f"true_{name}"].to_numpy();p=data[f"pred_{name}"].to_numpy();lo=min(y.min(),p.min());hi=max(y.max(),p.max());pad=.04*(hi-lo)
        ax.scatter(y,p,s=10,c="#74add1",alpha=.62,edgecolors="none");ax.plot([lo-pad,hi+pad],[lo-pad,hi+pad],"--",lw=1.4,c="#e6550d")
        ax.set_xlim(lo-pad,hi+pad);ax.set_ylim(lo-pad,hi+pad);ax.set_xlabel(f"Actual {name} ({unit})");ax.set_ylabel(f"Predicted {name} ({unit})")
        ax.text(.035,.965,f"$R^2$ = {row['R2']:.3f}\nRMSE = {row['RMSE']:,.2f}\nMAE = {row['MAE']:,.2f}\nn = {len(data):,}",transform=ax.transAxes,va="top")
        ax.grid(True,color="#dddddd",lw=.55,alpha=.7)
    fig.suptitle("Route I: Improved Annual Prediction (Internal Building-Level Test Set)",fontsize=13,fontweight="bold")
    fig.tight_layout(rect=(0,0,1,.94))
    for ext,kwargs in [("png",{"dpi":600}),("svg",{}),("tif",{"dpi":600,"pil_kwargs":{"compression":"tiff_lzw"}})]:
        fig.savefig(OUT/f"Route_I_improved_internal_test.{ext}",bbox_inches="tight",**kwargs)
    plt.close(fig)

    comparison=pd.DataFrame([
        {"evaluation":"Campus-grouped robustness test","EUI_R2":.586679,"Epv_R2":.910730,"CEI_R2":.767191},
        {"evaluation":"Building-level internal test","EUI_R2":rows[0]["R2"],"Epv_R2":rows[1]["R2"],"CEI_R2":rows[2]["R2"]},
    ])
    comparison.to_csv(OUT/"split_policy_comparison.csv",index=False)
    print(metrics.to_string(index=False),flush=True)

if __name__=="__main__":main()
