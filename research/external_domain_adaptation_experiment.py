from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor, RandomForestRegressor, StackingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT=Path(__file__).resolve().parent
BASE=ROOT/"leakage_free_retraining"
OUT=BASE/"external_domain_adaptation"
OUT.mkdir(parents=True,exist_ok=True)

NUM=[
 "Model predicted EUI","Construction_year","A.Building area","B.Building footprint","C.Building height","D.Layer","E.Height",
 "F.Building length","G.Building width","H.Orientation","I.Enclosure method","J.Shape coefficient",
 "K.Roof thermal coefficient","L.Wall thermal coefficient","M.Ground thermal coefficient","N.Window U-value",
 "Spacing","Campus_area","Density","FAR","Temp_mean","Humidity_mean","Wind_mean","HDD18","CDD26",
 "DNI_kWh_m2","DHI_kWh_m2","GHI_kWh_m2","OBJ vertices","OBJ faces",
]
CAT=["City_code","School_type"]

def add_physics(x):
    x=x.copy();area=x["A.Building area"].clip(lower=1);foot=x["B.Building footprint"].clip(lower=1)
    h=x["C.Building height"].clip(lower=.1);L=x["F.Building length"].clip(lower=.1);W=x["G.Building width"].clip(lower=.1)
    wall=2*(L+W)*h; ang=np.deg2rad(x["H.Orientation"])
    x["area_foot_ratio"]=area/foot;x["wall_floor_ratio"]=wall/area;x["surface_floor_ratio"]=(wall+2*foot)/area
    x["orientation_sin"]=np.sin(ang);x["orientation_cos"]=np.cos(ang)
    x["heat_loss_index"]=(.7*wall*x["L.Wall thermal coefficient"]+.3*wall*x["N.Window U-value"]+foot*(x["K.Roof thermal coefficient"]+x["M.Ground thermal coefficient"]))/area
    x["heating_exposure"]=x["heat_loss_index"]*x["HDD18"]
    x["cooling_exposure"]=x["surface_floor_ratio"]*x["CDD26"]
    x["age_2026"]=2026-x["Construction_year"]
    return x

def metric(y,p):return {"R2":r2_score(y,p),"RMSE":mean_squared_error(y,p)**.5,"MAE":mean_absolute_error(y,p)}

def main():
    d=pd.read_csv(BASE/"external_61_true_model_predictions.csv")
    x=add_physics(d[NUM+CAT]); y=d["Observed EUI"].to_numpy(float); base=d["Model predicted EUI"].to_numpy(float)
    num=[c for c in x.columns if c not in CAT]
    prep=ColumnTransformer([("num",Pipeline([("imp",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num),
                            ("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),CAT)])
    candidates={
      "Ridge":Ridge(alpha=20),
      "GradientBoosting":GradientBoostingRegressor(n_estimators=180,max_depth=2,learning_rate=.025,loss="huber",random_state=42),
      "RandomForest":RandomForestRegressor(n_estimators=700,max_depth=4,min_samples_leaf=3,max_features=.7,random_state=42,n_jobs=12),
      "ExtraTrees":ExtraTreesRegressor(n_estimators=700,max_depth=5,min_samples_leaf=2,max_features=.8,random_state=42,n_jobs=12),
    }
    cv=KFold(n_splits=5,shuffle=True,random_state=42)
    rows=[{"model":"Uncalibrated Route I",**metric(y,base)}]; predictions={"Uncalibrated Route I":base}
    for mode in ("direct","residual"):
      target=y if mode=="direct" else y-base
      for name,reg in candidates.items():
        pipe=Pipeline([("prep",prep),("model",reg)])
        raw=cross_val_predict(pipe,x,target,cv=cv,n_jobs=1)
        pred=raw if mode=="direct" else base+raw
        label=f"{mode}-{name}";predictions[label]=pred;rows.append({"model":label,**metric(y,pred)})
    result=pd.DataFrame(rows).sort_values("RMSE");result.to_csv(OUT/"cross_fitted_model_comparison.csv",index=False)
    best=result.iloc[0]["model"];p=predictions[best]
    out=d[["Validation ID","Number","Project","City_code","Observed EUI","Observed CEI","Model predicted EUI"]].copy()
    out["Domain-adapted predicted EUI"]=p;out["Domain-adapted predicted CEI"]=p*.5942
    out["EUI residual"]=p-y;out.to_csv(OUT/"cross_fitted_predictions_61.csv",index=False,encoding="utf-8-sig")
    (OUT/"selection.json").write_text(json.dumps({"method":best,"protocol":"5-fold cross-fitted; each prediction excludes its own measured target","metrics":metric(y,p)},indent=2),encoding="utf-8")
    print(result.to_string(index=False),flush=True)

if __name__=="__main__":main()
