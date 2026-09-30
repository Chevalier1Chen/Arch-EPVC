from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, r"C:\pt251")
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(r"C:\pt251\torch\lib")

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import retrain_leakage_free_routes as core
from optimize_route1_eui import add_campus_features, engineered

OUT = core.OUT / "route1_building_split" / "epv_yield_head"
OUT.mkdir(parents=True, exist_ok=True)


def score(y, p):
    return r2_score(y, p), mean_squared_error(y, p) ** .5, mean_absolute_error(y, p)


def main():
    frame=core.load_training_frame();core.prepare_route2_cache(frame);frame=core.add_annual_climate(frame);frame=add_campus_features(frame)
    split=json.loads((core.OUT/"route1_building_split"/"split.json").read_text(encoding="utf-8"))
    train,val,test=(np.asarray(split[k],int) for k in ("train","val","test"))
    x=engineered(frame); x["roof_area_m2"]=frame["roof_area_m2"].astype(float)
    cat=["city","enclosure","school_type"]
    roof=frame["roof_area_m2"].to_numpy(float).clip(1)
    total=frame["Epv"].to_numpy(float); intensity=total/roof
    rows=[];best_model=None;best_val=float("inf")
    for depth in (4,5,6,7,8):
        model=CatBoostRegressor(iterations=1800,depth=depth,learning_rate=.03,loss_function="RMSE",l2_leaf_reg=6,
                                random_seed=42,random_strength=.2,verbose=False,allow_writing_files=False)
        model.fit(x.iloc[train],intensity[train],cat_features=cat,eval_set=(x.iloc[val],intensity[val]),early_stopping_rounds=120)
        pv=np.maximum(model.predict(x.iloc[val]),0)*roof[val]
        pt=np.maximum(model.predict(x.iloc[test]),0)*roof[test]
        vr2,vrmse,vmae=score(total[val],pv);tr2,trmse,tmae=score(total[test],pt)
        rows.append({"depth":depth,"best_iteration":model.get_best_iteration(),"val_R2":vr2,"val_RMSE":vrmse,"val_MAE":vmae,
                     "test_R2":tr2,"test_RMSE":trmse,"test_MAE":tmae})
        if vrmse<best_val:best_val=vrmse;best_model=model;best_pred=pt
    result=pd.DataFrame(rows).sort_values("val_RMSE");result.to_csv(OUT/"comparison.csv",index=False)
    best_model.save_model(str(OUT/"best_epv_yield_catboost.cbm"))
    pd.DataFrame({"row_index":test,"building_id":frame.iloc[test]["Number"].to_numpy(),"true_Epv":total[test],"pred_Epv":best_pred}).to_csv(OUT/"test_predictions.csv",index=False,encoding="utf-8-sig")
    print(result.to_string(index=False),flush=True)


if __name__=="__main__":main()
