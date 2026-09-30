from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from catboost import CatBoostRegressor

base=Path(r"C:\Users\DELL\Desktop\SCI8")
d=pd.read_excel(base/"数据集完成"/"numerical data.xlsx",sheet_name="selected_variables")
g=pd.read_csv(base/"shandong_teaching_campus_overview_package"/"data"/"gh_building_energy_inputs.csv",encoding="utf-8-sig")
g=g.rename(columns={"obj_object_name_prefix":"Number"})
g=g.drop_duplicates("Number")[["Number","sample_uid","city","school_name","construction_year_numeric","roof_area_m2"]]
d=d.merge(g,on="Number",how="left")
years=pd.read_excel(base/"Simulation"/"gh_building_simulation_min_inputs.xlsx",sheet_name=0).iloc[:,:3]; years.columns=["Number","story","construction_year_user"]
d=d.merge(years[["Number","construction_year_user"]],on="Number",how="left")
y=d[["EUI","Epv","CEI-PV system"]].to_numpy(); idx=np.arange(len(d)); tr,te=train_test_split(idx,test_size=.15,random_state=42)
sets={
 "current":[c for c in d.columns if c not in ["Number","EUI","Epv","CEI","CEI-no PV system","CEI-PV system","sample_uid","city","school_name","construction_year_numeric","construction_year_user","roof_area_m2"]],
 "plus_city":None,
 "plus_campus":None,
}
sets["plus_city"]=sets["current"]+["city"]
sets["plus_campus"]=sets["current"]+["city","construction_year_user","roof_area_m2","sample_uid"]
for name,cols in sets.items():
    part=d[cols].copy()
    cats=[c for c in cols if not pd.api.types.is_numeric_dtype(part[c])]
    for c in cats: part[c]=part[c].astype(str).fillna("missing")
    x=pd.get_dummies(part,columns=cats,dtype=float).fillna(-1)
    m=ExtraTreesRegressor(n_estimators=400,min_samples_leaf=1,max_features=.9,n_jobs=-1,random_state=42).fit(x.iloc[tr],y[tr])
    p=m.predict(x.iloc[te]); print(name,[round(r2_score(y[te,j],p[:,j]),4) for j in range(3)])
    cbpart=d[cols].copy(); cats=[c for c in cols if not pd.api.types.is_numeric_dtype(cbpart[c])]
    for c in cats: cbpart[c]=cbpart[c].astype(str).fillna("missing")
    cbpart=cbpart.fillna(-1)
    preds=[]
    for j in range(3):
        cb=CatBoostRegressor(iterations=800,depth=8,learning_rate=.05,loss_function='RMSE',verbose=False,random_seed=42)
        cb.fit(cbpart.iloc[tr],y[tr,j],cat_features=cats); preds.append(cb.predict(cbpart.iloc[te]))
    cp=np.column_stack(preds); print(name,'catboost',[round(r2_score(y[te,j],cp[:,j]),4) for j in range(3)])
