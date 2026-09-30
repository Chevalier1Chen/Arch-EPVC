from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader

import train_route1 as core

OUT=core.ROOT/"ablation"/"route1"; OUT.mkdir(parents=True,exist_ok=True)


def load_data():
    df=pd.read_excel(core.TABLE,sheet_name="selected_variables")
    gh=pd.read_csv(r"C:\Users\DELL\Desktop\SCI8\shandong_teaching_campus_overview_package\data\gh_building_energy_inputs.csv",encoding="utf-8-sig")
    gh=gh.rename(columns={"obj_object_name_prefix":"Number"}).drop_duplicates("Number")[["Number","city","roof_area_m2"]]
    years=pd.read_excel(r"C:\Users\DELL\Desktop\SCI8\Simulation\gh_building_simulation_min_inputs.xlsx",sheet_name=0).iloc[:,:3]; years.columns=["Number","story","construction_year_user"]
    df=df.merge(gh,on="Number",how="left",validate="one_to_one").merge(years[["Number","construction_year_user"]],on="Number",how="left",validate="one_to_one")
    df[core.CAT]=df["city"].astype(str)+" | "+df["I.Enclosure method"].astype(str)
    split=json.loads((core.OUT/"split.json").read_text(encoding="utf-8")); import joblib
    x=joblib.load(core.OUT/"continuous_scaler.joblib").transform(df[core.CONT]).astype(np.float32)
    cmap={v:i for i,v in enumerate(split["categories"])}; cat=df[core.CAT].map(cmap).fillna(0).to_numpy(np.int64)
    raw=df[core.TARGETS].to_numpy(np.float64); z=raw.copy(); z[:,1]=np.log1p(z[:,1]); s=np.load(core.OUT/"target_scaler.npz"); y=((z-s["mean"])/s["std"]).astype(np.float32)
    return df,np.load(core.VOXELS,mmap_mode="r"),x,cat,raw,y,split,s


class TabularOnly(nn.Module):
    def __init__(self,nc,ncat): super().__init__(); self.enc=core.TabTransformer(nc,ncat); self.head=nn.Sequential(nn.Linear(128,96),nn.SiLU(),nn.Dropout(.08),nn.Linear(96,3))
    def forward(self,v,x,c): return self.head(self.enc(x,c))


class GeometryOnly(nn.Module):
    def __init__(self,nc,ncat): super().__init__(); self.enc=core.Geometry3DCNN(); self.head=nn.Sequential(nn.Linear(128,96),nn.SiLU(),nn.Dropout(.08),nn.Linear(96,3))
    def forward(self,v,x,c): return self.head(self.enc(v))


class TabularMLP(nn.Module):
    def __init__(self,nc,ncat):
        super().__init__(); self.cat=nn.Embedding(ncat,16); self.net=nn.Sequential(nn.Linear(nc+16,128),nn.LayerNorm(128),nn.SiLU(),nn.Dropout(.10),nn.Linear(128,128),nn.SiLU())
    def forward(self,x,c): return self.net(torch.cat([x,self.cat(c)],1))


class FusionMLP(nn.Module):
    def __init__(self,nc,ncat):
        super().__init__(); self.g=core.Geometry3DCNN(); self.t=TabularMLP(nc,ncat); self.gate=nn.Sequential(nn.Linear(256,128),nn.Sigmoid()); self.head=nn.Sequential(nn.Linear(512,256),nn.LayerNorm(256),nn.SiLU(),nn.Dropout(.12),nn.Linear(256,96),nn.SiLU(),nn.Linear(96,3))
    def forward(self,v,x,c):
        g,t=self.g(v),self.t(x,c); a=self.gate(torch.cat([g,t],1)); return self.head(torch.cat([a*g,(1-a)*t,g*t,torch.abs(g-t)],1))


@torch.no_grad()
def pred(model,loader):
    model.eval(); ps=[]; ids=[]
    for v,x,c,y,i in loader: ps.append(model(v,x,c).numpy()); ids.append(i.numpy())
    return np.concatenate(ids).astype(int),np.concatenate(ps)


def fit(name,model,loaders,y,raw,scaler):
    path=OUT/f"{name}.pt"; opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=2e-4); lossfn=nn.SmoothL1Loss(beta=.5); best=1e9; stale=0
    for epoch in range(1,51):
        model.train(); total=0; n=0
        for v,x,c,target,_ in loaders["train"]:
            opt.zero_grad(set_to_none=True); q=model(v,x,c); loss=lossfn(q,target); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),3); opt.step(); total+=loss.item()*len(v); n+=len(v)
        vi,vp=pred(model,loaders["val"]); vm=float(np.mean((vp-y[vi])**2)); print(f"{name} epoch={epoch:02d} train={total/n:.5f} val={vm:.5f}",flush=True)
        if vm<best-1e-5: best=vm; stale=0; torch.save(model.state_dict(),path)
        else:
            stale+=1
            if stale>=8: break
    model.load_state_dict(torch.load(path,weights_only=True,map_location="cpu")); ti,tp=pred(model,loaders["test"]); z=tp*scaler["std"]+scaler["mean"]; z[:,1]=np.expm1(z[:,1]); truth=raw[ti]
    rows=[]
    for j,target in enumerate(core.TARGET_LABELS): rows.append({"route":"Route I","model":name,"target":target,"R2":r2_score(truth[:,j],z[:,j]),"RMSE":mean_squared_error(truth[:,j],z[:,j])**.5,"MAE":mean_absolute_error(truth[:,j],z[:,j]),"n":len(ti)})
    return rows


def main():
    core.seed_all(42); torch.set_num_threads(min(16,torch.get_num_threads())); df,vox,x,cat,raw,y,split,scaler=load_data()
    loaders={k:DataLoader(core.Samples(vox,x,cat,y,np.asarray(split[k])),batch_size=56,shuffle=k=="train",num_workers=0) for k in ["train","val","test"]}
    models={"TabTransformer only":TabularOnly(len(core.CONT),len(split["categories"])),"3D-CNN only":GeometryOnly(len(core.CONT),len(split["categories"])),"3D-CNN + MLP":FusionMLP(len(core.CONT),len(split["categories"]))}
    rows=[]
    for name,model in models.items(): rows+=fit(name,model,loaders,y,raw,scaler)
    proposed=pd.read_csv(core.OUT/"internal_test_metrics.csv");
    for _,r in proposed.iterrows(): rows.append({"route":"Route I","model":"3D-CNN + TabTransformer","target":r.target,"R2":r.R2,"RMSE":r.RMSE,"MAE":r.MAE,"n":r.n})
    pd.DataFrame(rows).to_csv(OUT/"route1_ablation_metrics.csv",index=False,encoding="utf-8-sig"); print(pd.DataFrame(rows).to_string(index=False))


if __name__=="__main__": main()
