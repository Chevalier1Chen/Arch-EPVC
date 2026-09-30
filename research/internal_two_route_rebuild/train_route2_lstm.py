from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset

import train_route1 as r1

ROOT=Path(r"C:\Users\DELL\Documents\论文\internal_two_route_rebuild")
CACHE=ROOT/"route2_cache"; OUT=ROOT/"route2_lstm"; HOURS=8760; SEED=42


def seed_all(s): random.seed(s); np.random.seed(s); torch.manual_seed(s)


@torch.no_grad()
def build_static():
    df=pd.read_excel(r1.TABLE,sheet_name="selected_variables")
    gh=pd.read_csv(r"C:\Users\DELL\Desktop\SCI8\shandong_teaching_campus_overview_package\data\gh_building_energy_inputs.csv",encoding="utf-8-sig")
    gh=gh.rename(columns={"obj_object_name_prefix":"Number"}).drop_duplicates("Number")[["Number","city","roof_area_m2"]]
    years=pd.read_excel(r"C:\Users\DELL\Desktop\SCI8\Simulation\gh_building_simulation_min_inputs.xlsx",sheet_name=0).iloc[:,:3]; years.columns=["Number","story","construction_year_user"]
    df=df.merge(gh,on="Number",how="left",validate="one_to_one").merge(years[["Number","construction_year_user"]],on="Number",how="left",validate="one_to_one")
    df[r1.CAT]=df["city"].astype(str)+" | "+df["I.Enclosure method"].astype(str)
    vox=np.load(r1.VOXELS,mmap_mode="r")
    split=json.loads((r1.OUT/"split.json").read_text(encoding="utf-8"))
    import joblib
    scaler=joblib.load(r1.OUT/"continuous_scaler.joblib")
    x=scaler.transform(df[r1.CONT]).astype(np.float32)
    cmap={v:i for i,v in enumerate(split["categories"])}
    cat=df[r1.CAT].astype(str).map(cmap).fillna(0).to_numpy(np.int64)
    model=r1.Route1(len(r1.CONT),len(cmap)); model.load_state_dict(torch.load(r1.OUT/"best_route1_3dcnn_tabtransformer.pt",weights_only=True,map_location="cpu")); model.eval()
    chunks=[]
    for start in range(0,len(df),48):
        end=min(start+48,len(df)); p,z=model(torch.from_numpy(np.asarray(vox[start:end],dtype=np.float32)),torch.from_numpy(x[start:end]),torch.from_numpy(cat[start:end]),True)
        chunks.append(torch.cat([z,p],1).numpy())
    static=np.concatenate(chunks).astype(np.float32); np.save(OUT/"static_condition_259.npy",static); return static,split


def stream_stats(a,indices,log=False):
    total=np.zeros(a.shape[-1],np.float64); total2=np.zeros_like(total); count=0
    for block in np.array_split(np.asarray(indices),max(1,len(indices)//64)):
        z=np.asarray(a[block],dtype=np.float32)
        if log: z=np.log1p(np.maximum(z,0))
        total+=z.sum((0,1)); total2+=(z.astype(np.float64)**2).sum((0,1)); count+=z.shape[0]*z.shape[1]
    mean=total/count; std=np.sqrt(np.maximum(total2/count-mean**2,1e-8)); return mean.astype(np.float32),std.astype(np.float32)


class Windows(Dataset):
    def __init__(self,weather,targets,static,indices,xm,xs,ym,ys,n=7000,length=72,seed=0):
        self.w,self.y,self.s=weather,targets,static; self.idx=np.asarray(indices); self.xm,self.xs,self.ym,self.ys=xm,xs,ym,ys; self.n,self.length,self.seed=n,length,seed
    def __len__(self): return self.n
    def __getitem__(self,k):
        rng=np.random.default_rng(self.seed+k*104729); i=int(self.idx[rng.integers(len(self.idx))]); start=int(rng.integers(HOURS-self.length+1)); sl=slice(start,start+self.length)
        x=(np.asarray(self.w[i,sl])-self.xm)/self.xs; y=(np.log1p(np.maximum(np.asarray(self.y[i,sl]),0))-self.ym)/self.ys
        return torch.from_numpy(self.s[i]),torch.from_numpy(x.astype(np.float32)),torch.from_numpy(y.astype(np.float32))


class HourlyLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.static=nn.Sequential(nn.Linear(259,64),nn.LayerNorm(64),nn.SiLU())
        self.weather=nn.Sequential(nn.Linear(10,32),nn.LayerNorm(32),nn.SiLU())
        self.fuse=nn.Sequential(nn.Linear(96,64),nn.SiLU())
        self.lstm=nn.LSTM(64,64,num_layers=2,batch_first=True,dropout=.10)
        self.head=nn.Sequential(nn.Linear(64,48),nn.SiLU(),nn.Linear(48,3))
    def forward(self,s,x,state=None):
        a=self.static(s).unsqueeze(1).expand(-1,x.shape[1],-1); z=self.fuse(torch.cat([a,self.weather(x)],-1)); h,state=self.lstm(z,state); return self.head(h),state


@torch.no_grad()
def validate_windows(model,loader):
    model.eval(); total=0.; count=0
    for s,x,y in loader:
        p,_=model(s,x); total+=nn.functional.mse_loss(p,y,reduction="sum").item(); count+=y.numel()
    return total/count


@torch.no_grad()
def infer_test(model,weather,targets,static,test,xm,xs,ym,ys):
    model.eval(); truths=[]; preds=[]
    for bstart in range(0,len(test),12):
        batch=np.asarray(test[bstart:bstart+12]); parts=[]; state=None
        for start in range(0,HOURS,336):
            end=min(start+336,HOURS); x=((np.asarray(weather[batch,start:end])-xm)/xs).astype(np.float32)
            p,state=model(torch.from_numpy(static[batch]),torch.from_numpy(x),state); state=tuple(v.detach() for v in state); parts.append(p.numpy())
        z=np.concatenate(parts,axis=1); preds.append(np.maximum(np.expm1(z*ys+ym),0)); truths.append(np.asarray(targets[batch]))
        print(f"test inference {min(bstart+len(batch),len(test))}/{len(test)}",flush=True)
    truth=np.concatenate(truths); pred=np.concatenate(preds)
    return truth,pred,None,None


def main():
    seed_all(SEED); OUT.mkdir(parents=True,exist_ok=True); torch.set_num_threads(min(16,torch.get_num_threads()))
    weather=np.load(CACHE/"weather.npy",mmap_mode="r"); targets=np.load(CACHE/"targets.npy",mmap_mode="r")
    static,split=build_static(); valid=np.load(CACHE/"valid.npy")
    train=np.asarray(split["train"]); train=train[valid[train]]
    val=np.asarray(split["val"]); val=val[valid[val]]
    test=np.asarray(split["test"]); test=test[valid[test]]
    xm,xs=stream_stats(weather,train); ym,ys=stream_stats(targets,train,log=True); np.savez(OUT/"scalers.npz",input_mean=xm,input_std=xs,label_mean=ym,label_std=ys)
    train_ds=Windows(weather,targets,static,train,xm,xs,ym,ys,n=7000,length=72,seed=SEED)
    val_ds=Windows(weather,targets,static,val,xm,xs,ym,ys,n=1200,length=168,seed=SEED+999)
    train_loader=DataLoader(train_ds,batch_size=48,shuffle=True,num_workers=0); val_loader=DataLoader(val_ds,batch_size=48,num_workers=0)
    model=HourlyLSTM()
    if not (OUT/"best_route2_lstm.pt").exists():
        opt=torch.optim.AdamW(model.parameters(),lr=7e-4,weight_decay=1e-4); best=1e9; stale=0; hist=[]
        for epoch in range(1,31):
            model.train(); total=0.; count=0
            for s,x,y in train_loader:
                opt.zero_grad(set_to_none=True); p,_=model(s,x); loss=nn.functional.smooth_l1_loss(p,y,beta=.5); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),2.); opt.step(); total+=loss.item()*y.numel(); count+=y.numel()
            vm=validate_windows(model,val_loader); hist.append((epoch,total/count,vm)); print(f"epoch={epoch:02d} train={total/count:.6f} val_mse={vm:.6f}",flush=True)
            if vm<best-1e-5: best=vm; stale=0; torch.save(model.state_dict(),OUT/"best_route2_lstm.pt")
            else:
                stale+=1
                if stale>=6: break
        pd.DataFrame(hist,columns=["epoch","train_loss","val_mse"]).to_csv(OUT/"history.csv",index=False)
    model.load_state_dict(torch.load(OUT/"best_route2_lstm.pt",weights_only=True,map_location="cpu"))
    truth,pred,building,hour=infer_test(model,weather,targets,static,test,xm,xs,ym,ys)
    names=["Hourly energy","Hourly PV generation","Hourly carbon emission"]; rows=[]
    for j,name in enumerate(names):
        rows.append({"target":name,"R2":r2_score(truth[:,:,j].ravel(),pred[:,:,j].ravel()),"RMSE":mean_squared_error(truth[:,:,j].ravel(),pred[:,:,j].ravel())**.5,"MAE":mean_absolute_error(truth[:,:,j].ravel(),pred[:,:,j].ravel()),"n":truth.shape[0]*HOURS})
    pd.DataFrame(rows).to_csv(OUT/"internal_test_metrics.csv",index=False)
    np.savez_compressed(OUT/"internal_test_predictions_8760x3.npz",row_indices=test.astype(np.int32),hour=np.arange(1,HOURS+1,dtype=np.int16),true=truth.astype(np.float32),prediction=pred.astype(np.float32))
    print(pd.DataFrame(rows).to_string(index=False),flush=True)


if __name__=="__main__": main()
