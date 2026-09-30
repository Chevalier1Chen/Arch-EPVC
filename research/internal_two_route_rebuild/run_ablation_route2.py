from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader

import train_route2_lstm as core

OUT=core.ROOT/"ablation"/"route2"; OUT.mkdir(parents=True,exist_ok=True)


class InputFusion(nn.Module):
    def __init__(self):
        super().__init__(); self.s=nn.Sequential(nn.Linear(259,64),nn.LayerNorm(64),nn.SiLU()); self.w=nn.Sequential(nn.Linear(10,32),nn.LayerNorm(32),nn.SiLU()); self.f=nn.Sequential(nn.Linear(96,64),nn.SiLU())
    def forward(self,s,x): return self.f(torch.cat([self.s(s).unsqueeze(1).expand(-1,x.shape[1],-1),self.w(x)],-1))


class HourlyMLP(nn.Module):
    def __init__(self): super().__init__(); self.input=InputFusion(); self.head=nn.Sequential(nn.Linear(64,64),nn.SiLU(),nn.Dropout(.08),nn.Linear(64,3))
    def forward(self,s,x,state=None): return self.head(self.input(s,x)),None


class HourlyGRU(nn.Module):
    def __init__(self): super().__init__(); self.input=InputFusion(); self.gru=nn.GRU(64,64,num_layers=1,batch_first=True); self.head=nn.Sequential(nn.Linear(64,48),nn.SiLU(),nn.Linear(48,3))
    def forward(self,s,x,state=None): h,state=self.gru(self.input(s,x),state); return self.head(h),state


class HourlyTransformer(nn.Module):
    def __init__(self):
        super().__init__(); self.input=InputFusion(); layer=nn.TransformerEncoderLayer(64,4,128,dropout=.10,activation="gelu",batch_first=True,norm_first=True); self.enc=nn.TransformerEncoder(layer,2); self.head=nn.Sequential(nn.LayerNorm(64),nn.Linear(64,48),nn.SiLU(),nn.Linear(48,3))
    def forward(self,s,x,state=None): return self.head(self.enc(self.input(s,x))),None


@torch.no_grad()
def validation(model,loader):
    model.eval(); total=0;n=0
    for s,x,y in loader: p,_=model(s,x); total+=nn.functional.mse_loss(p,y,reduction="sum").item(); n+=y.numel()
    return total/n


def fit(name,model,train_loader,val_loader):
    path=OUT/(name.replace(" ","_")+".pt"); opt=torch.optim.AdamW(model.parameters(),lr=7e-4,weight_decay=1e-4); best=1e9; stale=0
    for epoch in range(1,26):
        model.train(); total=0;n=0
        for s,x,y in train_loader:
            opt.zero_grad(set_to_none=True); p,_=model(s,x); loss=nn.functional.smooth_l1_loss(p,y,beta=.5); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),2); opt.step(); total+=loss.item()*y.numel(); n+=y.numel()
        vm=validation(model,val_loader); print(f"{name} epoch={epoch:02d} train={total/n:.5f} val={vm:.5f}",flush=True)
        if vm<best-1e-5: best=vm;stale=0;torch.save(model.state_dict(),path)
        else:
            stale+=1
            if stale>=5: break
    model.load_state_dict(torch.load(path,weights_only=True,map_location="cpu")); return model


@torch.no_grad()
def full_metrics(name,model,weather,targets,static,test,xm,xs,ym,ys):
    truths=[];preds=[];model.eval()
    for bstart in range(0,len(test),12):
        batch=np.asarray(test[bstart:bstart+12]); parts=[];state=None
        for start in range(0,core.HOURS,336):
            end=min(start+336,core.HOURS); x=((np.asarray(weather[batch,start:end])-xm)/xs).astype(np.float32); p,state=model(torch.from_numpy(static[batch]),torch.from_numpy(x),state); parts.append(p.numpy());
            if state is not None: state=state.detach() if isinstance(state,torch.Tensor) else tuple(v.detach() for v in state)
        preds.append(np.maximum(np.expm1(np.concatenate(parts,axis=1)*ys+ym),0)); truths.append(np.asarray(targets[batch]))
    t=np.concatenate(truths);p=np.concatenate(preds); rows=[]; names=["Energy","PV","Carbon"]
    for j,target in enumerate(names): rows.append({"route":"Route II","model":name,"target":target,"R2":r2_score(t[:,:,j].ravel(),p[:,:,j].ravel()),"RMSE":mean_squared_error(t[:,:,j].ravel(),p[:,:,j].ravel())**.5,"MAE":mean_absolute_error(t[:,:,j].ravel(),p[:,:,j].ravel()),"n":t.shape[0]*core.HOURS})
    print(pd.DataFrame(rows).to_string(index=False),flush=True); return rows


def main():
    core.seed_all(42); torch.set_num_threads(min(16,torch.get_num_threads())); weather=np.load(core.CACHE/"weather.npy",mmap_mode="r"); targets=np.load(core.CACHE/"targets.npy",mmap_mode="r"); static=np.load(core.OUT/"static_condition_259.npy")
    split=json.loads((Path(r"C:\Users\DELL\Documents\论文\internal_two_route_rebuild\route1\split.json")).read_text(encoding="utf-8")); valid=np.load(core.CACHE/"valid.npy")
    train=np.asarray(split["train"]);train=train[valid[train]];val=np.asarray(split["val"]);val=val[valid[val]];test=np.asarray(split["test"]);test=test[valid[test]]
    s=np.load(core.OUT/"scalers.npz");xm,xs,ym,ys=[s[k] for k in ["input_mean","input_std","label_mean","label_std"]]
    train_ds=core.Windows(weather,targets,static,train,xm,xs,ym,ys,n=7000,length=72,seed=42); val_ds=core.Windows(weather,targets,static,val,xm,xs,ym,ys,n=1200,length=168,seed=1041)
    tl=DataLoader(train_ds,batch_size=48,shuffle=True,num_workers=0);vl=DataLoader(val_ds,batch_size=48,num_workers=0)
    rows=[]
    for name,model in [("Hourly MLP",HourlyMLP()),("GRU",HourlyGRU()),("Transformer",HourlyTransformer())]: rows+=full_metrics(name,fit(name,model,tl,vl),weather,targets,static,test,xm,xs,ym,ys)
    proposed=pd.read_csv(core.OUT/"internal_test_metrics.csv")
    mapping={"Hourly energy":"Energy","Hourly PV generation":"PV","Hourly carbon emission":"Carbon"}
    for _,r in proposed.iterrows(): rows.append({"route":"Route II","model":"LSTM","target":mapping[r.target],"R2":r.R2,"RMSE":r.RMSE,"MAE":r.MAE,"n":r.n})
    pd.DataFrame(rows).to_csv(OUT/"route2_ablation_metrics.csv",index=False,encoding="utf-8-sig"); print(pd.DataFrame(rows).to_string(index=False))


if __name__=="__main__": main()
