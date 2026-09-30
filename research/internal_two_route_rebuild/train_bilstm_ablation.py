from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import train_route2_lstm as core
import run_ablation_route2 as abl

class BiLSTM(nn.Module):
    def __init__(self):
        super().__init__(); self.input=abl.InputFusion(); self.lstm=nn.LSTM(64,64,num_layers=2,batch_first=True,dropout=.10,bidirectional=True); self.head=nn.Sequential(nn.Linear(128,64),nn.SiLU(),nn.Linear(64,3))
    def forward(self,s,x,state=None): h,state=self.lstm(self.input(s,x),state); return self.head(h),state

def main():
    core.seed_all(42);torch.set_num_threads(min(16,torch.get_num_threads()));weather=np.load(core.CACHE/"weather.npy",mmap_mode="r");targets=np.load(core.CACHE/"targets.npy",mmap_mode="r");static=np.load(core.OUT/"static_condition_259.npy")
    split=json.loads((core.ROOT/"route1"/"split.json").read_text(encoding="utf-8"));valid=np.load(core.CACHE/"valid.npy")
    train=np.asarray(split["train"]);train=train[valid[train]];val=np.asarray(split["val"]);val=val[valid[val]];test=np.asarray(split["test"]);test=test[valid[test]]
    s=np.load(core.OUT/"scalers.npz");xm,xs,ym,ys=[s[k] for k in ["input_mean","input_std","label_mean","label_std"]]
    tl=DataLoader(core.Windows(weather,targets,static,train,xm,xs,ym,ys,n=7000,length=72,seed=42),batch_size=48,shuffle=True,num_workers=0);vl=DataLoader(core.Windows(weather,targets,static,val,xm,xs,ym,ys,n=1200,length=168,seed=1041),batch_size=48,num_workers=0)
    model=BiLSTM(); checkpoint=abl.OUT/"BiLSTM_proposed.pt"
    if checkpoint.exists(): model.load_state_dict(torch.load(checkpoint,weights_only=True,map_location="cpu"))
    else: model=abl.fit("BiLSTM_proposed",model,tl,vl)
    truths=[];preds=[];model.eval()
    with torch.no_grad():
        for start in range(0,len(test),8):
            batch=np.asarray(test[start:start+8]);x=((np.asarray(weather[batch])-xm)/xs).astype(np.float32);p,_=model(torch.from_numpy(static[batch]),torch.from_numpy(x));preds.append(np.maximum(np.expm1(p.numpy()*ys+ym),0));truths.append(np.asarray(targets[batch]));print(f"full-year inference {min(start+len(batch),len(test))}/{len(test)}",flush=True)
    t=np.concatenate(truths);p=np.concatenate(preds);rows=[]
    from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
    for j,target in enumerate(["Energy","PV","Carbon"]):rows.append({"route":"Route II","model":"LSTM (proposed)","target":target,"R2":r2_score(t[:,:,j].ravel(),p[:,:,j].ravel()),"RMSE":mean_squared_error(t[:,:,j].ravel(),p[:,:,j].ravel())**.5,"MAE":mean_absolute_error(t[:,:,j].ravel(),p[:,:,j].ravel()),"n":t.shape[0]*core.HOURS})
    pd.DataFrame(rows).to_csv(abl.OUT/"bilstm_proposed_metrics.csv",index=False,encoding="utf-8-sig")

if __name__=='__main__':main()
