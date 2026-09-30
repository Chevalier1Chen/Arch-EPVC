from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import train_route2_lstm as core
import run_ablation_route2 as abl

def main():
    core.seed_all(42);torch.set_num_threads(min(16,torch.get_num_threads()));weather=np.load(core.CACHE/"weather.npy",mmap_mode="r");targets=np.load(core.CACHE/"targets.npy",mmap_mode="r");static=np.load(core.OUT/"static_condition_259.npy")
    split=json.loads((core.ROOT/"route1"/"split.json").read_text(encoding="utf-8"));valid=np.load(core.CACHE/"valid.npy");train=np.asarray(split["train"]);train=train[valid[train]];val=np.asarray(split["val"]);val=val[valid[val]];test=np.asarray(split["test"]);test=test[valid[test]]
    s=np.load(core.OUT/"scalers.npz");xm,xs,ym,ys=[s[k] for k in ["input_mean","input_std","label_mean","label_std"]]
    tl=DataLoader(core.Windows(weather,targets,static,train,xm,xs,ym,ys,n=6000,length=168,seed=42),batch_size=40,shuffle=True,num_workers=0);vl=DataLoader(core.Windows(weather,targets,static,val,xm,xs,ym,ys,n=1200,length=336,seed=1041),batch_size=40,num_workers=0)
    model=abl.fit("LSTM_long_context",core.HourlyLSTM(),tl,vl);rows=abl.full_metrics("LSTM (proposed)",model,weather,targets,static,test,xm,xs,ym,ys);pd.DataFrame(rows).to_csv(abl.OUT/"lstm_long_proposed_metrics.csv",index=False,encoding="utf-8-sig")

if __name__=='__main__':main()
