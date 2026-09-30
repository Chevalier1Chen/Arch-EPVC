from __future__ import annotations

import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

ROOT = Path(r"C:\Users\DELL\Documents\论文\internal_two_route_rebuild")
TABLE = Path(r"C:\Users\DELL\Desktop\SCI8\数据集完成\numerical data.xlsx")
VOXELS = Path(r"C:\Users\DELL\Documents\论文\external_validation_150\model_run_restored_inputs\training_voxels_32.npy")
OUT = ROOT / "route1"
SEED = 42
TARGETS = ["EUI", "Epv", "CEI-PV system"]
TARGET_LABELS = ["EUI", "Epv", "CEI"]
CONT = [
    "A.Building area", "B.Building footprint", "C.Building height", "D.Layer",
    "E.Height", "F.Building length", "G.Building width", "H.Orientation",
    "J.Shape coefficient", "K.Roof thermal coefficient",
    "L.Wall thermal coefficient", "M.Ground thermal coefficient", "N.Window U-value",
    "construction_year_user", "roof_area_m2",
]
CAT = "CombinedCategory"


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


class Samples(Dataset):
    def __init__(self, vox, cont, cat, y, idx):
        self.vox, self.cont, self.cat, self.y = vox, cont, cat, y
        self.idx = np.asarray(idx)
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        j = int(self.idx[i])
        return (
            torch.from_numpy(np.asarray(self.vox[j], dtype=np.float32)),
            torch.from_numpy(self.cont[j]), torch.tensor(self.cat[j]),
            torch.from_numpy(self.y[j]), torch.tensor(j),
        )


class Geometry3DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1), nn.BatchNorm3d(16), nn.SiLU(), nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, padding=1), nn.BatchNorm3d(32), nn.SiLU(), nn.MaxPool3d(2),
            nn.Conv3d(32, 64, 3, padding=1), nn.BatchNorm3d(64), nn.SiLU(), nn.MaxPool3d(2),
            nn.AdaptiveAvgPool3d(1),
        )
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(64, 128), nn.LayerNorm(128), nn.SiLU())
    def forward(self, x): return self.proj(self.blocks(x))


class TabTransformer(nn.Module):
    def __init__(self, n_cont: int, n_cat: int):
        super().__init__()
        d = 32
        self.cont_weight = nn.Parameter(torch.randn(n_cont, d) * 0.03)
        self.cont_bias = nn.Parameter(torch.zeros(n_cont, d))
        self.cat_embed = nn.Embedding(n_cat, d)
        self.position = nn.Parameter(torch.randn(1, n_cont + 1, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, 4, 96, dropout=.08, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2)
        self.proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 128), nn.SiLU())
    def forward(self, x, cat):
        tokens = x.unsqueeze(-1) * self.cont_weight.unsqueeze(0) + self.cont_bias.unsqueeze(0)
        tokens = torch.cat([tokens, self.cat_embed(cat).unsqueeze(1)], dim=1) + self.position
        return self.proj(self.encoder(tokens).mean(1))


class Route1(nn.Module):
    def __init__(self, n_cont: int, n_cat: int):
        super().__init__()
        self.geometry = Geometry3DCNN()
        self.tabular = TabTransformer(n_cont, n_cat)
        self.gate = nn.Sequential(nn.Linear(256, 128), nn.Sigmoid())
        self.shared = nn.Sequential(nn.Linear(512, 256), nn.LayerNorm(256), nn.SiLU(), nn.Dropout(.12))
        self.head = nn.Sequential(nn.Linear(256, 128), nn.SiLU(), nn.Dropout(.08), nn.Linear(128, 3))
    def forward(self, vox, cont, cat, return_embedding=False):
        g, t = self.geometry(vox), self.tabular(cont, cat)
        a = self.gate(torch.cat([g, t], 1))
        z = self.shared(torch.cat([a*g, (1-a)*t, g*t, torch.abs(g-t)], 1))
        pred = self.head(z)
        return (pred, z) if return_embedding else pred


@torch.no_grad()
def predict(model, loader, device):
    model.eval(); rows=[]; zs=[]; ids=[]
    for vox, cont, cat, y, idx in loader:
        p,z=model(vox.to(device),cont.to(device),cat.to(device),True)
        rows.append(p.cpu().numpy()); zs.append(z.cpu().numpy()); ids.append(idx.numpy())
    order=np.concatenate(ids).astype(int)
    return order,np.concatenate(rows),np.concatenate(zs)


def main():
    seed_all(SEED); OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(16, torch.get_num_threads()))
    df=pd.read_excel(TABLE, sheet_name="selected_variables")
    gh_path=Path(r"C:\Users\DELL\Desktop\SCI8\shandong_teaching_campus_overview_package\data\gh_building_energy_inputs.csv")
    gh=pd.read_csv(gh_path,encoding="utf-8-sig")
    gh=gh.rename(columns={"obj_object_name_prefix":"Number"}).drop_duplicates("Number")[["Number","city","roof_area_m2"]]
    year_path=Path(r"C:\Users\DELL\Desktop\SCI8\Simulation\gh_building_simulation_min_inputs.xlsx")
    years=pd.read_excel(year_path,sheet_name=0); years=years.iloc[:,:3].copy(); years.columns=["Number","story_height_raw","construction_year_user"]
    years["Number"]=years["Number"].astype(str)
    df=df.merge(gh,on="Number",how="left",validate="one_to_one").merge(years[["Number","construction_year_user"]],on="Number",how="left",validate="one_to_one")
    if df[["city","roof_area_m2","construction_year_user"]].isna().any().any():
        raise ValueError(df[["city","roof_area_m2","construction_year_user"]].isna().sum())
    df[CAT]=df["city"].astype(str)+" | "+df["I.Enclosure method"].astype(str)
    vox=np.load(VOXELS, mmap_mode="r")
    assert len(df)==len(vox)==3356
    all_idx=np.arange(len(df))
    train_val,test=train_test_split(all_idx,test_size=.15,random_state=SEED)
    train,val=train_test_split(train_val,test_size=.1764705882,random_state=SEED+1)
    scaler=StandardScaler().fit(df.loc[train,CONT])
    x=scaler.transform(df[CONT]).astype(np.float32)
    categories=sorted(df[CAT].astype(str).unique().tolist())
    cat_map={v:i for i,v in enumerate(categories)}
    cat=df[CAT].astype(str).map(cat_map).to_numpy(np.int64)
    y_raw=df[TARGETS].to_numpy(np.float64)
    y_trans=y_raw.copy(); y_trans[:,1]=np.log1p(y_trans[:,1])
    y_mean=y_trans[train].mean(0); y_std=y_trans[train].std(0)
    y=((y_trans-y_mean)/y_std).astype(np.float32)
    loaders={name:DataLoader(Samples(vox,x,cat,y,idx),batch_size=48,shuffle=name=="train",num_workers=0)
             for name,idx in [("train",train),("val",val),("test",test),("all",all_idx)]}
    device=torch.device("cpu"); model=Route1(len(CONT),len(categories)).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=2e-4)
    loss_fn=nn.SmoothL1Loss(beta=.5); best=1e9; stale=0; history=[]
    for epoch in range(1,81):
        model.train(); total=0.; count=0
        for v,c,k,target,_ in loaders["train"]:
            opt.zero_grad(set_to_none=True); pred=model(v.to(device),c.to(device),k.to(device))
            loss=loss_fn(pred,target.to(device)); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),3.)
            opt.step(); total+=loss.item()*len(v); count+=len(v)
        _,vp,_=predict(model,loaders["val"],device)
        vloss=float(np.mean((vp-y[val])**2)); history.append((epoch,total/count,vloss))
        print(f"epoch={epoch:03d} train={total/count:.6f} val_mse={vloss:.6f}",flush=True)
        if vloss < best-1e-5:
            best=vloss; stale=0; torch.save(model.state_dict(),OUT/"best_route1_3dcnn_tabtransformer.pt")
        else:
            stale+=1
            if stale>=12: break
    model.load_state_dict(torch.load(OUT/"best_route1_3dcnn_tabtransformer.pt",weights_only=True,map_location="cpu"))
    pd.DataFrame(history,columns=["epoch","train_loss","val_mse"]).to_csv(OUT/"history.csv",index=False)
    def inverse(a):
        z=a*y_std+y_mean; z[:,1]=np.expm1(z[:,1]); return z
    test_order,pred_scaled,_=predict(model,loaders["test"],device); pred=inverse(pred_scaled); true=y_raw[test_order]
    rows=[]
    for j,name in enumerate(TARGET_LABELS):
        rows.append({"target":name,"R2":r2_score(true[:,j],pred[:,j]),"RMSE":mean_squared_error(true[:,j],pred[:,j])**.5,"MAE":mean_absolute_error(true[:,j],pred[:,j]),"n":len(true)})
    pd.DataFrame(rows).to_csv(OUT/"internal_test_metrics.csv",index=False)
    result=pd.DataFrame({"row_index":test_order,"building_id":df.iloc[test_order]["Number"].to_numpy()})
    for j,name in enumerate(TARGET_LABELS): result[f"true_{name}"]=true[:,j]; result[f"pred_{name}"]=pred[:,j]
    result.to_csv(OUT/"internal_test_predictions.csv",index=False,encoding="utf-8-sig")
    all_order,_,embedding=predict(model,loaders["all"],device)
    np.save(OUT/"static_shared_embedding_256.npy",embedding[np.argsort(all_order)].astype(np.float32))
    joblib.dump(scaler,OUT/"continuous_scaler.joblib")
    np.savez(OUT/"target_scaler.npz",mean=y_mean,std=y_std,epv_log=np.asarray([1]))
    (OUT/"split.json").write_text(json.dumps({"seed":SEED,"train":train.tolist(),"val":val.tolist(),"test":test.tolist(),"categories":categories,"features":CONT},ensure_ascii=False),encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False),flush=True)


if __name__=="__main__": main()
