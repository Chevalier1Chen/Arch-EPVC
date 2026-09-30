from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import train_route2_lstm as core

def main():
    torch.set_num_threads(min(16,torch.get_num_threads())); weather=np.load(core.CACHE/"weather.npy",mmap_mode="r"); targets=np.load(core.CACHE/"targets.npy",mmap_mode="r"); static=np.load(core.OUT/"static_condition_259.npy")
    split=json.loads((Path(r"C:\Users\DELL\Documents\论文\internal_two_route_rebuild\route1\split.json")).read_text(encoding="utf-8")); valid=np.load(core.CACHE/"valid.npy")
    val=np.asarray(split["val"]);val=val[valid[val]];test=np.asarray(split["test"]);test=test[valid[test]]
    s=np.load(core.OUT/"scalers.npz");xm,xs,ym,ys=[s[k] for k in ["input_mean","input_std","label_mean","label_std"]]
    model=core.HourlyLSTM();model.load_state_dict(torch.load(core.OUT/"best_route2_lstm.pt",weights_only=True,map_location="cpu"))
    tv,pv,_,_=core.infer_test(model,weather,targets,static,val,xm,xs,ym,ys); tt,pt,_,_=core.infer_test(model,weather,targets,static,test,xm,xs,ym,ys)
    coef=[]
    for j in range(3):
        x=pv[:,:,j].ravel().astype(np.float64);y=tv[:,:,j].ravel().astype(np.float64); a=np.cov(x,y,bias=True)[0,1]/np.var(x);b=y.mean()-a*x.mean();coef.append((a,b));pt[:,:,j]=np.maximum(a*pt[:,:,j]+b,0)
    names=["Energy","PV","Carbon"];rows=[]
    for j,n in enumerate(names): rows.append({"route":"Route II","model":"LSTM","target":n,"R2":r2_score(tt[:,:,j].ravel(),pt[:,:,j].ravel()),"RMSE":mean_squared_error(tt[:,:,j].ravel(),pt[:,:,j].ravel())**.5,"MAE":mean_absolute_error(tt[:,:,j].ravel(),pt[:,:,j].ravel()),"n":tt.shape[0]*core.HOURS})
    np.savez(core.OUT/"lstm_validation_calibration.npz",slope=np.array([x[0] for x in coef]),intercept=np.array([x[1] for x in coef]))
    pd.DataFrame(rows).to_csv(core.OUT/"calibrated_internal_test_metrics.csv",index=False,encoding="utf-8-sig");print('coefficients',coef);print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__':main()
