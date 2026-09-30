import json
from pathlib import Path
import numpy as np
import pandas as pd

root=Path(r"C:\Users\DELL\Documents\论文\internal_two_route_rebuild")
html=Path(r"C:\Users\DELL\.codex\visualizations\2026\07\31\019fb653-43ca-7a30-9493-f0d87bd84060\internal-two-route-scatter.html")
r1=pd.read_csv(root/"route1"/"internal_test_predictions.csv")
r2=pd.read_csv(root/"figures"/"Route_II_Internal_Test_Display_Sample.csv")
rng=np.random.default_rng(42)
if len(r2)>1800: r2=r2.iloc[np.sort(rng.choice(len(r2),1800,replace=False))]
def pairs(a,b): return [[round(float(x),4),round(float(y),4)] for x,y in zip(a,b)]
data={"route1":{
    "eui":pairs(r1.true_EUI,r1.pred_EUI),"epv":pairs(r1.true_Epv,r1.pred_Epv),"cei":pairs(r1.true_CEI,r1.pred_CEI)},
    "route2":{"energy":pairs(r2.actual_energy,r2.pred_energy),"pv":pairs(r2.actual_pv,r2.pred_pv),"carbon":pairs(r2.actual_carbon,r2.pred_carbon)}}
text=html.read_text(encoding="utf-8")
text=text.replace("__DATA_PLACEHOLDER__",json.dumps(data,separators=(",",":"),ensure_ascii=False))
html.write_text(text,encoding="utf-8")
print(html.stat().st_size)
