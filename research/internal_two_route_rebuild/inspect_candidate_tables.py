from pathlib import Path
import pandas as pd

paths = [
    Path(r"C:\Users\DELL\Desktop\SCI8\shandong_teaching_campus_overview_package\outputs\numerical data.xlsx"),
    Path(r"C:\Users\DELL\Desktop\SCI8\Simulation\6-9 自变量统计表.xlsx"),
    Path(r"C:\Users\DELL\Desktop\SCI8\Simulation\各学校建设年代.xlsx"),
    Path(r"C:\Users\DELL\Desktop\SCI8\Simulation\construction_year_collection_results.xlsx"),
    Path(r"C:\Users\DELL\Desktop\SCI8\shandong_teaching_campus_overview_package\data\gh_building_energy_inputs.csv"),
    Path(r"C:\Users\DELL\Desktop\SCI8\shandong_teaching_campus_overview_package\data\teaching_building_metrics.csv"),
]
for p in paths:
    print("\n===", p)
    try:
        if p.suffix.lower()==".csv":
            d=pd.read_csv(p,nrows=3); print("shape?",sum(1 for _ in p.open('r',encoding='utf-8-sig'))-1); print(d.columns.tolist()); print(d.head(2).to_string())
        else:
            x=pd.ExcelFile(p); print(x.sheet_names)
            for s in x.sheet_names[:5]:
                d=pd.read_excel(p,sheet_name=s,nrows=3); print(s,d.columns.tolist()); print(d.head(2).to_string())
    except Exception as e: print(type(e).__name__,e)
