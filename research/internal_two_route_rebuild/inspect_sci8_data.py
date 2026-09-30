from pathlib import Path
import pandas as pd

NUM = Path(r"C:\Users\DELL\Desktop\SCI8\数据集完成\numerical data.xlsx")
TS = Path(r"C:\Users\DELL\Desktop\SCI8\数据集完成\Time series data\SD0001_T001_1hao.xlsx")

x = pd.ExcelFile(NUM, engine="openpyxl")
print("NUM_SHEETS", x.sheet_names)
for sheet in x.sheet_names:
    data = pd.read_excel(NUM, sheet_name=sheet, nrows=3, engine="openpyxl")
    print("NUM", sheet, data.shape, list(data.columns))

y = pd.ExcelFile(TS, engine="openpyxl")
print("TS_SHEETS", y.sheet_names)
for sheet in y.sheet_names:
    data = pd.read_excel(TS, sheet_name=sheet, nrows=8, engine="openpyxl")
    print("TS", sheet, data.shape, list(data.columns))
    print(data.head(5).to_string())

hourly = pd.read_excel(TS, sheet_name="Sheet1", header=1, engine="openpyxl")
print("HOURLY_SHAPE", hourly.shape)
for index, column in enumerate(hourly.columns):
    print("COL", index, str(column).encode("unicode_escape").decode(), "SUM", pd.to_numeric(hourly.iloc[:, index], errors="coerce").sum())
selected = pd.read_excel(NUM, sheet_name="selected_variables", engine="openpyxl")
row = selected.loc[selected["Number"].astype(str).eq("SD0001_T001_1hao")].iloc[0]
print("ANNUAL_ROW", row.to_dict())
