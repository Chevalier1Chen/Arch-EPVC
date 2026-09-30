# Arch-EPVC：学校建筑能耗、光伏发电与碳排放双尺度预测数据与平台

[打开 Arch-EPVC 在线评价平台](https://arch-epvc.vercel.app)

本仓库用于公开论文相关的数据、模型说明、模拟流程和平台展示页面。研究对象为山东省学校教学建筑，目标是同时服务于**新建学校建筑方案阶段评价**与**既有教学楼低碳改造/光伏潜力评估**。

## 仓库内容

- `data/processed/`：结构化数值数据与年度性能指标。
- `data/geometry/`：教学楼 OBJ 几何模型汇总包。
- `data/samples/time_series/`：逐时性能数据样例。
- `data/metadata/`：字段字典、逐时数据清单和数据包清单。
- `simulation/`：Grasshopper 文件与 Rhino/GH 辅助脚本。
- `models/`：年度模型、逐时模型的模型卡与配置模板。
- `platform/`：Arch-EPVC 平台展示材料。
- `research/`：已找回的训练源码、平台实际引用的两路模型权重、预处理参数，以及含 ONNX 模型的平台源码。
- `docs/`：GitHub Pages 项目主页。

## 重要说明

当前是本地待发布包，尚未上传 GitHub。请先看 `PUBLICATION_STATUS.md`：原训练代码仍含本机路径，平台既有建筑逐时模块存在资源缺失，不能将此包标为全部功能已复现通过。已核实的浏览器模型版本与其他实验版本按来源区分。

完整逐时数据约数 GB，不建议直接放入普通 GitHub 仓库。本仓库先公开样例文件和完整清单；正式公开时建议使用 GitHub Releases、Git LFS、Zenodo 或 OSF 承载完整数据包。

如果卫星底图来自第三方地图服务，公开前需要确认再分发许可；默认公开版暂不包含原始卫星底图。
