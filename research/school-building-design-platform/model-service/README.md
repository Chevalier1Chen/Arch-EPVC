# 模型服务资产

浏览器部署包由正式双路模型导出：

- 第一路：表格 MLP、Tabular-ResNet、OBJ-3D 特征 + TabTransformer 集成，输出年度 EUI、Epv、CEI。
- 第二路：259 维静态融合特征 + 8760×10 时间/气象特征，经两层 LSTM 输出逐时能耗、光伏与运行碳排放；逐时曲线按第一路年度预测进行总量一致性校准。
- 第二路测试精度：逐时 R² 为 0.9332 / 0.9864 / 0.9346；年度累计 R² 为 0.9617 / 0.9913 / 0.9672。

`export_browser_models.py` 从正式 PyTorch checkpoint 导出 Route I 与 Route II ONNX 文件，并同时导出连续变量、年度目标及逐时输入/输出的标准化参数。

前端对新设计执行：OBJ 解析 → `32 × 32 × 32` 实心占据体素化 → 3D-CNN–TabTransformer 年度预测与 256 维共享特征 → 259 维静态条件 → LSTM 8,760 小时预测。`verify_browser_models.py` 用于对比 PyTorch 与导出的 ONNX 数值。
