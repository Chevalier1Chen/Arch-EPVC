"use client";

import { useEffect } from "react";

const TEXT:Record<string,string>={
  "Building Performance Intelligence":"建筑性能智能评价",
  "Shandong dataset connected":"山东省数据集已连接",
  "3,360 buildings":"3,360 栋建筑",
  "Model status":"模型状态",
  "MULTIMODAL DESIGN INTELLIGENCE":"多模态设计智能",
  "From school buildings to":"从学校建筑到",
  "energy—PV—carbon":"能耗—光伏—碳排放",
  "digital profiles":"数字画像",
  "Evaluate an existing school from the research database or map, or upload a new OBJ design with project parameters and an EPW file for native dual-route prediction.":"可从研究数据库或地图选择既有学校建筑进行评价，也可上传新建方案的OBJ模型、设计参数与EPW气象文件，完成双路模型预测。",
  "Existing school assessment":"既有学校建筑评价","Research database or map selection":"研究数据库或地图选择",
  "New design assessment":"新建建筑方案评价","OBJ + design parameters + EPW":"OBJ模型＋设计参数＋EPW气象文件",
  "Use one of 3,360 verified school buildings unchanged, or select a new map footprint and provide its measured height and floor count before evaluation.":"可直接选用3,360栋已验证学校建筑，或在地图中选择新的建筑底面，并输入实测高度和层数后进行评价。",
  "Select":"选择","Model":"建模","Evaluate":"评价",
  "Teaching building database":"教学楼数据库",
  "994 campuses · 3,360 verified building records":"994个校区 · 3,360条已验证建筑记录",
  "City and campus directory":"城市与校区目录",
  "Search campus or school":"搜索校区或学校",
  "Geometry and attributes are read from the formal research package. The map is no longer used to create or estimate buildings.":"几何与属性均读取自正式研究数据包；原数据集方案不通过地图重新生成或估算。",
  "EPW weather input":"EPW气象数据输入",
  "8,760-hour climate sequence":"8,760小时气象序列",
  "Custom EPW file":"自定义EPW文件",
  "(Required for new design)":"（新建方案必填）",
  "(User input, optional)":"（用户自填，可选）",
  "(User input)":"（用户自填）",
  "Select an .epw file":"选择EPW文件",
  "EnergyPlus Weather File":"EnergyPlus气象文件",
  "Temperature":"温度","Humidity":"相对湿度","Wind":"风速","Wind speed":"风速",
  "Binzhou":"滨州","Dezhou":"德州","Dongying":"东营","Heze":"菏泽","Jinan":"济南","Jining":"济宁","Liaocheng":"聊城","Linyi":"临沂","Qingdao":"青岛","Rizhao":"日照","Tai'an":"泰安","Weihai":"威海","Weifang":"潍坊","Yantai":"烟台","Zaozhuang":"枣庄","Zibo":"淄博",
  "DNI":"法向直接辐照度","DHI":"水平散射辐照度","GHI":"水平总辐照度",
  "CAMPUS DIGITAL MODEL":"校区数字模型",
  "high":"高可信度","unknown":"未知",
  "Retrofit PV scenario":"既有建筑改造光伏情景","New-build PV scenario":"新建建筑光伏情景","Mapped-building PV scenario":"地图建筑光伏情景",
  "Switch the active assessment, 3D roof, PV profile and carbon profile together":"切换评价情景时，同步更新三维屋顶、逐时光伏和逐时碳排放","Complete the building geometry before adding rooftop PV":"完成建筑几何建模后即可添加屋顶光伏",
  "No rooftop PV":"不设置屋顶光伏","Full-roof PV":"屋顶满铺光伏","Rooftop PV scenario":"屋顶光伏情景",
  "Click a building in the 3D campus model":"点击三维校区模型中的建筑进行选择",
  "Selected building":"当前选中建筑",
  "Select a map footprint":"选择地图建筑底面",
  "Click the map and select a building footprint":"请在地图上点击并选择建筑底面",
  "Research dataset · original merged OBJ and stored geometry attributes":"研究数据集 · 原始合并OBJ及已存储的几何属性",
  "Select a building":"选择建筑",
  "Download cleaned OBJ":"下载清洗后的OBJ",
  "Height and floors required":"需要输入高度和层数",
  "dataset matched":"数据集已匹配",
  "Footprint":"建筑底面积","Height":"建筑高度","Floors":"建筑层数","Window ratio":"窗墙比","Geometry":"几何来源",
  "Source OBJ":"原始OBJ","OSM":"开放地图数据",
  "BUILDING SOURCE":"建筑数据来源",
  "Research database":"研究数据库","Map selection":"地图选择","Select an existing building from the map":"从地图选择既有建筑",
  "Select a new building footprint":"选择新的建筑底面",
  "Select a teaching building":"选择教学楼",
  "Dataset building":"数据集建筑","New from map":"地图新建建筑",
  "Building ID":"建筑编号","Campus ID":"校区编号","City":"城市","Coordinates":"经纬度坐标",
  "Geometry type":"几何类型","Real existing geometry":"真实既有几何",
  "Height source":"高度来源","Source 3D geometry · Z extent":"原始三维几何 · Z轴高度范围",
  "Floor method":"层数计算方法","Height / floor-to-floor height":"建筑高度 / 层高",
  "OBJ source":"OBJ来源","Per-campus merged OBJ":"按校区合并的OBJ",
  "The original research-dataset geometry and parameters remain unchanged. Adjacent parts were merged before model features were calculated.":"原研究数据集的几何与参数保持不变；相邻体块在模型特征计算前完成合并。",
  "Select a campus and teaching building":"请选择校区和教学楼",
  "DUAL-ROUTE PREDICTION":"双路预测",
  "Performance evaluation":"性能评价",
  "Height and floors require confirmation":"需要确认建筑高度和层数",
  "Mapped-building assessment":"地图建筑评价",
  "Mapped-building prediction":"地图建筑预测",
  "Running building-specific 8,760-hour model":"正在运行建筑专属8,760小时模型",
  "Building-specific annual + 8,760-hour output":"建筑专属全年与8,760小时输出",
  "First-route profile · hourly model unavailable":"第一路预测 · 逐时模型暂不可用",
  "Full first-route annual prediction":"完整第一路年度预测",
  "ANNUAL CORE INDICATORS":"年度核心指标",
  "Whole-building prediction, normalized by floor area where applicable":"整栋建筑预测；适用指标按总建筑面积归一化",
  "Annual EUI":"年度建筑能耗强度 EUI","Annual Epv":"年度光伏发电量 Epv","Full-roof PV generation":"屋顶满铺光伏发电量",
  "CEI before PV offset":"光伏抵消前碳排放强度 CEI","CEI after PV offset":"光伏抵消后碳排放强度 CEI",
  "CEI without rooftop PV":"无屋顶光伏的碳排放强度 CEI","CEI with full-roof PV":"屋顶满铺光伏后的碳排放强度 CEI",
  "Full-roof PV impact":"屋顶满铺光伏效益",
  "Enter the measured building height and number of floors to calculate annual indicators.":"请输入实测建筑高度和层数以计算年度指标。",
  "ECONOMIC AND CARBON SAVINGS":"经济与减碳效益",
  "Editable assessment assumptions":"可编辑的评价参数",
  "Electricity tariff":"购电电价","Feed-in tariff":"余电上网电价","Grid emission factor":"电网碳排放因子",
  "Gross electricity cost":"总购电费用","Total PV economic value":"光伏总经济价值",
  "Surplus PV export revenue":"余电上网收益","Net electricity cost":"净电费",
  "Cost saving rate":"费用节省率","Carbon reduction rate (CRR)":"减碳率（CRR）","PV yield intensity":"光伏发电强度",
  "Economic and carbon results are waiting for complete geometry.":"完成建筑几何输入后生成经济与减碳结果。",
  "Economic and carbon results are waiting for a completed prediction.":"完成模型预测后生成经济与减碳结果。",
  "EUI = annual electricity / total floor area. CEI = annual operational carbon / total floor area. CRR = (carbon before PV − carbon after PV) / carbon before PV × 100%. Net electricity cost = grid-import cost − surplus-PV export revenue.":"EUI＝年度用电量/总建筑面积；CEI＝年度运行碳排放/总建筑面积；CRR＝（光伏抵消前碳排放－抵消后碳排放）/抵消前碳排放×100%；净电费＝购电费用－余电上网收益。",
  "Complete parameters":"完整设计参数","Geometry, site, PV and envelope data":"几何、场地、光伏与围护结构数据",
  "Building height":"建筑高度","Measured total height in metres":"实测建筑总高度（米）","Enter measured height":"输入实测高度",
  "Number of floors":"建筑层数","Measured above-ground floor count":"实测地上层数","Enter floor count":"输入建筑层数",
  "Geometry input required":"需要补充几何参数","Geometry confirmed and linked":"几何参数已确认并联动",
  "The map supplies the footprint only. Prediction and final OBJ remain locked.":"地图仅提供建筑底面；完成高度和层数输入前不生成最终OBJ与预测结果。",
  "Floor area, mean floor height, façades, shape factor, windows, model inputs and 8,760-hour results have been recalculated.":"建筑面积、平均层高、立面、体形系数、窗户、模型输入及8,760小时结果均已联动重算。",
  "Length":"长度","Width":"宽度","Roof area":"屋顶面积","Floor area":"总建筑面积","levels":"层",
  "Geometry, site and photovoltaics":"几何、场地与光伏参数",
  "Roof PV scenario":"屋顶光伏情景",
  "Geometry source":"几何来源","OpenStreetMap footprint + user height / floors":"开放地图底面＋用户输入高度/层数","Research dataset":"研究数据集",
  "Longitude / Latitude":"经度 / 纬度","Orientation":"建筑朝向","Footprint area":"建筑底面积","Footprint perimeter":"底面周长",
  "Gross wall area":"外墙总面积","Window area (WWR 0.30)":"窗面积（窗墙比0.30）","Opaque wall area":"非透明外墙面积",
  "Window-to-wall ratio":"窗墙比","Floor-to-floor height":"层高","Building spacing":"建筑间距",
  "Local site envelope":"局部场地范围","Campus area":"校区面积","Local footprint density":"局部建筑密度","Campus density":"校区建筑密度",
  "Local FAR":"局部容积率","Campus FAR":"校区容积率","Context buildings":"周边建筑数量",
  "School type":"学校类型","Enclosure type":"围合方式","Merged massing parts":"合并体块数量","Window model":"窗户建模方式",
  "Per-floor modules · target WWR 0.30":"逐层模块化开窗 · 目标窗墙比0.30","Dataset geometry":"数据集几何",
  "Envelope and data quality":"围护结构与数据质量","Construction year":"建造年代",
  "Shape factor":"体形系数","Roof U-value":"屋面传热系数","Wall U-value":"外墙传热系数","Ground U-value":"地面传热系数",
  "Window U-value":"外窗传热系数","Window SHGC":"外窗太阳得热系数","Envelope age band":"围护结构年代分组","Height data quality":"高度数据质量",
  "Required user measurements missing":"缺少用户实测值","User-entered height and floor count":"用户输入的高度与层数","Research dataset record":"研究数据集记录",
  "Linked geometry and model-input record ready":"几何与模型输入记录已就绪","Waiting for required geometry inputs":"等待必要的几何参数",
  "Assigned from the dataset standard table":"由数据集标准表赋值","Year-and-shape prior; verification required":"基于年代与体形的先验值，需核验",
  "Export model-input JSON":"导出模型输入JSON","Select a campus and building to view all parameters":"选择校区和建筑后查看全部参数",
  "Upload a new teaching-building OBJ":"上传新建教学楼OBJ模型","The uploaded OBJ will be rebuilt here with floor-by-floor windows":"上传的OBJ将在此重建为逐层开窗模型","Waiting for a metric OBJ file":"等待上传采用米制单位的OBJ文件","Uploaded OBJ":"已上传OBJ",
  "New design input":"新建方案输入","Upload metric OBJ model":"上传米制OBJ模型","The platform extracts geometry, voxelizes it and rebuilds a windowed LoD1 model":"平台将提取几何、进行体素化，并重建带窗LoD1模型",
  "Project name":"项目名称","Orientation (°)":"建筑朝向（°）","Two-sided":"双面围合","Three-sided":"三面围合","Four-sided":"四面围合",
  "OBJ geometry":"OBJ几何","Dual-route output":"双路模型输出","Run dual-route prediction":"运行双路模型预测","Re-run dual-route prediction":"重新运行双路模型预测","Running annual and hourly models…":"正在运行年度与逐时模型……",
  "Waiting for OBJ, parameters and EPW":"等待OBJ、设计参数与EPW文件","Upload the OBJ and EPW, complete the design parameters, then run the two-route model.":"请上传OBJ与EPW文件、完善设计参数后运行双路模型。",
  "Uploaded OBJ + generated windows":"上传OBJ＋自动生成窗户","New-design inputs are isolated from the research records":"新建方案输入与研究数据记录相互独立","Run the new-design dual-route prediction to generate the 8,760-hour profile.":"运行新建方案双路预测后生成8,760小时性能曲线。",
  "INTERACTIVE 8,760-HOUR RESULTS":"交互式8,760小时结果","Annual hourly performance profile":"全年逐时性能曲线","Monthly performance summary":"逐月性能汇总",
  "Export 8,760h CSV":"导出8,760小时CSV","Energy (kWh)":"能耗（kWh）","PV (kWh)":"光伏发电（kWh）",
  "Carbon (kgCO₂)":"碳排放（kgCO₂）","Carbon without roof PV":"无屋顶光伏碳排放","Carbon with roof PV":"有屋顶光伏碳排放","Net cost (CNY)":"净电费（元）","Export revenue (CNY)":"余电上网收益（元）",
  "No-PV hourly carbon is shown separately with a dashed curve":"无光伏逐时碳排放独立显示为虚线","Full-roof-PV hourly carbon is shown separately with a solid curve":"屋顶满铺光伏后的逐时碳排放独立显示为实线",
  "Without rooftop PV":"无屋顶光伏","Annual carbon without PV":"无光伏年碳排放","With full-roof PV":"屋顶满铺光伏后",
  "Click the curve to inspect an hourly value":"点击曲线查看具体小时数值",
  "Complete the height and floor inputs above to generate the 8,760-hour prediction profile.":"完成上方建筑高度与层数输入后生成8,760小时预测曲线。",
  "Selected hour":"选中时刻","Annual maximum":"全年最大值","Annual minimum":"全年最小值",
  "Arch-EPVC Research Platform":"Arch-EPVC 研究平台","16 cities · 994 campuses · 3,360 teaching buildings":"16个城市 · 994个校区 · 3,360栋教学楼",
  "Click a building footprint on the map":"请在地图上点击建筑底面",
  "Retrieving and cleaning nearby building footprints…":"正在获取并清洗附近建筑底面……",
  "No valid public building footprint was found at this location.":"当前位置未找到有效的公共建筑底面。",
  "This footprint could not be cleaned. Select another building outline.":"该底面无法完成清洗，请选择其他建筑轮廓。",
  "Dataset building location · orange footprint extracted from the source OBJ":"数据集建筑位置 · 橙色轮廓提取自原始OBJ",
  "Select a building to load its campus OBJ":"选择建筑以加载校区OBJ",
  "Due South":"正南","Primary / secondary school":"中小学","Not specified":"未注明",
  "Row layout":"行列式","Two-sided enclosure":"双向围合式","Three-sided enclosure":"三面围合式","Four-sided enclosure":"四面围合式","Other enclosure":"其他围合方式",
  "Annual EUI, Epv and CEI are direct outputs from the first-route 3D-CNN–TabTransformer ensemble. The LSTM predicts only the 8,760-hour distribution, which is calibrated to those annual outputs. CRR = (carbon before PV − carbon after PV) / carbon before PV × 100%.":"年度 EUI、Epv 和 CEI 直接采用第一路 3D-CNN–TabTransformer 集成模型的预测结果。LSTM 只预测 8,760 小时分布，并校准至第一路年度总量。CRR＝（光伏抵消前碳排放－抵消后碳排放）/抵消前碳排放×100%。",
  "16 cities · 994 campuses · 3,360 school buildings":"16个城市 · 994个校区 · 3,360栋学校建筑",
  "annual mean":"年平均值","Min":"最小","Max":"最大"
};

function translateDynamic(value:string){
  let match=value.match(/^Campus (.+?) · (\d+) buildings · selected Building (\d+)$/);if(match)return `校区 ${match[1]} · 共${match[2]}栋 · 当前选择${match[3]}号楼`;
  match=value.match(/^Campus (.+?) · (.+)$/);if(match)return `校区 ${match[1]} · ${match[2]}`;
  match=value.match(/^Building (\d+)$/);if(match)return `${match[1]}号楼`;
  match=value.match(/^(.+?) · (\d+) buildings$/);if(match)return `${match[1]} · ${match[2]} 栋建筑`;
  match=value.match(/^(\d[\d,]*) hours$/);if(match)return `${match[1]} 小时`;
  match=value.match(/^(.+?) · (\d[\d,]*) hours$/);if(match)return `${match[1]} · ${match[2]} 小时`;
  match=value.match(/^(.+?) · automatic windowed LoD1 model$/);if(match)return `${match[1]} · 自动开窗LoD1模型`;
  match=value.match(/^User OBJ · (.+?) vertices · automatic windows at WWR (.+)$/);if(match)return `用户OBJ · ${match[1]}个顶点 · 按窗墙比${match[2]}自动开窗`;
  match=value.match(/^South ([\d.]+)° (East|West)$/);if(match)return `南偏${match[2]==="East"?"东":"西"} ${match[1]}°`;
  match=value.match(/^(.+?) selected · ([\d.]+) m² · (\d+) massing parts?$/);if(match)return `${match[1]} 已选择 · ${match[2]} m² · ${match[3]} 个体块`;
  match=value.match(/^(.+?) · (\d+) massing parts?$/);if(match)return `${match[1]} · ${match[2]} 个体块`;
  if(value.startsWith("Building service unavailable:"))return value.replace("Building service unavailable:","建筑数据服务暂不可用：");
  if(value.startsWith("Dataset geometry fallback ·"))return value.replace("Dataset geometry fallback ·","数据集几何回退 ·");
  return value;
}

function translateValue(value:string){return TEXT[value]??translateDynamic(value)}

function translateText(node:Text){
  const value=node.nodeValue??"",trimmed=value.trim();if(!trimmed)return;
  const translated=translateValue(trimmed);if(translated===trimmed)return;
  node.nodeValue=value.slice(0,value.indexOf(trimmed))+translated+value.slice(value.indexOf(trimmed)+trimmed.length);
}

function translateTree(root:Node){
  if(root instanceof Text){translateText(root);return}
  if(root instanceof Element){
    for(const attribute of ["placeholder","title","aria-label"]){const value=root.getAttribute(attribute);if(value){const translated=translateValue(value);if(translated!==value)root.setAttribute(attribute,translated)}}
  }
  const walker=document.createTreeWalker(root,NodeFilter.SHOW_TEXT);let node:Node|null;
  while((node=walker.nextNode()))translateText(node as Text);
}

export function DomesticChinese({enabled=false}:{enabled?:boolean}){
  useEffect(()=>{
    const hostname=window.location.hostname.toLowerCase(),forced=new URLSearchParams(window.location.search).get("lang")==="zh";
    if(!enabled&&!hostname.endsWith(".trycloudflare.com")&&!forced)return;
    document.documentElement.lang="zh-CN";document.title="Arch-EPVC · 学校建筑性能智能评价";
    translateTree(document.body);
    const observer=new MutationObserver(records=>{for(const record of records){if(record.type==="characterData")translateText(record.target as Text);record.addedNodes.forEach(translateTree)}});
    observer.observe(document.body,{subtree:true,childList:true,characterData:true});
    return()=>observer.disconnect();
  },[enabled]);
  return null;
}
