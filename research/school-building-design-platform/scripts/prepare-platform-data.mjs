import fs from "node:fs";
import path from "node:path";

const project = process.cwd();
const source = path.join(process.env.LOCALAPPDATA, "Temp", "school_platform_inspect");
const inputCsv = path.join(source, "data", "gh_building_energy_inputs.csv");
const schoolsCsv = path.join(source, "data", "schools_teaching_campus.csv");
const predictionCsv = path.resolve(project, "..", "outputs", "eui_restored", "experiment_outputs", "second_route_temporal_fusion", "test_predictions.csv");
const hourlyCsv = path.resolve(project, "..", "outputs", "eui_restored", "experiment_outputs", "hourly_sequence_multimodal", "example_SD0011_T001_1hao_hourly_prediction.csv");

function parseCsv(text) {
  const rows = [];
  let row = [], field = "", quoted = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (ch === '"') {
      if (quoted && text[i + 1] === '"') { field += '"'; i++; }
      else quoted = !quoted;
    } else if (ch === "," && !quoted) { row.push(field); field = ""; }
    else if ((ch === "\n" || ch === "\r") && !quoted) {
      if (ch === "\r" && text[i + 1] === "\n") i++;
      row.push(field); field = "";
      if (row.some((v) => v !== "")) rows.push(row);
      row = [];
    } else field += ch;
  }
  if (field || row.length) { row.push(field); rows.push(row); }
  const headers = rows.shift().map((h, i) => i === 0 ? h.replace(/^\uFEFF/, "") : h);
  return rows.map((r) => Object.fromEntries(headers.map((h, i) => [h, r[i] ?? ""])));
}

const read = (p) => parseCsv(fs.readFileSync(p, "utf8"));
const num = (v) => v === "" || v == null ? null : Number(v);

const schools = read(schoolsCsv);
const schoolById = new Map(schools.map((r) => [r.sample_uid, r]));
const preds = new Map(read(predictionCsv).map((r) => [r["教学楼序号"], r]));

const buildings = read(inputCsv).map((r) => {
  const school = schoolById.get(r.sample_uid) ?? {};
  const pred = preds.get(r.obj_object_name_prefix);
  return {
    id: r.obj_object_name_prefix,
    teachingId: r.teaching_building_id,
    campusId: r.sample_uid || r.teaching_building_id?.split("_")[0],
    campusBuildingNo: num(r.campus_building_no),
    sourceBuildingIds: r.source_building_ids || null,
    sourceBuildingCount: num(r.source_building_count),
    city: r.city,
    school: r.school_name,
    lat: num(school.lat), lon: num(school.lon),
    obj: r.obj_path?.replaceAll("\\", "/").replace(/^models\//, "/models/"),
    length: num(r.length_m), width: num(r.width_m), height: num(r.height_m),
    floors: num(r.estimated_floor_count), year: num(r.construction_year_numeric), constructionYearRaw: r.construction_year_raw || null,
    shapeFactor: num(r.shape_factor), orientation: num(r.orientation_deg),
    footprintArea: num(r.footprint_area_m2), buildingArea: num(r.building_area_m2), roofArea: num(r.roof_area_m2),
    southFacadeArea: num(r.south_facade_area_net_m2), wwr: num(r.wwr_assumption),
    roofPvArea: num(r.roof_pv_area_60pct_m2), facadePvArea: num(r.south_facade_pv_area_65pct_m2),
    enclosureType: r.A8_enclosure_type || null, confidence: r.confidence || null, filterReason: r.filter_reason || null,
    schoolType: school["A2.学校类型"] || null,
    averageFloorHeight: num(school["A6.建筑层高"]),
    buildingSpacing: num(school["A12.教学楼间距"]),
    campusArea: num(school["学校场地面积"]),
    campusDensity: num(school["A14.建筑密度"]),
    campusFar: num(school["容积率"]),
    envelopeBucket: r.envelope_standard_bucket || null, shapeBin: r.shape_factor_bin || null,
    thermal: {
      roofU: num(r.roof_u_w_m2k), wallU: num(r.wall_u_w_m2k), groundU: num(r.ground_u_w_m2k),
      windowU: num(r.window_u_w_m2k), shgc: num(r.window_shgc), status: r.envelope_param_status || "needs_user_assignment"
    },
    prediction: pred ? {
      trueEui: num(pred.true_EUI), annualEui: num(pred.temporal_fusion_EUI),
      trueEpv: num(pred.true_Epv), annualEpv: num(pred.temporal_fusion_Epv),
      trueCei: num(pred.true_CEI), annualCei: num(pred.temporal_fusion_CEI),
      firstEui: num(pred.first_route_EUI), firstEpv: num(pred.first_route_Epv), firstCei: num(pred.first_route_CEI)
    } : null
  };
}).filter((b) => Number.isFinite(b.lat) && Number.isFinite(b.lon));

const hourly = read(hourlyCsv).map((r) => ({
  hour: num(r.HourIndex), energy: num(r["pred_建筑逐时能耗"]), pv: num(r["pred_光伏系统逐时发电量"]), carbon: num(r["pred_逐时运行碳排放"])
}));

fs.mkdirSync(path.join(project, "public", "data"), { recursive: true });
fs.writeFileSync(path.join(project, "public", "data", "buildings.json"), JSON.stringify(buildings));
fs.writeFileSync(path.join(project, "public", "data", "demo-hourly.json"), JSON.stringify(hourly));
fs.writeFileSync(path.join(project, "public", "data", "dataset-summary.json"), JSON.stringify({
  generatedAt: new Date().toISOString(), buildings: buildings.length,
  campuses: new Set(buildings.map((b) => b.campusId)).size,
  cities: new Set(buildings.map((b) => b.city)).size,
  validatedPredictions: buildings.filter((b) => b.prediction).length,
  coordinateLevel: "campus"
}, null, 2));

console.log(`Prepared ${buildings.length} buildings and ${hourly.length} hourly records.`);
