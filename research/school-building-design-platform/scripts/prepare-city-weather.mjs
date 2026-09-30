import fs from "node:fs";
import path from "node:path";

const sources = {
  "济南市":"1.Jinan.epw", "青岛市":"2.Qingdao.epw", "淄博市":"3.Zibo.epw", "枣庄市":"4.Zaozhuang.epw",
  "东营市":"5.Dongying.epw", "烟台市":"6.Yantai.epw", "潍坊市":"7.Weifang.epw", "济宁市":"8.Jining.epw",
  "泰安市":"9.Taian.epw", "威海市":"10.Weihai.epw", "临沂市":"11.Linyi.epw", "德州市":"12.Dezhou.epw",
  "日照市":"13.Rizhao.epw", "滨州市":"14.Binzhou.epw", "聊城市":"15.Liaocheng.epw", "菏泽市":"16.Heze.epw",
};
const desktop = "C:/Users/DELL/Desktop";
const output = {};
for (const [city, filename] of Object.entries(sources)) {
  const lines = fs.readFileSync(path.join(desktop, filename), "utf8").replace(/^\uFEFF/, "").split(/\r?\n/).filter(Boolean);
  const points = lines.slice(8).map(line => line.split(",")).filter(row => row.length >= 22).slice(0, 8760).map((r, i) => [
    Math.floor(i/24)+1, Number(r[3]), Number(r[6]), Number(r[8]), Number(r[21]), Number(r[14]), Number(r[15]), Number(r[13]),
  ]);
  if (points.length !== 8760) throw new Error(`${filename}: ${points.length} rows`);
  output[city] = { filename, location: lines[0].split(",").slice(1,4).join(" · "), points };
}
const target = path.resolve("public/data/city-weather.json");
fs.mkdirSync(path.dirname(target), { recursive: true });
fs.writeFileSync(target, JSON.stringify(output));
console.log(`Prepared ${Object.keys(output).length} cities.`);
