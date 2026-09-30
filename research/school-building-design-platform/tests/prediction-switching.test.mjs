import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";

const buildings=JSON.parse(fs.readFileSync(new URL("../public/data/buildings.json",import.meta.url),"utf8"));
const weather=JSON.parse(fs.readFileSync(new URL("../public/data/city-weather.json",import.meta.url),"utf8"));

test("every dataset building has a traceable first-route prediction and static index",()=>{
  assert.equal(buildings.length,3360);
  assert.ok(buildings.every((building,index)=>building.prediction&&building.staticFeatureIndex===index));
  assert.ok(buildings.every(building=>building.prediction.source?.startsWith("full-")));
});

test("switching buildings changes annual model inputs and predictions",()=>{
  const first=buildings[0],sameCampus=buildings.find(building=>building.campusId===first.campusId&&building.id!==first.id);
  assert.ok(sameCampus);
  assert.notDeepEqual(
    [first.prediction.annualEui,first.prediction.annualEpv,first.prediction.annualCei],
    [sameCampus.prediction.annualEui,sameCampus.prediction.annualEpv,sameCampus.prediction.annualCei],
  );
  const bytes=fs.readFileSync(new URL("../public/models/second-route-static.f32",import.meta.url));
  const staticData=new Float32Array(bytes.buffer,bytes.byteOffset,bytes.byteLength/4);
  const vector=index=>Array.from(staticData.slice(index*259,(index+1)*259));
  assert.notDeepEqual(vector(first.staticFeatureIndex),vector(sameCampus.staticFeatureIndex));
});

test("each city weather sequence supplies exactly 8,760 second-route hours",()=>{
  const cities=new Set(buildings.map(building=>building.city));
  for(const city of cities)assert.equal(weather[city]?.points?.length,8760,city);
  assert.ok(fs.statSync(new URL("../public/models/second-route-hourly-lstm.onnx",import.meta.url)).size>500_000);
});
