import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);
  const fetchHandler = typeof worker === "function" ? worker : worker.fetch.bind(worker);
  return fetchHandler(
    new Request("http://localhost/", { headers: { accept: "text/html" } }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("server-renders the finished Arch-EPVC platform", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);
  const html = await response.text();
  assert.match(html, /<title>Arch-EPVC · Building Performance Intelligence<\/title>/i);
  assert.match(html, /From school buildings to/);
  assert.match(html, /ANNUAL CORE INDICATORS/);
  assert.match(html, /INTERACTIVE 8,760-HOUR RESULTS/);
  assert.match(html, /3D-CNN \+ TabTransformer \/ LSTM/);
  assert.doesNotMatch(html, /Your site is taking shape|react-loading-skeleton|codex-preview/i);
});

test("uses first-route annual outputs and the LSTM only for hourly profiles", async () => {
  const [platform, inference, page] = await Promise.all([
    readFile(new URL("../app/components/DesignPlatform.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/components/hourlyInference.ts", import.meta.url), "utf8"),
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
  ]);
  assert.match(platform, /calibrateHourlyToFirstRoute/);
  assert.match(platform, /first\?\.firstEui/);
  assert.match(platform, /prediction\.firstEpv/);
  assert.match(platform, /first\?\.firstCei/);
  assert.doesNotMatch(platform, /const annual=hourly\.reduce/);
  assert.match(inference, /second-route-hourly-lstm\.onnx/);
  assert.doesNotMatch(page, /codex-preview|_sites-preview/);
});

test("supports separate no-PV and full-roof-PV results across all modeled-building workflows", async () => {
  const [platform, chart, massing, campus] = await Promise.all([
    readFile(new URL("../app/components/DesignPlatform.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/components/HourlyChart.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/components/MassingViewer.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/components/ObjViewer.tsx", import.meta.url), "utf8"),
  ]);
  assert.match(platform, /No rooftop PV/);
  assert.match(platform, /Full-roof PV/);
  assert.match(platform, /carbonWithoutPv/);
  assert.match(platform, /carbonWithPv/);
  assert.match(platform, /Carbon without roof PV/);
  assert.match(platform, /Carbon with roof PV/);
  assert.match(platform, /Mapped-building PV scenario/);
  assert.match(platform, /Complete the building geometry before adding rooftop PV/);
  assert.match(platform, /useState<"none"\|"full">\("none"\)/);
  assert.match(platform, /activatePvScenario/);
  assert.match(platform, /next==="full"\?"carbonWithPv":"carbonWithoutPv"/);
  assert.match(platform, /pvScenario==="full"\?pvFull:pvWithout/);
  assert.match(platform, /pvScenario==="full"\?carbonWithPv:carbonWithoutPv/);
  assert.match(chart, /carbonWithoutPv/);
  assert.match(chart, /carbonWithPv/);
  assert.match(massing, /pvCoverage/);
  assert.match(platform, /pvCoverage=\{pvScenario==="full"\?1:0\}/);
  assert.match(campus, /showRoofPv/);
});

test("renders rooftop PV as a tilted module array and differentiates carbon scenarios by line style", async () => {
  const [array, chart] = await Promise.all([
    readFile(new URL("../app/components/roofPvArray.ts", import.meta.url), "utf8"),
    readFile(new URL("../app/components/HourlyChart.tsx", import.meta.url), "utf8"),
  ]);
  assert.match(array, /InstancedMesh/);
  assert.match(array, /degToRad\(8\)/);
  assert.match(array, /roof-boundary-full-coverage-pv-array/);
  assert.match(array, /roof's longest boundary/);
  assert.match(array, /MAX_MODULES=1600/);
  assert.doesNotMatch(array, /centers\.length<220/);
  assert.match(chart, /setLineDash\(isCarbonWithout\?\[6,4\]:\[\]\)/);
});

test("provides a dedicated Chinese mirror with localized dynamic and canvas labels", async () => {
  const [zhPage, platform, translator, chart, staticEntry, vercel] = await Promise.all([
    readFile(new URL("../app/zh/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/components/DesignPlatform.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/components/DomesticChinese.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/components/HourlyChart.tsx", import.meta.url), "utf8"),
    readFile(new URL("../static-site/src/main.tsx", import.meta.url), "utf8"),
    readFile(new URL("../vercel.json", import.meta.url), "utf8"),
  ]);
  assert.match(zhPage, /DesignPlatform language="zh"/);
  assert.match(platform, /href=\{language==="zh"\?"\/":"\/zh"\}/);
  assert.match(translator, /enabled=false/);
  assert.match(translator, /root instanceof Text/);
  assert.match(chart, /1月/);
  assert.match(chart, /最大值/);
  assert.match(chart, /最小值/);
  assert.match(staticEntry, /window\.location\.pathname/);
  assert.match(staticEntry, /DesignPlatform language=\{language\}/);
  assert.match(vercel, /"source": "\/zh"/);
});

test("offers hourly and monthly views with a monthly bar and point-line chart", async () => {
  const [platform, monthly] = await Promise.all([
    readFile(new URL("../app/components/DesignPlatform.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/components/MonthlyChart.tsx", import.meta.url), "utf8"),
  ]);
  assert.match(platform, /chartView/);
  assert.match(platform, /Hourly/);
  assert.match(platform, /Monthly/);
  assert.match(platform, /MonthlyChart/);
  assert.match(monthly, /fillRect/);
  assert.match(monthly, /lineTo/);
  assert.match(monthly, /ctx\.arc/);
  assert.match(monthly, /Monthly total/);
  assert.match(monthly, /Hourly mean/);
  assert.match(monthly, /Date\.UTC\(2025,0,1,hourIndex\)/);
  assert.doesNotMatch(monthly, /hourIndex\/24/);
});

test("preserves concave OBJ footprints and courtyard holes throughout reconstruction", async () => {
  const [inference, massing, pvArray, platform, mapTypes] = await Promise.all([
    readFile(new URL("../app/components/designInference.ts", import.meta.url), "utf8"),
    readFile(new URL("../app/components/MassingViewer.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/components/roofPvArray.ts", import.meta.url), "utf8"),
    readFile(new URL("../app/components/DesignPlatform.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/components/OpenBuildingMap.tsx", import.meta.url), "utf8"),
  ]);
  assert.match(inference, /extractBoundaryRings/);
  assert.match(inference, /obj-face-topology/);
  assert.match(inference, /footprintHoles/);
  assert.doesNotMatch(inference, /const hull=convexHull\(vertices\.map/);
  assert.match(mapTypes, /footprintHoles\?/);
  assert.match(massing, /shape\.holes\.push/);
  assert.match(massing, /\[xy,\.\.\.holeRings\]/);
  assert.match(pvArray, /holes:THREE\.Vector2\[\]\[\]=\[\]/);
  assert.match(pvArray, /!localHoles\.some/);
  assert.match(platform, /ShapeUtils\.triangulateShape\(xy\.map[\s\S]*holeXy\.map/);
});
