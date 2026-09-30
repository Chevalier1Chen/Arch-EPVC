"use client";

import { useEffect, useMemo, useState } from "react";
import { ShapeUtils, Vector2 } from "three";
import {
  Activity, Building2, CheckCircle2, ChevronRight, Database, Download,
  FileUp, Search, ShieldCheck, Sparkles,
} from "lucide-react";
import { HourlyChart } from "./HourlyChart";
import { MonthlyChart } from "./MonthlyChart";
import { DomesticChinese } from "./DomesticChinese";
import { OpenBuildingMap, type ExtractedBuilding } from "./OpenBuildingMap";
import { MassingViewer } from "./MassingViewer";
import { ObjViewer } from "./ObjViewer";
import { predictHourly } from "./hourlyInference";
import { parseDesignObj, predictDesign, type DesignGeometry, type DesignInputs } from "./designInference";
import type { Building, EpwData, HourlyPerformancePoint, HourlyPoint, Thermal } from "./types";

const fmt = new Intl.NumberFormat("zh-CN", { maximumFractionDigits: 2 });
const haversine = (a:number,b:number,c:number,d:number) => {
  const r=6371,p=Math.PI/180,da=(c-a)*p,db=(d-b)*p;
  const x=Math.sin(da/2)**2+Math.cos(a*p)*Math.cos(c*p)*Math.sin(db/2)**2;
  return 2*r*Math.asin(Math.sqrt(x));
};
type CampusRegistration={rotation:number;dx:number;dy:number};
const ringFrame=(ring:Array<[number,number]>,lat0:number,lon0:number)=>ring.slice(0,-1).map(p=>({x:(p[1]-lon0)*111320*Math.cos(lat0*Math.PI/180),y:(p[0]-lat0)*110540}));
const frameStats=(points:Array<{x:number;y:number}>)=>{const x=points.reduce((s,p)=>s+p.x,0)/points.length,y=points.reduce((s,p)=>s+p.y,0)/points.length;let xx=0,yy=0,xy=0;points.forEach(p=>{xx+=(p.x-x)**2;yy+=(p.y-y)**2;xy+=(p.x-x)*(p.y-y)});return {x,y,angle:.5*Math.atan2(2*xy,xx-yy)}};
function applyRegistration(massing:ExtractedBuilding,building:Building,registration?:CampusRegistration){if(!registration)return massing;const lat0=building.lat,lon0=building.lon,c=Math.cos(registration.rotation),s=Math.sin(registration.rotation),footprint=massing.footprint.map(p=>{const x=(p[1]-lon0)*111320*Math.cos(lat0*Math.PI/180),y=(p[0]-lat0)*110540,rx=x*c-y*s+registration.dx,ry=x*s+y*c+registration.dy;return [lat0+ry/110540,lon0+rx/(111320*Math.cos(lat0*Math.PI/180))] as [number,number]});return {...massing,footprint,heightSource:"Original OBJ footprint · map control-point registered"}}
function solveRegistration(source:ExtractedBuilding,target:ExtractedBuilding,building:Building):CampusRegistration{const lat0=building.lat,lon0=building.lon,a=frameStats(ringFrame(source.footprint,lat0,lon0)),b=frameStats(ringFrame(target.footprint,lat0,lon0));let rotation=b.angle-a.angle;while(rotation>Math.PI/2)rotation-=Math.PI;while(rotation<-Math.PI/2)rotation+=Math.PI;const c=Math.cos(rotation),s=Math.sin(rotation);return {rotation,dx:b.x-(a.x*c-a.y*s),dy:b.y-(a.x*s+a.y*c)}};

function parseEpw(file: File): Promise<EpwData> {
  return file.text().then((text) => {
    const lines=text.replace(/^\uFEFF/,"").split(/\r?\n/).filter(Boolean);
    const location=lines[0]?.split(",").slice(1,4).join(" · ")||"Unknown weather station";
    const rows=lines.slice(8).map(line=>line.split(",")).filter(r=>r.length>=22);
    if(rows.length<8760) throw new Error(`Only ${rows.length} valid hourly records were found; 8,760 are required.`);
    const points=rows.slice(0,8760).map((r,i)=>({
      day:Math.floor(i/24)+1,hour:Number(r[3]),temperature:Number(r[6]),
      humidity:Number(r[8]),wind:Number(r[21]),dni:Number(r[14]),
      dhi:Number(r[15]),ghi:Number(r[13]),
    }));
    if(points.some(p=>![p.temperature,p.humidity,p.wind,p.dni,p.dhi,p.ghi].every(Number.isFinite))) {
      throw new Error("The EPW contains weather fields that could not be parsed.");
    }
    return {name:file.name,location,rows:points.length,points};
  });
}

function thermalPrior(year:number, shape:number): Thermal {
  const base=year<1986?[1,1.5,.65,5.7,.82]:year<2006?[.7,1,.55,3.5,.7]:year<2016?[.45,.55,.4,2.5,.6]:[.3,.4,.3,1.8,.5];
  const factor=shape>.35?.92:shape>.25?.96:1;
  return {roofU:+(base[0]*factor).toFixed(2),wallU:+(base[1]*factor).toFixed(2),groundU:base[2],windowU:+(base[3]*factor).toFixed(2),shgc:base[4],status:"platform_prior"};
}

function scaledHourly(base:HourlyPoint[], building:Building|null, epw:EpwData|null):HourlyPoint[] {
  if(!base.length||!building?.prediction) return [];
  const eui=building.prediction.firstEui,cei=building.prediction.firstCei,epv=building.prediction.firstEpv;
  const area=Math.max(building.buildingArea,1);
  const maxGhi=epw?epw.points.reduce((m,x)=>Math.max(m,x.ghi),1):1;
  const shaped=base.map((p,i)=>{
    const solar=epw?Math.max(0,epw.points[i].ghi)/maxGhi:1;
    const weather=epw?.points[i];
    const climate=weather?Math.max(.55,(1+.035*Math.max(18-weather.temperature,0)+.03*Math.max(weather.temperature-26,0)+.002*Math.max(weather.humidity-60,0)-.01*Math.min(weather.wind,6))/1.32):1;
    return {hour:p.hour,energyWeight:p.energy*climate,pvWeight:p.pv*(epw?(.25+.75*solar):1),carbonWeight:p.carbon*climate};
  });
  const sums=shaped.reduce((a,p)=>[a[0]+p.energyWeight,a[1]+p.pvWeight,a[2]+p.carbonWeight],[0,0,0]);
  return shaped.map(p=>({
    hour:p.hour,
    energy:p.energyWeight*(eui*area/Math.max(sums[0],1e-9)),
    pv:p.pvWeight*(epv/Math.max(sums[1],1e-9)),
    carbon:p.carbonWeight*(cei*area/Math.max(sums[2],1e-9)),
  }));
}

function calibrateHourlyToFirstRoute(points:HourlyPoint[],building:Building|null):HourlyPoint[]{
  if(!points.length||!building?.prediction)return points;
  const area=Math.max(building.buildingArea,1),targets={
    energy:Math.max(building.prediction.firstEui,0)*area,
    pv:Math.max(building.prediction.firstEpv,0),
    carbon:Math.max(building.prediction.firstCei,0)*area,
  };
  const sums=points.reduce((a,p)=>({energy:a.energy+p.energy,pv:a.pv+p.pv,carbon:a.carbon+p.carbon}),{energy:0,pv:0,carbon:0});
  return points.map(p=>({hour:p.hour,
    energy:p.energy*targets.energy/Math.max(sums.energy,1e-9),
    pv:p.pv*targets.pv/Math.max(sums.pv,1e-9),
    carbon:p.carbon*targets.carbon/Math.max(sums.carbon,1e-9),
  }));
}

function footprintObj(building:ExtractedBuilding,wwr=.3){
  const ring=building.footprint.slice(0,-1),lat0=ring.reduce((s,p)=>s+p[0],0)/ring.length,lon0=ring.reduce((s,p)=>s+p[1],0)/ring.length;
  const xy=ring.map(p=>[(p[1]-lon0)*111320*Math.cos(lat0*Math.PI/180),(p[0]-lat0)*110540]);
  const holeXy=(building.footprintHoles??[]).map(hole=>hole.slice(0,-1).map(p=>[(p[1]-lon0)*111320*Math.cos(lat0*Math.PI/180),(p[0]-lat0)*110540]));
  const lines=[`# Arch-EPVC footprint extrusion with facade openings`,`# Z-up; height=${building.height}m; floors=${building.floors}; target_WWR=${wwr}`,`o ${building.osmId}_ROOF_AND_SLABS`];let vertex=0;
  const quad=(a:[number,number,number],b:[number,number,number],c:[number,number,number],d:[number,number,number])=>{for(const p of [a,b,c,d])lines.push(`v ${p[0].toFixed(5)} ${p[1].toFixed(5)} ${p[2].toFixed(5)}`);lines.push(`f ${vertex+1} ${vertex+2} ${vertex+3} ${vertex+4}`);vertex+=4};
  const triangle=(a:number[],b:number[],c:number[])=>{for(const p of [a,b,c])lines.push(`v ${p[0].toFixed(5)} ${p[1].toFixed(5)} ${p[2].toFixed(5)}`);lines.push(`f ${vertex+1} ${vertex+2} ${vertex+3}`);vertex+=3};
  const flat=[...xy,...holeXy.flat()],triangles=ShapeUtils.triangulateShape(xy.map(p=>new Vector2(p[0],p[1])),holeXy.map(hole=>hole.map(p=>new Vector2(p[0],p[1]))));triangles.forEach(t=>{triangle([...flat[t[2]],0],[...flat[t[1]],0],[...flat[t[0]],0]);triangle([...flat[t[0]],building.height],[...flat[t[1]],building.height],[...flat[t[2]],building.height])});
  lines.push(`o ${building.osmId}_OPAQUE_WALLS`);const floorHeight=building.height/Math.max(1,building.floors),windowHeight=floorHeight*.5,coverage=Math.min(.9,Math.max(.1,wwr/.5));
  const windows:Array<{a:[number,number,number];b:[number,number,number];c:[number,number,number];d:[number,number,number]}>=[];
  for(const wallRing of [xy,...holeXy]){const signedArea=wallRing.reduce((sum,p,i)=>sum+p[0]*wallRing[(i+1)%wallRing.length][1]-wallRing[(i+1)%wallRing.length][0]*p[1],0)/2;for(let i=0;i<wallRing.length;i++){const a=wallRing[i],b=wallRing[(i+1)%wallRing.length],dx=b[0]-a[0],dy=b[1]-a[1],length=Math.hypot(dx,dy);if(length<.5)continue;const ux=dx/length,uy=dy/length,modules=Math.max(1,Math.min(24,Math.floor(length/3.2))),desiredWindowWidth=length*coverage/modules,gap=Math.max(.08,(length-desiredWindowWidth*modules)/(modules+1)),windowWidth=Math.max(.1,(length-gap*(modules+1))/modules);
    const point=(distance:number,z:number):[number,number,number]=>[a[0]+ux*distance,a[1]+uy*distance,z],inward:[number,number]=signedArea>=0?[-uy,ux]:[uy,-ux],glassPoint=(distance:number,z:number):[number,number,number]=>[a[0]+ux*distance+inward[0]*.08,a[1]+uy*distance+inward[1]*.08,z];
    for(let floor=0;floor<building.floors;floor++){const z0=floor*floorHeight,z1=(floor+1)*floorHeight,wb=z0+floorHeight*.25,wt=Math.min(z1-.12,wb+windowHeight);quad(point(0,z0),point(length,z0),point(length,wb),point(0,wb));quad(point(0,wt),point(length,wt),point(length,z1),point(0,z1));
      let cursor=0;for(let module=0;module<modules;module++){const start=gap+module*(windowWidth+gap),end=Math.min(length-gap,start+windowWidth);if(start>cursor)quad(point(cursor,wb),point(start,wb),point(start,wt),point(cursor,wt));windows.push({a:glassPoint(start,wb),b:glassPoint(end,wb),c:glassPoint(end,wt),d:glassPoint(start,wt)});cursor=end}if(cursor<length)quad(point(cursor,wb),point(length,wb),point(length,wt),point(cursor,wt));}}}
  lines.push(`o ${building.osmId}_WINDOW_GLAZING`,`g WINDOWS_WWR_${Math.round(wwr*100)}`);for(const w of windows)quad(w.a,w.b,w.c,w.d);return lines.join("\n");
}

function fallbackDatasetMassing(building:Building):ExtractedBuilding{
  const lat0=building.lat,lon0=building.lon,length=Math.max(building.length||20,5),width=Math.max(building.width||10,5),angle=(90-(building.orientation||0))*Math.PI/180,c=Math.cos(angle),s=Math.sin(angle);
  const local=[[-length/2,-width/2],[length/2,-width/2],[length/2,width/2],[-length/2,width/2]];
  const footprint=local.map(([u,v])=>{const x=u*c-v*s,y=u*s+v*c;return [lat0+y/110540,lon0+x/(111320*Math.cos(lat0*Math.PI/180))] as [number,number]});
  footprint.push(footprint[0]);
  return {osmId:`FALLBACK-${building.id}`,footprint,footprintArea:building.footprintArea||length*width,length,width,orientation:building.orientation||0,perimeter:2*(length+width),height:Math.max(building.height||12,3),floors:Math.max(1,Math.round(building.floors||4)),heightSource:"Fallback rectangle: source OBJ footprint unavailable",mergedParts:Math.max(1,building.sourceBuildingCount||1)};
}

function sourceObjMassing(text:string,building:Building):ExtractedBuilding{
  let objectName="";const vertices:Array<[number,number,number]>=[],boundary:Array<[number,number,number]>=[];
  for(const raw of text.split(/\r?\n/)){const line=raw.trim();if(line.startsWith("o ")){objectName=line.slice(2).trim();continue}if(line.startsWith("v ")){const vertex=line.slice(2).trim().split(/\s+/).map(Number) as [number,number,number];if(!vertex.every(Number.isFinite))continue;if(objectName===building.id)vertices.push(vertex);if(objectName===`${building.campusId}_campus_boundary`)boundary.push(vertex)}}
  if(vertices.length<6)throw new Error(`Object ${building.id} was not found in its campus OBJ`);
  const minZ=Math.min(...vertices.map(v=>v[2])),bottom=vertices.filter(v=>Math.abs(v[2]-minZ)<.01),unique=bottom.filter((v,i)=>i===0||Math.hypot(v[0]-bottom[i-1][0],v[1]-bottom[i-1][1])>.001);
  if(unique.length>2&&Math.hypot(unique[0][0]-unique.at(-1)![0],unique[0][1]-unique.at(-1)![1])<.001)unique.pop();
  if(unique.length<3)throw new Error(`No valid bottom polygon was found for ${building.id}`);
  const anchors=boundary.length?boundary:unique,cx=anchors.reduce((s,v)=>s+v[0],0)/anchors.length,cy=anchors.reduce((s,v)=>s+v[1],0)/anchors.length,lat0=building.lat,lon0=building.lon;
  const footprint=unique.map(v=>[lat0+(v[1]-cy)/110540,lon0+(v[0]-cx)/(111320*Math.cos(lat0*Math.PI/180))] as [number,number]);footprint.push(footprint[0]);
  let area=0,perimeter=0;for(let i=0;i<unique.length;i++){const a=unique[i],b=unique[(i+1)%unique.length];area+=a[0]*b[1]-b[0]*a[1];perimeter+=Math.hypot(b[0]-a[0],b[1]-a[1])}
  return {osmId:`OBJ-${building.id}`,footprint,footprintArea:Math.abs(area)/2,length:building.length,width:building.width,orientation:building.orientation||0,perimeter,height:Math.max(building.height||Math.max(...vertices.map(v=>v[2]))-minZ,3),floors:Math.max(1,Math.round(building.floors||4)),heightSource:"Original OBJ footprint · campus-level georeferencing",mergedParts:Math.max(1,building.sourceBuildingCount||1)};
}

export function DesignPlatform({language="en"}:{language?:"en"|"zh"}){
  const [buildings,setBuildings]=useState<Building[]>([]);
  const [demo,setDemo]=useState<HourlyPoint[]>([]);
  const [lat,setLat]=useState("36.6700");
  const [lon,setLon]=useState("117.0200");
  const [campus,setCampus]=useState<string|null>(null);
  const [selectedId,setSelectedId]=useState<string|null>(null);
  const [mode,setMode]=useState<"map"|"design"|"existing">("existing");
  const [extracted,setExtracted]=useState<ExtractedBuilding|null>(null);
  const [generatedObjUrl,setGeneratedObjUrl]=useState<string|null>(null);
  const [cityFilter,setCityFilter]=useState("济南市");
  const [epw,setEpw]=useState<EpwData|null>(null);
  const [cityWeather,setCityWeather]=useState<Record<string,EpwData>>({});
  const [epwError,setEpwError]=useState("");
  const [metric,setMetric]=useState<"energy"|"pv"|"carbonWithoutPv"|"carbonWithPv"|"netCost"|"exportRevenue">("energy");
  const [chartView,setChartView]=useState<"hourly"|"monthly">("hourly");
  const [year,setYear]=useState("");
  const [mappedHeight,setMappedHeight]=useState("");
  const [mappedFloors,setMappedFloors]=useState("");
  const [tariff,setTariff]=useState(0.8);
  const [feedInTariff,setFeedInTariff]=useState(0.35);
  const [gridFactor,setGridFactor]=useState(0.57);
  const [pvScenario,setPvScenario]=useState<"none"|"full">("none");
  const [datasetFootprint,setDatasetFootprint]=useState<ExtractedBuilding|null>(null);
  const [datasetGeometryStatus,setDatasetGeometryStatus]=useState("Select a dataset building");
  const [registrations,setRegistrations]=useState<Record<string,CampusRegistration>>({});
  const [registrationMode,setRegistrationMode]=useState(false);
  const [datasetQuery,setDatasetQuery]=useState("");
  const [modelHourly,setModelHourly]=useState<HourlyPoint[]|null>(null);
  const [hourlyStatus,setHourlyStatus]=useState<"idle"|"running"|"ready"|"error">("idle");
  const [designGeometry,setDesignGeometry]=useState<DesignGeometry|null>(null);
  const [designError,setDesignError]=useState("");
  const [designStatus,setDesignStatus]=useState<"idle"|"running"|"ready"|"error">("idle");
  const [designPrediction,setDesignPrediction]=useState<Building["prediction"]>(null);
  const [designHourly,setDesignHourly]=useState<HourlyPoint[]>([]);
  const [designInputs,setDesignInputs]=useState<DesignInputs>({name:"New teaching building",floors:4,year:2025,enclosure:1,wwr:.30,pvRatio:1,orientation:0,thermal:{roofU:.30,wallU:.40,groundU:.30,windowU:1.80,shgc:.50,status:"user_input"}});

  useEffect(()=>{
    Promise.all([
      fetch("/data/buildings.json").then(r=>r.json() as Promise<Building[]>),
      fetch("/data/demo-hourly.json").then(r=>r.json() as Promise<HourlyPoint[]>),
      fetch("/data/city-weather.json").then(r=>r.json() as Promise<Record<string,{filename:string;location:string;points:number[][]}>>),
    ]).then(([b,h,w])=>{
      setBuildings(b);setDemo(h);
      const initial=b.find(x=>x.prediction)??b[0];
      if(initial){setCampus(initial.campusId);setSelectedId(initial.id);setCityFilter(initial.city);setLat(String(initial.lat));setLon(String(initial.lon));setMode("existing")}
      setCityWeather(Object.fromEntries(Object.entries(w).map(([city,v])=>[city,{name:v.filename,location:v.location,rows:v.points.length,points:v.points.map(p=>({day:p[0],hour:p[1],temperature:p[2],humidity:p[3],wind:p[4],dni:p[5],dhi:p[6],ghi:p[7]}))}])));
    });
  },[]);

  const campuses=useMemo(()=>{
    const map=new Map<string,Building>();
    buildings.forEach(b=>{if(b.campusId&&!map.has(b.campusId)) map.set(b.campusId,b)});
    return [...map.values()];
  },[buildings]);
  const nearest=useMemo(()=>{
    const a=Number(lat),o=Number(lon);
    if(!Number.isFinite(a)||!Number.isFinite(o)) return [];
    return campuses.map(b=>({b,d:haversine(a,o,b.lat,b.lon)})).sort((x,y)=>x.d-y.d).slice(0,5);
  },[campuses,lat,lon]);
  const cities=useMemo(()=>[...new Set(campuses.map(b=>b.city))].sort((a,b)=>a.localeCompare(b,"zh-CN")),[campuses]);
  const cityCampuses=useMemo(()=>campuses.filter(b=>b.city===cityFilter).sort((a,b)=>a.school.localeCompare(b.school,"zh-CN")),[campuses,cityFilter]);
  const visibleCampuses=useMemo(()=>{const q=datasetQuery.trim().toLowerCase();if(!q)return cityCampuses;return cityCampuses.filter(b=>`${b.campusId} ${b.school} ${b.id}`.toLowerCase().includes(q))},[cityCampuses,datasetQuery]);
  const campusBuildings=useMemo(()=>buildings.filter(b=>b.campusId===campus),[buildings,campus]);
  const selected=useMemo(()=>buildings.find(b=>b.id===selectedId)??campusBuildings[0]??null,[buildings,selectedId,campusBuildings]);

  const estimate=useMemo(()=>{
    const rows=nearest.flatMap(n=>buildings.filter(b=>b.campusId===n.b.campusId).map(b=>({b,w:1/(n.d+.5)})));
    const avg=(key:keyof Building)=>{const v=rows.filter(r=>Number.isFinite(Number(r.b[key])));const w=v.reduce((s,r)=>s+r.w,0);return w?v.reduce((s,r)=>s+Number(r.b[key])*r.w,0)/w:0};
    const predicted=rows.filter(r=>r.b.prediction);const pw=predicted.reduce((s,r)=>s+r.w,0);
    const pavg=(key:"annualEui"|"annualEpv"|"annualCei")=>pw?predicted.reduce((s,r)=>s+r.b.prediction![key]*r.w,0)/pw:0;
    const euis=predicted.map(r=>r.b.prediction!.annualEui);
    return rows.length?{
      samples:rows.length,campuses:nearest.length,length:avg("length"),width:avg("width"),height:avg("height"),floors:avg("floors"),
      footprintArea:avg("footprintArea"),buildingArea:avg("buildingArea"),roofArea:avg("roofArea"),shapeFactor:avg("shapeFactor"),
      orientation:avg("orientation"),wwr:avg("wwr"),roofPvArea:avg("roofPvArea"),eui:pavg("annualEui"),epv:pavg("annualEpv"),cei:pavg("annualCei"),
      euiLow:euis.length?Math.min(...euis):0,euiHigh:euis.length?Math.max(...euis):0,
    }:null;
  },[nearest,buildings]);

  useEffect(()=>{if(selected)setYear(String(selected.year??2020))},[selected]);
  const mapGeometryReady=mode==="design"?Boolean(designGeometry):!extracted||(Number(mappedHeight)>2&&Number(mappedFloors)>=1);
  const mappedHeightValue=Number(mappedHeight)>2?Number(mappedHeight):3.3;
  const mappedFloorsValue=Number(mappedFloors)>=1?Math.round(Number(mappedFloors)):1;
  const heightNeedsConfirmation=Boolean(extracted&&!mapGeometryReady);
  const evaluatedBuilding=useMemo<Building|null>(()=>{
    if(mode==="map"&&extracted){
      const height=Math.max(mappedHeightValue,3),floors=Math.max(mappedFloorsValue,1),area=extracted.footprintArea,shape=(2*area+extracted.perimeter*height)/(area*height),prediction=estimate&&estimate.eui>0?{trueEui:estimate.eui,annualEui:estimate.eui,trueEpv:estimate.epv,annualEpv:estimate.epv,trueCei:estimate.cei,annualCei:estimate.cei,firstEui:estimate.eui,firstEpv:estimate.epv,firstCei:estimate.cei}:null;
      const base:Building={id:extracted.osmId,teachingId:extracted.osmId,campusId:"MAP",city:cityFilter,school:"New mapped school building",campusBuildingNo:1,sourceBuildingIds:null,sourceBuildingCount:extracted.mergedParts,lat:Number(lat),lon:Number(lon),obj:"",length:extracted.length,width:extracted.width,height:extracted.height,floors:extracted.floors,year:Number(year)||2020,constructionYearRaw:year||"2020",shapeFactor:shape,orientation:extracted.orientation,footprintArea:area,buildingArea:area*extracted.floors,roofArea:area,southFacadeArea:0,wwr:.3,roofPvArea:area*.6,facadePvArea:0,enclosureType:extracted.enclosureType??"Row layout",confidence:"map-extracted",filterReason:null,schoolType:"Mapped school building",averageFloorHeight:extracted.height/extracted.floors,buildingSpacing:extracted.buildingSpacing??null,campusArea:extracted.localSiteArea??null,campusDensity:extracted.localDensity??null,campusFar:extracted.localFar??null,envelopeBucket:null,shapeBin:null,thermal:{roofU:null,wallU:null,groundU:null,windowU:null,shgc:null,status:"platform_prior"},prediction};
      return {...base,id:extracted.osmId,teachingId:extracted.osmId,city:cityFilter,lat:Number(lat),lon:Number(lon),length:extracted.length,width:extracted.width,height,floors,footprintArea:area,roofArea:area,buildingArea:area*floors,roofPvArea:area*.6,facadePvArea:extracted.width*height*.65,southFacadeArea:extracted.width*height*.7,wwr:.3,shapeFactor:shape,orientation:extracted.orientation,averageFloorHeight:height/floors,buildingSpacing:extracted.buildingSpacing??null,campusArea:extracted.localSiteArea??null,campusDensity:extracted.localDensity??null,campusFar:extracted.localFar??null,enclosureType:extracted.enclosureType??"Row layout",sourceBuildingCount:extracted.mergedParts,prediction};
    }
    if(mode==="design"&&designGeometry){const g=designGeometry.massing,floors=Math.max(1,Math.round(designInputs.floors)),height=g.height,area=g.footprintArea,shape=(2*area+g.perimeter*height)/(area*height);return {id:g.osmId,teachingId:g.osmId,campusId:"DESIGN",city:cityFilter,school:designInputs.name,campusBuildingNo:1,sourceBuildingIds:null,sourceBuildingCount:1,lat:Number(lat),lon:Number(lon),obj:"",length:g.length,width:g.width,height,floors,year:designInputs.year,constructionYearRaw:String(designInputs.year),shapeFactor:shape,orientation:designInputs.orientation,footprintArea:area,buildingArea:area*floors,roofArea:area,southFacadeArea:g.width*height,wwr:designInputs.wwr,roofPvArea:area*designInputs.pvRatio,facadePvArea:0,enclosureType:["Row layout","Two-sided enclosure","Three-sided enclosure","Four-sided enclosure"][designInputs.enclosure-1]??"Row layout",confidence:"user-design",filterReason:null,schoolType:"New teaching building",averageFloorHeight:height/floors,buildingSpacing:null,campusArea:null,campusDensity:null,campusFar:null,envelopeBucket:"User-defined",shapeBin:null,thermal:designInputs.thermal,prediction:designPrediction};}
    if(!selected)return null;let value=selected;
    if(mode==="map"&&estimate)value={...value,length:estimate.length,width:estimate.width,height:estimate.height,floors:estimate.floors,footprintArea:estimate.footprintArea,buildingArea:estimate.buildingArea,roofArea:estimate.roofArea,shapeFactor:estimate.shapeFactor,orientation:estimate.orientation,wwr:estimate.wwr,roofPvArea:estimate.roofPvArea,prediction:{trueEui:estimate.eui,annualEui:estimate.eui,trueEpv:estimate.epv,annualEpv:estimate.epv,trueCei:estimate.cei,annualCei:estimate.cei,firstEui:estimate.eui,firstEpv:estimate.epv,firstCei:estimate.cei}};
    return value;
  },[mode,selected,estimate,extracted,cityFilter,lat,lon,year,mappedHeightValue,mappedFloorsValue,designGeometry,designInputs,designPrediction]);
  const thermal=evaluatedBuilding?(mode==="design"?designInputs.thermal:mode==="map"?thermalPrior(Number(year)||2020,evaluatedBuilding.shapeFactor):evaluatedBuilding.thermal.wallU?evaluatedBuilding.thermal:thermalPrior(Number(year)||2020,evaluatedBuilding.shapeFactor)):null;
  const inputMassing=mode==="design"?designGeometry?.massing??null:extracted;
  const modelInput=useMemo(()=>evaluatedBuilding&&thermal?{
    building_id:evaluatedBuilding.teachingId,City:cityFilter,
    "A.Building area":evaluatedBuilding.buildingArea,"B.Building footprint":evaluatedBuilding.footprintArea,"C.Building height":evaluatedBuilding.height,"D.Layer":evaluatedBuilding.floors,"E.Height":evaluatedBuilding.height/evaluatedBuilding.floors,
    "F.Building length":evaluatedBuilding.length,"G.Building width":evaluatedBuilding.width,"H.Orientation":evaluatedBuilding.orientation,"I.Enclosure method":enclosureGroup(evaluatedBuilding.enclosureType),
    "J.Shape coefficient":evaluatedBuilding.shapeFactor,"K.Roof thermal coefficient":thermal.roofU,"L.Wall thermal coefficient":thermal.wallU,"M.Ground thermal coefficient":thermal.groundU,"N.Window U-value":thermal.windowU,
    construction_year_user:mode==="design"?designInputs.year:Number(year)||2020,"屋顶光伏发电总面积":pvScenario==="full"?evaluatedBuilding.roofArea:0,roof_pv_area:pvScenario==="full"?evaluatedBuilding.roofArea:0,window_to_wall_ratio:evaluatedBuilding.wwr,
    average_floor_height_m:evaluatedBuilding.averageFloorHeight,
    derived_gross_wall_area_m2:inputMassing?inputMassing.perimeter*evaluatedBuilding.height:null,
    derived_window_area_m2:inputMassing?inputMassing.perimeter*evaluatedBuilding.height*evaluatedBuilding.wwr:null,
    derived_opaque_wall_area_m2:inputMassing?inputMassing.perimeter*evaluatedBuilding.height*(1-evaluatedBuilding.wwr):null,
    longitude:Number(lon),latitude:Number(lat),geometry_source:mode==="design"?"User-uploaded OBJ + automatically generated facade windows":extracted?"OpenStreetMap footprint + user height and floors":"Research dataset OBJ",context_buildings:extracted?.contextBuildingCount??null,
  }:null,[evaluatedBuilding,thermal,cityFilter,year,lon,lat,extracted,mode,designInputs.year,inputMassing,pvScenario]);
  useEffect(()=>{let disposed=false;if(mode!=="existing"||!selected?.obj){setDatasetFootprint(null);return}setDatasetFootprint(null);setDatasetGeometryStatus("Loading the selected research-dataset OBJ geometry…");fetch(selected.obj).then(r=>{if(!r.ok)throw new Error(`HTTP ${r.status}`);return r.text()}).then(text=>{if(disposed)return;const registered=registrations[selected.campusId],massing=applyRegistration(sourceObjMassing(text,selected),selected,registered),ring=massing.footprint.slice(0,-1),centerLat=ring.reduce((s,p)=>s+p[0],0)/ring.length,centerLon=ring.reduce((s,p)=>s+p[1],0)/ring.length;setDatasetFootprint(massing);setLat(centerLat.toFixed(7));setLon(centerLon.toFixed(7));setDatasetGeometryStatus("Research dataset · original merged OBJ and stored geometry attributes")}).catch(error=>{if(disposed)return;setDatasetFootprint(fallbackDatasetMassing(selected));setDatasetGeometryStatus(`Dataset geometry fallback · ${String(error)}`)});return()=>{disposed=true}},[mode,selected,registrations]);
  const viewerMassing=useMemo(()=>mode==="design"&&designGeometry?{...designGeometry.massing,floors:Math.max(1,Math.round(designInputs.floors))}:extracted?{...extracted,height:mappedHeightValue,floors:mappedFloorsValue}:(mode==="existing"?datasetFootprint:null),[extracted,mode,datasetFootprint,mappedHeightValue,mappedFloorsValue,designGeometry,designInputs.floors]);
  const roofPvReady=mode==="existing"?Boolean(selected):Boolean(viewerMassing)&&mapGeometryReady;
  useEffect(()=>{if(!viewerMassing||(extracted&&!mapGeometryReady)){setGeneratedObjUrl(null);return}const wwr=mode==="design"?designInputs.wwr:.3,url=URL.createObjectURL(new Blob([footprintObj(viewerMassing,wwr)],{type:"text/plain"}));setGeneratedObjUrl(url);return()=>URL.revokeObjectURL(url)},[viewerMassing,extracted,mapGeometryReady,mode,designInputs.wwr]);
  useEffect(()=>{if(mode!=="map"||!extracted||!nearest[0])return;const inferredCity=nearest[0].b.city;if(inferredCity!==cityFilter){setCityFilter(inferredCity);setEpw(null)}},[mode,extracted,nearest,cityFilter]);
  const activeWeather=mode==="design"?epw:epw??cityWeather[cityFilter]??null;
  const weatherStats=useMemo(()=>{if(!activeWeather?.points.length)return [];const specs=[
    {key:"temperature",label:"Temperature",unit:"°C",color:"#d97706"},{key:"humidity",label:"Humidity",unit:"%",color:"#2563eb"},{key:"wind",label:"Wind speed",unit:"m/s",color:"#0f766e"},
    {key:"dni",label:"DNI",unit:"W/m²",color:"#c2410c"},{key:"dhi",label:"DHI",unit:"W/m²",color:"#7c3aed"},{key:"ghi",label:"GHI",unit:"W/m²",color:"#16a34a"},
  ] as const;return specs.map(spec=>{const values=activeWeather.points.map(point=>point[spec.key]).filter(Number.isFinite),sum=values.reduce((a,b)=>a+b,0);return {...spec,average:sum/values.length,min:Math.min(...values),max:Math.max(...values),values}})},[activeWeather]);
  useEffect(()=>{
    let cancelled=false;
    setModelHourly(null);
    if(mode!=="existing"||!evaluatedBuilding||!activeWeather||evaluatedBuilding.staticFeatureIndex===undefined){setHourlyStatus("idle");return()=>{cancelled=true}}
    setHourlyStatus("running");
    predictHourly(evaluatedBuilding,activeWeather).then(points=>{if(!cancelled){setModelHourly(points);setHourlyStatus("ready")}}).catch(error=>{if(!cancelled){console.error("Second-route hourly inference failed",error);setHourlyStatus("error")}});
    return()=>{cancelled=true};
  },[mode,evaluatedBuilding?.id,evaluatedBuilding?.staticFeatureIndex,activeWeather]);
  const hourly=useMemo(()=>mode==="design"?designHourly:modelHourly?calibrateHourlyToFirstRoute(modelHourly,evaluatedBuilding):scaledHourly(demo,evaluatedBuilding,activeWeather),[mode,designHourly,modelHourly,demo,evaluatedBuilding,activeWeather]);
  const fullRoofPvScale=useMemo(()=>{if(!evaluatedBuilding)return 1;const modeledArea=Math.max(evaluatedBuilding.roofPvArea,0),fullArea=Math.max(evaluatedBuilding.roofArea,0);if(mode==="design")return 1;if(modeledArea>1e-6)return Math.max(fullArea/modeledArea,0);return fullArea>0?1:0},[evaluatedBuilding,mode]);
  const performanceHourly=useMemo<HourlyPerformancePoint[]>(()=>hourly.map(p=>{
    const pvFull=Math.max(p.pv*fullRoofPvScale,0),pvWithout=0,carbonWithoutPv=Math.max(p.carbon,0),carbonWithPv=Math.max(carbonWithoutPv-Math.min(p.energy,pvFull)*Math.max(gridFactor,0),0),pv=pvScenario==="full"?pvFull:pvWithout,carbon=pvScenario==="full"?carbonWithPv:carbonWithoutPv;
    const gridImport=Math.max(p.energy-pv,0),surplusExport=Math.max(pv-p.energy,0),exportRevenue=surplusExport*Math.max(feedInTariff,0);
    return {...p,pv,carbon,pvWithout,pvFull,carbonWithoutPv,carbonWithPv,exportRevenue,netCost:gridImport*Math.max(tariff,0)-exportRevenue};
  }),[hourly,fullRoofPvScale,gridFactor,pvScenario,tariff,feedInTariff]);
  const annualAssessment=useMemo(()=>{
    const area=Math.max(evaluatedBuilding?.buildingArea??0,1);
    const first=evaluatedBuilding?.prediction;
    const eui=Math.max(first?.firstEui??0,0),ceiBefore=Math.max(first?.firstCei??0,0);
    const annualEnergy=eui*area,annualCarbon=ceiBefore*area;
    const hourlyAccounting=performanceHourly.reduce((a,p)=>({fullPv:a.fullPv+p.pvFull,selfConsumed:a.selfConsumed+Math.min(p.energy,p.pvFull),exportRevenue:a.exportRevenue+p.exportRevenue,netCost:a.netCost+p.netCost,avoidedCarbon:a.avoidedCarbon+Math.max(p.carbonWithoutPv-p.carbonWithPv,0)}),{fullPv:0,selfConsumed:0,exportRevenue:0,netCost:0,avoidedCarbon:0});
    const usedPv=pvScenario==="full"?hourlyAccounting.selfConsumed:0;
    const avoidedCarbon=hourlyAccounting.avoidedCarbon;
    const carbonAfterPv=Math.max(annualCarbon-avoidedCarbon,0);
    const grossCost=annualEnergy*Math.max(tariff,0);
    const pvValue=pvScenario==="full"?usedPv*Math.max(tariff,0)+hourlyAccounting.exportRevenue:0;
    const fullCrr=annualCarbon>0?Math.min(avoidedCarbon/annualCarbon*100,100):0;
    return {
      eui,epv:pvScenario==="full"?hourlyAccounting.fullPv:0,epvFull:hourlyAccounting.fullPv,epvIntensity:pvScenario==="full"?hourlyAccounting.fullPv/area:0,
      ceiBefore,ceiAfter:carbonAfterPv/area,
      grossCost,pvValue,exportRevenue:pvScenario==="full"?hourlyAccounting.exportRevenue:0,netCost:hourlyAccounting.netCost,
      savingRate:grossCost>0?(grossCost-hourlyAccounting.netCost)/grossCost*100:0,
      crr:pvScenario==="full"?fullCrr:0,fullCrr,
      carbonReduction:avoidedCarbon,
    };
  },[evaluatedBuilding?.buildingArea,evaluatedBuilding?.prediction,tariff,performanceHourly,pvScenario]);
  const predictionReady=Boolean(evaluatedBuilding?.prediction)&&mapGeometryReady;

  const selectCampus=(id:string,nextMode:"map"|"existing"="existing")=>{
    setMode(nextMode);if(nextMode==="existing")setExtracted(null);setCampus(id);
    const rows=buildings.filter(b=>b.campusId===id);
    const target=rows.find(b=>b.prediction)??rows[0];setSelectedId(target?.id??null);
    if(nextMode==="existing"&&target){setLat(String(target.lat));setLon(String(target.lon));setCityFilter(target.city)}
  };
  const selectDatasetBuilding=(building:Building)=>{setMode("existing");setExtracted(null);setSelectedId(building.id);setLat(String(building.lat));setLon(String(building.lon));setCityFilter(building.city)};
  const selectCampusModelBuilding=(id:string)=>{const building=buildings.find(item=>item.id===id);if(building)selectDatasetBuilding(building)};
  const activatePvScenario=(next:"none"|"full")=>{setPvScenario(next);setMetric(current=>current==="carbonWithoutPv"||current==="carbonWithPv"?(next==="full"?"carbonWithPv":"carbonWithoutPv"):current)};
  const selectCity=(city:string)=>{
    setCityFilter(city);setEpw(null);setMode("existing");
    const first=campuses.find(b=>b.city===city);
    if(first){setLat(String(first.lat));setLon(String(first.lon));selectCampus(first.campusId,"existing")}
  };
  const analyseCoordinates=()=>{
    if(!nearest[0])return;
    setExtracted(null);
    setCityFilter(nearest[0].b.city);setEpw(null);
    selectCampus(nearest[0].b.campusId,"map");
  };
  const updateDesign=(patch:Partial<DesignInputs>)=>{setDesignInputs(current=>({...current,...patch}));setDesignPrediction(null);setDesignHourly([]);setDesignStatus("idle")};
  const updateDesignThermal=(key:Exclude<keyof Thermal,"status">,value:number)=>{setDesignInputs(current=>({...current,thermal:{...current.thermal,[key]:value,status:"user_input"}}));setDesignPrediction(null);setDesignHourly([]);setDesignStatus("idle")};
  const uploadDesignObj=async(file:File)=>{try{setDesignError("");setDesignStatus("idle");setDesignPrediction(null);setDesignHourly([]);const geometry=await parseDesignObj(file,Number(lat)||36.67,Number(lon)||117.02);setDesignGeometry(geometry);setDesignInputs(current=>({...current,name:file.name.replace(/\.[^.]+$/,"")||current.name,floors:Math.max(1,Math.round(geometry.sourceHeight/3.6)),orientation:geometry.massing.orientation}))}catch(error){setDesignGeometry(null);setDesignError(String(error))}};
  const runDesignPrediction=async()=>{if(!designGeometry){setDesignError("Upload a metric OBJ model first.");return}if(!epw){setDesignError("Upload the project's 8,760-hour EPW file before prediction.");return}try{setDesignError("");setDesignStatus("running");const result=await predictDesign(designGeometry,{...designInputs,pvRatio:1},epw);setDesignPrediction(result.prediction);setDesignHourly(result.hourly);setDesignStatus("ready")}catch(error){console.error(error);setDesignError(String(error));setDesignStatus("error")}};
  const alignDatasetToMap=(target:ExtractedBuilding)=>{if(!registrationMode||!selected||!datasetFootprint)return false;const registration=solveRegistration(datasetFootprint,target,selected);setRegistrations(current=>({...current,[selected.campusId]:registration}));setRegistrationMode(false);setDatasetGeometryStatus("Applying map control-point registration to this campus…");return true};
  const exportCsv=()=>{
    if(!hourly.length)return;
    const head="HourIndex,pred_energy_kWh,PV_no_rooftop_kWh,PV_full_roof_kWh,carbon_no_PV_kgCO2,carbon_full_roof_PV_kgCO2,active_scenario_PV_kWh,active_scenario_carbon_kgCO2,net_electricity_cost_CNY,surplus_PV_export_revenue_CNY\n";
    const body=performanceHourly.map(p=>`${p.hour},${p.energy},${p.pvWithout},${p.pvFull},${p.carbonWithoutPv},${p.carbonWithPv},${p.pv},${p.carbon},${p.netCost},${p.exportRevenue}`).join("\n");
    const a=document.createElement("a");a.href=URL.createObjectURL(new Blob(["\uFEFF"+head+body],{type:"text/csv"}));
    a.download=`${selected?.id??"new-school"}_8760h_prediction.csv`;a.click();URL.revokeObjectURL(a.href);
  };
  const exportModelInput=()=>{if(!modelInput)return;const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([JSON.stringify(modelInput,null,2)],{type:"application/json"}));a.download=`${evaluatedBuilding?.teachingId??"new-school"}_model_input.json`;a.click();URL.revokeObjectURL(a.href)};

  return <main><DomesticChinese enabled={language==="zh"}/>
    <header className="topbar">
      <div className="brand"><span className="brandmark"><Building2 size={20}/></span><span>Arch-EPVC</span><small>Building Performance Intelligence</small></div>
      <div className="status"><span className="dot"/> Shandong dataset connected <b>3,360 buildings</b></div>
      <a className="language-switch" href={language==="zh"?"/":"/zh"} hrefLang={language==="zh"?"en":"zh-CN"}>{language==="zh"?"English":"中文"}</a><button className="ghost"><ShieldCheck size={16}/> Model status</button>
    </header>
    <section className="hero">
      <div><div className="eyebrow"><Sparkles size={14}/> MULTIMODAL DESIGN INTELLIGENCE</div><h1>From school buildings to<br/><em>energy—PV—carbon</em> digital profiles</h1><p>Evaluate an existing school from the research database or map, or upload a new OBJ design with project parameters and an EPW file for native dual-route prediction.</p></div>
      <div className="hero-flow"><span>Select</span><ChevronRight/><span>Model</span><ChevronRight/><span>Evaluate</span></div>
    </section>
    <section className="evaluation-modes"><button className={mode!=="design"?"active":""} onClick={()=>setMode("existing")}><Database size={18}/><span><b>Existing school assessment</b><small>Research database or map selection</small></span></button><button className={mode==="design"?"active":""} onClick={()=>{setMode("design");setExtracted(null);setEpw(null)}}><FileUp size={18}/><span><b>New design assessment</b><small>OBJ + design parameters + EPW</small></span></button></section>
    <section className="workspace">
      <div className="left-rail">
      <aside className="panel locator">
        <div className="panel-title"><span>01</span><div><h2>Teaching building database</h2><p>994 campuses · 3,360 verified building records</p></div></div>
        <div className="school-directory"><div className="mode-label">City and campus directory</div><select value={cityFilter} onChange={e=>selectCity(e.target.value)}>{cities.map(c=><option value={c} key={c}>{cityName(c)}</option>)}</select><label className="dataset-search"><Search size={14}/><input value={datasetQuery} onChange={e=>setDatasetQuery(e.target.value)} placeholder="Search campus or school"/></label><div className="school-list">{visibleCampuses.map(s=><button className={campus===s.campusId?"active":""} key={s.campusId} onClick={()=>selectCampus(s.campusId,"existing")}><span>{s.school||`Campus ${s.campusId}`}</span><small>{s.campusId} · {buildings.filter(b=>b.campusId===s.campusId).length} buildings</small></button>)}</div></div>
        <div className="note"><Database size={15}/><span>Geometry and attributes are read from the formal research package. The map is no longer used to create or estimate buildings.</span></div>
      </aside>
      <section className="panel epw-compact">
        <div className="panel-title"><span>00</span><div><h2>EPW weather input</h2><p>8,760-hour climate sequence</p></div></div>
        <label className="upload"><strong className="manual-title">Custom EPW file <em>{mode==="design"?"(Required for new design)":"(User input, optional)"}</em></strong><FileUp size={20}/><b>{epw?.name||activeWeather?.name||"Select an .epw file"}</b><small>{activeWeather?`${activeWeather.location} · ${activeWeather.rows} hours`:"EnergyPlus Weather File"}</small><input type="file" accept=".epw" onChange={async e=>{const f=e.target.files?.[0];if(!f)return;try{setEpwError("");const parsed=await parseEpw(f);setEpw(parsed);setDesignPrediction(null);setDesignHourly([]);setDesignStatus("idle")}catch(err){setEpwError(String(err))}}}/></label>
        {weatherStats.length?<div className="weather-summary-grid">{weatherStats.map(({key,...stat})=><WeatherMini key={key} {...stat}/>)}</div>:<div className="epw-fields">{["Temperature","Humidity","Wind","DNI","DHI","GHI"].map(x=><span key={x}><CheckCircle2 size={11}/>{x}</span>)}</div>}
        {epwError&&<p className="error">{epwError}</p>}
      </section>
      </div>
      <section className="panel viewer-panel">
        <div className="panel-head"><div><span className="kicker">CAMPUS DIGITAL MODEL</span><h2>{mode==="design"?(designGeometry?`${designInputs.name} · automatic windowed LoD1 model`:"Upload a new teaching-building OBJ"):extracted?`${extracted.osmId} · ${extracted.mergedParts} massing part${extracted.mergedParts>1?"s":""}`:mode==="map"?"Select a map footprint":selected?`Campus ${selected.campusId} · ${campusBuildings.length} buildings · selected Building ${selected.campusBuildingNo}`:"Select a building"}</h2></div>{generatedObjUrl&&mode!=="existing"?<a className="obj-download" href={generatedObjUrl} download={`${viewerMassing?.osmId??"Arch-EPVC-building"}-windowed.obj`}>Download windowed OBJ</a>:extracted?<span className="confidence warning">Height and floors required</span>:selected&&<span className="confidence"><CheckCircle2 size={14}/> {selected.confidence||"dataset matched"}</span>}</div>
        <div className="pv-scenario-bar"><div><span>{mode==="design"?"New-build PV scenario":mode==="map"?"Mapped-building PV scenario":"Retrofit PV scenario"}</span><small>{roofPvReady?"Switch the active assessment, 3D roof, PV profile and carbon profile together":"Complete the building geometry before adding rooftop PV"}</small></div><div className="pv-scenario-buttons" role="group" aria-label="Rooftop PV scenario"><button disabled={!roofPvReady} className={pvScenario==="none"?"active":""} onClick={()=>activatePvScenario("none")}>No rooftop PV</button><button disabled={!roofPvReady} className={pvScenario==="full"?"active":""} onClick={()=>activatePvScenario("full")}>Full-roof PV</button></div></div>
        {mode==="existing"&&selected?.obj?<ObjViewer url={selected.obj} buildings={campusBuildings} selectedId={selected.id} onSelectBuilding={selectCampusModelBuilding} showRoofPv={pvScenario==="full"}/>:viewerMassing?<MassingViewer building={viewerMassing} wwr={mode==="design"?designInputs.wwr:.3} pvCoverage={pvScenario==="full"?1:0}/>:<div className="obj-viewer"><div className="viewer-empty">{mode==="existing"?datasetGeometryStatus:mode==="design"?"The uploaded OBJ will be rebuilt here with floor-by-floor windows":"Click the map and select a building footprint"}</div></div>}
        <div className="geometry-source-line"><span>{mode==="design"?(designGeometry?`User OBJ · ${designGeometry.vertexCount} vertices · automatic windows at WWR ${designInputs.wwr.toFixed(2)}`:"Waiting for a metric OBJ file"):extracted?(mapGeometryReady?"Map footprint · user-confirmed height and floors":"Footprint preview · enter height and floors to generate the final OBJ"):datasetGeometryStatus}</span></div>
        {evaluatedBuilding&&<div className="viewer-facts"><div><span>Footprint</span><b>{fmt.format(evaluatedBuilding.footprintArea)} <small>m²</small></b></div><div><span>Height</span><b>{fmt.format(evaluatedBuilding.height)} <small>m</small></b></div><div><span>Floors</span><b>{fmt.format(evaluatedBuilding.floors)} <small>levels</small></b></div><div><span>Window ratio</span><b>{fmt.format(evaluatedBuilding.wwr)} <small>WWR</small></b></div><div><span>Geometry</span><b>{mode==="design"?"Uploaded OBJ":extracted?"OSM":"Source OBJ"}</b></div></div>}
        {mode==="existing"&&<div className="building-tabs">{campusBuildings.map(b=><button className={selected?.id===b.id?"active":""} onClick={()=>selectDatasetBuilding(b)} key={b.id} title={`${b.id} · ${b.floors} floors`}>{`Building ${b.campusBuildingNo}`}{b.prediction&&<i/>}</button>)}</div>}
      </section>
      <section className="map-section panel dataset-record"><div className="map-section-head"><div><span className="kicker">BUILDING SOURCE</span><h2>{mode==="design"?"New design input":mode==="map"?"Select an existing building from the map":selected?.school||"Select a teaching building"}</h2></div>{mode!=="design"&&<div className="source-switch"><button className={mode==="existing"?"active":""} onClick={()=>selected?selectDatasetBuilding(selected):setMode("existing")}>Research database</button><button className={mode==="map"?"active":""} onClick={()=>{setMode("map");setExtracted(null);setMappedHeight("");setMappedFloors("");setYear("")}}>Map selection</button></div>}</div>{mode==="design"?<DesignInputPanel geometry={designGeometry} inputs={designInputs} pvScenario={pvScenario} epwReady={Boolean(epw)} status={designStatus} error={designError} onFile={uploadDesignObj} onChange={updateDesign} onThermal={updateDesignThermal} onPredict={runDesignPrediction}/>:mode==="map"?<OpenBuildingMap lat={Number(lat)||36.67} lon={Number(lon)||117.02} datasetFootprint={null} onPick={(a,o)=>{setLat(a.toFixed(7));setLon(o.toFixed(7));setExtracted(null);setMappedHeight("");setMappedFloors("")} } onBuilding={b=>{setExtracted(b);setMappedHeight("");setMappedFloors("");setMode("map")}}/>:selected?<div className="dataset-record-body"><div><span>Building ID</span><b>{selected.id}</b></div><div><span>Campus ID</span><b>{selected.campusId}</b></div><div><span>City</span><b>{cityName(selected.city)}</b></div><div><span>Coordinates</span><b>{selected.lat.toFixed(6)}, {selected.lon.toFixed(6)}</b></div><div><span>Geometry type</span><b>Real existing geometry</b></div><div><span>Height source</span><b>Source 3D geometry · Z extent</b></div><div><span>Floor method</span><b>Height / floor-to-floor height</b></div><div><span>OBJ source</span><b>Per-campus merged OBJ</b></div><p>The original research-dataset geometry and parameters remain unchanged. Adjacent parts were merged before model features were calculated.</p></div>:<div className="viewer-empty">Select a campus and teaching building</div>}</section>
      <div className="panel result-panel">
        <div className="result-head"><div><span className="kicker">DUAL-ROUTE PREDICTION</span><h2>Performance evaluation</h2></div><span className={predictionReady&&hourlyStatus!=="error"?"model-badge ready":"model-badge"}><Activity size={14}/>{mode==="design"?(designStatus==="running"?"Running Route I + Route II in the browser":designStatus==="ready"?"Native annual + 8,760-hour prediction ready":"Waiting for OBJ, parameters and EPW"):heightNeedsConfirmation?"Height and floors require confirmation":mode==="map"?"Mapped-building assessment":hourlyStatus==="running"?"Running building-specific 8,760-hour model":hourlyStatus==="ready"?"Building-specific annual + 8,760-hour output":hourlyStatus==="error"?"First-route profile · hourly model unavailable":"Full first-route annual prediction"}</span></div>
        <div className="result-section-head core-head"><span>ANNUAL CORE INDICATORS</span><small>Whole-building prediction, normalized by floor area where applicable</small></div>
        {predictionReady?<><div className="annual-core-grid"><Result label="Annual EUI" value={annualAssessment.eui} unit="kWh/m²·year" tone="orange"/><Result label="Full-roof PV generation" value={annualAssessment.epvFull} unit="kWh/year" tone="green"/><Result label="CEI without rooftop PV" value={annualAssessment.ceiBefore} unit="kgCO₂/m²·year" tone="blue"/><Result label="CEI with full-roof PV" value={annualAssessment.ceiAfter} unit="kgCO₂/m²·year" tone="teal"/></div><div className="pv-impact-strip"><span>Full-roof PV impact</span><b>{fmt.format(annualAssessment.carbonReduction)} kgCO₂/year avoided</b><small>{fmt.format(annualAssessment.fullCrr)}% reduction · active view: {pvScenario==="full"?"full-roof PV":"no rooftop PV"}</small></div></>:<div className="prediction-gate annual-gate">{mode==="design"?"Upload the OBJ and EPW, complete the design parameters, then run the two-route model.":"Enter the measured building height and number of floors to calculate annual indicators."}</div>}
        <div className="result-section-head compact economics-head"><span>ECONOMIC AND CARBON SAVINGS</span><small>Editable assessment assumptions</small></div>
        <div className="assumption-row"><label><strong className="manual-title">Electricity tariff <em>(User input)</em></strong><span>CNY/kWh</span><input type="number" min="0" step="0.01" value={tariff} onChange={e=>setTariff(Number(e.target.value)||0)}/></label><label><strong className="manual-title">Feed-in tariff <em>(User input)</em></strong><span>CNY/kWh</span><input type="number" min="0" step="0.01" value={feedInTariff} onChange={e=>setFeedInTariff(Number(e.target.value)||0)}/></label><label><strong className="manual-title">Grid emission factor <em>(User input)</em></strong><span>kgCO₂/kWh</span><input type="number" min="0" step="0.01" value={gridFactor} onChange={e=>setGridFactor(Number(e.target.value)||0)}/></label></div>
        {predictionReady?<div className="economic-grid"><Economic label="Gross electricity cost" value={annualAssessment.grossCost} unit="CNY/year"/><Economic label="Total PV economic value" value={annualAssessment.pvValue} unit="CNY/year"/><Economic label="Surplus PV export revenue" value={annualAssessment.exportRevenue} unit="CNY/year"/><Economic label="Net electricity cost" value={annualAssessment.netCost} unit="CNY/year"/><Economic label="Cost saving rate" value={annualAssessment.savingRate} unit="%"/><Economic label="Carbon reduction rate (CRR)" value={annualAssessment.crr} unit="%"/><Economic label="PV yield intensity" value={annualAssessment.epvIntensity} unit="kWh/m²·year"/></div>:<div className="prediction-gate compact economic-gate">Economic and carbon results are waiting for a completed prediction.</div>}
        <p className="accounting-note">Annual EUI, Epv and CEI are direct outputs from the first-route 3D-CNN–TabTransformer ensemble. The LSTM predicts only the 8,760-hour distribution, which is calibrated to those annual outputs. CRR = (carbon before PV − carbon after PV) / carbon before PV × 100%.</p>
        <div className="embedded-parameters">
        <div className="panel-title"><span>02</span><div><h2>Complete parameters</h2><p>Geometry, site, PV and envelope data</p></div></div>
        {evaluatedBuilding?<>
          {extracted&&mode==="map"&&<div className={heightNeedsConfirmation?"geometry-inputs needs-confirmation":"geometry-inputs"}><label><strong className="manual-title">Building height <em>(User input)</em></strong><span>Measured total height in metres</span><input type="number" min="3" step="0.1" value={mappedHeight} placeholder="Enter measured height" onChange={e=>setMappedHeight(e.target.value)}/></label><label><strong className="manual-title">Number of floors <em>(User input)</em></strong><span>Measured above-ground floor count</span><input type="number" min="1" max="40" step="1" value={mappedFloors} placeholder="Enter floor count" onChange={e=>setMappedFloors(e.target.value)}/></label><div><b>{heightNeedsConfirmation?"Geometry input required":"Geometry confirmed and linked"}</b><span>{heightNeedsConfirmation?"The map supplies the footprint only. Prediction and final OBJ remain locked.":"Floor area, mean floor height, façades, shape factor, windows, model inputs and 8,760-hour results have been recalculated."}</span></div></div>}
          <div className="metric-grid"><Metric label="Length" value={evaluatedBuilding.length} unit="m"/><Metric label="Width" value={evaluatedBuilding.width} unit="m"/><Metric label="Height" value={evaluatedBuilding.height} unit="m"/><Metric label="Floors" value={evaluatedBuilding.floors} unit="levels"/><Metric label="Roof area" value={evaluatedBuilding.roofArea} unit="m²"/><Metric label="Floor area" value={evaluatedBuilding.buildingArea} unit="m²"/></div>
          <div className="section-label">Geometry, site and photovoltaics</div>
          <div className="parameter-table"><Param label="Building ID" value={evaluatedBuilding.teachingId}/><Param label="Geometry source" value={mode==="design"?"Uploaded OBJ + generated windows":extracted?"OpenStreetMap footprint + user height / floors":"Research dataset"}/><Param label="Longitude / Latitude" value={`${Number(lon).toFixed(6)} / ${Number(lat).toFixed(6)}`}/><Param label="Orientation" value={orientationName(evaluatedBuilding.orientation)}/><Param label="Footprint area" value={evaluatedBuilding.footprintArea} unit="m²"/><Param label="Opaque wall area" value={viewerMassing?viewerMassing.perimeter*evaluatedBuilding.height*(1-evaluatedBuilding.wwr):null} unit="m²"/><Param label="Window-to-wall ratio" value={evaluatedBuilding.wwr}/><Param label="Floor-to-floor height" value={evaluatedBuilding.averageFloorHeight} unit="m"/><Param label="Building spacing" value={evaluatedBuilding.buildingSpacing} unit="m"/><Param label={extracted?"Local site envelope":"Campus area"} value={evaluatedBuilding.campusArea} unit="m²"/><Param label="School type" value={schoolTypeName(evaluatedBuilding.schoolType)}/><Param label="Enclosure type" value={enclosureName(evaluatedBuilding.enclosureType)}/><Param label="Merged massing parts" value={extracted?.mergedParts??evaluatedBuilding.sourceBuildingCount}/><Param label="Window model" value={mode==="design"?`Per-floor modules · target WWR ${designInputs.wwr.toFixed(2)}`:extracted?"Per-floor modules · target WWR 0.30":"Dataset geometry"}/></div>
          <div className="section-label">Envelope and data quality</div>
          {mode!=="design"&&<label><strong className="manual-title">Construction year <em>(User input)</em></strong><input value={year} onChange={e=>setYear(e.target.value)}/></label>}
          <div className="parameter-table"><Param label="Shape factor" value={evaluatedBuilding.shapeFactor}/><Param label="Roof U-value" value={thermal?.roofU} unit="W/(m²·K)"/><Param label="Wall U-value" value={thermal?.wallU} unit="W/(m²·K)"/><Param label="Ground U-value" value={thermal?.groundU} unit="W/(m²·K)"/><Param label="Window U-value" value={thermal?.windowU} unit="W/(m²·K)"/><Param label="Window SHGC" value={thermal?.shgc}/><Param label="Envelope age band" value={evaluatedBuilding.envelopeBucket}/><Param label="Height data quality" value={extracted?(heightNeedsConfirmation?"Required user measurements missing":"User-entered height and floor count"):"Research dataset record"}/></div>
          <div className="source-chip"><span><Database size={14}/> {mode==="design"?(designStatus==="ready"?"User geometry, parameters, EPW and both prediction routes completed":"New-design inputs are isolated from the research records"):extracted?(mapGeometryReady?"Linked geometry and model-input record ready":"Waiting for required geometry inputs"):selected?.thermal.wallU?"Assigned from the dataset standard table":"Year-and-shape prior; verification required"}</span><button className="feature-download" disabled={!mapGeometryReady} onClick={exportModelInput}>Export model-input JSON</button></div>
        </>:<div className="empty"><Building2/><p>Select a campus and building to view all parameters</p></div>}
        </div>
      </div>
    </section>
    <section className="performance-stack">
      <div className="panel timeline-panel">
        <div className="result-head"><div><span className="kicker">INTERACTIVE 8,760-HOUR RESULTS</span><h2>{chartView==="monthly"?"Monthly performance summary":"Annual hourly performance profile"}</h2></div><button className="download" disabled={!predictionReady} onClick={exportCsv}><Download size={15}/> Export 8,760h CSV</button></div>
        {predictionReady?<><div className="chart-bar"><div>{([['energy','Energy (kWh)'],['pv','PV (kWh)'],['carbonWithoutPv','Carbon without roof PV'],['carbonWithPv','Carbon with roof PV'],['netCost','Net cost (CNY)'],['exportRevenue','Export revenue (CNY)']] as const).map(([k,n])=><button className={metric===k?"active":""} onClick={()=>setMetric(k)} key={k}>{n}</button>)}</div><div className="chart-view-switch" role="group" aria-label={language==="zh"?"时间尺度":"Time resolution"}><button className={chartView==="hourly"?"active":""} onClick={()=>setChartView("hourly")}>{language==="zh"?"逐时":"Hourly"}</button><button className={chartView==="monthly"?"active":""} onClick={()=>setChartView("monthly")}>{language==="zh"?"逐月":"Monthly"}</button></div></div>{chartView==="monthly"?<MonthlyChart data={performanceHourly} metric={metric} language={language}/>:<HourlyChart data={performanceHourly} metric={metric} language={language}/>}</>:<div className="prediction-gate hourly">{mode==="design"?"Run the new-design dual-route prediction to generate the 8,760-hour profile.":"Complete the height and floor inputs above to generate the 8,760-hour prediction profile."}</div>}
      </div>
    </section>
    <footer><span>Arch-EPVC Research Platform</span><span>16 cities · 994 campuses · 3,360 school buildings</span><span>3D-CNN + TabTransformer / LSTM</span></footer>
  </main>;
}

function DesignInputPanel({geometry,inputs,pvScenario,epwReady,status,error,onFile,onChange,onThermal,onPredict}:{geometry:DesignGeometry|null;inputs:DesignInputs;pvScenario:"none"|"full";epwReady:boolean;status:"idle"|"running"|"ready"|"error";error:string;onFile:(file:File)=>void;onChange:(patch:Partial<DesignInputs>)=>void;onThermal:(key:Exclude<keyof Thermal,"status">,value:number)=>void;onPredict:()=>void}){
  const number=(value:string,fallback:number)=>Number.isFinite(Number(value))?Number(value):fallback;
  return <div className="design-input-card">
    <label className="design-obj-upload"><FileUp size={22}/><span><b>{geometry?.filename||"Upload metric OBJ model"}</b><small>{geometry?`${geometry.vertexCount} vertices · ${geometry.faceCount} faces · ${fmt.format(geometry.sourceHeight)} m high`:"The platform extracts geometry, voxelizes it and rebuilds a windowed LoD1 model"}</small></span><input type="file" accept=".obj" onChange={event=>{const file=event.target.files?.[0];if(file)onFile(file)}}/></label>
    <div className="design-field-grid">
      <label className="wide"><span>Project name</span><input value={inputs.name} onChange={e=>onChange({name:e.target.value})}/></label>
      <label><span>Floors</span><input type="number" min="1" max="40" step="1" value={inputs.floors} onChange={e=>onChange({floors:Math.max(1,Math.round(number(e.target.value,inputs.floors)))})}/></label>
      <label><span>Construction year</span><input type="number" min="1950" max="2100" value={inputs.year} onChange={e=>onChange({year:Math.round(number(e.target.value,inputs.year))})}/></label>
      <label><span>Orientation (°)</span><input type="number" min="-90" max="90" step="1" value={inputs.orientation} onChange={e=>onChange({orientation:number(e.target.value,inputs.orientation)})}/></label>
      <label><span>Enclosure type</span><select value={inputs.enclosure} onChange={e=>onChange({enclosure:Number(e.target.value)})}><option value="1">Row layout</option><option value="2">Two-sided</option><option value="3">Three-sided</option><option value="4">Four-sided</option></select></label>
      <label><span>Window-to-wall ratio</span><input type="number" min="0.05" max="0.80" step="0.01" value={inputs.wwr} onChange={e=>onChange({wwr:Math.min(.8,Math.max(.05,number(e.target.value,inputs.wwr)))})}/></label>
      <label><span>Roof PV scenario</span><input value={pvScenario==="full"?"1.00 · full roof":"0.00 · no PV"} readOnly/></label>
      <label><span>Roof U-value</span><input type="number" min="0.05" step="0.01" value={inputs.thermal.roofU??0} onChange={e=>onThermal("roofU",number(e.target.value,Number(inputs.thermal.roofU)))}/></label>
      <label><span>Wall U-value</span><input type="number" min="0.05" step="0.01" value={inputs.thermal.wallU??0} onChange={e=>onThermal("wallU",number(e.target.value,Number(inputs.thermal.wallU)))}/></label>
      <label><span>Ground U-value</span><input type="number" min="0.05" step="0.01" value={inputs.thermal.groundU??0} onChange={e=>onThermal("groundU",number(e.target.value,Number(inputs.thermal.groundU)))}/></label>
      <label><span>Window U-value</span><input type="number" min="0.05" step="0.01" value={inputs.thermal.windowU??0} onChange={e=>onThermal("windowU",number(e.target.value,Number(inputs.thermal.windowU)))}/></label>
      <label><span>Window SHGC</span><input type="number" min="0" max="1" step="0.01" value={inputs.thermal.shgc??0} onChange={e=>onThermal("shgc",number(e.target.value,Number(inputs.thermal.shgc)))}/></label>
    </div>
    <div className="design-readiness"><span className={geometry?"done":""}><CheckCircle2 size={13}/> OBJ geometry</span><span className={epwReady?"done":""}><CheckCircle2 size={13}/> 8,760-hour EPW</span><span className={status==="ready"?"done":""}><CheckCircle2 size={13}/> Dual-route output</span></div>
    {error&&<p className="error design-error">{error}</p>}
    <button className="primary design-predict" disabled={!geometry||!epwReady||status==="running"} onClick={onPredict}><Activity size={16}/>{status==="running"?"Running annual and hourly models…":status==="ready"?"Re-run dual-route prediction":"Run dual-route prediction"}</button>
  </div>;
}

function Metric({label,value,unit}:{label:string,value:number|null,unit:string}){
  return <div className="metric"><span>{label}</span><b>{value==null?"—":fmt.format(value)}</b><small>{unit}</small></div>;
}
function Param({label,value,unit=""}:{label:string,value:string|number|null|undefined,unit?:string}){
  return <div><span>{label}</span><b>{value==null||value===""?"—":typeof value==="number"?fmt.format(value):value}{unit&&<small> {unit}</small>}</b></div>;
}
function Result({label,value,unit,tone}:{label:string,value:number,unit:string,tone:string}){
  return <div className={`result ${tone}`}><span>{label}</span><b>{fmt.format(value)}</b><small>{unit}</small></div>;
}
function Economic({label,value,unit}:{label:string,value:number,unit:string}){
  return <div className="economic"><span>{label}</span><b>{fmt.format(value)}</b><small>{unit}</small></div>;
}
function WeatherMini({label,unit,color,average,min,max,values}:{label:string;unit:string;color:string;average:number;min:number;max:number;values:number[]}){
  const width=120,height=34,bins=180,step=Math.max(1,Math.ceil(values.length/bins)),sampled:Array<number>=[];for(let i=0;i<values.length;i+=step){const slice=values.slice(i,i+step);sampled.push(slice.reduce((a,b)=>a+b,0)/slice.length)}const low=Math.min(...sampled),high=Math.max(...sampled),range=Math.max(high-low,1e-9),points=sampled.map((value,index)=>`${index/(sampled.length-1||1)*width},${height-(value-low)/range*(height-3)-1.5}`).join(" "),area=`0,${height} ${points} ${width},${height}`;
  return <article className="weather-mini"><div className="weather-mini-head"><span>{label}</span><i style={{background:color}}/></div><div className="weather-average"><b>{fmt.format(average)}</b><small>{unit} annual mean</small></div><svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${label} 8,760-hour profile`} preserveAspectRatio="none"><polygon points={area} fill={color} opacity=".09"/><polyline points={points} fill="none" stroke={color} strokeWidth="1.4" vectorEffect="non-scaling-stroke"/></svg><div className="weather-range"><span>Min {fmt.format(min)}</span><span>Max {fmt.format(max)}</span></div></article>;
}

const CITY_NAMES:Record<string,string>={"济南市":"Jinan","青岛市":"Qingdao","淄博市":"Zibo","枣庄市":"Zaozhuang","东营市":"Dongying","烟台市":"Yantai","潍坊市":"Weifang","济宁市":"Jining","泰安市":"Tai'an","威海市":"Weihai","临沂市":"Linyi","德州市":"Dezhou","日照市":"Rizhao","滨州市":"Binzhou","聊城市":"Liaocheng","菏泽市":"Heze"};
function cityName(value:string){return CITY_NAMES[value]??value}
function orientationName(value:number){
  const angle=((value+90)%180+180)%180-90,magnitude=Math.abs(angle);
  if(magnitude<.05)return "Due South";
  return `South ${fmt.format(magnitude)}° ${angle>0?"East":"West"}`;
}
function enclosureGroup(value:string|null){const text=String(value??"");if(text.includes("Four")||text.includes("四面"))return 4;if(text.includes("Three")||text.includes("三面")||text.includes("S形"))return 3;if(text.includes("Two")||text.includes("双面")||text.includes("双向"))return 2;return 1}
function schoolTypeName(value:string|null){return value?"Primary / secondary school":"Not specified"}
function enclosureName(value:string|null){
  if(!value)return "Not specified";
  if(["Row layout","Two-sided enclosure","Three-sided enclosure","Four-sided enclosure"].includes(value))return value;
  if(value.includes("四面"))return "Four-sided enclosure";
  if(value.includes("三面"))return "Three-sided enclosure";
  if(value.includes("双向")||value.includes("两面"))return "Two-sided enclosure";
  if(value.includes("行列"))return "Row layout";
  return "Other enclosure";
}
