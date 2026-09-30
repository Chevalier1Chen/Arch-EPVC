import type { EpwData, HourlyPoint, Prediction, Thermal } from "./types";
import type { ExtractedBuilding } from "./OpenBuildingMap";

type Point3 = [number, number, number];
type OrtTensor = { data: Float32Array; dims: readonly number[] };
type OrtSession = { run(feeds: Record<string, unknown>): Promise<Record<string, OrtTensor>> };
type OrtApi = {
  env: { wasm: { wasmPaths: string; numThreads: number } };
  Tensor: new (type: "float32"|"int64", data: Float32Array|BigInt64Array, dims: number[]) => unknown;
  InferenceSession: { create(path: string, options?: Record<string, unknown>): Promise<OrtSession> };
};

export type DesignInputs = {
  name:string; floors:number; year:number; enclosure:number; wwr:number;
  pvRatio:number; orientation:number; thermal:Thermal;
};

export type DesignGeometry = {
  filename:string; massing:ExtractedBuilding; voxel:Float32Array;
  sourceHeight:number; vertexCount:number; faceCount:number;
};

type Preprocessing = {
  features:string[]; categories:string[]; continuous_mean:number[]; continuous_std:number[];
  target_mean:number[]; target_std:number[]; hourly_input_mean:number[];
  hourly_input_std:number[]; hourly_label_mean:number[]; hourly_label_std:number[];
};

let runtimePromise:Promise<OrtApi>|null=null;
let route1Promise:Promise<OrtSession>|null=null;
let route2Promise:Promise<OrtSession>|null=null;
let preprocessingPromise:Promise<Preprocessing>|null=null;

function runtime(){
  if(runtimePromise)return runtimePromise;
  const current=()=> (window as unknown as {ort?:OrtApi}).ort;
  runtimePromise=new Promise<OrtApi>((resolve,reject)=>{
    if(current()){resolve(current()!);return}
    const script=document.createElement("script");
    script.src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.min.js";
    script.async=true;
    script.onload=()=>current()?resolve(current()!):reject(new Error("ONNX Runtime did not initialize."));
    script.onerror=()=>reject(new Error("ONNX Runtime could not be downloaded."));
    document.head.appendChild(script);
  }).then(ort=>{ort.env.wasm.wasmPaths="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";ort.env.wasm.numThreads=1;return ort});
  return runtimePromise;
}

async function resources(){
  const ort=await runtime();
  route1Promise??=ort.InferenceSession.create("/models/browser/route1-annual.onnx",{executionProviders:["wasm"]});
  route2Promise??=ort.InferenceSession.create("/models/browser/route2-hourly.onnx",{executionProviders:["wasm"]});
  preprocessingPromise??=fetch("/models/browser/preprocessing.json").then(r=>{if(!r.ok)throw new Error("Model preprocessing metadata are unavailable.");return r.json()});
  const [route1,route2,preprocessing]=await Promise.all([route1Promise,route2Promise,preprocessingPromise]);
  return {ort,route1,route2,preprocessing};
}

function convexHull(points:Array<[number,number]>){
  const unique=[...new Map(points.map(p=>[`${p[0]},${p[1]}`,p])).values()].sort((a,b)=>a[0]-b[0]||a[1]-b[1]);
  if(unique.length<3)throw new Error("OBJ footprint contains fewer than three unique XY vertices.");
  const cross=(o:[number,number],a:[number,number],b:[number,number])=>(a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0]);
  const lower:Array<[number,number]>=[],upper:Array<[number,number]>=[];
  for(const p of unique){while(lower.length>=2&&cross(lower.at(-2)!,lower.at(-1)!,p)<=0)lower.pop();lower.push(p)}
  for(const p of [...unique].reverse()){while(upper.length>=2&&cross(upper.at(-2)!,upper.at(-1)!,p)<=0)upper.pop();upper.push(p)}
  return [...lower.slice(0,-1),...upper.slice(0,-1)];
}

function signedPolygonArea(points:Array<[number,number]>){
  return points.reduce((sum,a,index)=>{const b=points[(index+1)%points.length];return sum+a[0]*b[1]-b[0]*a[1]},0)/2;
}

function simplifyRing(points:Array<[number,number]>){
  const clean=points.filter((point,index)=>index===0||Math.hypot(point[0]-points[index-1][0],point[1]-points[index-1][1])>1e-6);
  let changed=true;
  while(changed&&clean.length>3){changed=false;for(let i=0;i<clean.length;i++){const a=clean[(i-1+clean.length)%clean.length],b=clean[i],c=clean[(i+1)%clean.length],cross=(b[0]-a[0])*(c[1]-b[1])-(b[1]-a[1])*(c[0]-b[0]);if(Math.abs(cross)<=1e-6*Math.max(1,Math.hypot(b[0]-a[0],b[1]-a[1])+Math.hypot(c[0]-b[0],c[1]-b[1]))){clean.splice(i,1);changed=true;break}}}
  return clean;
}

function orientRing(points:Array<[number,number]>,clockwise:boolean){
  const isClockwise=signedPolygonArea(points)<0;
  return isClockwise===clockwise?points:[...points].reverse();
}

function extractBoundaryRings(vertices:Point3[],faces:number[][],minZ:number,maxZ:number){
  const tolerance=Math.max(1e-4,(maxZ-minZ)*1e-5),horizontal=(target:number)=>faces.filter(face=>face.length>=3&&face.every(index=>Math.abs(vertices[index][2]-target)<=tolerance));
  const surfaceFaces=horizontal(maxZ).length?horizontal(maxZ):horizontal(minZ);
  const pointByKey=new Map<string,[number,number]>(),edges=new Map<string,{a:string;b:string;count:number}>();
  const key=(point:[number,number])=>`${point[0].toFixed(6)},${point[1].toFixed(6)}`;
  for(const face of surfaceFaces){for(let i=0;i<face.length;i++){const pa=vertices[face[i]],pb=vertices[face[(i+1)%face.length]],a:[number,number]=[pa[0],pa[1]],b:[number,number]=[pb[0],pb[1]],ka=key(a),kb=key(b);if(ka===kb)continue;pointByKey.set(ka,a);pointByKey.set(kb,b);const edgeKey=ka<kb?`${ka}|${kb}`:`${kb}|${ka}`,current=edges.get(edgeKey);if(current)current.count+=1;else edges.set(edgeKey,{a:ka,b:kb,count:1})}}
  const boundary=[...edges.values()].filter(edge=>edge.count===1);
  if(!boundary.length)return {outer:orientRing(convexHull(vertices.map(v=>[v[0],v[1]])),false),holes:[] as Array<Array<[number,number]>>,source:"convex-hull-fallback"};
  const adjacency=new Map<string,Set<string>>();
  const connect=(a:string,b:string)=>{if(!adjacency.has(a))adjacency.set(a,new Set());adjacency.get(a)!.add(b)};
  boundary.forEach(edge=>{connect(edge.a,edge.b);connect(edge.b,edge.a)});
  const visited=new Set<string>(),edgeKey=(a:string,b:string)=>a<b?`${a}|${b}`:`${b}|${a}`,loops:Array<Array<[number,number]>>=[];
  for(const edge of boundary){if(visited.has(edgeKey(edge.a,edge.b)))continue;const start=edge.a,keys=[start];let current=start,next=edge.b,guard=0;while(guard++<boundary.length+2){visited.add(edgeKey(current,next));keys.push(next);if(next===start)break;const candidates=[...(adjacency.get(next)??[])].filter(candidate=>candidate!==current&&!visited.has(edgeKey(next,candidate)));const closing=[...(adjacency.get(next)??[])].find(candidate=>candidate===start&&!visited.has(edgeKey(next,candidate)));const following=candidates[0]??closing;if(!following)break;current=next;next=following}if(keys.at(-1)===start){keys.pop();const ring=simplifyRing(keys.map(item=>pointByKey.get(item)!));if(ring.length>=3&&Math.abs(signedPolygonArea(ring))>1e-4)loops.push(ring)}}
  if(!loops.length)return {outer:orientRing(convexHull(vertices.map(v=>[v[0],v[1]])),false),holes:[] as Array<Array<[number,number]>>,source:"convex-hull-fallback"};
  loops.sort((a,b)=>Math.abs(signedPolygonArea(b))-Math.abs(signedPolygonArea(a)));const outer=orientRing(loops[0],false),holes=loops.slice(1).filter(ring=>{const sample=ring[0];return insidePolygon(sample[0],sample[1],outer)}).map(ring=>orientRing(ring,true));
  return {outer,holes,source:"obj-face-topology"};
}

function polygonMetrics(points:Array<[number,number]>){
  let area=0,perimeter=0;
  points.forEach((a,i)=>{const b=points[(i+1)%points.length];area+=a[0]*b[1]-b[0]*a[1];perimeter+=Math.hypot(b[0]-a[0],b[1]-a[1])});
  return {area:Math.abs(area)/2,perimeter};
}

function frameMetrics(points:Array<[number,number]>){
  const mx=points.reduce((s,p)=>s+p[0],0)/points.length,my=points.reduce((s,p)=>s+p[1],0)/points.length;
  let xx=0,yy=0,xy=0;points.forEach(p=>{xx+=(p[0]-mx)**2;yy+=(p[1]-my)**2;xy+=(p[0]-mx)*(p[1]-my)});
  const angle=.5*Math.atan2(2*xy,xx-yy),c=Math.cos(angle),s=Math.sin(angle),rotated=points.map(p=>({u:(p[0]-mx)*c+(p[1]-my)*s,v:-(p[0]-mx)*s+(p[1]-my)*c}));
  const u=rotated.map(p=>p.u),v=rotated.map(p=>p.v),a=Math.max(...u)-Math.min(...u),b=Math.max(...v)-Math.min(...v);
  return {length:Math.max(a,b),width:Math.min(a,b),orientation:angle*180/Math.PI};
}

function insidePolygon(x:number,y:number,polygon:Array<[number,number]>){
  let inside=false,j=polygon.length-1;
  for(let i=0;i<polygon.length;i++){const a=polygon[i],b=polygon[j];if((a[1]>y)!==(b[1]>y)&&x<(b[0]-a[0])*(y-a[1])/(b[1]-a[1]+1e-12)+a[0])inside=!inside;j=i}
  return inside;
}

function voxelize(vertices:Point3[],faces:number[][],size=32,margin=.08){
  const mins:[number,number,number]=[0,1,2].map(k=>Math.min(...vertices.map(v=>v[k]))) as Point3;
  const maxs:[number,number,number]=[0,1,2].map(k=>Math.max(...vertices.map(v=>v[k]))) as Point3;
  const center=mins.map((v,k)=>(v+maxs[k])/2),largest=Math.max(...maxs.map((v,k)=>v-mins[k]),1e-6);
  const normalized=vertices.map(v=>v.map((x,k)=>(x-center[k])/largest*(1-2*margin)+.5) as Point3),grid=new Float32Array(size**3),minZ=Math.min(...normalized.map(v=>v[2]));
  const roofs=faces.map(face=>face.map(i=>normalized[i]).filter(Boolean)).filter(points=>points.length>=3&&Math.max(...points.map(p=>p[2]))-Math.min(...points.map(p=>p[2]))<1e-4&&points.reduce((s,p)=>s+p[2],0)/points.length>minZ+1e-4);
  const set=(z:number,y:number,x:number)=>{grid[z*size*size+y*size+x]=1};
  if(!roofs.length){const lo=mins.map((_,k)=>Math.max(0,Math.floor(Math.min(...normalized.map(v=>v[k]))*size))),hi=maxs.map((_,k)=>Math.min(size,Math.ceil(Math.max(...normalized.map(v=>v[k]))*size)));for(let z=lo[2];z<hi[2];z++)for(let y=lo[1];y<hi[1];y++)for(let x=lo[0];x<hi[0];x++)set(z,y,x);return grid}
  for(const roof of roofs){const polygon=roof.map(p=>[p[0],p[1]] as [number,number]),top=roof.reduce((s,p)=>s+p[2],0)/roof.length,x0=Math.max(0,Math.floor(Math.min(...polygon.map(p=>p[0]))*size)),x1=Math.min(size,Math.ceil(Math.max(...polygon.map(p=>p[0]))*size)),y0=Math.max(0,Math.floor(Math.min(...polygon.map(p=>p[1]))*size)),y1=Math.min(size,Math.ceil(Math.max(...polygon.map(p=>p[1]))*size)),z0=Math.max(0,Math.floor(minZ*size)),z1=Math.min(size,Math.ceil(top*size));for(let y=y0;y<y1;y++)for(let x=x0;x<x1;x++)if(insidePolygon((x+.5)/size,(y+.5)/size,polygon))for(let z=z0;z<z1;z++)set(z,y,x)}
  return grid;
}

export async function parseDesignObj(file:File,lat:number,lon:number):Promise<DesignGeometry>{
  const text=await file.text(),vertices:Point3[]=[],faces:number[][]=[];
  for(const raw of text.split(/\r?\n/)){const line=raw.trim();if(line.startsWith("v ")){const p=line.slice(2).trim().split(/\s+/).slice(0,3).map(Number) as Point3;if(p.every(Number.isFinite))vertices.push(p)}else if(line.startsWith("f ")){const face=line.slice(2).trim().split(/\s+/).map(token=>Number(token.split("/")[0])).map(index=>index<0?vertices.length+index:index-1).filter(index=>index>=0&&index<vertices.length);if(face.length>=3)faces.push(face)}}
  if(vertices.length<6)throw new Error("OBJ must contain at least six valid vertices.");
  const minZ=Math.min(...vertices.map(v=>v[2])),maxZ=Math.max(...vertices.map(v=>v[2])),height=maxZ-minZ;
  const rings=extractBoundaryRings(vertices,faces,minZ,maxZ),outerMetrics=polygonMetrics(rings.outer),holeMetrics=rings.holes.map(polygonMetrics),area=outerMetrics.area-holeMetrics.reduce((sum,item)=>sum+item.area,0),perimeter=outerMetrics.perimeter+holeMetrics.reduce((sum,item)=>sum+item.perimeter,0),frame=frameMetrics(rings.outer);
  if(area<=1||height<=2)throw new Error("OBJ footprint area or height is outside the supported range (metres required).");
  const cx=rings.outer.reduce((s,p)=>s+p[0],0)/rings.outer.length,cy=rings.outer.reduce((s,p)=>s+p[1],0)/rings.outer.length,toGeo=(ring:Array<[number,number]>)=>{const result=ring.map(p=>[lat+(p[1]-cy)/110540,lon+(p[0]-cx)/(111320*Math.cos(lat*Math.PI/180))] as [number,number]);result.push(result[0]);return result},footprint=toGeo(rings.outer),footprintHoles=rings.holes.map(toGeo);
  return {filename:file.name,sourceHeight:height,vertexCount:vertices.length,faceCount:faces.length,voxel:voxelize(vertices,faces),massing:{osmId:`DESIGN-${file.name.replace(/\.[^.]+$/,"")}`,footprint,footprintHoles,footprintArea:area,length:frame.length,width:frame.width,orientation:frame.orientation,perimeter,height,floors:Math.max(1,Math.round(height/3.6)),heightSource:`User OBJ · metric geometry · ${rings.source}`,mergedParts:1}};
}

function climate(epw:EpwData){
  const n=epw.points.length,sum=(pick:(p:EpwData["points"][number])=>number)=>epw.points.reduce((s,p)=>s+pick(p),0);
  return {Temp_mean:sum(p=>p.temperature)/n,Humidity_mean:sum(p=>p.humidity)/n,Wind_mean:sum(p=>p.wind)/n,HDD18:sum(p=>Math.max(18-p.temperature,0))/24,CDD26:sum(p=>Math.max(p.temperature-26,0))/24,DNI_kWh_m2:sum(p=>p.dni)/1000,DHI_kWh_m2:sum(p=>p.dhi)/1000,GHI_kWh_m2:sum(p=>p.ghi)/1000};
}

export async function predictDesign(geometry:DesignGeometry,inputs:DesignInputs,epw:EpwData):Promise<{prediction:Prediction;hourly:HourlyPoint[]}>{
  if(epw.points.length!==8760)throw new Error("Exactly 8,760 EPW records are required.");
  const {ort,route1,route2,preprocessing}=await resources(),m=geometry.massing,height=m.height,floors=Math.max(1,Math.round(inputs.floors)),floorArea=m.footprintArea*floors,shape=(2*m.footprintArea+m.perimeter*height)/(m.footprintArea*height),weather=climate(epw);
  const values:Record<string,number>={"A.Building area":floorArea,"B.Building footprint":m.footprintArea,"C.Building height":height,"D.Layer":floors,"E.Height":height/floors,"F.Building length":m.length,"G.Building width":m.width,"H.Orientation":inputs.orientation,"J.Shape coefficient":shape,"K.Roof thermal coefficient":Number(inputs.thermal.roofU),"L.Wall thermal coefficient":Number(inputs.thermal.wallU),"M.Ground thermal coefficient":Number(inputs.thermal.groundU),"N.Window U-value":Number(inputs.thermal.windowU),construction_year_user:inputs.year,roof_area_m2:m.footprintArea,...weather};
  const continuous=Float32Array.from(preprocessing.features.map((name,index)=>(values[name]-preprocessing.continuous_mean[index])/Math.max(preprocessing.continuous_std[index],1e-9))),categoryIndex=Math.max(0,preprocessing.categories.indexOf(String(Math.min(4,Math.max(1,inputs.enclosure)))));
  const r1=await route1.run({voxel:new ort.Tensor("float32",geometry.voxel,[1,1,32,32,32]),continuous:new ort.Tensor("float32",continuous,[1,continuous.length]),category:new ort.Tensor("int64",BigInt64Array.from([BigInt(categoryIndex)]),[1])});
  const scaled=r1.annual_scaled.data,annual=[0,1,2].map(i=>scaled[i]*preprocessing.target_std[i]+preprocessing.target_mean[i]);annual[1]=Math.max(Math.expm1(annual[1]),0);annual[0]=Math.max(annual[0],0);annual[2]=Math.max(annual[2],0);
  const staticInput=new Float32Array(259);staticInput.set(r1.shared_embedding.data.slice(0,256),0);staticInput.set(scaled.slice(0,3),256);
  const weatherInput=new Float32Array(8760*10);epw.points.forEach((p,index)=>{const raw=[Math.sin(2*Math.PI*(p.day-1)/365),Math.cos(2*Math.PI*(p.day-1)/365),Math.sin(2*Math.PI*p.hour/24),Math.cos(2*Math.PI*p.hour/24),p.temperature,p.humidity,p.wind,p.dni,p.dhi,p.ghi];raw.forEach((value,channel)=>weatherInput[index*10+channel]=(value-preprocessing.hourly_input_mean[channel])/Math.max(preprocessing.hourly_input_std[channel],1e-9))});
  const r2=await route2.run({static:new ort.Tensor("float32",staticInput,[1,259]),weather:new ort.Tensor("float32",weatherInput,[1,8760,10])}),hourlyScaled=r2.hourly_scaled.data;
  const rawHourly=Array.from({length:8760},(_,index)=>{const intensity=[0,1,2].map(channel=>Math.max(Math.expm1(hourlyScaled[index*3+channel]*preprocessing.hourly_label_std[channel]+preprocessing.hourly_label_mean[channel]),0));return {hour:index+1,energy:intensity[0]*floorArea,pv:intensity[1]*Math.max(m.footprintArea*inputs.pvRatio,1e-6),carbon:intensity[2]*floorArea}});
  const targetTotals=[annual[0]*floorArea,annual[1]*Math.max(inputs.pvRatio,0),annual[2]*floorArea],sums=rawHourly.reduce((a,p)=>[a[0]+p.energy,a[1]+p.pv,a[2]+p.carbon],[0,0,0]);
  const hourly=rawHourly.map(p=>({hour:p.hour,energy:p.energy*targetTotals[0]/Math.max(sums[0],1e-9),pv:p.pv*targetTotals[1]/Math.max(sums[1],1e-9),carbon:p.carbon*targetTotals[2]/Math.max(sums[2],1e-9)}));
  const prediction:Prediction={annualEui:annual[0],annualEpv:targetTotals[1],annualCei:annual[2],firstEui:annual[0],firstEpv:targetTotals[1],firstCei:annual[2],source:"Browser inference · Route I + Route II"};
  return {prediction,hourly};
}
