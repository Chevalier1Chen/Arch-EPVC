import type { Building, EpwData, HourlyPoint } from "./types";

type OrtTensor = { data: Float32Array; dims: readonly number[] };
type OrtSession = { run(feeds: Record<string, unknown>): Promise<Record<string, OrtTensor>> };
type OrtApi = {
  env: { wasm: { wasmPaths: string; numThreads: number } };
  Tensor: new (type: "float32", data: Float32Array, dims: number[]) => unknown;
  InferenceSession: { create(path: string, options?: Record<string, unknown>): Promise<OrtSession> };
};

declare global { interface Window { ort?: OrtApi } }

type Scalers = { input_mean:number[]; input_std:number[]; label_mean:number[]; label_std:number[] };
let runtimePromise:Promise<OrtApi>|null=null, sessionPromise:Promise<OrtSession>|null=null;
let staticPromise:Promise<Float32Array>|null=null, scalersPromise:Promise<Scalers>|null=null;

function runtime(){
  if(runtimePromise)return runtimePromise;
  runtimePromise=new Promise<OrtApi>((resolve,reject)=>{
    if(window.ort){resolve(window.ort);return}
    const script=document.createElement("script");script.src="/vendor/onnx/ort.min.js";script.async=true;
    script.onload=()=>window.ort?resolve(window.ort):reject(new Error("The hourly inference runtime did not initialize."));
    script.onerror=()=>reject(new Error("The hourly inference runtime could not be loaded."));document.head.appendChild(script);
  }).then(ort=>{ort.env.wasm.wasmPaths="/vendor/onnx/";ort.env.wasm.numThreads=1;return ort});
  return runtimePromise;
}

async function resources(){
  const ort=await runtime();
  sessionPromise??=ort.InferenceSession.create("/models/second-route-hourly-lstm.onnx",{executionProviders:["wasm"]});
  staticPromise??=fetch("/models/second-route-static.f32").then(r=>{if(!r.ok)throw new Error("Static features are unavailable.");return r.arrayBuffer()}).then(b=>new Float32Array(b));
  scalersPromise??=fetch("/models/second-route-scalers.json").then(r=>{if(!r.ok)throw new Error("Hourly scalers are unavailable.");return r.json()});
  const [session,features,scalers]=await Promise.all([sessionPromise,staticPromise,scalersPromise]);return {ort,session,features,scalers};
}

export async function predictHourly(building:Building,weather:EpwData):Promise<HourlyPoint[]>{
  if(building.staticFeatureIndex===undefined)throw new Error("This building has no archived second-route feature vector.");
  if(weather.points.length!==8760)throw new Error("The second route requires exactly 8,760 weather records.");
  const {ort,session,features,scalers}=await resources(),offset=building.staticFeatureIndex*259;
  if(offset+259>features.length)throw new Error("The building static-feature index is out of range.");
  const staticInput=features.slice(offset,offset+259),weatherInput=new Float32Array(8760*10);
  weather.points.forEach((point,index)=>{
    const raw=[Math.sin(2*Math.PI*(point.day-1)/365),Math.cos(2*Math.PI*(point.day-1)/365),Math.sin(2*Math.PI*point.hour/24),Math.cos(2*Math.PI*point.hour/24),point.temperature,point.humidity,point.wind,point.dni,point.dhi,point.ghi];
    raw.forEach((value,channel)=>weatherInput[index*10+channel]=(value-scalers.input_mean[channel])/scalers.input_std[channel]);
  });
  const output=await session.run({static:new ort.Tensor("float32",staticInput,[1,259]),weather:new ort.Tensor("float32",weatherInput,[1,8760,10])});
  const tensor=output.normalized_hourly??Object.values(output)[0],data=tensor.data;
  const area=Math.max(building.buildingArea,1),pvArea=Math.max(building.roofPvArea,1e-6);
  return Array.from({length:8760},(_,index)=>{
    const intensity=[0,1,2].map(channel=>Math.max(Math.expm1(data[index*3+channel]*scalers.label_std[channel]+scalers.label_mean[channel]),0));
    return {hour:index+1,energy:intensity[0]*area,pv:intensity[1]*pvArea,carbon:intensity[2]*area};
  });
}
