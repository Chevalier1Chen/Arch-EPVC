"use client";

import { useEffect, useRef, useState } from "react";
import type { Map as LeafletMap, Layer } from "leaflet";
import "leaflet/dist/leaflet.css";

const footprintCache=new Map<string,any>();

export type ExtractedBuilding = {
  osmId:string; footprint:Array<[number,number]>; footprintArea:number; length:number; width:number;
  footprintHoles?:Array<Array<[number,number]>>;
  orientation:number; perimeter:number; height:number; floors:number; heightSource:string; mergedParts:number;
  buildingSpacing?:number|null; localSiteArea?:number|null; localDensity?:number|null; localFar?:number|null;
  enclosureType?:string|null; contextBuildingCount?:number;
};

function metrics(id:string,geometry:Array<{lat:number;lon:number}>,tags:Record<string,string>):ExtractedBuilding{
  const valid=geometry.filter(p=>Number.isFinite(p.lat)&&Number.isFinite(p.lon)),clean=valid.filter((p,i)=>i===0||Math.hypot(p.lat-valid[i-1].lat,p.lon-valid[i-1].lon)>1e-8);
  if(clean.length<3)throw new Error("Building footprint has fewer than three valid vertices");
  if(clean[0].lat!==clean.at(-1)!.lat||clean[0].lon!==clean.at(-1)!.lon)clean.push({...clean[0]});
  const lat0=clean.reduce((s,p)=>s+p.lat,0)/clean.length,lon0=clean.reduce((s,p)=>s+p.lon,0)/clean.length;
  const points=clean.map(p=>({x:(p.lon-lon0)*111320*Math.cos(lat0*Math.PI/180),y:(p.lat-lat0)*110540}));
  let area=0,perimeter=0;for(let i=0;i<points.length-1;i++){area+=points[i].x*points[i+1].y-points[i+1].x*points[i].y;perimeter+=Math.hypot(points[i+1].x-points[i].x,points[i+1].y-points[i].y)}area=Math.abs(area)/2;
  const mx=points.reduce((s,p)=>s+p.x,0)/points.length,my=points.reduce((s,p)=>s+p.y,0)/points.length;let xx=0,yy=0,xy=0;points.forEach(p=>{xx+=(p.x-mx)**2;yy+=(p.y-my)**2;xy+=(p.x-mx)*(p.y-my)});
  const angle=.5*Math.atan2(2*xy,xx-yy),c=Math.cos(angle),s=Math.sin(angle),rotated=points.map(p=>({u:(p.x-mx)*c+(p.y-my)*s,v:-(p.x-mx)*s+(p.y-my)*c}));
  const a=Math.max(...rotated.map(p=>p.u))-Math.min(...rotated.map(p=>p.u)),b=Math.max(...rotated.map(p=>p.v))-Math.min(...rotated.map(p=>p.v));
  const taggedHeight=Number.parseFloat(tags.height),taggedFloors=Number.parseInt(tags["building:levels"]),hasHeight=Number.isFinite(taggedHeight)&&taggedHeight>2,hasFloors=Number.isFinite(taggedFloors)&&taggedFloors>0;
  const floors=Math.min(40,Math.max(1,hasFloors?taggedFloors:hasHeight?Math.round(taggedHeight/3.3):4)),height=Math.min(200,hasHeight?taggedHeight:floors*3.3);
  const heightSource=hasHeight&&hasFloors?"OSM height and levels tags":hasHeight?"OSM height tag; levels inferred from height ÷ 3.3 m":hasFloors?"OSM levels × 3.3 m":"Teaching-building prior: 4 levels × 3.3 m";
  return {osmId:`OSM-${id}`,footprint:clean.map(p=>[p.lat,p.lon]),footprintArea:area,length:Math.max(a,b),width:Math.min(a,b),orientation:angle*180/Math.PI,perimeter,height,floors,heightSource,mergedParts:1};
}

function vertexDistance(a:ExtractedBuilding,b:ExtractedBuilding){let best=Infinity;for(const p of a.footprint)for(const q of b.footprint){const y=(p[0]-q[0])*110540,x=(p[1]-q[1])*111320*Math.cos((p[0]+q[0])/2*Math.PI/180);best=Math.min(best,Math.hypot(x,y))}return best}
function containsPoint(building:ExtractedBuilding,lat:number,lon:number){
  const ring=building.footprint;let inside=false;
  for(let i=0,j=ring.length-1;i<ring.length;j=i++){
    const [yi,xi]=ring[i],[yj,xj]=ring[j];
    if(((yi>lat)!==(yj>lat))&&(lon<(xj-xi)*(lat-yi)/(yj-yi||Number.EPSILON)+xi))inside=!inside;
  }
  return inside;
}
function pointDistance(building:ExtractedBuilding,lat:number,lon:number){
  if(containsPoint(building,lat,lon))return 0;
  return building.footprint.reduce((best,p)=>{const y=(p[0]-lat)*110540,x=(p[1]-lon)*111320*Math.cos((p[0]+lat)/2*Math.PI/180);return Math.min(best,Math.hypot(x,y))},Infinity);
}
async function reverseFootprint(lat:number,lon:number){
  const url=new URL("https://nominatim.openstreetmap.org/reverse");
  url.search=new URLSearchParams({lat:String(lat),lon:String(lon),format:"jsonv2",zoom:"18",polygon_geojson:"1"}).toString();
  const response=await fetch(url,{headers:{Accept:"application/json","Accept-Language":"en"},signal:AbortSignal.timeout(9000)});
  if(!response.ok)throw new Error(`Nominatim: HTTP ${response.status}`);const item:any=await response.json(),geo=item.geojson;
  if(!geo||!(geo.type==="Polygon"||geo.type==="MultiPolygon"))throw new Error("Nominatim did not return a building polygon");
  const coordinates=geo.type==="Polygon"?geo.coordinates[0]:geo.coordinates.sort((a:any,b:any)=>b[0].length-a[0].length)[0][0];
  return {elements:[{id:`nominatim-${item.osm_id??item.place_id}`,geometry:coordinates.map((p:number[])=>({lat:p[1],lon:p[0]})),tags:{height:item.extratags?.height,"building:levels":item.extratags?.building_levels}}]};
}
async function osmMapFootprints(lat:number,lon:number){
  const url=`/api/osm-map?lat=${encodeURIComponent(lat)}&lon=${encodeURIComponent(lon)}`;
  const response=await fetch(url,{headers:{Accept:"application/xml"},signal:AbortSignal.timeout(12000)});if(!response.ok)throw new Error(`OSM map API: HTTP ${response.status}`);
  const xml=new DOMParser().parseFromString(await response.text(),"application/xml");if(xml.querySelector("parsererror"))throw new Error("OSM map API returned invalid XML");
  const nodes=new Map<string,{lat:number;lon:number}>();xml.querySelectorAll("node").forEach(node=>nodes.set(node.getAttribute("id")??"",{lat:Number(node.getAttribute("lat")),lon:Number(node.getAttribute("lon"))}));
  const elements:any[]=[];xml.querySelectorAll("way").forEach(way=>{const tags=Object.fromEntries([...way.querySelectorAll(":scope > tag")].map(tag=>[tag.getAttribute("k")??"",tag.getAttribute("v")??""]));if(!tags.building)return;const geometry=[...way.querySelectorAll(":scope > nd")].map(nd=>nodes.get(nd.getAttribute("ref")??"")).filter(Boolean);if(geometry.length>3)elements.push({id:`way-${way.getAttribute("id")}`,geometry,tags})});
  if(!elements.length)throw new Error("OSM map API returned no building ways");return {elements};
}
async function overpassFootprints(endpoint:string,query:string){
  const response=await fetch(endpoint,{method:"POST",headers:{"Content-Type":"application/x-www-form-urlencoded;charset=UTF-8"},body:`data=${encodeURIComponent(query)}`,signal:AbortSignal.timeout(12000)});if(!response.ok)throw new Error(`${endpoint}: HTTP ${response.status}`);const data:any=await response.json();if(!data.elements?.some((e:any)=>e.geometry?.length>3))throw new Error(`${endpoint}: no building footprints`);return data;
}
function convexHull(buildings:ExtractedBuilding[]){
  const raw=buildings.flatMap(b=>b.footprint.slice(0,-1)),lat0=raw.reduce((s,p)=>s+p[0],0)/raw.length,lon0=raw.reduce((s,p)=>s+p[1],0)/raw.length;
  const pts=raw.map(p=>({lat:p[0],lon:p[1],x:(p[1]-lon0)*111320*Math.cos(lat0*Math.PI/180),y:(p[0]-lat0)*110540})).sort((a,b)=>a.x-b.x||a.y-b.y),cross=(o:any,a:any,b:any)=>(a.x-o.x)*(b.y-o.y)-(a.y-o.y)*(b.x-o.x);
  const lower:any[]=[],upper:any[]=[];for(const p of pts){while(lower.length>=2&&cross(lower.at(-2),lower.at(-1),p)<=0)lower.pop();lower.push(p)}for(const p of [...pts].reverse()){while(upper.length>=2&&cross(upper.at(-2),upper.at(-1),p)<=0)upper.pop();upper.push(p)}const hull=[...lower.slice(0,-1),...upper.slice(0,-1)];hull.push(hull[0]);return hull.map(p=>({lat:p.lat,lon:p.lon}));
}
function mergeCluster(seed:ExtractedBuilding,all:ExtractedBuilding[]){
  const cluster=[seed],remaining=all.filter(x=>x!==seed);let changed=true;while(changed){changed=false;for(let i=remaining.length-1;i>=0;i--){if(cluster.some(item=>vertexDistance(item,remaining[i])<=3.5&&Math.abs(item.floors-remaining[i].floors)<=1)){cluster.push(remaining[i]);remaining.splice(i,1);changed=true}}}
  if(cluster.length===1)return seed;const merged=metrics(cluster.map(x=>x.osmId.replace("OSM-","")).join("+"),convexHull(cluster),{"building:levels":String(Math.round(cluster.reduce((s,x)=>s+x.floors,0)/cluster.length)),height:String(Math.max(...cluster.map(x=>x.height)))});return {...merged,mergedParts:cluster.length,heightSource:`Merged ${cluster.length} adjacent massing parts; ${merged.heightSource}`};
}
function centroid(building:ExtractedBuilding){const ring=building.footprint.slice(0,-1);return {lat:ring.reduce((s,p)=>s+p[0],0)/ring.length,lon:ring.reduce((s,p)=>s+p[1],0)/ring.length}}
function contextFeatures(building:ExtractedBuilding,all:ExtractedBuilding[]){
  const center=centroid(building),neighbours=all.filter(item=>item!==building).map(item=>({item,distance:vertexDistance(building,item),center:centroid(item)})).filter(x=>x.distance>.2&&x.distance<=80&&!containsPoint(building,x.center.lat,x.center.lon)),spacing=neighbours.length?Math.min(...neighbours.map(x=>x.distance)):null;
  let localSiteArea:number|null=null,localDensity:number|null=null,localFar:number|null=null;
  if(all.length>1){try{const site=metrics("local-site",convexHull(all),{});localSiteArea=site.footprintArea;if(localSiteArea>0){localDensity=all.reduce((s,x)=>s+x.footprintArea,0)/localSiteArea;localFar=all.reduce((s,x)=>s+x.footprintArea*x.floors,0)/localSiteArea}}catch{/* local context remains unavailable */}}
  const sectors=new Set(neighbours.map(x=>{const y=(x.center.lat-center.lat)*110540,xm=(x.center.lon-center.lon)*111320*Math.cos(center.lat*Math.PI/180),angle=(Math.atan2(y,xm)*180/Math.PI+360)%360;return Math.floor((angle+45)%360/90)})),count=sectors.size,enclosureType=count>=4?"Four-sided enclosure":count===3?"Three-sided enclosure":count===2?"Two-sided enclosure":"Row layout";
  return {buildingSpacing:spacing,localSiteArea,localDensity,localFar,enclosureType,contextBuildingCount:all.length};
}

export function OpenBuildingMap({lat,lon,onPick,onBuilding,datasetFootprint}:{lat:number;lon:number;onPick:(lat:number,lon:number)=>void;onBuilding:(b:ExtractedBuilding)=>void;datasetFootprint?:Array<[number,number]>|null}){
  const host=useRef<HTMLDivElement>(null),mapRef=useRef<LeafletMap|null>(null),layers=useRef<Layer[]>([]),datasetLayer=useRef<Layer|null>(null),requestVersion=useRef(0);const [status,setStatus]=useState("Click a building footprint on the map");
  useEffect(()=>{let disposed=false;(async()=>{if(!host.current||mapRef.current)return;const L=await import("leaflet");if(disposed||!host.current)return;const map=L.map(host.current,{zoomControl:true}).setView([lat,lon],18);L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png",{maxZoom:19,attribution:'&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'}).addTo(map);mapRef.current=map;
    map.on("click",async event=>{const requestId=++requestVersion.current,point=event.latlng;onPick(point.lat,point.lng);setStatus("Retrieving and cleaning nearby building footprints…");layers.current.forEach(layer=>map.removeLayer(layer));layers.current=[];
      try{const cacheKey=`${point.lat.toFixed(4)},${point.lng.toFixed(4)}`,cached=footprintCache.get(cacheKey);const query=`[out:json][timeout:12];way(around:80,${point.lat},${point.lng})[building];out tags geom;`;let data:any=cached??null;if(!data){const endpoints=["https://overpass.private.coffee/api/interpreter","https://overpass-api.de/api/interpreter"],primary=[osmMapFootprints(point.lat,point.lng),...endpoints.map(endpoint=>overpassFootprints(endpoint,query))];try{data=await Promise.any(primary)}catch{data=await reverseFootprint(point.lat,point.lng)}footprintCache.set(cacheKey,data)}
        if(requestId!==requestVersion.current)return;const found=(data.elements??[]).filter((e:any)=>e.geometry?.length>3&&e.geometry.length<=300).flatMap((e:any)=>{try{const metric=metrics(String(e.id),e.geometry,e.tags??{});return [{metric,distance:pointDistance(metric,point.lat,point.lng)}]}catch{return []}}).filter((x:any)=>Number.isFinite(x.distance)&&x.metric.footprintArea>=10&&x.metric.footprintArea<=100000).sort((a:any,b:any)=>a.distance-b.distance).slice(0,120);if(!found.length){setStatus("No valid public building footprint was found at this location.");return}const all:ExtractedBuilding[]=found.map((x:any)=>x.metric);
        const select=(seed:ExtractedBuilding)=>{if(requestId!==requestVersion.current)return;try{const merged=mergeCluster(seed,all),enriched={...merged,...contextFeatures(merged,all)};onBuilding(enriched);setStatus(`${enriched.osmId} selected · ${enriched.footprintArea.toFixed(1)} m² · ${enriched.mergedParts} massing part${enriched.mergedParts>1?"s":""}`);const selectedLayer=L.polygon(enriched.footprint,{color:"#0f766e",weight:3,fillOpacity:.32}).addTo(map);layers.current.push(selectedLayer)}catch{setStatus("This footprint could not be cleaned. Select another building outline.")}};
        all.forEach(item=>{const layer=L.polygon(item.footprint,{color:"#71807b",weight:1,fillOpacity:.08,bubblingMouseEvents:false}).addTo(map);layers.current.push(layer);layer.on("click",()=>select(item))});select(found[0].metric);
      }catch(error){if(requestId===requestVersion.current)setStatus(`Building service unavailable: ${String(error)}`)}});
  })();return()=>{disposed=true;requestVersion.current++;mapRef.current?.remove();mapRef.current=null}},[]);
  useEffect(()=>{mapRef.current?.setView([lat,lon],18)},[lat,lon]);
  useEffect(()=>{let disposed=false;(async()=>{const map=mapRef.current;if(!map)return;const L=await import("leaflet");if(disposed)return;if(datasetLayer.current){map.removeLayer(datasetLayer.current);datasetLayer.current=null}if(datasetFootprint&&datasetFootprint.length>3){const layer=L.polygon(datasetFootprint,{color:"#d97706",weight:4,fillColor:"#f59e0b",fillOpacity:.28}).addTo(map);datasetLayer.current=layer;map.fitBounds(layer.getBounds(),{padding:[45,45],maxZoom:19});setStatus("Dataset building location · orange footprint extracted from the source OBJ")}})();return()=>{disposed=true}},[datasetFootprint]);
  return <div className="open-map"><div ref={host}/><span>{status}</span></div>;
}
