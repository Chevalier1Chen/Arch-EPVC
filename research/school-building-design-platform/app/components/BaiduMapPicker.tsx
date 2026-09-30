"use client";

import { useEffect, useRef, useState } from "react";
import { KeyRound, MapPinned } from "lucide-react";

declare global { interface Window { BMapGL?: any; BMAP_SATELLITE_MAP?: any; __archEpvcBaiduReady?: () => void } }

function bd09ToWgs84(bdLon:number,bdLat:number){
  const x=bdLon-.0065,y=bdLat-.006,z=Math.sqrt(x*x+y*y)-.00002*Math.sin(y*Math.PI*3000/180),theta=Math.atan2(y,x)-.000003*Math.cos(x*Math.PI*3000/180);
  const gcjLon=z*Math.cos(theta),gcjLat=z*Math.sin(theta);
  const a=6378245,ee=.00669342162296594323,dLat=transformLat(gcjLon-105,gcjLat-35),dLon=transformLon(gcjLon-105,gcjLat-35),rad=gcjLat/180*Math.PI;
  let magic=Math.sin(rad);magic=1-ee*magic*magic;const sqrt=Math.sqrt(magic);
  const mgLat=gcjLat+(dLat*180)/((a*(1-ee))/(magic*sqrt)*Math.PI),mgLon=gcjLon+(dLon*180)/(a/sqrt*Math.cos(rad)*Math.PI);
  return {lat:gcjLat*2-mgLat,lon:gcjLon*2-mgLon};
}
function transformLat(x:number,y:number){let r=-100+2*x+3*y+.2*y*y+.1*x*y+.2*Math.sqrt(Math.abs(x));r+=(20*Math.sin(6*x*Math.PI)+20*Math.sin(2*x*Math.PI))*2/3;r+=(20*Math.sin(y*Math.PI)+40*Math.sin(y/3*Math.PI))*2/3;r+=(160*Math.sin(y/12*Math.PI)+320*Math.sin(y*Math.PI/30))*2/3;return r}
function transformLon(x:number,y:number){let r=300+x+2*y+.1*x*x+.1*x*y+.1*Math.sqrt(Math.abs(x));r+=(20*Math.sin(6*x*Math.PI)+20*Math.sin(2*x*Math.PI))*2/3;r+=(20*Math.sin(x*Math.PI)+40*Math.sin(x/3*Math.PI))*2/3;r+=(150*Math.sin(x/12*Math.PI)+300*Math.sin(x/30*Math.PI))*2/3;return r}
function wgs84ToBd09(lon:number,lat:number){
  const a=6378245,ee=.00669342162296594323,rad=lat/180*Math.PI;let magic=Math.sin(rad);magic=1-ee*magic*magic;const sqrt=Math.sqrt(magic);
  const dLat=(transformLat(lon-105,lat-35)*180)/((a*(1-ee))/(magic*sqrt)*Math.PI),dLon=(transformLon(lon-105,lat-35)*180)/(a/sqrt*Math.cos(rad)*Math.PI);
  const gLon=lon+dLon,gLat=lat+dLat,x=gLon,y=gLat,z=Math.sqrt(x*x+y*y)+.00002*Math.sin(y*Math.PI*3000/180),theta=Math.atan2(y,x)+.000003*Math.cos(x*Math.PI*3000/180);
  return {lon:z*Math.cos(theta)+.0065,lat:z*Math.sin(theta)+.006};
}

export function BaiduMapPicker({lat,lon,onPick}:{lat:number;lon:number;onPick:(lat:number,lon:number)=>void}){
  const host=useRef<HTMLDivElement>(null),mapRef=useRef<any>(null);
  const [ak,setAk]=useState(""),[draft,setDraft]=useState(""),[error,setError]=useState("");
  useEffect(()=>setAk(localStorage.getItem("arch_epvc_baidu_ak")||""),[]);
  useEffect(()=>{
    if(!ak||!host.current)return;
    const start=()=>{
      if(!host.current||!window.BMapGL)return;
      const center=wgs84ToBd09(lon,lat);const map=new window.BMapGL.Map(host.current);map.centerAndZoom(new window.BMapGL.Point(center.lon,center.lat),18);map.enableScrollWheelZoom(true);map.setMapType(window.BMAP_SATELLITE_MAP);mapRef.current=map;
      let marker:any=null;
      map.addEventListener("click",(event:any)=>{if(marker)map.removeOverlay(marker);marker=new window.BMapGL.Marker(event.latlng);map.addOverlay(marker);const wgs=bd09ToWgs84(event.latlng.lng,event.latlng.lat);onPick(wgs.lat,wgs.lon)});
    };
    if(window.BMapGL){start();return}
    window.__archEpvcBaiduReady=start;
    const script=document.createElement("script");script.src=`https://api.map.baidu.com/api?type=webgl&v=1.0&ak=${encodeURIComponent(ak)}&callback=__archEpvcBaiduReady`;script.onerror=()=>setError("Baidu Maps could not be loaded. Check the AK and allowed referrer.");document.head.appendChild(script);
  },[ak]);
  useEffect(()=>{if(mapRef.current&&window.BMapGL){const p=wgs84ToBd09(lon,lat);mapRef.current.panTo(new window.BMapGL.Point(p.lon,p.lat))}},[lat,lon]);
  const save=()=>{if(!draft.trim())return;localStorage.setItem("arch_epvc_baidu_ak",draft.trim());location.reload()};
  if(!ak)return <div className="map-key"><MapPinned size={27}/><b>Connect Baidu Maps</b><p>Enter a browser-side Baidu Maps AK to enable satellite building selection.</p><div><KeyRound size={15}/><input value={draft} onChange={e=>setDraft(e.target.value)} placeholder="Baidu Maps browser AK"/><button onClick={save}>Connect map</button></div></div>;
  return <div className="baidu-map"><div ref={host}/><span>Satellite mode · click a teaching building to set WGS84 coordinates</span>{error&&<p>{error}</p>}</div>;
}
