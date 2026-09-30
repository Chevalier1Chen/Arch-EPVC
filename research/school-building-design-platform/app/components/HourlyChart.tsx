"use client";

import { useEffect, useRef, useState } from "react";
import type { HourlyPerformancePoint } from "./types";

const COLORS={energy:"#ea6a47",pv:"#19a974",carbonWithoutPv:"#7b8794",carbonWithPv:"#168a7d",netCost:"#8b5cf6",exportRevenue:"#d18b16"};
const UNITS={energy:"kWh",pv:"kWh",carbonWithoutPv:"kgCO₂",carbonWithPv:"kgCO₂",netCost:"CNY",exportRevenue:"CNY"};
const stamp=(hourIndex:number,language:"en"|"zh")=>{const index=Math.max(0,hourIndex-1),date=new Date(Date.UTC(2025,0,1,Math.floor(index))),hour=index%24;if(language==="zh")return `${date.toLocaleDateString("zh-CN",{month:"numeric",day:"numeric",timeZone:"UTC"})} ${String(hour).padStart(2,"0")}:00（第${index+1}小时）`;return `${date.toLocaleDateString("en-US",{month:"short",day:"numeric",timeZone:"UTC"})}, ${String(hour).padStart(2,"0")}:00 (Hour ${index+1})`};

export function HourlyChart({data,metric,language="en"}:{data:HourlyPerformancePoint[];metric:keyof typeof COLORS;language?:"en"|"zh"}){
  const ref=useRef<HTMLCanvasElement>(null);
  const [selectedIndex,setSelectedIndex]=useState<number|null>(null);
  const values=data.map(d=>d[metric]),maximum=values.length?Math.max(...values):0,minimum=values.length?Math.min(...values):0;
  const maximumIndex=values.indexOf(maximum),minimumIndex=values.indexOf(minimum),isCarbonWithout=metric==="carbonWithoutPv";

  useEffect(()=>{
    const canvas=ref.current;if(!canvas||!data.length)return;const ratio=Math.min(devicePixelRatio,2),w=canvas.clientWidth,h=canvas.clientHeight;canvas.width=w*ratio;canvas.height=h*ratio;const ctx=canvas.getContext("2d")!;ctx.scale(ratio,ratio);ctx.clearRect(0,0,w,h);
    const pad={l:62,r:24,t:48,b:38};
    const series=[{label:"",color:COLORS[metric],values:data.map(d=>d[metric])}];
    const all=series.flatMap(item=>item.values),maximum=Math.max(...all),minimum=Math.min(...all),max=Math.max(maximum,1),min=Math.min(minimum,0),x=(i:number)=>pad.l+(w-pad.l-pad.r)*i/(data.length-1),y=(v:number)=>pad.t+(h-pad.t-pad.b)*(max-v)/(max-min||1);
    ctx.fillStyle="#586560";ctx.font="bold 11px Arial";ctx.fillText(UNITS[metric],4,16);
    ctx.strokeStyle="#d9e0dd";ctx.lineWidth=1;ctx.fillStyle="#73807c";ctx.font="11px Arial";for(let i=0;i<5;i++){const py=pad.t+(h-pad.t-pad.b)*i/4;ctx.beginPath();ctx.moveTo(pad.l,py);ctx.lineTo(w-pad.r,py);ctx.stroke();ctx.fillText((max-(max-min)*i/4).toFixed(1),3,py+4)}
    const monthLabels=language==="zh"?[["1月",0],["4月",.25],["7月",.5],["10月",.75],["12月",1]]:[["Jan",0],["Apr",.25],["Jul",.5],["Oct",.75],["Dec",1]];monthLabels.forEach(([label,pos])=>ctx.fillText(String(label),pad.l+(w-pad.l-pad.r)*Number(pos)-(Number(pos)===1?20:0),h-9));
    for(const item of series){ctx.strokeStyle=item.color;ctx.lineWidth=isCarbonWithout?1.45:metric==="carbonWithPv"?1.9:1.2;ctx.setLineDash(isCarbonWithout?[6,4]:[]);ctx.globalAlpha=isCarbonWithout?.84:1;ctx.beginPath();item.values.forEach((value,i)=>i?ctx.lineTo(x(i),y(value)):ctx.moveTo(x(i),y(value)));ctx.stroke();ctx.setLineDash([]);ctx.globalAlpha=1}
    const marker=(index:number,value:number,label:string,above:boolean)=>{const px=x(index),py=y(value);ctx.fillStyle=COLORS[metric];ctx.beginPath();ctx.arc(px,py,4,0,Math.PI*2);ctx.fill();const markerText=`${label} ${value.toFixed(2)} · ${stamp(data[index].hour,language)}`;ctx.font="bold 10px Arial";const tw=ctx.measureText(markerText).width,tx=Math.max(4,Math.min(w-tw-4,px-tw/2)),ty=above?Math.max(28,py-10):Math.min(h-18,py+17);ctx.fillStyle="#15201d";ctx.fillText(markerText,tx,ty)};const active=data.map(d=>d[metric]),high=Math.max(...active),low=Math.min(...active);marker(active.indexOf(high),high,language==="zh"?"最大值":"MAX",true);marker(active.indexOf(low),low,language==="zh"?"最小值":"MIN",false);
    if(selectedIndex!=null&&data[selectedIndex]){const px=x(selectedIndex);ctx.strokeStyle="#15201d88";ctx.setLineDash([4,4]);ctx.beginPath();ctx.moveTo(px,pad.t);ctx.lineTo(px,h-pad.b);ctx.stroke();ctx.setLineDash([]);for(const item of series){const py=y(item.values[selectedIndex]);ctx.fillStyle="#fff";ctx.strokeStyle=item.color;ctx.lineWidth=3;ctx.beginPath();ctx.arc(px,py,5,0,Math.PI*2);ctx.fill();ctx.stroke()}}
  },[data,metric,selectedIndex,isCarbonWithout,language]);

  const inspect=(event:React.MouseEvent<HTMLCanvasElement>)=>{if(!data.length)return;const bounds=event.currentTarget.getBoundingClientRect(),left=62,right=24,pointer=Math.max(left,Math.min(bounds.width-right,event.clientX-bounds.left)),index=Math.round((pointer-left)/(bounds.width-left-right)*(data.length-1));setSelectedIndex(index)};
  const selected=selectedIndex!=null?data[selectedIndex]:null;
  return <div className="hourly-chart-wrap"><canvas className="hourly-chart" ref={ref} onClick={inspect}/>{selected&&<div className="selected-hour"><span>Selected hour</span><b className={metric==="carbonWithPv"?"with-pv":undefined}>{selected[metric].toFixed(3)} <small>{UNITS[metric]}</small></b><strong>{stamp(selected.hour,language)}</strong></div>}{data.length>0&&<div className="hourly-extrema"><span><i style={{background:COLORS[metric]}}/>Annual maximum <b>{maximum.toFixed(2)} {UNITS[metric]}</b> · {stamp(data[maximumIndex].hour,language)}</span><span><i style={{background:COLORS[metric]}}/>Annual minimum <b>{minimum.toFixed(2)} {UNITS[metric]}</b> · {stamp(data[minimumIndex].hour,language)}</span></div>}</div>;
}
