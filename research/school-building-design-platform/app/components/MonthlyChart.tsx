"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { HourlyPerformancePoint } from "./types";

const COLORS={energy:"#ea6a47",pv:"#19a974",carbonWithoutPv:"#7b8794",carbonWithPv:"#168a7d",netCost:"#8b5cf6",exportRevenue:"#d18b16"};
const UNITS={energy:"kWh",pv:"kWh",carbonWithoutPv:"kgCO₂",carbonWithPv:"kgCO₂",netCost:"CNY",exportRevenue:"CNY"};
type Metric=keyof typeof COLORS;
type Language="en"|"zh";

function monthName(index:number,language:Language,short=false){
  if(language==="zh")return `${index+1}月`;
  const names=short?["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]:["January","February","March","April","May","June","July","August","September","October","November","December"];
  return names[index];
}

export function MonthlyChart({data,metric,language="en"}:{data:HourlyPerformancePoint[];metric:Metric;language?:Language}){
  const ref=useRef<HTMLCanvasElement>(null);
  const [selectedMonth,setSelectedMonth]=useState<number|null>(null);
  const monthly=useMemo(()=>{
    const buckets=Array.from({length:12},(_,month)=>({month,total:0,average:0,count:0}));
    data.forEach((point,index)=>{
      const hourIndex=Math.max(0,(Number.isFinite(point.hour)?point.hour:index+1)-1);
      const month=new Date(Date.UTC(2025,0,1,hourIndex)).getUTCMonth();
      const value=point[metric];
      if(Number.isFinite(value)){buckets[month].total+=value;buckets[month].count+=1}
    });
    buckets.forEach(bucket=>{bucket.average=bucket.count?bucket.total/bucket.count:0});
    return buckets;
  },[data,metric]);

  useEffect(()=>{
    const canvas=ref.current;if(!canvas||!data.length)return;
    const ratio=Math.min(devicePixelRatio,2),w=canvas.clientWidth,h=canvas.clientHeight;
    canvas.width=w*ratio;canvas.height=h*ratio;
    const ctx=canvas.getContext("2d");if(!ctx)return;
    ctx.scale(ratio,ratio);ctx.clearRect(0,0,w,h);
    const pad={l:70,r:68,t:46,b:42},plotW=w-pad.l-pad.r,plotH=h-pad.t-pad.b;
    const maxTotal=Math.max(...monthly.map(item=>item.total),1)*1.12,maxAverage=Math.max(...monthly.map(item=>item.average),1)*1.18;
    const step=plotW/12,barW=Math.min(54,step*.58),x=(index:number)=>pad.l+step*(index+.5),yTotal=(value:number)=>pad.t+plotH*(1-value/maxTotal),yAverage=(value:number)=>pad.t+plotH*(1-value/maxAverage);

    ctx.font="11px Arial";ctx.fillStyle="#586560";
    ctx.fillText(language==="zh"?`月累计（${UNITS[metric]}）`:`Monthly total (${UNITS[metric]})`,4,17);
    const rightLabel=language==="zh"?`逐时均值（${UNITS[metric]}/h）`:`Hourly mean (${UNITS[metric]}/h)`;
    ctx.fillText(rightLabel,Math.max(pad.l,w-ctx.measureText(rightLabel).width-4),17);
    for(let tick=0;tick<5;tick++){
      const py=pad.t+plotH*tick/4;
      ctx.strokeStyle="#d9e0dd";ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(pad.l,py);ctx.lineTo(w-pad.r,py);ctx.stroke();
      ctx.fillStyle="#73807c";ctx.font="11px Arial";
      const left=(maxTotal*(1-tick/4)).toFixed(maxTotal>=100?0:1),right=(maxAverage*(1-tick/4)).toFixed(maxAverage>=100?0:1);
      ctx.fillText(left,4,py+4);ctx.fillText(right,w-pad.r+8,py+4);
    }

    monthly.forEach((item,index)=>{
      const px=x(index),top=yTotal(item.total);
      ctx.fillStyle=`${COLORS[metric]}b8`;ctx.fillRect(px-barW/2,top,barW,pad.t+plotH-top);
      if(selectedMonth===index){ctx.strokeStyle="#15201d";ctx.lineWidth=2;ctx.strokeRect(px-barW/2-2,top-2,barW+4,pad.t+plotH-top+4)}
      ctx.fillStyle="#66736f";ctx.font="11px Arial";const label=monthName(index,language,true);ctx.fillText(label,px-ctx.measureText(label).width/2,h-12);
    });

    ctx.strokeStyle="#164f49";ctx.lineWidth=2.2;ctx.beginPath();monthly.forEach((item,index)=>index?ctx.lineTo(x(index),yAverage(item.average)):ctx.moveTo(x(index),yAverage(item.average)));ctx.stroke();
    monthly.forEach((item,index)=>{ctx.fillStyle="#fff";ctx.strokeStyle="#164f49";ctx.lineWidth=2;ctx.beginPath();ctx.arc(x(index),yAverage(item.average),4,0,Math.PI*2);ctx.fill();ctx.stroke()});
  },[data,language,metric,monthly,selectedMonth]);

  const inspect=(event:React.MouseEvent<HTMLCanvasElement>)=>{const bounds=event.currentTarget.getBoundingClientRect(),left=70,right=68,pointer=Math.max(left,Math.min(bounds.width-right,event.clientX-bounds.left)),index=Math.max(0,Math.min(11,Math.floor((pointer-left)/(bounds.width-left-right)*12)));setSelectedMonth(index)};
  const selected=selectedMonth==null?null:monthly[selectedMonth];
  const maximum=monthly.reduce((best,item)=>item.total>best.total?item:best,monthly[0]);
  const minimum=monthly.reduce((best,item)=>item.total<best.total?item:best,monthly[0]);

  return <div className="hourly-chart-wrap monthly-chart-wrap">
    <div className="monthly-legend"><span><i className="monthly-bar-key" style={{background:COLORS[metric]}}/>{language==="zh"?"月累计":"Monthly total"}</span><span><i className="monthly-line-key"/>{language==="zh"?"逐时均值":"Hourly mean"}</span></div>
    <canvas className="hourly-chart monthly-chart" ref={ref} onClick={inspect} aria-label={language==="zh"?"逐月柱状与点线组合图":"Monthly bar and point-line chart"}/>
    {selected?<div className="selected-hour selected-month"><span>{language==="zh"?"选中月份":"Selected month"}</span><b>{selected.total.toFixed(2)} <small>{UNITS[metric]}</small></b><strong>{monthName(selected.month,language)} · {language==="zh"?"逐时均值":"hourly mean"} {selected.average.toFixed(3)} {UNITS[metric]}/h</strong></div>:null}
    <div className="hourly-extrema"><span><i style={{background:COLORS[metric]}}/>{language==="zh"?"月累计最大值":"Maximum monthly total"} <b>{maximum.total.toFixed(2)} {UNITS[metric]}</b> · {monthName(maximum.month,language)}</span><span><i style={{background:COLORS[metric]}}/>{language==="zh"?"月累计最小值":"Minimum monthly total"} <b>{minimum.total.toFixed(2)} {UNITS[metric]}</b> · {monthName(minimum.month,language)}</span></div>
  </div>;
}
