import { NextRequest, NextResponse } from "next/server";

export async function GET(request:NextRequest){
  const lat=Number(request.nextUrl.searchParams.get("lat")),lon=Number(request.nextUrl.searchParams.get("lon"));
  if(!Number.isFinite(lat)||!Number.isFinite(lon))return NextResponse.json({error:"Valid lat and lon are required"},{status:400});
  const dy=90/110540,dx=90/(111320*Math.cos(lat*Math.PI/180)),bbox=`${lon-dx},${lat-dy},${lon+dx},${lat+dy}`;
  try{
    const upstream=await fetch(`https://api.openstreetmap.org/api/0.6/map?bbox=${bbox}`,{headers:{Accept:"application/xml","User-Agent":"Arch-EPVC-research-platform/1.0"},signal:AbortSignal.timeout(15000)});
    if(!upstream.ok)throw new Error(`OSM HTTP ${upstream.status}`);
    return new NextResponse(await upstream.text(),{headers:{"Content-Type":"application/xml; charset=utf-8","Cache-Control":"public, max-age=300"}});
  }catch(error){return NextResponse.json({error:String(error)},{status:502})}
}
