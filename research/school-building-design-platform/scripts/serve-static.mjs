import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { execFile } from "node:child_process";
import { promisify } from "node:util";

const root = path.resolve(process.cwd(), "local-dist");
const execFileAsync=promisify(execFile);
const types = { ".html":"text/html; charset=utf-8", ".js":"text/javascript; charset=utf-8", ".css":"text/css; charset=utf-8", ".json":"application/json; charset=utf-8", ".obj":"text/plain; charset=utf-8", ".svg":"image/svg+xml", ".wasm":"application/wasm", ".onnx":"application/octet-stream", ".f32":"application/octet-stream" };
http.createServer(async(req,res)=>{
  const requestUrl=new URL(req.url,"http://localhost"),pathname=decodeURIComponent(requestUrl.pathname);
  if(pathname==="/api/osm-map"){
    const lat=Number(requestUrl.searchParams.get("lat")),lon=Number(requestUrl.searchParams.get("lon"));
    if(!Number.isFinite(lat)||!Number.isFinite(lon)){res.writeHead(400,{"Content-Type":"application/json"});res.end(JSON.stringify({error:"Valid lat and lon are required"}));return}
    const dy=90/110540,dx=90/(111320*Math.cos(lat*Math.PI/180)),bbox=`${lon-dx},${lat-dy},${lon+dx},${lat+dy}`;
    try{const command=`$ProgressPreference='SilentlyContinue'; (Invoke-WebRequest -UseBasicParsing 'https://api.openstreetmap.org/api/0.6/map?bbox=${bbox}' -TimeoutSec 15).Content`,result=await execFileAsync("powershell.exe",["-NoProfile","-NonInteractive","-Command",command],{timeout:20000,maxBuffer:20*1024*1024}),xml=result.stdout;if(!xml.includes("<osm"))throw new Error("OSM returned no XML map data");res.writeHead(200,{"Content-Type":"application/xml; charset=utf-8","Cache-Control":"public, max-age=300"});res.end(xml)}catch(error){res.writeHead(502,{"Content-Type":"application/json"});res.end(JSON.stringify({error:String(error)}))}return;
  }
  const requested=path.resolve(root,"."+pathname);
  const file=requested.startsWith(root)&&fs.existsSync(requested)&&fs.statSync(requested).isFile()?requested:path.join(root,"index.html");
  res.setHeader("Content-Type",types[path.extname(file)]||"application/octet-stream");
  fs.createReadStream(file).pipe(res);
}).listen(4173,"127.0.0.1",()=>console.log("EduTwin local: http://127.0.0.1:4173"));
