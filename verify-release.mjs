import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {fileURLToPath} from 'node:url';
const root=path.dirname(fileURLToPath(import.meta.url));
function walk(dir) {
  return fs.readdirSync(dir,{withFileTypes:true}).flatMap(e=>e.name==='.git'?[]:e.isDirectory()?walk(path.join(dir,e.name)):[path.join(dir,e.name)]);
}
const files=walk(root);
const problems=[];
for(const f of files){
  if(fs.statSync(f).size>=100*1024*1024) problems.push(`Large repository file: ${path.relative(root,f)}`);
  if(/^\.env(?:\.|$)/.test(path.basename(f))) problems.push(`Environment file: ${path.relative(root,f)}`);
}
const html=fs.readFileSync(path.join(root,'docs/index.html'),'utf8');
for(const [,link] of html.matchAll(/(?:src|href)="([^"]+)"/g)){
  if(/^(#|https?:)/.test(link)) continue;
  if(!fs.existsSync(path.resolve(root,'docs',link))) problems.push(`Missing page resource: ${link}`);
}
const rows=files.filter(f=>!['release-file-manifest.json','release-verification.json'].includes(path.basename(f))).map(f=>({path:path.relative(root,f).replaceAll('\\','/'),bytes:fs.statSync(f).size,sha256:crypto.createHash('sha256').update(fs.readFileSync(f)).digest('hex')}));
fs.writeFileSync(path.join(root,'release-file-manifest.json'),JSON.stringify(rows,null,2)+'\n');
const report={checked_at:new Date().toISOString(),file_count:rows.length,bytes:rows.reduce((n,x)=>n+x.bytes,0),page_local_links_valid:!problems.some(x=>x.startsWith('Missing')),problems,scope:'Packaging and local page resources only; not model accuracy, dataset quality or platform runtime verification.'};
fs.writeFileSync(path.join(root,'release-verification.json'),JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify(report,null,2));
if(problems.length)process.exitCode=1;
