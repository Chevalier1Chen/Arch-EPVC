import * as THREE from "three";

function moduleMaterial(){
  const canvas=document.createElement("canvas");
  canvas.width=128;canvas.height=192;
  const ctx=canvas.getContext("2d")!;
  const gradient=ctx.createLinearGradient(0,0,128,192);
  gradient.addColorStop(0,"#0b3157");gradient.addColorStop(.55,"#164f82");gradient.addColorStop(1,"#082844");
  ctx.fillStyle=gradient;ctx.fillRect(0,0,128,192);
  ctx.strokeStyle="#79aaca";ctx.lineWidth=2;
  for(let x=0;x<=128;x+=32){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,192);ctx.stroke()}
  for(let y=0;y<=192;y+=32){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(128,y);ctx.stroke()}
  ctx.strokeStyle="#d4e4ed";ctx.lineWidth=5;ctx.strokeRect(2,2,124,188);
  const map=new THREE.CanvasTexture(canvas);map.colorSpace=THREE.SRGBColorSpace;
  return new THREE.MeshStandardMaterial({map,color:0xffffff,roughness:.28,metalness:.38});
}

function inside(point:THREE.Vector2,polygon:THREE.Vector2[]){
  let hit=false;
  for(let i=0,j=polygon.length-1;i<polygon.length;j=i++){
    const a=polygon[i],b=polygon[j];
    if((a.y>point.y)!==(b.y>point.y)&&point.x<(b.x-a.x)*(point.y-a.y)/(b.y-a.y)+a.x)hit=!hit;
  }
  return hit;
}

export function createRoofPvArray(polygon:THREE.Vector2[],roofZ:number,holes:THREE.Vector2[][]=[]){
  const array=new THREE.Group();array.name="roof-boundary-full-coverage-pv-array";
  const origin=new THREE.Box2().setFromPoints(polygon).getCenter(new THREE.Vector2());
  let roofAngle=0,longestEdge=0;
  for(let i=0;i<polygon.length;i++){
    const a=polygon[i],b=polygon[(i+1)%polygon.length],edgeLength=a.distanceToSquared(b);
    if(edgeLength>longestEdge){longestEdge=edgeLength;roofAngle=Math.atan2(b.y-a.y,b.x-a.x)}
  }
  const rotate=(point:THREE.Vector2,angle:number)=>point.clone().sub(origin).rotateAround(new THREE.Vector2(),angle);
  const localPolygon=polygon.map(point=>rotate(point,-roofAngle));
  const localHoles=holes.map(hole=>hole.map(point=>rotate(point,-roofAngle)));
  const bounds=new THREE.Box2().setFromPoints(localPolygon),size=bounds.getSize(new THREE.Vector2());
  const MAX_MODULES=1600,roofArea=Math.max(0,Math.abs(THREE.ShapeUtils.area(polygon))-holes.reduce((sum,hole)=>sum+Math.abs(THREE.ShapeUtils.area(hole)),0)),baseModuleArea=1.75*1.08;
  const moduleScale=Math.max(1,Math.sqrt(roofArea/baseModuleArea/MAX_MODULES));
  // Align the module rows to the roof's longest boundary instead of imposing a compass direction.
  const panelX=1.75*moduleScale,panelY=1.08*moduleScale;
  const gapX=.08*moduleScale,gapY=.12*moduleScale,setback=Math.max(.06,.08*moduleScale);
  const centers:THREE.Vector2[]=[];
  const corners=(x:number,y:number)=>[
    new THREE.Vector2(x-panelX/2,y-panelY/2),new THREE.Vector2(x+panelX/2,y-panelY/2),
    new THREE.Vector2(x+panelX/2,y+panelY/2),new THREE.Vector2(x-panelX/2,y+panelY/2),
  ];
  for(let y=bounds.min.y+setback+panelY/2;y<=bounds.max.y-setback-panelY/2&&centers.length<MAX_MODULES;y+=panelY+gapY){
    for(let x=bounds.min.x+setback+panelX/2;x<=bounds.max.x-setback-panelX/2&&centers.length<MAX_MODULES;x+=panelX+gapX){
      if(corners(x,y).every(point=>inside(point,localPolygon)&&!localHoles.some(hole=>inside(point,hole))))centers.push(new THREE.Vector2(x,y));
    }
  }
  if(!centers.length){const center=bounds.getCenter(new THREE.Vector2());if(inside(center,localPolygon)&&!localHoles.some(hole=>inside(center,hole)))centers.push(center)}
  const width=centers.length===1?Math.max(.45,Math.min(panelX,size.x*.55)):panelX;
  const depth=centers.length===1?Math.max(.7,Math.min(panelY,size.y*.55)):panelY;
  const moduleGeometry=new THREE.BoxGeometry(width,depth,.075),frameGeometry=new THREE.BoxGeometry(width+.09,depth+.09,.045);
  const modules=new THREE.InstancedMesh(moduleGeometry,moduleMaterial(),centers.length),frames=new THREE.InstancedMesh(frameGeometry,new THREE.MeshStandardMaterial({color:0xc8d1d6,roughness:.35,metalness:.78}),centers.length);
  const tilt=THREE.MathUtils.degToRad(8),tiltQuaternion=new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1,0,0),tilt);
  const quaternion=new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0,0,1),roofAngle).multiply(tiltQuaternion),scale=new THREE.Vector3(1,1,1),matrix=new THREE.Matrix4();
  const lift=depth*Math.sin(tilt)/2+.16;
  centers.forEach((center,index)=>{
    const worldCenter=center.clone().rotateAround(new THREE.Vector2(),roofAngle).add(origin);
    matrix.compose(new THREE.Vector3(worldCenter.x,worldCenter.y,roofZ+lift),quaternion,scale);modules.setMatrixAt(index,matrix);
    matrix.compose(new THREE.Vector3(worldCenter.x,worldCenter.y,roofZ+lift-.055),quaternion,scale);frames.setMatrixAt(index,matrix);
  });
  modules.instanceMatrix.needsUpdate=true;frames.instanceMatrix.needsUpdate=true;
  modules.castShadow=true;modules.receiveShadow=true;array.add(frames,modules);array.userData.panelCount=centers.length;
  return array;
}
