"use client";

import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import type { ExtractedBuilding } from "./OpenBuildingMap";
import { createIsometricCamera, fitIsometricCamera } from "./isometricCamera";
import { createRoofPvArray } from "./roofPvArray";

type Point3 = [number, number, number];

function addQuad(positions:number[],indices:number[],a:Point3,b:Point3,c:Point3,d:Point3){
  const start=positions.length/3;
  positions.push(...a,...b,...c,...d);
  indices.push(start,start+1,start+2,start,start+2,start+3);
}

function surfaceMesh(positions:number[],indices:number[],material:THREE.Material){
  const geometry=new THREE.BufferGeometry();
  geometry.setAttribute("position",new THREE.Float32BufferAttribute(positions,3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  return new THREE.Mesh(geometry,material);
}

export function MassingViewer({building,wwr=.3,pvCoverage=0}:{building:ExtractedBuilding;wwr?:number;pvCoverage?:number}){
  const host=useRef<HTMLDivElement>(null);
  const [error,setError]=useState("");
  useEffect(()=>{
    if(!host.current)return;
    const rawRing=building.footprint.slice(0,-1).filter(p=>p.length===2&&p.every(Number.isFinite));
    const ring=rawRing.filter((p,i)=>i===0||Math.hypot(p[0]-rawRing[i-1][0],p[1]-rawRing[i-1][1])>1e-9);
    if(ring.length<3||!Number.isFinite(building.height)||building.height<=0){setError("This footprint is invalid and was not sent to the 3D renderer.");return}
    setError("");
    const container=host.current,scene=new THREE.Scene();scene.background=new THREE.Color(0xf3f6f5);
    const camera=createIsometricCamera(),renderer=new THREE.WebGLRenderer({antialias:true});
    renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.setSize(container.clientWidth,container.clientHeight);container.replaceChildren(renderer.domElement);
    scene.add(new THREE.HemisphereLight(0xffffff,0x60716c,2.4));const light=new THREE.DirectionalLight(0xffffff,2.4);light.position.set(3,5,4);scene.add(light);

    const lat0=ring.reduce((s,p)=>s+p[0],0)/ring.length,lon0=ring.reduce((s,p)=>s+p[1],0)/ring.length;
    const xy=ring.map(p=>new THREE.Vector2((p[1]-lon0)*111320*Math.cos(lat0*Math.PI/180),(p[0]-lat0)*110540));
    const holeRings=(building.footprintHoles??[]).map(hole=>hole.slice(0,-1).filter(p=>p.length===2&&p.every(Number.isFinite)).map(p=>new THREE.Vector2((p[1]-lon0)*111320*Math.cos(lat0*Math.PI/180),(p[0]-lat0)*110540))).filter(hole=>hole.length>=3);
    const signedArea=THREE.ShapeUtils.area(xy);
    if(!Number.isFinite(signedArea)||Math.abs(signedArea)<1||Math.abs(signedArea)>250000){renderer.dispose();renderer.forceContextLoss();setError("This map footprint has an invalid or unsupported area.");return}

    const group=new THREE.Group(),shape=new THREE.Shape(xy);holeRings.forEach(hole=>shape.holes.push(new THREE.Path(hole)));
    const wallMaterial=new THREE.MeshStandardMaterial({color:0xcbd7d3,roughness:.82,metalness:.02,side:THREE.DoubleSide});
    const slabMaterial=new THREE.MeshStandardMaterial({color:0xdbe6e2,roughness:.88,metalness:.01,side:THREE.DoubleSide});
    const windowMaterial=new THREE.MeshStandardMaterial({color:0x244b58,roughness:.25,metalness:.1,side:THREE.DoubleSide});
    const base=new THREE.Mesh(new THREE.ShapeGeometry(shape),slabMaterial),roof=new THREE.Mesh(new THREE.ShapeGeometry(shape),slabMaterial);roof.position.z=building.height;group.add(base,roof);
    if(pvCoverage>0)group.add(createRoofPvArray(xy,building.height,holeRings));

    const wallPositions:number[]=[],wallIndices:number[]=[],windowPositions:number[]=[],windowIndices:number[]=[];
    const floorHeight=building.height/Math.max(1,building.floors),windowHeight=floorHeight*.5,coverage=Math.min(.9,Math.max(.1,wwr/.5));
    for(const wallRing of [xy,...holeRings])for(let i=0;i<wallRing.length;i++){
      const a=wallRing[i],b=wallRing[(i+1)%wallRing.length],dx=b.x-a.x,dy=b.y-a.y,length=Math.hypot(dx,dy);if(length<.5)continue;
      const ux=dx/length,uy=dy/length,modules=Math.max(1,Math.min(24,Math.floor(length/3.2)));
      const desiredWindowWidth=length*coverage/modules,gap=Math.max(.08,(length-desiredWindowWidth*modules)/(modules+1));
      const windowWidth=Math.max(.1,(length-gap*(modules+1))/modules);
      const point=(distance:number,z:number,offset=0):Point3=>[a.x+ux*distance-uy*offset,a.y+uy*distance+ux*offset,z];
      const ringArea=THREE.ShapeUtils.area(wallRing),outwardOffset=(ringArea>=0?-1:1)*.025;
      for(let floor=0;floor<building.floors;floor++){
        const z0=floor*floorHeight,z1=(floor+1)*floorHeight,wb=z0+floorHeight*.25,wt=Math.min(z1-.12,wb+windowHeight);
        addQuad(wallPositions,wallIndices,point(0,z0),point(length,z0),point(length,wb),point(0,wb));
        addQuad(wallPositions,wallIndices,point(0,wt),point(length,wt),point(length,z1),point(0,z1));
        let cursor=0;
        for(let module=0;module<modules;module++){
          const start=gap+module*(windowWidth+gap),end=Math.min(length-gap,start+windowWidth);
          if(start>cursor)addQuad(wallPositions,wallIndices,point(cursor,wb),point(start,wb),point(start,wt),point(cursor,wt));
          addQuad(windowPositions,windowIndices,point(start,wb,outwardOffset),point(end,wb,outwardOffset),point(end,wt,outwardOffset),point(start,wt,outwardOffset));
          cursor=end;
        }
        if(cursor<length)addQuad(wallPositions,wallIndices,point(cursor,wb),point(length,wb),point(length,wt),point(cursor,wt));
      }
    }
    group.add(surfaceMesh(wallPositions,wallIndices,wallMaterial),surfaceMesh(windowPositions,windowIndices,windowMaterial));

    const outlinePositions:number[]=[];
    for(const outlineRing of [xy,...holeRings])for(let i=0;i<outlineRing.length;i++){
      const a=outlineRing[i],b=outlineRing[(i+1)%outlineRing.length];
      outlinePositions.push(a.x,a.y,0,b.x,b.y,0,a.x,a.y,building.height,b.x,b.y,building.height,a.x,a.y,0,a.x,a.y,building.height);
    }
    const outlineGeometry=new THREE.BufferGeometry();outlineGeometry.setAttribute("position",new THREE.Float32BufferAttribute(outlinePositions,3));
    group.add(new THREE.LineSegments(outlineGeometry,new THREE.LineBasicMaterial({color:0x60716c,transparent:true,opacity:.42})));

    group.rotation.x=-Math.PI/2;group.updateMatrixWorld(true);const box=new THREE.Box3().setFromObject(group),center=box.getCenter(new THREE.Vector3()),size=box.getSize(new THREE.Vector3());group.position.sub(center);scene.add(group);
    const span=Math.max(size.x,size.y,size.z,20);fitIsometricCamera(camera,container,span);const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;
    let frame=0;const animate=()=>{controls.update();renderer.render(scene,camera);frame=requestAnimationFrame(animate)};animate();
    let resizeFrame=0;const observer=new ResizeObserver(()=>{cancelAnimationFrame(resizeFrame);resizeFrame=requestAnimationFrame(()=>{fitIsometricCamera(camera,container,span);renderer.setSize(container.clientWidth,container.clientHeight)})});observer.observe(container);
    return()=>{cancelAnimationFrame(frame);cancelAnimationFrame(resizeFrame);observer.disconnect();controls.dispose();group.traverse(object=>{if(object instanceof THREE.Mesh||object instanceof THREE.LineSegments){object.geometry?.dispose();const materials=Array.isArray(object.material)?object.material:[object.material];materials.forEach(material=>{if(material instanceof THREE.MeshStandardMaterial)material.map?.dispose();material?.dispose()})}});renderer.dispose();renderer.forceContextLoss();container.replaceChildren()};
  },[building,wwr,pvCoverage]);
  return <div className="obj-viewer"><div className="viewer-canvas-host" ref={host}/>{error&&<div className="viewer-error">{error}</div>}</div>;
}
