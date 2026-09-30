import * as THREE from "three";

export function createIsometricCamera() {
  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 100000);
  camera.up.set(0, 1, 0);
  return camera;
}

export function fitIsometricCamera(
  camera: THREE.OrthographicCamera,
  container: HTMLElement,
  span: number,
) {
  const width = Math.max(container.clientWidth, 1);
  const height = Math.max(container.clientHeight, 1);
  const aspect = width / height;
  const paddedHalfSpan = Math.max(span, 20) * 0.9;
  const halfHeight = Math.max(paddedHalfSpan, paddedHalfSpan / aspect);
  camera.left = -halfHeight * aspect;
  camera.right = halfHeight * aspect;
  camera.top = halfHeight;
  camera.bottom = -halfHeight;
  const distance = Math.max(span * 3, 100);
  camera.position.set(distance, distance, distance);
  camera.lookAt(0, 0, 0);
  camera.updateProjectionMatrix();
}
