# -*- coding: utf-8 -*-
# GhPython helper: split building meshes into roof faces and south facade faces.
#
# Inputs:
#   M                Mesh or list of Mesh from OBJ reader
#   N                object names, optional
#   roof_z_min       optional, default 0.75
#   south_angle_deg  optional, default 60
#   min_area         optional, default 0.01
#
# Outputs:
#   RoofM
#   SouthM
#   WallM
#   RoofA
#   SouthA
#   log

import math
import Rhino.Geometry as rg


def to_list(value):
    if value is None:
        return []
    if isinstance(value, rg.Mesh):
        return [value]
    if isinstance(value, (list, tuple)):
        return list(value)
    if hasattr(value, "Branches"):
        result = []
        for branch in value.Branches:
            result.extend(list(branch))
        return result
    try:
        return list(value)
    except:
        return [value]


def clamp(value, low, high):
    return max(low, min(high, value))


def mesh_area(mesh):
    if mesh is None or mesh.Faces.Count == 0:
        return 0.0
    amp = rg.AreaMassProperties.Compute(mesh)
    if amp is None:
        return 0.0
    return amp.Area


def face_vertex_ids(face):
    if face.IsQuad:
        return [face.A, face.B, face.C, face.D]
    return [face.A, face.B, face.C]


def make_submesh(source, face_indices):
    sub = rg.Mesh()
    local = {}

    def local_id(global_id):
        if global_id not in local:
            p = source.Vertices[global_id]
            local[global_id] = sub.Vertices.Add(p.X, p.Y, p.Z)
        return local[global_id]

    for face_index in face_indices:
        face = source.Faces[face_index]
        ids = [local_id(i) for i in face_vertex_ids(face)]
        if len(ids) == 3:
            sub.Faces.AddFace(ids[0], ids[1], ids[2])
        elif len(ids) == 4:
            sub.Faces.AddFace(ids[0], ids[1], ids[2], ids[3])

    sub.Normals.ComputeNormals()
    sub.Compact()
    return sub


def face_center(mesh, face):
    ids = face_vertex_ids(face)
    x = y = z = 0.0
    for i in ids:
        p = mesh.Vertices[i]
        x += p.X
        y += p.Y
        z += p.Z
    n = float(len(ids))
    return rg.Point3d(x / n, y / n, z / n)


def corrected_face_normal(mesh, face_index, building_center):
    n = rg.Vector3d(mesh.FaceNormals[face_index])
    face = mesh.Faces[face_index]
    c = face_center(mesh, face)
    radial = rg.Vector3d(c.X - building_center.X, c.Y - building_center.Y, 0)
    n_horizontal = rg.Vector3d(n.X, n.Y, 0)

    if radial.Length > 1e-9 and n_horizontal.Length > 1e-9:
        if rg.Vector3d.Multiply(radial, n_horizontal) < 0:
            n.Reverse()

    if n.Length > 1e-9:
        n.Unitize()
    return n, c


def split_mesh(mesh, roof_z_threshold, south_angle, min_face_area):
    mesh.FaceNormals.ComputeFaceNormals()
    mesh.Normals.ComputeNormals()

    bbox = mesh.GetBoundingBox(True)
    building_center = bbox.Center
    max_z = bbox.Max.Z
    z_tol = max(0.05, (bbox.Max.Z - bbox.Min.Z) * 0.02)

    roof_faces = []
    south_faces = []
    wall_faces = []

    for i in range(mesh.Faces.Count):
        face = mesh.Faces[i]
        n, c = corrected_face_normal(mesh, i, building_center)

        # Quick area check on a temporary one-face mesh.
        single = make_submesh(mesh, [i])
        if mesh_area(single) < min_face_area:
            continue

        is_top_zone = c.Z >= max_z - z_tol
        is_roof = (n.Z >= roof_z_threshold) or (is_top_zone and abs(n.Z) > 0.45)

        if is_roof:
            roof_faces.append(i)
            continue

        is_vertical = abs(n.Z) <= 0.35
        if is_vertical:
            wall_faces.append(i)
            horizontal = rg.Vector3d(n.X, n.Y, 0)
            if horizontal.Length > 1e-9:
                horizontal.Unitize()
                # South is negative Y in this dataset.
                south_dot = clamp(-horizontal.Y, -1.0, 1.0)
                angle = math.degrees(math.acos(south_dot))
                if angle <= south_angle:
                    south_faces.append(i)

    return (
        make_submesh(mesh, roof_faces),
        make_submesh(mesh, south_faces),
        make_submesh(mesh, wall_faces),
        len(roof_faces),
        len(south_faces),
        len(wall_faces),
    )


meshes = to_list(M)
names = to_list(N)

if roof_z_min is None:
    roof_z_min = 0.75
if south_angle_deg is None:
    south_angle_deg = 60.0
if min_area is None:
    min_area = 0.01

roof_z_min = float(roof_z_min)
south_angle_deg = float(south_angle_deg)
min_area = float(min_area)

RoofM = []
SouthM = []
WallM = []
RoofA = []
SouthA = []
messages = []

for index, mesh in enumerate(meshes):
    if mesh is None or mesh.Faces.Count == 0:
        continue

    name = names[index] if index < len(names) else "mesh_{}".format(index)
    roof, south, walls, roof_count, south_count, wall_count = split_mesh(
        mesh,
        roof_z_min,
        south_angle_deg,
        min_area,
    )

    RoofM.append(roof)
    SouthM.append(south)
    WallM.append(walls)
    RoofA.append(round(mesh_area(roof), 3))
    SouthA.append(round(mesh_area(south), 3))

    messages.append(
        "{} | roof_faces={} south_faces={} wall_faces={} roof_area={} south_area={}".format(
            name,
            roof_count,
            south_count,
            wall_count,
            RoofA[-1],
            SouthA[-1],
        )
    )

log = "\n".join(messages)
