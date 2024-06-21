#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import open3d as o3d


def getCameraBoundaryMesh(boundary_point_array, camera_point):
    verts = np.vstack((boundary_point_array, camera_point.reshape(1, -1)))

    triangles = np.array([[0, 2, 4], [2, 3, 4], [3, 1, 4], [1, 0, 4]])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()

    o3d.visualization.draw_geometries([mesh])
    return mesh
