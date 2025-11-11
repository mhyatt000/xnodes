import numpy as np
import open3d as o3d

X = np.load("X_world.npy")

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(X)

pcd.paint_uniform_color([0.2, 0.7, 1.0])

axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)

o3d.visualization.draw_geometries([pcd, axes])
