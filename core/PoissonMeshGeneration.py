import open3d as o3d
import numpy as np
from utils.meshUtils import smooth_point_cloud
from utils.meshUtils import densify_pcd

def poissonDepth():
    pcd = o3d.io.read_point_cloud("output/sparse_points.ply")
    if len(pcd.points) == 0:
        raise ValueError("Point cloud is empty.")

    # 1. cleaning
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=2)
    pcd = smooth_point_cloud(pcd, k=8, alpha=0.1, iterations=2)

    # 2. Adaptive normals
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=avg_dist * 3,
            max_nn=30
        )
    )
    pcd.orient_normals_consistent_tangent_plane(10)

    # --- Poisson ---
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9, linear_fit=True
    )
    densities = np.asarray(densities)
    mesh.remove_vertices_by_mask(densities < np.quantile(densities, 0.1))

    # Final smoothing
    mesh = mesh.filter_smooth_taubin(number_of_iterations=7)

    return mesh
