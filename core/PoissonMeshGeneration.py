import open3d as o3d
import numpy as np
from utils.meshUtils import smooth_point_cloud

def poissonDepth():
    pcd = o3d.io.read_point_cloud("output/sparse_points.ply")
    if len(pcd.points) == 0:
        raise ValueError("Point cloud is empty.")

    # 1. cleaning
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd = smooth_point_cloud(pcd, k=8, alpha=0.15, iterations=2)

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

    # --- First Poisson (LOWER depth) ---
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=6, linear_fit=True
    )
    densities = np.asarray(densities)
    mesh.remove_vertices_by_mask(densities < np.quantile(densities, 0.02))

    # --- Artificial densification od point cloud ---
    pcd_dense = mesh.sample_points_poisson_disk(number_of_points=120000)

    # Adaptive normals again
    distances = pcd_dense.compute_nearest_neighbor_distance()
    avg_dist_dense = np.mean(distances)

    pcd_dense.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=avg_dist_dense * 2,
            max_nn=50
        )
    )
    pcd_dense.orient_normals_consistent_tangent_plane(20)

    # cleanup
    pcd_dense, _ = pcd_dense.remove_statistical_outlier(nb_neighbors=40, std_ratio=1.5)
    pcd_dense = smooth_point_cloud(pcd_dense, k=8, alpha=0.15, iterations=3)

    # --- Second Poisson ---
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd_dense, depth=8, linear_fit=True
    )
    densities = np.asarray(densities)
    mesh.remove_vertices_by_mask(densities < np.quantile(densities, 0.01))

    # Final smoothing
    mesh = mesh.filter_smooth_taubin(number_of_iterations=5)

    return mesh