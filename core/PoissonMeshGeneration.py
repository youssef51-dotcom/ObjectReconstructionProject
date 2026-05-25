import open3d as o3d
import numpy as np
from utils.meshUtils import smooth_point_cloud

def poissonDepth(pointCloud):

    pcd = o3d.io.read_point_cloud(pointCloud)
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

def poissonDepthDense(pointCloud):

    pcd = o3d.io.read_point_cloud(pointCloud)

    if len(pcd.points) == 0:
        raise ValueError("Point cloud is empty.")

    print(f"Initial points: {len(pcd.points)}")

    # -------------------------------------------------
    # 1. Remove outliers
    # -------------------------------------------------

    pcd, ind = pcd.remove_statistical_outlier(
        nb_neighbors=20,
        std_ratio=1.5
    )

    pcd, ind = pcd.remove_radius_outlier(
        nb_points=8,
        radius=0.01
    )

    print(f"After cleanup: {len(pcd.points)}")

    # -------------------------------------------------
    # 2. Optional smoothing
    # -------------------------------------------------

    pcd = smooth_point_cloud(
        pcd,
        k=8,
        alpha=0.1,
        iterations=2
    )

    # -------------------------------------------------
    # 3. Recompute normals
    # -------------------------------------------------

    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)

    normal_radius = avg_dist * 4

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius,
            max_nn=40
        )
    )

    pcd.normalize_normals()

    # Very important for Poisson
    pcd.orient_normals_consistent_tangent_plane(50)

    # -------------------------------------------------
    # 4. Poisson reconstruction
    # -------------------------------------------------

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=10,
        width=0,
        scale=1.1,
        linear_fit=True
    )

    densities = np.asarray(densities)

    # Remove low-density vertices
    density_threshold = np.quantile(densities, 0.05)

    vertices_to_remove = densities < density_threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)

    # -------------------------------------------------
    # 5. Cleanup mesh
    # -------------------------------------------------

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    # Optional
    mesh = mesh.filter_smooth_taubin(
        number_of_iterations=10
    )

    mesh.compute_vertex_normals()

    print(mesh)

    return mesh