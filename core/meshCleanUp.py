import os
import numpy as np
import open3d as o3d
import pymeshlab

from utils.meshUtils import fill_holes_and_smooth


def remove_nan_vertices(mesh):
    vertices = np.asarray(mesh.vertices)

    # Keep only finite vertices
    valid_mask = np.isfinite(vertices).all(axis=1)

    invalid_count = np.sum(~valid_mask)

    print(f"Invalid vertices removed: {invalid_count}")

    mesh.remove_vertices_by_mask(~valid_mask)

    return mesh


def cleanMesh(mesh, outputPath):

    print("=== CLEANING MESH ===")

    # --------------------------------------------------
    # Keep largest connected component
    # --------------------------------------------------

    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)

    largest_cluster_idx = np.argmax(cluster_n_triangles)

    mesh.remove_triangles_by_mask(
        triangle_clusters != largest_cluster_idx
    )

    # --------------------------------------------------
    # Basic cleanup
    # --------------------------------------------------

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()

    # --------------------------------------------------
    # Remove NaN / Inf vertices
    # --------------------------------------------------

    mesh = remove_nan_vertices(mesh)

    # --------------------------------------------------
    # Smooth
    # --------------------------------------------------

    mesh = mesh.filter_smooth_laplacian(
        number_of_iterations=3
    )

    mesh.compute_vertex_normals()

    # --------------------------------------------------
    # Decimate
    # --------------------------------------------------

    target_triangles = min(
        50000,
        len(mesh.triangles)
    )

    mesh = mesh.simplify_quadric_decimation(
        target_number_of_triangles=target_triangles
    )

    mesh.compute_vertex_normals()

    # --------------------------------------------------
    # Final cleanup after decimation
    # --------------------------------------------------

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()

    mesh = remove_nan_vertices(mesh)

    # --------------------------------------------------
    # Save intermediate mesh
    # --------------------------------------------------

    os.makedirs("output", exist_ok=True)

    temp_mesh_path = "output/mesh_before_holes.ply"

    success = o3d.io.write_triangle_mesh(
        temp_mesh_path,
        mesh,
        write_ascii=True
    )

    print(f"Mesh saved: {success}")
    print(f"Path exists: {os.path.exists(temp_mesh_path)}")

    # --------------------------------------------------
    # Verify mesh validity
    # --------------------------------------------------

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    print(f"Vertices: {len(vertices)}")
    print(f"Triangles: {len(triangles)}")

    if len(vertices) == 0:
        raise ValueError("Mesh has no vertices")

    if len(triangles) == 0:
        raise ValueError("Mesh has no triangles")

    if np.isnan(vertices).any():
        raise ValueError("Mesh still contains NaN values")

    # --------------------------------------------------
    # Fill holes with PyMeshLab
    # --------------------------------------------------

    print("=== FILLING HOLES ===")

    fill_holes_and_smooth(
        input_mesh_path=temp_mesh_path,
        output_mesh_path=outputPath,
        max_hole_size=100000,
        smooth_iterations=3
    )

    print("=== CLEANING FINISHED ===")

    return mesh