import open3d as o3d
import numpy as np
import pymeshlab

from utils.meshUtils import fill_big_holes_planar
from utils.meshUtils import fill_holes_and_smooth
from utils.meshUtils import remove_nan_vertices

def cleanMesh(mesh,outputPath):
    # --- Keep largest cluster ---
    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    largest_cluster_idx = np.argmax(cluster_n_triangles)
    mesh.remove_triangles_by_mask(triangle_clusters != largest_cluster_idx)

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    # --- Fill big holes safely ---
    #mesh = fill_big_holes_planar(mesh)

    # --- Remove any NaNs in vertices ---
    #mesh = remove_nan_vertices(mesh)

    # --- Smooth ---
    mesh = mesh.filter_smooth_laplacian(number_of_iterations=3)
    mesh.compute_vertex_normals()

    # --- Decimate ---
    mesh = mesh.simplify_quadric_decimation(50000)
    mesh.compute_vertex_normals()

    # Extract vertices and faces as NumPy arrays
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)

    # -----------------------------
    # 2. Load mesh into PyMeshLab
    # -----------------------------
    """
    print("create mesh in pymeshlab")
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces), "mesh_from_o3d")
    print("mesh created in pymeshlab")

    # Fill holes and smooth
    fill_holes_and_smooth(
        ms,
        output_mesh_path=outputPath,
        max_hole_size=100000,
        smooth_iterations=3
    )
    """
    # After your Poisson reconstruction
    o3d.io.write_triangle_mesh("output/mesh_before_holes.ply", mesh)

    # Fill holes and smooth
    fill_holes_and_smooth(
        input_mesh_path="output/mesh_before_holes.ply",
        output_mesh_path=outputPath,
        max_hole_size=100000,
        smooth_iterations=3
    )

    return mesh