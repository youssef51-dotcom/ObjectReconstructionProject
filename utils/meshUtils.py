import open3d as o3d
from scipy.spatial import Delaunay
import pymeshlab


def smooth_point_cloud(pcd, k=10, alpha=0.3, iterations=1):
    points = np.asarray(pcd.points)
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    for _ in range(iterations):
        new_points = points.copy()
        for i, p in enumerate(points):
            _, idx, _ = kdtree.search_knn_vector_3d(p, k)
            neighbors = points[idx[1:]]  # exclude itself
            mean = neighbors.mean(axis=0)

            # Move slightly toward neighbor mean
            new_points[i] = p + alpha * (mean - p)

        points = new_points

    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def fill_holes_and_smooth(input_mesh_path, output_mesh_path, max_faces = 50000, max_hole_size=1000000, smooth_iterations=3):
    """
    Fills holes in a mesh, smooths it, and recomputes normals using PyMeshLab.

    Parameters:
        input_mesh_path (str): Path to input mesh (PLY/OBJ).
        output_mesh_path (str): Path to save the processed mesh.
        max_hole_size (int): Maximum hole size to fill (number of edges on the boundary loop).
        smooth_iterations (int): Number of smoothing iterations (Laplacian).
    """
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(input_mesh_path)
    # Fill holes
    ms.meshing_close_holes(maxholesize=int(max_hole_size))

    # Laplacian smoothing
    for _ in range(smooth_iterations):
        ms.apply_filter('apply_coord_laplacian_smoothing_surface_preserving')

    # Recompute normals
    ms.apply_filter('compute_normal_per_face')
    ms.apply_filter('compute_normal_per_vertex')

    # Decimate mesh to <= max_faces (Quadric Edge Collapse)
    current_faces = ms.current_mesh().face_number()
    if current_faces > max_faces:
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=max_faces, preservenormal=True, preservetopology=True)

    # Save final mesh
    ms.save_current_mesh(output_mesh_path)
    print(f"Mesh processed and saved: {output_mesh_path}")


import numpy as np


def get_mesh_boundaries(mesh):
    triangles = np.asarray(mesh.triangles)
    edges_count = {}

    # Count how many times each edge appears
    for tri in triangles:
        edges = [
            tuple(sorted([tri[0], tri[1]])),
            tuple(sorted([tri[1], tri[2]])),
            tuple(sorted([tri[2], tri[0]])),
        ]
        for edge in edges:
            edges_count[edge] = edges_count.get(edge, 0) + 1

    # Only edges that appear once are boundary edges
    boundary_edges = [edge for edge, count in edges_count.items() if count == 1]

    # Build adjacency list
    adjacency = {}
    for v1, v2 in boundary_edges:
        adjacency.setdefault(v1, []).append(v2)
        adjacency.setdefault(v2, []).append(v1)

    # Keep track of visited edges
    visited_edges = set()
    loops = []

    for v_start in adjacency.keys():
        for neighbor in adjacency[v_start]:
            edge = tuple(sorted([v_start, neighbor]))
            if edge in visited_edges:
                continue  # Already traversed this edge

            loop = [v_start]
            current = neighbor
            prev = v_start
            visited_edges.add(edge)

            while True:
                loop.append(current)

                # Find next neighbor that is not previous and edge not visited
                next_vertices = [
                    n for n in adjacency[current]
                    if n != prev and tuple(sorted([current, n])) not in visited_edges
                ]
                if not next_vertices:
                    break

                next_v = next_vertices[0]
                visited_edges.add(tuple(sorted([current, next_v])))
                prev, current = current, next_v

                # If we looped back to start, stop
                if current == v_start:
                    break

            if len(loop) > 2:
                loops.append(loop)

    return loops

def safe_extend_mesh(mesh, new_triangles):
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    combined_triangles = np.vstack([triangles, np.array(new_triangles)])
    new_mesh = o3d.geometry.TriangleMesh()
    new_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    new_mesh.triangles = o3d.utility.Vector3iVector(combined_triangles)

    # --- Clean up mesh ---
    new_mesh.remove_duplicated_vertices()
    new_mesh.remove_duplicated_triangles()
    new_mesh.remove_degenerate_triangles()
    new_mesh.remove_non_manifold_edges()
    new_mesh.compute_vertex_normals()
    return new_mesh

def fill_big_holes_planar(mesh):
    boundaries = get_mesh_boundaries(mesh)
    if len(boundaries) == 0:
        print("No boundary loops found")
        return mesh

    all_new_triangles = []

    for loop in boundaries:
        if len(loop) < 3:
            continue
        points = np.asarray(mesh.vertices)[loop, :]
        centroid = points.mean(axis=0)
        cov = np.cov(points.T)
        eigvals, eigvecs = np.linalg.eigh(cov)

        basis1 = eigvecs[:, 1]
        basis2 = eigvecs[:, 2]
        points_2d = np.dot(points - centroid, np.vstack([basis1, basis2]).T)

        try:
            tri = Delaunay(points_2d)
        except:
            continue  # Skip problematic loops

        triangles = tri.simplices
        new_triangles = [[loop[i] for i in tri] for tri in triangles]
        all_new_triangles.extend(new_triangles)

    mesh = safe_extend_mesh(mesh, all_new_triangles)
    return mesh

def remove_nan_vertices(mesh):
    verts = np.asarray(mesh.vertices)
    mask = np.isfinite(verts).all(axis=1)
    mesh.remove_vertices_by_mask(~mask)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    return mesh

def remove_nan_triangles(mesh):
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)

    # 1D boolean mask: True if all three vertices of the triangle are finite
    mask = np.all(np.isfinite(verts[tris]), axis=1)  # shape: (n_triangles,)
    mask = mask.tolist()  # Open3D expects a list[bool]

    mesh.remove_triangles_by_mask([not m for m in mask])  # invert: True means REMOVE
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    return mesh
