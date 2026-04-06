from typing import Tuple
import os

import numpy as np
from PIL import Image
import open3d as o3d

def rigid3d_to_cam2world_matrix(rigid: "pycolmap._core.Rigid3d") -> np.ndarray:
    """
    Converts a pycolmap Rigid3d (3x4) to a 4x4 camera-to-world matrix.
    """
    mat34 = rigid.matrix()  # 3x4
    R = mat34[:, :3]        # 3x3 rotation
    t = mat34[:, 3]         # 3x1 translation
    cam2world = np.eye(4)   # 4x4 homogeneous
    cam2world[:3, :3] = R.T          # invert rotation
    cam2world[:3, 3] = -R.T @ t      # invert translation
    return cam2world

def save_sparse_points_ply(reconstruction, out_path):
    """
    Save sparse points from COLMAP reconstruction as a PLY.
    Points are in world coordinates.
    """
    if not hasattr(reconstruction, "points3D"):
        print("[COLMAP] Reconstruction has no points3D attribute")
        return

    points = reconstruction.points3D  # dict of Point3D objects
    if not points:
        print("[COLMAP] No points to save in PLY")
        return

    all_points = np.array([p.xyz for p in points.values()])
    all_colors = np.array([p.color for p in points.values()]) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)
    o3d.io.write_point_cloud(str(out_path), pcd)
    print(f"[COLMAP] Saved sparse points PLY to {out_path}")


def save_camera_poses_ply(poses_list, out_path):
    """
    Save camera centers as PLY (red points) for debug visualization.
    Poses should be camera-to-world 4x4 matrices.
    """
    camera_centers = np.array([pose[:3, 3] for pose in poses_list])
    if camera_centers.size == 0:
        print("[COLMAP] No camera poses to save in PLY")
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(camera_centers)
    pcd.colors = o3d.utility.Vector3dVector(
        np.tile(np.array([[1.0, 0.0, 0.0]]), (len(camera_centers), 1))
    )
    o3d.io.write_point_cloud(str(out_path), pcd)
    print(f"[COLMAP] Saved camera poses PLY to {out_path}")


def resize_images(
        input_dir: str,
        output_dir: str,
        target_size: Tuple[int, int] = (960, 540)
):
    """
    Resize all images in input_dir and save to output_dir.

    Args:
        input_dir (str): Folder containing original images.
        output_dir (str): Folder to save resized images.
        target_size (Tuple[int, int]): (width, height) in pixels.
    """
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(input_dir, filename)
            img = Image.open(img_path)
            img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
            img_resized.save(os.path.join(output_dir, filename))

    print(f"Resized all images to {target_size} and saved in {output_dir}")