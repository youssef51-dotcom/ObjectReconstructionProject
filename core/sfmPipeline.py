import pycolmap
from pathlib import Path
import numpy as np
from utils.image_utils import rigid3d_to_cam2world_matrix
from utils.image_utils import save_camera_poses_ply
from utils.image_utils import save_sparse_points_ply

def run_colmap_camera_poses(image_dir, output_dir, save_ply=True):
    """
    Compute camera poses from RGB images using COLMAP + pycolmap.
    Returns camera-to-world extrinsic matrices and the sparse reconstruction object.

    Args:
        image_dir (str): folder containing input RGB images.
        output_dir (str): folder to save database and reconstruction.
        save_ply (bool): save PLY debug files.

    Returns:
        poses_list (list of np.ndarray): camera-to-world 4x4 matrices.
        reconstruction (pycolmap.Reconstruction): sparse reconstruction object.
    """
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    database_path = output_dir / "database.db"

    # 1. Extract features
    print("[COLMAP] Extracting features...")
    pycolmap.extract_features(database_path, image_dir)

    # 2. Match features
    print("[COLMAP] Matching features...")
    pycolmap.match_exhaustive(database_path)

    # 3. Incremental mapping
    print("[COLMAP] Running incremental mapping...")
    reconstructions = pycolmap.incremental_mapping(database_path, image_dir, output_dir)
    if not reconstructions:
        raise RuntimeError("[COLMAP] No reconstruction was computed")

    reconstruction = reconstructions[0]  # pick the first model
    num_points = len(reconstruction.points3D) if hasattr(reconstruction, "points3D") else 0
    print(f"[COLMAP] Reconstruction complete with {len(reconstruction.images)} cameras and {num_points} points")

    # 4. Extract camera-to-world poses and intrinsics
    poses_dict = {}
    intrinsics_dict = {}

    for img_id, img in reconstruction.images.items():
        name = img.name

        if not img.has_pose:
            print(f"[COLMAP] Skipping image {name} — no pose estimated")
            continue

        try:
            cam2world = rigid3d_to_cam2world_matrix(img.cam_from_world())
            poses_dict[name] = cam2world

            # Extract camera intrinsics
            cam = reconstruction.cameras[img.camera_id]
            if cam.model.name in ["PINHOLE", "SIMPLE_PINHOLE"]:
                fx, fy, cx, cy = cam.params
            elif cam.model.name in ["SIMPLE_RADIAL", "RADIAL", "OPENCV"]:
                fx = fy = cam.params[0]
                cx, cy = cam.params[1:3]
            else:
                raise NotImplementedError(f"[COLMAP] Camera model {cam.model.name} not supported")

            K = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]])

            intrinsics_dict[name] = K

        except Exception as e:
            print(f"[COLMAP] Skipping image {name} — error converting to cam2world or intrinsics: {e}")

    if not poses_dict:
        raise RuntimeError("[COLMAP] No valid camera poses could be extracted")

    # 5. Optional: save PLY debug files
    if save_ply:
        save_sparse_points_ply(reconstruction, output_dir / "sparse_points.ply")
        save_camera_poses_ply(list(poses_dict.values()), output_dir / "camera_poses.ply")

    # Save reconstruction files for future use
    reconstruction.write(output_dir)

    return reconstruction, poses_dict, intrinsics_dict