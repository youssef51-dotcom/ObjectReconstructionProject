import subprocess
from pathlib import Path

import numpy as np
import pycolmap

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

def run_colmap_dense_reconstruction(image_dir, output_dir, save_ply=True):
    """
    Run COLMAP sparse + dense reconstruction pipeline.

    Returns:
        reconstruction: sparse reconstruction
        poses_dict: camera-to-world matrices
        intrinsics_dict: camera intrinsics
        dense_model_path: path to fused dense point cloud
    """

    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    database_path = output_dir / "database.db"
    sparse_dir = output_dir / "sparse"
    dense_dir = output_dir / "dense"

    sparse_dir.mkdir(exist_ok=True)
    dense_dir.mkdir(exist_ok=True)

    # ---------------------------------------------------
    # 1. Feature extraction
    # ---------------------------------------------------
    print("[COLMAP] Extracting features...")
    pycolmap.extract_features(
        database_path=database_path,
        image_path=image_dir,
    )

    # ---------------------------------------------------
    # 2. Feature matching
    # ---------------------------------------------------
    print("[COLMAP] Matching features...")
    pycolmap.match_exhaustive(database_path)

    # ---------------------------------------------------
    # 3. Sparse reconstruction
    # ---------------------------------------------------
    print("[COLMAP] Running incremental mapping...")

    reconstructions = pycolmap.incremental_mapping(
        database_path=database_path,
        image_path=image_dir,
        output_path=sparse_dir,
    )

    if not reconstructions:
        raise RuntimeError("No sparse reconstruction computed")

    reconstruction = reconstructions[0]

    print(
        f"[COLMAP] Sparse reconstruction complete: "
        f"{len(reconstruction.images)} images, "
        f"{len(reconstruction.points3D)} sparse points"
    )

    # ---------------------------------------------------
    # 4. Extract poses + intrinsics
    # ---------------------------------------------------
    poses_dict = {}
    intrinsics_dict = {}

    for img_id, img in reconstruction.images.items():

        if not img.has_pose:
            continue

        name = img.name

        try:
            cam2world = rigid3d_to_cam2world_matrix(
                img.cam_from_world()
            )

            poses_dict[name] = cam2world

            cam = reconstruction.cameras[img.camera_id]

            if cam.model.name == "PINHOLE":
                fx, fy, cx, cy = cam.params

            elif cam.model.name == "SIMPLE_PINHOLE":
                f, cx, cy = cam.params
                fx = fy = f

            elif cam.model.name in ["SIMPLE_RADIAL", "RADIAL", "OPENCV"]:
                fx = fy = cam.params[0]
                cx, cy = cam.params[1:3]

            else:
                raise NotImplementedError(
                    f"Unsupported model: {cam.model.name}"
                )

            K = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ])

            intrinsics_dict[name] = K

        except Exception as e:
            print(f"Skipping {name}: {e}")

    # ---------------------------------------------------
    # 5. Dense reconstruction
    # ---------------------------------------------------

    fused_path = dense_dir / "fused.ply"

    print("[COLMAP] Undistorting images...")

    pycolmap.undistort_images(
        image_path=image_dir,
        input_path=sparse_dir / "0",
        output_path=dense_dir,
    )

    print("[COLMAP] Running PatchMatch stereo...")

    subprocess.run([
    "colmap",
    "patch_match_stereo",
    "--workspace_path", str(dense_dir),
    "--workspace_format", "COLMAP",
    "--PatchMatchStereo.gpu_index", "0",
    "--PatchMatchStereo.geom_consistency", "true",
    "--PatchMatchStereo.num_iterations", "3",
    "--PatchMatchStereo.num_samples", "8"
    ], check=True)

    print("[COLMAP] Running stereo fusion...")

    subprocess.run([
        "colmap",
        "stereo_fusion",
        "--workspace_path", str(dense_dir),
        "--workspace_format", "COLMAP",
        "--input_type", "geometric",
        "--output_path", str(fused_path)
    ], check=True)

    print(f"[COLMAP] Dense reconstruction saved to: {fused_path}")

    # ---------------------------------------------------
    # 6. Optional debug outputs
    # ---------------------------------------------------

    if save_ply:
        save_sparse_points_ply(
            reconstruction,
            output_dir / "sparse_points.ply"
        )

        save_camera_poses_ply(
            list(poses_dict.values()),
            output_dir / "camera_poses.ply"
        )

    reconstruction.write(sparse_dir)

    return (
        reconstruction,
        poses_dict,
        intrinsics_dict,
        fused_path,
    )