---

# 3D Reconstruction Pipeline

This repository implements a 3D reconstruction pipeline using COLMAP for Structure-from-Motion (SfM) and
Poisson surface reconstruction for generating meshes from sparse point clouds. The pipeline also includes mesh cleaning
and smoothing using Open3D and PyMeshLab.

---

## Features

* Automatic SfM using COLMAP + pycolmap.
* Sparse point cloud generation from images.
* Poisson mesh reconstruction with artificial densification.
* Mesh cleaning, smoothing, and hole-filling.
* Works on CPU-only setups (GPU can accelerate COLMAP if available).

---

## Requirements

* Python 3.11+
* [COLMAP](https://colmap.github.io/) installed and accessible via binaries.
* Python packages (can be installed via pip):

  pip install torch numpy open3d pycolmap pymeshlab sympy
---

## Project Structure
.
├── run.py                   # Main entry point
├── core
│   ├── sfmPipeline.py       # COLMAP SfM pipeline
│   ├── PoissonMeshGeneration.py  # Poisson mesh reconstruction
│   └── meshCleanUp.py       # Mesh cleaning and smoothing
├── utils
│   ├── meshUtils.py         # Mesh processing helpers
│   ├── platformInfo.py      # Logs system/CPU/GPU info
│   └── image_utils.py       # contains code useful for colmap
└── dataset                  # Input images (user-provided)


---

## Usage

Run the main pipeline with:

python run.py --input ./dataset --output ./output/clean_mesh.ply

### Arguments

* --input: Path to folder containing input images.
* --output: Path to save the final cleaned mesh (PLY or OBJ).
* --resize_width: Width to resize images before processing (default: 960).
* --resize_height: Height to resize images before processing (default: 540).

The pipeline will also generate intermediate files in the `output` directory:

* sparse_points.ply – Sparse point cloud from SfM.
* camera_poses.ply – Camera positions for debug visualization.
* mesh_before_holes.ply – Mesh before hole-filling and final smoothing.

---

## Pipeline Steps

1. SfM with COLMAP

   * Extracts features from images.
   * Matches features and performs incremental mapping.
   * Computes camera-to-world poses and sparse point cloud.

2. Poisson Mesh Generation

   * Reads sparse point cloud.
   * Cleans and smooths point cloud.
   * Estimates normals and performs two-stage Poisson reconstruction for better mesh density.

3. Mesh Cleaning & Smoothing

   * Keeps largest connected component.
   * Removes degenerate triangles and non-manifold edges.
   * Smooths mesh with Laplacian filter.
   * Decimates mesh for reduced size.
   * Uses PyMeshLab to fill holes and final smooth.

---

## Notes / Limitations

* Works best with textured objects. Non-textured objects may require a deep learning-based depth estimation pipeline.
* Dense reconstruction in COLMAP requires GPU; this pipeline compensates by artificially densifying the point cloud.
* PyMeshLab is used to handle large holes; adjust parameters for extremely sparse or noisy reconstructions.

---

## Example Output

After running:

python run.py --input ./dataset --output ./output/clean_mesh.ply

You will get:

* Cleaned and smoothed mesh: clean_mesh.ply.
* Sparse point cloud: sparse_points.ply.
* Mesh before hole filling: mesh_before_holes.ply.
* Camera poses visualization: camera_poses.ply.
