import argparse
import time
import torch
from utils.platformInfo import log_system_info
from core.sfmPipeline import run_colmap_camera_poses
from core.PoissonMeshGeneration import poissonDepth
from core.meshCleanUp import cleanMesh


def main():
    total_start_time = time.time()
    parser = argparse.ArgumentParser(description="3D Reconstruction Pipeline")
    parser.add_argument("--input", type=str, required=True, help="Input folder with images")
    parser.add_argument("--output", type=str, required=True, help="Output mesh file (OBJ/PLY)")
    args = parser.parse_args()

    #platform info
    log_system_info()

    # Step 1: SfM
    print("[Step 1] Running SfM...")
    start_time = time.time()
    #on cuda I would have enabled cuda based colmap that is faster and also enables to
    # generate a dense point cloud, that would have made mesh reconstruction much easier and more precise
    #I would also have had a more independant batch minded way to do things
    #N.B: this approach works mainly with textured objects for non textured ones I would use a DL method
    #that generates depth maps from images and maybe combine even with a very sparse reconstruction to constraint scale
    reconstruction, poses, intrinsics = run_colmap_camera_poses(args.input, "./output")

    # lot of information unused from colmap for now with better proc/gpu i would use a method that generates
    # mesh based on reporjection minimization error like NeRF / NeuS (and exploit better all the rich information sfm gives
    end_time = time.time()
    print(f"[Step 1] SfM stage finished in {end_time - start_time:.2f} seconds")

    # Step 2: Depth Estimation
    print("[Step 2] Running Poisson Mesh Generation...")
    start_time = time.time()
    # basic approach meant to fit my proc constraints
    mesh = poissonDepth()
    end_time = time.time()
    print(f"[Step 2] Poisson Mesh Generation finished in {end_time - start_time:.2f} seconds")

    # Step 3: Mesh Cleansing
    print("[Step 3] Running Mesh cleanse...")
    start_time = time.time()
    # basic approach meant to fit my proc constraints would be a nice init for another approach
    cleanMesh(mesh,args.output)
    end_time = time.time()
    print(f"[Step 3] Running Mesh cleanse finished in {end_time - start_time:.2f} seconds")

    total_end_time = time.time()
    print(f" all process finished in {total_end_time - total_start_time:.2f} seconds")

if __name__ == "__main__":
    main()
