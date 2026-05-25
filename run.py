import argparse
import time
import torch
from utils.platformInfo import log_system_info
from core.sfmPipeline import run_colmap_camera_poses, run_colmap_dense_reconstruction
from core.PoissonMeshGeneration import poissonDepth, poissonDepthDense
from core.meshCleanUp import cleanMesh
from utils.folderUtils import clear_folder


def main():
    total_start_time = time.time()
    parser = argparse.ArgumentParser(description="3D Reconstruction Pipeline")
    parser.add_argument("--input", type=str, required=True, help="Input folder with images")
    parser.add_argument("--output", type=str, required=True, help="output folder")
    parser.add_argument("--outputName", type=str, required=True, help="Output mesh file (OBJ/PLY)")
    parser.add_argument("--reconstructionType", type=str, required=True, help="Type of reconstruction Colmap ColmapDense Nerf...")
    args = parser.parse_args()

    #platform info
    log_system_info()

    clear_folder(args.output)

    # Step 1: SfM
    print("[Step 1] Running SfM...")
    start_time = time.time()
    #on cuda I would have enabled cuda based colmap that is faster and also enables to
    # generate a dense point cloud, that would have made mesh reconstruction much easier and more precise
    #I would also have had a more independant batch minded way to do things
    #N.B: this approach works mainly with textured objects for non textured ones I would use a DL method
    #that generates depth maps from images and maybe combine even with a very sparse reconstruction to constraint scale

    outputPointCloudPath = ""

    if args.reconstructionType == "Colmap":
        reconstruction, poses, intrinsics = run_colmap_camera_poses(
        args.input, args.output, args.outputName)
        outputPointCloudPath = "output/sparse_points.ply"
    elif args.reconstructionType == "ColmapDense":
        print("dummy function")
        #Now that I have cuda I can use dense reconstruction
        reconstruction, poses, intrinsics, fused_path = run_colmap_dense_reconstruction(
        args.input, args.output, args.outputName)
        outputPointCloudPath = "output/dense/fused.ply"  
    else:
        raise ValueError(f"Unknown reconstruction type: {args.reconstructionType}")

    # lot of information unused from colmap for now with better proc/gpu i would use a method that generates
    # mesh based on reporjection minimization error like NeRF / NeuS (and exploit better all the rich information sfm gives
    end_time = time.time()

    print(f"[Step 1] SfM stage finished in {end_time - start_time:.2f} seconds")

    if args.reconstructionType in ("Colmap", "ColmapDense"):
        # Step 2: Depth Estimation
        print("[Step 2] Running Poisson Mesh Generation...")
        start_time = time.time()
        # basic approach meant to fit my proc constraints

        if args.reconstructionType == "Colmap":
            mesh = poissonDepth(outputPointCloudPath)
        elif args.reconstructionType == "ColmapDense":
            mesh = poissonDepthDense(outputPointCloudPath)
        else:
            raise ValueError(f"Unknown reconstruction type: {args.reconstructionType}")

        end_time = time.time()
        print(f"[Step 2] Poisson Mesh Generation finished in {end_time - start_time:.2f} seconds")

        # Step 3: Mesh Cleansing
        print("[Step 3] Running Mesh cleanse...")
        start_time = time.time()
        # basic approach meant to fit my proc constraints would be a nice init for another approach
        outputMeshPathName = args.output+"/"+args.outputName
        cleanMesh(mesh,outputMeshPathName)
        end_time = time.time()
        print(f"[Step 3] Running Mesh cleanse finished in {end_time - start_time:.2f} seconds")

        total_end_time = time.time()
        print(f" all process finished in {total_end_time - total_start_time:.2f} seconds")

if __name__ == "__main__":
    main()
