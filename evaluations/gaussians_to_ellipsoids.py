""""
Usage: Put this in the root directory of your gaussian splatting. You'll need to use a splatting-compatible environment.
Then run with your .ply file saved from splatting.

Arguments:

- scale_factor: scales the ellipsoids. I used 1.5
- min_opacity: Skips splats under this value. I used 0.1.
- Skip_step: Only keeps 1 out of this many gaussians. I used 3.

Using the mesh in blender (I used 4.1):
- file -> implort the ply
- To view the colors, go to the dropdown in the top-left that says "object mode" and switch it to "vertex paint"
- To set it up for renders, go to the shader editor (tiny button in top left)
    - Select the imported mesh
    - Click new in the top middle
    - Click the main screen, then shift+A and search for input->attribute
    - Connect the color output of the attribute to the base color input of the material (both nodes should be yellow)
    - Change the name of the attribute to Col

"""

# Replace this with [[x_min, x_max], [y_min, y_max], [z_min, z_max]] if wanted

import pandas as pd
import os

import numpy as np
from scipy.spatial.transform import Rotation as R
import tqdm
from scipy.spatial import cKDTree



def SH2RGB(sh):
    C0 = 0.28209479177387814
    return sh * C0 + 0.5

def gaussians_to_ellipsoids(input_csv, out_ply="./ellipsoid_mesh.ply",
           bounds = None, scale_factor=1,min_opacity=None,
           skip_step=3, max_size=None, opa_quantile=None,
           voxel_size=None, max_dist=None):

    df = pd.read_csv(input_csv)

    means = df[["mu_x", "mu_y", "mu_z"]].values
    colors = df[["r", "g", "b"]].values
    opacities = df[["opacity"]].values
    scalings = df[["s_x", "s_y", "s_z"]].values
    rotations = df[["q_w", "q_x", "q_y", "q_z"]].values
    

    import trimesh
    def create_ellipsoid_mesh(mean, scaling, rotation_quat, color):

        # Create a unit sphere mesh
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=1)
        
        # Scale the sphere to create an ellipsoid based on the eigenvalues
        scale_factors = scaling * scale_factor
        sphere.apply_scale(scale_factors)
        
        # Rotate the ellipsoid to align with the eigenvectors
        # trimesh expects rotation as a 4x4 matrix, where the upper left 3x3 is the rotation matrix
        T = np.eye(4)
        rotation_matrix = R.from_quat(rotation_quat).as_matrix()
        T[:3, :3] = rotation_matrix
        sphere.apply_transform(T)
        sphere.visual.vertex_colors = (color.flatten()*255).tolist() + [255] # RGB to RGBA

        
        # Translate the ellipsoid to its mean position
        sphere.apply_translation(mean)
        
        return sphere

    meshes = []
    means = means[::skip_step]
    scalings = scalings[::skip_step]
    rotations = rotations[::skip_step]
    opacities = opacities[::skip_step]
    colors = colors[::skip_step]

    if bounds is not None:
        x_mask = np.logical_and(means[:,0] >= bounds[0][0], means[:,0] <= bounds[0][1])
        y_mask = np.logical_and(means[:,1] >= bounds[1][0], means[:,1] <= bounds[1][1])
        z_mask = np.logical_and(means[:,2] >= bounds[2][0], means[:,2] <= bounds[2][1])
        valid_mask = np.logical_and(x_mask, np.logical_and(y_mask, z_mask))
        means = means[valid_mask]
        scalings = scalings[valid_mask]
        rotations = rotations[valid_mask]
        opacities = opacities[valid_mask]
        colors = colors[valid_mask]
        

    if max_size is not None:
        valid_mask = (scalings*scale_factor < max_size).all(-1)
        means = means[valid_mask]
        scalings = scalings[valid_mask]
        rotations = rotations[valid_mask]
        opacities = opacities[valid_mask]
        colors = colors[valid_mask]

    if opa_quantile is not None:
        min_opacity = max(min_opacity, np.percentile(opacities, opa_quantile*100))
    
    if voxel_size is not None:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(means)
        selected_indices = pcd.voxel_down_sample_and_trace(voxel_size, pcd.get_min_bound(), pcd.get_max_bound())[1]
        selected_indices = selected_indices[:,0]
        means = means[selected_indices]
        scalings = scalings[selected_indices]
        rotations = rotations[selected_indices]
        opacities = opacities[selected_indices]
        colors = colors[selected_indices]
    
    if max_dist is not None:
        kdtree = cKDTree(means)
        distances, indices = kdtree.query(means, k=2)
        selected_indices = np.where(distances[:, 1] <= max_dist)[0]
        means = means[selected_indices]
        scalings = scalings[selected_indices]
        rotations = rotations[selected_indices]
        opacities = opacities[selected_indices]
        colors = colors[selected_indices]

    for i, (mean, scaling, rotation, color, opa) in tqdm.tqdm(enumerate(zip(means, scalings, rotations, colors, opacities)), total=means.shape[0], dynamic_ncols=True):

        if min_opacity is not None and opa <  min_opacity:
            continue

        try:
            color = color.clip(0, 1)
            meshes.append(create_ellipsoid_mesh(mean, scaling, rotation, color))
        except Exception as e:
            print(f"Skipping gaussian: {type(e).__name__}: {e}")
    if not meshes:
        raise RuntimeError("No ellipsoids survived filtering; loosen --min_opacity or the crop box.")

    out_dir = os.path.dirname(os.path.abspath(out_ply))
    os.makedirs(out_dir, exist_ok=True)
    trimesh.util.concatenate(meshes).export(out_ply)
    print(f"Wrote {len(meshes)} ellipsoids to {out_ply}")
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument("input_csv")
    parser.add_argument("output_ply")
    parser.add_argument("--scale_factor", default=1, type=float, required=False)
    parser.add_argument("--min_opacity", type=float, default=0.1, required=False)
    parser.add_argument("--skip_step", type=int, default=3, required=False)
    parser.add_argument("--max_size", default=None, type=float)
    parser.add_argument("--opa_quantile", type=float, default=None, required=False)
    parser.add_argument("--voxel_size", type=float, default=None, required=False)
    parser.add_argument("--max_dist", type=float, default=None, required=False)
    parser.add_argument("--box_center", type=float, nargs=3, default=None,
                        help="Crop to a box centred here; requires --box_dim.")
    parser.add_argument("--box_dim", type=float, nargs=3, default=None,
                        help="Crop box side lengths; requires --box_center.")

    args = parser.parse_args()

    if (args.box_center is None) != (args.box_dim is None):
        parser.error("--box_center and --box_dim must be given together")
    bounds = None
    if args.box_center is not None:
        bounds = [[args.box_center[i] - args.box_dim[i] / 2,
                   args.box_center[i] + args.box_dim[i] / 2] for i in range(3)]

    gaussians_to_ellipsoids(args.input_csv, args.output_ply, bounds=bounds, min_opacity=args.min_opacity, skip_step=args.skip_step,
                            scale_factor=args.scale_factor, max_size=args.max_size, opa_quantile=args.opa_quantile,
                            voxel_size=args.voxel_size, max_dist=args.max_dist)
    
    
    