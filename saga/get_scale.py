from pathlib import Path
import torch

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

from normalized_splatting.utils.camera_utils import Camera



from argparse import ArgumentParser

from normalized_splatting.arguments import ModelParams, PipelineParams, get_combined_args
from normalized_splatting.scene import Scene, GaussianModel
import normalized_splatting.gaussian_renderer as gaussian_renderer

torch.set_grad_enabled(False)


def generate_grid_index(depth):
    """Per-pixel (u, v) image coordinates, shaped (H, W, 2).

    meshgrid([arange(h), arange(w)]) yields (row, col), so the column index --
    the horizontal coordinate u -- is element 1, not element 0.
    """
    h, w = depth.shape
    rows, cols = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    return torch.stack((cols, rows), dim=-1)

def get_masks_for_view(view, dataset):
    """
    Fetches the masks corresponding to the given view.

    Args:
        view: The current view object containing metadata such as image_name or a unique identifier.
        dataset: An object or structure containing paths to the dataset resources (e.g., source_path).

    Returns:
        A torch.Tensor containing the masks for the current view.
    """
    image_name = Path(view.image_name).stem  # Extract the base name without extension
    mask_path = os.path.join(
        dataset.source_path,
        'saga',
        'sam_masks', 
        image_name + '.pt'
    )
    return torch.load(mask_path, weights_only=False).cpu().float()


if __name__ == '__main__':
    
    parser = ArgumentParser(description="Get scales for SAM masks")

    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--segment", action="store_true")
    parser.add_argument('--idx', default=0, type=int)
    parser.add_argument('--precomputed_mask', default=None, type=str)

    args = get_combined_args(parser)

    dataset = model.extract(args)
    scene_gaussians = GaussianModel(dataset.sh_degree)

    load_iteration = args.iteration if args.model_path is not None else None
    scene = Scene(dataset, scene_gaussians, load_iteration=load_iteration, shuffle=False, mode='eval')


    # assert os.path.exists(os.path.join(dataset.source_path, 'images')) and "Please specify a valid image root."
    assert os.path.join(dataset.source_path, 'saga', 'sam_masks') and "Please run extract_segment_everything_masks first."

    from tqdm import tqdm


    OUTPUT_DIR = os.path.join(dataset.source_path, 'saga', 'mask_scales')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cameras = scene.getTrainCameras()

    background = torch.zeros(scene_gaussians.get_mask.shape[0], 3, device = 'cuda')

    for it, view in enumerate(tqdm(cameras, dynamic_ncols=True)):
        view: Camera
        if view.depth is not None:
            depth = view.depth.cpu().squeeze()
        else:
            rendered_pkg = gaussian_renderer.render(view, scene_gaussians, pipeline.extract(args), background)
            depth = rendered_pkg['depth'].cpu().squeeze()
            del rendered_pkg  # Free GPU memory if not needed

        # Generate the grid index and compute 3D points
        grid_index = generate_grid_index(depth)
        K = view.get_K
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        
        z = depth
        u, v = grid_index[:, :, 0], grid_index[:, :, 1]
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        points_in_3D = torch.stack((x, y, z), dim=-1)

        # Process corresponding masks directly
        corresponding_masks = get_masks_for_view(view, dataset)  # Replace with the logic to get masks for the current view
        upsampled_mask = torch.nn.functional.interpolate(
            corresponding_masks.unsqueeze(1),
            mode='bilinear',
            size=(depth.shape[0], depth.shape[1]),
            align_corners=False
        )
        eroded_masks = torch.conv2d(
            upsampled_mask.float(),
            torch.full((3, 3), 1.0).view(1, 1, 3, 3),
            padding=1
        )
        eroded_masks = (eroded_masks >= 5).squeeze(1)
        # Compute scale for each mask
        scale = torch.zeros(len(corresponding_masks))
        for mask_id in range(len(corresponding_masks)):
            point_in_3D_in_mask = points_in_3D[eroded_masks[mask_id] == 1]
            scale[mask_id] = (point_in_3D_in_mask.std(dim=0) * 2).norm()
        # Empty masks give a NaN std; zero them once, not once per mask.
        scale = scale.nan_to_num(0, 0, 0)

        # Save the computed scale
        torch.save(scale.half(), os.path.join(OUTPUT_DIR, view.image_name + '.pt'))
