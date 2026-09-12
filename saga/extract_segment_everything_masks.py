import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

from pathlib import Path

import torch
from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np
from normalized_splatting.arguments import PipelineParams, ModelParams, get_combined_args
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from normalized_splatting.scene import Scene, GaussianModel

if __name__ == '__main__':
    
    parser = ArgumentParser(description="SAM segment everything masks extracting params")
    
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--sam_checkpoint_path",
                        default=os.environ.get("SAM_CHECKPOINT", "sam_vit_h_4b8939.pth"),
                        type=str,
                        help="Path to the SAM checkpoint (download from the segment-anything "
                             "repo). Defaults to $SAM_CHECKPOINT, else ./sam_vit_h_4b8939.pth.")
    parser.add_argument("--sam_arch", default="vit_h", type=str)
    parser.add_argument("--force_overwrite", action='store_true')

    # parser.add_argument("--downsample", default=1, type=int)
    # parser.add_argument("--downsample_type", default='image', type=str, choices=['image', 'mask'], help="Downsample then segment, or segment then downsample.")

    args = get_combined_args(parser)
    
    gaussians = GaussianModel(3)
    scene = Scene(model.extract(args), gaussians)
    
    print("Initializing SAM...")
    model_type = args.sam_arch
    sam = sam_model_registry[model_type](checkpoint=args.sam_checkpoint_path).to('cuda')
    
    # custom
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.88,
        box_nms_thresh=0.7,
        stability_score_thresh=0.95,
        crop_n_layers=0,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=100,
    )
    # assert os.path.exists(IMAGE_DIR) and "Please specify a valid image root"
    OUTPUT_DIR = os.path.join(args.source_path, 'saga', 'sam_masks')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Extracting SAM segment everything masks...")
    
    for cam in tqdm(scene.getTrainCameras(), dynamic_ncols=True):
        pt_path = os.path.join(OUTPUT_DIR, Path(cam.image_name).stem+'.pt')
        if not args.force_overwrite and os.path.exists(pt_path):
            continue
        
        img = (cam.original_image.permute(1,2,0)*255).int().cpu().numpy().astype(np.uint8)
        masks = mask_generator.generate(img)
        # print(len(masks))
        mask_list = []
        for m in masks:
            m_score = torch.from_numpy(m['segmentation']).float().to('cuda')

            if len(m_score.unique()) < 2:
                continue
            else:
                mask_list.append(m_score.bool())
        masks = torch.stack(mask_list, dim=0)

        torch.save(masks, pt_path)