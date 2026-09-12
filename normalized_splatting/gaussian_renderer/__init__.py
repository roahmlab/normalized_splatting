#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from ..scene.gaussian_model import GaussianModel
from ..utils.sh_utils import eval_sh




def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, 
           scaling_modifier = 1.0, override_color = None, viewpoint_color = -1,
           norm_point_features=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!

    viewpoint_color is the ID of the viewpoints.
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        # viewmatrix=viewpoint_camera.world_view_transform,
        # projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.get_camera_center(pc.scale_factor),
        prefiltered=False,
        debug=pipe.debug,
        filter_radius=pc.filter_radius_2d,
        normalize_gaussians=pc.normalized
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    
    if override_color is not None:
        colors_precomp = override_color
    elif pc.app_opt:
        # viewpoint_color selects the per-image appearance embedding. Callers that
        # know they are rendering a *training* view pass that view's index (see
        # train.py); everything else leaves it at -1, which averages the
        # embeddings -- the right choice for novel views, which have none.
        #
        # Do not infer it from viewpoint_camera.uid here: train and test camera
        # lists are each enumerated from 0, so uids collide across the two sets,
        # and a test uid can exceed the embedding table.
        colors_precomp = pc.get_colors_app_opt(image_ids=viewpoint_color,
                                               cam_centers=(viewpoint_camera.T).reshape(-1,3),
                                               means=means3D)
    elif pipe.convert_SHs_python:
        shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
        dir_pp = (pc.get_xyz - viewpoint_camera.get_camera_center().repeat(pc.get_features.shape[0], 1))
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    else:
        shs = pc.get_features
    if pc.is_saga:
        contrastive_features = pc.get_smoothed_point_features()
        if norm_point_features:
            contrastive_features = contrastive_features / (contrastive_features.norm(dim=1, keepdim=True) + 1e-9)
    else:
        contrastive_features = None

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, depth, alpha, contrastive_img = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
        viewmatrix = viewpoint_camera.get_world_view_transform(pc.scale_factor), # Add for gradient propagation
        projmatrix = viewpoint_camera.get_full_proj_transform(pc.scale_factor), # Add for gradient propagation
        contrastive_features=contrastive_features
        )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.

    return_dict = {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "depth": depth,
            "alpha": alpha,
            "contrastive_features": contrastive_img}

    return return_dict

