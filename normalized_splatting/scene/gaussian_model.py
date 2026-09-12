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

import json
import math
from typing import Union
import torch
import torch.nn.functional as F
import numpy as np
from ..utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from ..utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from ..utils.sh_utils import RGB2SH, SH2RGB
from simple_knn._C import distCUDA2
from ..utils.graphics_utils import BasicPointCloud
from ..utils.general_utils import strip_symmetric, build_scaling_rotation
from enum import Enum
    
INIT_OPA = 0.1

class TrainingTarget(Enum):
    NO_SAGA = 0
    CONTRASTIVE_FEATURES = 1
    COARSE_SEG_EVERYTHING = 2
    TRAIN_EVERYTHING = 3


class GaussianModel:
    def precompute_gaussians(self, xyz, rgb, grid_size=0.2):

        xyz_offset = xyz.min(dim=0)[0]
        xyz_norm = xyz - xyz_offset
        grid_dim_idxs = torch.floor(xyz_norm / grid_size).long()
        n_cells_per_dim = torch.max(grid_dim_idxs, dim=0)[0] + 1

        grid_indices = grid_dim_idxs[:,2]*(n_cells_per_dim[0]*n_cells_per_dim[1]) \
                    + grid_dim_idxs[:,1]*n_cells_per_dim[0] \
                    + grid_dim_idxs[:,0]
                    
        unique_indices, inverse_indices = grid_indices.unique(return_inverse=True)
        mapping_tensor = torch.arange(unique_indices.size(0)).to(grid_indices.device)
        grid_indices = mapping_tensor[inverse_indices]

        n_grid_cells = grid_indices.max().item() + 1

        n_pts_per_cell = torch.zeros(n_grid_cells).long().cuda()
        mean_xyz = torch.zeros((n_grid_cells, 3)).cuda().float()
        mean_rgb = torch.zeros((n_grid_cells, 3)).cuda().float()
        cov = torch.zeros((n_grid_cells, 3, 3)).cuda().float()

        mean_xyz.index_add_(0, grid_indices, xyz.float())
        mean_rgb.index_add_(0, grid_indices, rgb.float())
        n_pts_per_cell.index_add_(0, grid_indices, torch.ones_like(grid_indices))

        mean_xyz /= n_pts_per_cell.unsqueeze(1)
        mean_rgb /= n_pts_per_cell.unsqueeze(1)

        
        centered_points = xyz - mean_xyz[grid_indices]

        cov.index_add_(0, grid_indices, (centered_points.unsqueeze(2) @ centered_points.unsqueeze(1)).float())
        cov /= n_pts_per_cell.unsqueeze(1).unsqueeze(1)-1
        cov[n_pts_per_cell == 1] = torch.eye(3).cuda()* cov[n_pts_per_cell!=1][:, [0,1,2], [0,1,2]].mean()
        

        mask = n_pts_per_cell > 0

        mean_xyz = mean_xyz[mask].detach()
        mean_rgb = mean_rgb[mask].detach()
        cov = cov[mask].detach()

        return mean_xyz, mean_rgb, cov

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation, return_full = False):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            if return_full:
                return actual_covariance
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        
        ### For normalized gaussian ###
        # self.opacity_activation = torch.relu # torch.sigmoid
        # self.inverse_opacity_activation = torch.relu #inverse_sigmoid
        
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        # self.opacity_activation = torch.abs
        # self.inverse_opacity_activation = torch.abs

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int, app_opt=False, feature_dim : int = 32):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._opacity_scale = torch.empty(0)
        self._mask = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.filter_radius_3d = 0
        self.filter_radius_2d = 0
        self.normalized = False
        self.learn_opacity_scale = False
        self.app_opt=app_opt
        self.scale_factor = 1.0
        
        # Contrastive SAGA stuff
        self.feature_dim = feature_dim
        self._fix_saga_features = False
        self._point_features = torch.empty(0)
        self.feature_smooth_map = None
        
        self.training_target = TrainingTarget.NO_SAGA

        self.setup_functions()
        
        self._eval_mode = False
        self.scale_factor = None
        
        # Segmentation rollback state (SAGA). Must be initialized here: segment()
        # appends to these and clear_segment() restores from them.
        self.old_xyz = []
        self.old_point_features = []
        self.old_opacity = []
        self.old_scaling = []
        self.old_rotation = []
        self.segment_times = 0

        # Lazily-built caches for interactive 3D queries (saga_gui click-to-segment).
        # Invalidated by segment()/clear_segment(), which change _xyz/_point_features.
        self._smoothed_point_features = None
        self._kd_tree = None

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._opacity_scale,
            self._mask,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.filter_radius_3d,
            self.filter_radius_2d,
            self.normalized,
            self.learn_opacity_scale,
            self.training_target
        )

    def restore(self, model_args, training_args = None):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self._opacity_scale,
        self._mask,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale,
        self.filter_radius_3d,
        self.filter_radius_2d,
        self.normalized,
        self.learn_opacity_scale,
        self.training_target) = model_args

        if training_args is not None:
            self.training_setup(training_args)
            self.optimizer.load_state_dict(opt_dict)

        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        
    # ---------- sub-selection ----------
    def _build_subset(self, mask: torch.Tensor) -> "GaussianModel":
        """New GaussianModel holding points where mask == True."""
        sub = GaussianModel(self.max_sh_degree, app_opt=self.app_opt)

        # meta
        sub.active_sh_degree = self.active_sh_degree
        sub.scale_factor = self.scale_factor
        sub.normalized = self.normalized
        sub.filter_radius_3d = self.filter_radius_3d
        sub.filter_radius_2d = self.filter_radius_2d
        sub.learn_opacity_scale = self.learn_opacity_scale

        # params
        to_param = lambda t: nn.Parameter(t[mask].detach().clone().requires_grad_(True))
        sub._xyz           = to_param(self._xyz)
        sub._features_dc   = to_param(self._features_dc)
        sub._features_rest = to_param(self._features_rest)
        sub._scaling       = to_param(self._scaling)
        sub._rotation      = to_param(self._rotation)
        sub._opacity       = to_param(self._opacity)

        # non-grad
        sub._opacity_scale = self._opacity_scale.clone().detach()
        sub._mask = self._mask[mask].clone()
        if self.max_radii2D.numel() > 0:
            sub.max_radii2D = self.max_radii2D[mask].clone()
        else:
            sub.max_radii2D = self.max_radii2D.clone()
        sub.optimizer = None
        return sub

    def subset(self, selection):
        """Return (inside, outside) given indices or bool mask."""
        if isinstance(selection, torch.Tensor) and selection.dtype == torch.bool:
            mask = selection
        else:
            idx = torch.as_tensor(selection, device=self._xyz.device, dtype=torch.long)
            mask = torch.zeros(self._xyz.size(0), device=self._xyz.device, dtype=torch.bool)
            mask[idx.unique()] = True
        return self._build_subset(mask), self._build_subset(~mask)

    def split_by_bounding_box(self, bounding_box):
        """Return (inside, outside) split by [[xmin,ymin,zmin],[xmax,ymax,zmax]]."""
        bb = torch.as_tensor(bounding_box, dtype=self._xyz.dtype, device=self._xyz.device)
        if bb.shape != (2, 3):
            raise ValueError("bounding_box must have shape (2,3)")
        bb_min, bb_max = bb
        xyz = self.get_xyz * self.scale_factor
        mask = ((xyz >= bb_min) & (xyz <= bb_max)).all(dim=1)
        return self._build_subset(mask), self._build_subset(~mask)
    
    def split_by_point_radius(self, points, radius):
        """
        Return (near, far) Gaussians w.r.t. *points* within *radius* using cKDTree.
        """
        import numpy as np
        from scipy.spatial import cKDTree
        xyz = self.get_xyz * self.scale_factor

        xyz_cpu = xyz.detach().cpu().numpy()
        tree = cKDTree(xyz_cpu)

        pts_cpu = (points.detach().cpu().numpy()
                   if torch.is_tensor(points) else np.asarray(points, dtype=np.float32))
        idx_lists = tree.query_ball_point(pts_cpu, r=radius)
        
        if len(idx_lists) > 0:
            sel_idx = np.unique(np.concatenate(idx_lists)).astype(int)
        else:
            sel_idx = np.empty(0, dtype=np.int64)

        mask = torch.zeros(xyz_cpu.shape[0], dtype=torch.bool, device=self._xyz.device)
        if sel_idx.size:
            mask[torch.from_numpy(sel_idx).to(self._xyz.device)] = True

        return self._build_subset(mask), self._build_subset(~mask)

    def split_by_convex_hull_radius(self, points, radius):
        """
        Return (near, far) Gaussians w.r.t. the convex hull of *points* inflated by *radius*.
        """
        import numpy as np
        from scipy.spatial import ConvexHull
        import torch

        xyz = self.get_xyz * self.scale_factor
        xyz_cpu = xyz.detach().cpu().numpy()

        pts_cpu = (points.detach().cpu().numpy()
                if torch.is_tensor(points) else np.asarray(points, dtype=np.float32))

        # need at least ndim+1 points to build a hull in ndim dimensions
        ndim = xyz_cpu.shape[1]
        if pts_cpu.shape[0] >= ndim + 1:
            hull = ConvexHull(pts_cpu)
            eq = hull.equations  # shape (n_facets, ndim+1): normals and offsets
            normals = eq[:, :ndim]
            offsets = eq[:, ndim]
            norms = np.linalg.norm(normals, axis=1, keepdims=True)

            # signed distance from each point to each hull facet
            d = (normals.dot(xyz_cpu.T) + offsets[:, None]) / norms
            max_d = d.max(axis=0)

            sel_idx = np.where(max_d <= radius)[0]
        else:
            sel_idx = np.empty(0, dtype=int)

        mask = torch.zeros(xyz_cpu.shape[0], dtype=torch.bool, device=self._xyz.device)
        if sel_idx.size:
            mask[torch.from_numpy(sel_idx).to(self._xyz.device)] = True

        return self._build_subset(mask), self._build_subset(~mask)

    @property
    def get_saga_features(self):
        return self._point_features
        
    def set_eval_mode(self, mode: bool = True):
        self._eval_mode = mode
        
    @property
    def is_saga(self):
        return self._eval_mode or self.training_target != TrainingTarget.NO_SAGA
    
    def set_training_target(self, training_args, target: TrainingTarget):
        self.training_target = target
        self.training_setup(training_args)
    
    def set_segmentation_enabled(self, training_args, 
                                 mode: TrainingTarget = TrainingTarget.CONTRASTIVE_FEATURES,
                                 fix_features: bool = False):
        self.training_target = mode
        self._fix_saga_features = fix_features
    
        self.training_setup(training_args)
        
        
    def change_to_segmentation_mode(self, training_args, target = TrainingTarget.CONTRASTIVE_FEATURES, fixed_feature = False):
        
        if target == TrainingTarget.COARSE_SEG_EVERYTHING:
            self._point_features.data[:,:] = 0
        elif target == TrainingTarget.CONTRASTIVE_FEATURES or target == TrainingTarget.TRAIN_EVERYTHING:
            self._point_features.data = torch.randn_like(self._point_features.data) * 1e-2
        else:
            raise ValueError("Unknown target", target)
        
        self.set_segmentation_enabled(training_args, target, fixed_feature)

       
    
    
    @torch.no_grad()
    def clear_segment(self):
        """Roll back to the state before the first segment() call. No-op if nothing
        has been segmented yet."""
        if not self.old_xyz:
            return

        self._xyz = self.old_xyz[0]
        self._point_features = self.old_point_features[0]
        self._opacity = self.old_opacity[0]
        self._scaling = self.old_scaling[0]
        self._rotation = self.old_rotation[0]

        self.old_xyz = []
        self.old_point_features = []
        self.old_opacity = []
        self.old_scaling = []
        self.old_rotation = []

        self.segment_times = 0
        self._mask = torch.ones((self._xyz.shape[0],), dtype=torch.float, device="cuda")
        self._invalidate_query_caches()
        
    def segment(self, mask=None):
        assert mask is not None and "Must input point cloud mask"
        mask = mask.squeeze()
        assert mask.shape[0] == self._xyz.shape[0]
        if torch.count_nonzero(mask) == 0:
            mask = ~mask
            print("Seems like the mask is empty, segmenting the whole point cloud. Please run seg.py first.")

        

        self.old_xyz.append(self._xyz)
        self.old_point_features.append(self._point_features)
        self.old_opacity.append(self._opacity)
        self.old_scaling.append(self._scaling)
        self.old_rotation.append(self._rotation)
        
        assert self.optimizer is None and "Please set optimizer to None"

        self._xyz = self._xyz[mask]

        self._opacity = self._opacity[mask]
        self._scaling = self._scaling[mask]
        self._rotation = self._rotation[mask]
        self._point_features = self._point_features[mask]

        self._invalidate_query_caches()

        self.segment_times += 1
        tmp = self._mask[self._mask == self.segment_times]
        tmp[mask] += 1
        self._mask[self._mask == self.segment_times] = tmp
        
    def apply_sort(self, order):
        self._xyz = self._xyz[order]
        self._features_dc = self._features_dc[order]
        self._features_rest = self._features_rest[order]
        self._scaling = self._scaling[order]
        self._rotation = self._rotation[order]
        self._opacity = self._opacity[order]

    def pop(self, num=1):
        self._xyz = self._xyz[num:]
        self._features_dc = self._features_dc[num:]
        self._features_rest = self._features_rest[num:]
        self._scaling = self._scaling[num:]
        self._rotation = self._rotation[num:]
        self._opacity = self._opacity[num:]

    def check_nan(self):
        def check(param: torch.nn.Parameter, name):
            if param.isnan().any():
                raise Exception(f"Param {name} is nan")
            if param.grad is not None and param.grad.isnan().any():
                raise Exception(f"Param {name} has nan grad")
        
        check(self._xyz, "xyz")
        check(self._features_dc, "features_dc")
        check(self._features_rest, "features_rest")
        check(self._scaling, "scaling")
        check(self._rotation, "rotation")
        check(self._opacity, "opacity")
        check(self._opacity_scale, "opacity_scale")

    def clip_grads(self, threshold):
        torch.nn.utils.clip_grad_norm_(self._xyz, threshold)
        torch.nn.utils.clip_grad_norm_(self._features_dc, threshold)
        torch.nn.utils.clip_grad_norm_(self._features_rest, threshold)
        torch.nn.utils.clip_grad_norm_(self._scaling, threshold)
        torch.nn.utils.clip_grad_norm_(self._rotation, threshold)
        torch.nn.utils.clip_grad_norm_(self._opacity, threshold)
        torch.nn.utils.clip_grad_norm_(self._opacity_scale, threshold)

    def filter_cov(self, min_cov):
        if min_cov is None or min_cov <= 0:
            return
        
        cov = self.get_scaling ** 2
        print("MAX: ", cov.max())
        mask = cov.max(dim=1)[0] >= min_cov
        print(mask.sum() / len(mask))
        
        self._xyz = self._xyz[mask]
        self._features_dc = self._features_dc[mask]
        self._features_rest = self._features_rest[mask]

        self._scaling = self._scaling[mask]
        self._rotation = self._rotation[mask]
        self._opacity = self._opacity[mask]   
        
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling) + math.sqrt(self.filter_radius_3d)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        # print(self.opacity_activation(self._opacity)*torch.relu(self._opacity_scale))
        return self.opacity_activation(self._opacity)*self.get_opacity_scale
    
    @property
    def get_rgb(self):
        if self.app_opt:
            return torch.sigmoid(self._features_dc)
        return SH2RGB(self._features_dc)

    @property
    def get_opacity_scale(self):
        if self.learn_opacity_scale:
            return torch.abs(self._opacity_scale)
        return torch.abs(self._opacity_scale).detach()

    def get_covariance(self, scaling_modifier = 1, return_full=False):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation, return_full)
    
    def get_full_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation, True)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
    
    def create_app_opt_model(self, training_cams_len, rgb_colors):

        ## make these nn.params
        app_features = torch.rand(self._features_dc.shape[0], 32).float().cuda()  # [N, feature_dim]
        rgbs = torch.logit(rgb_colors).float().cuda()  # [N, 3]

        self._features_rest = nn.Parameter(app_features.requires_grad_(True))
        self._features_dc = nn.Parameter(rgbs.requires_grad_(True))

        self.app_module = AppearanceOptModule(
                training_cams_len, 32, 16, self.max_sh_degree).to(self._features_dc.device)
        
        # initialize the last layer to be zero so that the initial output is zero.
        torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
        torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
        
        batch_size  = 1
        
        self.app_optimizers = [
            torch.optim.Adam(
                self.app_module.embeds.parameters(),
                lr=1e-3 * math.sqrt(batch_size) * 10.0, 
                weight_decay=1e-6,
            ),
            torch.optim.Adam(
                self.app_module.color_head.parameters(),
                lr=1e-3 * math.sqrt(batch_size),
            ),
            ]
        return
    
    def load_app_opt_module(self, model_path):
        data = torch.load(model_path)
        num_images = data["embeds.weight"].shape[0]

        self.app_module = AppearanceOptModule(
            num_images, 32, 16, self.max_sh_degree).to(self._features_dc.device)
        self.app_module.load_state_dict(data)

    def get_colors_app_opt(self, image_ids: int, 
                           cam_centers: Union[torch.Tensor, np.ndarray], 
                           means: torch.Tensor = None):
        if means is None:
            means = self.get_xyz
            
        assert len(cam_centers.shape)==2, "shape length is not 2"
        if isinstance(cam_centers, np.ndarray):
            cam_centers = torch.tensor(cam_centers).cuda()

        if image_ids == -1:
            image_ids = torch.arange(self.app_module.num_images).cuda()
        elif not 0 <= image_ids < self.app_module.num_images:
            raise IndexError(
                f"appearance embedding index {image_ids} is outside the table of "
                f"{self.app_module.num_images} training images. Pass -1 to average "
                "the embeddings when rendering a view the model was not trained on.")
        else:
            image_ids =  torch.tensor([image_ids]).cuda()
            
        colors = self.app_module(
                features=self._features_rest,
                embed_ids=image_ids,
                dirs=means[None, :, :] - cam_centers[:, None, :],
                sh_degree=self.max_sh_degree,
            )
        colors = colors + self._features_dc
        colors = torch.sigmoid(colors).squeeze()
        return colors

    def set_scale_factor(self, scale_factor):
        self.scale_factor = scale_factor
        
    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, 
                        filter_radius_3d: float = 0.0, opacity_scale: float = 1,
                        filter_radius_2d: float = 0.3, normalized: bool = False,
                        num_app_opt_cameras = None,
                        saga_feature_dim = None, scale_factor=1.0):
        
        self.set_scale_factor(scale_factor)
        
        if self.app_opt:
            assert num_app_opt_cameras is not None

        self.set_scale_factor(scale_factor)

        self.normalized = normalized
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points) / self.scale_factor).float().cuda()
        fused_color = torch.tensor(np.asarray(pcd.colors)).float().cuda()

        if self.app_opt:
            features = torch.zeros((fused_color.shape[0], 3 + 32))
            features[:, :3] = fused_color 
            self._features_dc = nn.Parameter(features[:,:3].contiguous().requires_grad_(True))
            self._features_rest = nn.Parameter(features[:,3:].contiguous().requires_grad_(True))
        else:
            features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
            features[:, :3, 0] = RGB2SH(fused_color)
            self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
            self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2)/ self.scale_factor)[...,None].repeat(1, 3) * (1 if self.normalized else 1.4) 
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        init_opa_factor = 1 if self.normalized else 0.6
        opacities = self.inverse_opacity_activation(INIT_OPA * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")) * init_opa_factor
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))

        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        opacity_scale = torch.full_like(self._opacity.flatten()[0], opacity_scale)
        self._opacity_scale = nn.Parameter(opacity_scale.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.filter_radius_3d = filter_radius_3d
        self.filter_radius_2d = filter_radius_2d
        if saga_feature_dim is not None:
            contrastive_features = torch.zeros((fused_point_cloud.shape[0], self.feature_dim)).float().cuda()
            self._point_features = nn.Parameter(contrastive_features.contiguous().requires_grad_(True))
        
        self.training_target = TrainingTarget.NO_SAGA
            
        self._mask = torch.ones((self._xyz.shape[0],), dtype=torch.float, device="cuda")

        if self.app_opt:
            self.create_app_opt_model(num_app_opt_cameras, fused_color)

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        self.learn_opacity_scale = training_args.opacity_scale_lr > 0

        def configure_param(param_list, param: nn.Parameter, lr: float, name: str):
            if lr > 0:
                if name != "poses":
                    param.requires_grad_(True)
                    param = [param]
                param_list.append({'params': param, 'lr': lr, "name": name})
            else:
                param.requires_grad_(False)
        
        param_groups = []
    
        train_gaussians = self.training_target in [TrainingTarget.NO_SAGA, TrainingTarget.TRAIN_EVERYTHING]

        if train_gaussians:
            param_groups += [
                (self._xyz, training_args.position_lr_init * self.spatial_lr_scale, "xyz"),
                (self._features_dc, training_args.feature_lr, "f_dc"),
                (self._features_rest, training_args.feature_lr / 20.0, "f_rest"),
                (self._opacity, training_args.opacity_lr, "opacity"),
                (self._opacity_scale, training_args.opacity_scale_lr, "opacity_scale"),
                (self._scaling, training_args.scaling_lr, "scaling"),
                (self._rotation, training_args.rotation_lr, "rotation")
            ]

        if self.is_saga and not self._fix_saga_features:
            param_groups.append((self._point_features, training_args.saga_feature_lr, "f_contrastive"))
            
        l = []
        for p in param_groups:
            configure_param(l, *p)

        if len(l) == 0:
            raise RuntimeError("Can't optimize with nothing to optimize :)")

        self.optimizer = torch.optim.Adam(l)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)


    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']

        dc_shape = torch.tensor(self._features_dc.shape[1:]).prod()
        rest_shape = torch.tensor(self._features_rest.shape[1:]).prod()

        for i in range(dc_shape):
            l.append('f_dc_{}'.format(i))
        for i in range(rest_shape):
            l.append('f_rest_{}'.format(i))
        # Add contrastive point features if available
        if (
            self._point_features is not None 
            and self._point_features.numel() > 0 
            and self._point_features.shape[0] == self._xyz.shape[0]
        ):
            contrast_shape = self._point_features.shape[1]
            for i in range(contrast_shape):
                l.append('f_contrastive_{}'.format(i))
            
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l                

    def save_ply(self, path, viewpoint_camera=None, saga_iteration = None):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)

        if self.app_opt:
            if viewpoint_camera is not None:
                with torch.no_grad():
                    means3D = self.get_xyz.detach()
                    rgbs = self.get_colors_app_opt(image_ids=viewpoint_camera.uid, 
                                                        cam_centers=(viewpoint_camera.T).reshape(-1,3),
                                                        means=means3D)
                
                f_dc = rgbs.cpu().numpy()
            else:
                f_dc = self._features_dc.detach().cpu().numpy()
            f_rest = self._features_rest.detach().cpu().numpy()
        else:
            f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()

        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        
        attr_list = [xyz, normals, f_dc, f_rest]
        if (
            self._point_features is not None 
            and self._point_features.numel() > 0 
            and self._point_features.shape[0] == xyz.shape[0]
        ):
            contrast_features = self._point_features.detach().cpu().numpy()
            attr_list.append(contrast_features)
            
        attr_list.extend([opacities, scale, rotation])
        attributes = np.concatenate(attr_list, axis=1)

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        
        
        if self.app_opt:
            app_opt_data_path = os.path.join(os.path.dirname(path), 'appearance_optimization_module.pt')
            torch.save(self.app_module.state_dict(), app_opt_data_path)

        metadata = {
            "normalized": self.normalized,
            "filter_radius_3d": self.filter_radius_3d,
            "filter_radius_2d": self.filter_radius_2d,
            "opacity_scale": self._opacity_scale.item(),
            "appearance_optimization": self.app_opt,
            "scale_factor": self.scale_factor,
            "saga_iterations": saga_iteration
        }
  
        with open(os.path.join(os.path.dirname(path), 'metadata.json'), 'w+') as f:
            json.dump(metadata, f)

    def save_mask(self, path):
        mkdir_p(os.path.dirname(path))
        mask = self._mask.detach().cpu().numpy()        
        np.save(path, mask)
        
    def reset_opacity(self):
        new_opa_factor = 1 if self.normalized else 0.1
        new_opa = INIT_OPA*new_opa_factor*self.get_opacity_scale
        real_opacity_values = torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*new_opa)
        pre_scaled_opacity_values = real_opacity_values / self.get_opacity_scale
        opacities_new = self.inverse_opacity_activation(pre_scaled_opacity_values)
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

        
    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        metadata_path = os.path.join(os.path.dirname(path), 'metadata.json')
        
        if os.path.exists(metadata_path):
            with open(metadata_path) as metadata_f:
                data = json.load(metadata_f)
                default_false = lambda x: data[x] if x in data else False

                opacity_scale = data["opacity_scale"]
                self.filter_radius_3d = data["filter_radius_3d"]
                self.filter_radius_2d = data["filter_radius_2d"]
                self.normalized = data["normalized"]

                if "scale_factor" in data:
                    self.set_scale_factor(data["scale_factor"])
                else:
                    self.set_scale_factor(1.0)
                self.app_opt = default_false("appearance_optimization")
        else:
            if "opacity_scale" in plydata.elements[0]:
                opacity_scale = np.asarray(plydata.elements[0]["opacity_scale"])[0].item()
            else:
                opacity_scale = 1.

            if "filter_radius_3d" in plydata.elements[0]:
                self.filter_radius_3d = np.asarray(plydata.elements[0]["filter_radius_3d"])[0].item()
            else:
                self.filter_radius_3d = 0.0

            if "fitler_radius_2d" in plydata.elements[0]:
                self.filter_radius_2d = np.asarray(plydata.elements[0]["filter_radius_2d"])[0].item()
            else:
                self.filter_radius_2d = 0.3

            self.normalized = False
            self.app_opt = False

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))

        if self.app_opt:
            assert len(extra_f_names)==32
        else:
            assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])

        contrastive_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_contrastive_")]
        contrastive_f_names = sorted(contrastive_f_names, key=lambda x: int(x.split('_')[-1]))
        contrastive_features = np.zeros((xyz.shape[0], len(contrastive_f_names)))
                
        for idx, attr_name in enumerate(contrastive_f_names):
            contrastive_features[:, idx] = np.asarray(plydata.elements[0][attr_name])
            
        if contrastive_features.flatten().size == 0:
            contrastive_features = np.zeros((xyz.shape[0], 32))
            
        self.training_target = TrainingTarget.NO_SAGA

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))

        if self.app_opt:
            self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").squeeze().contiguous().requires_grad_(True))
            self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").contiguous().requires_grad_(True))
        else:
            features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))
            self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
            self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity_scale = torch.full_like(self._opacity.flatten()[0], opacity_scale)
        self.active_sh_degree = self.max_sh_degree

        self._point_features = nn.Parameter(torch.tensor(contrastive_features, dtype=torch.float, device="cuda").contiguous().requires_grad_(True))
        
        if self.app_opt:
            self.load_app_opt_module(os.path.join(os.path.dirname(path), 'appearance_optimization_module.pt'))

        self._mask = torch.ones((self._xyz.shape[0],), dtype=torch.float, device="cuda")

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)

                if stored_state is not None and stored_state["exp_avg"] is not None:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                
                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "poses" or group["name"] == "opacity_scale":
                continue

            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def cull_by_eigenvalue(self, min_eigenvalue):
        eigenvalues = self.get_scaling.max(dim=1)[0]**2
        valid_points_mask = eigenvalues >= min_eigenvalue
        
        optimizable_tensors = self._prune_optimizer(valid_points_mask)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        
    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "poses":
                continue
            assert len(group["params"]) == 1
            
            if group["name"] == "poses":
                continue
            
            if group["name"] == "opacity_scale":
                continue

            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        # print("Splitting:", selected_pts_mask.sum())
        if selected_pts_mask.sum() == 0:
            return

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.scaling_activation(self._scaling)[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        
        if self.app_opt:
            new_shape = (N,1)
        else:
            new_shape = (N,1,1)

        new_features_dc = self._features_dc[selected_pts_mask].repeat(*new_shape)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(*new_shape)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)


        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    @property
    def get_mask(self):
        return self._mask

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        # print("Cloning:", selected_pts_mask.sum())
        new_xyz = self._xyz[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)


    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

        
    def _invalidate_query_caches(self):
        self._smoothed_point_features = None
        self._kd_tree = None

    def _build_kd_tree(self):
        from scipy.spatial import cKDTree
        return cKDTree(self.get_xyz.detach().cpu().numpy())

    def get_cached_smoothed_point_features(self, K = 16, dropout = 0.5):
        """Deterministic, cached smoothed features for interactive queries.

        Training uses get_smoothed_point_features(), which re-samples the dropout
        subset on every call; the GUI needs a stable answer instead.
        """
        if self._smoothed_point_features is None:
            self._smoothed_point_features = self.get_smoothed_point_features(K=K, dropout=dropout)
        return self._smoothed_point_features

    def get_nearest_feature(self, query_point, dist_thresh=0.25):
        """Smoothed contrastive feature of the Gaussian nearest to a 3D world point."""
        if self._kd_tree is None:
            self._kd_tree = self._build_kd_tree()

        _, nns = self._kd_tree.query(query_point, k=1, distance_upper_bound=dist_thresh)

        all_features = self.get_cached_smoothed_point_features()
        nn_feats = all_features[nns].reshape(-1, all_features.shape[1])
        return nn_feats.mean(0)

    def get_smoothed_point_features(self, K = 16, dropout = 0.5):
        if K <= 1:
            return self._point_features

        assert dropout < 0 or int(K*dropout) >= 1

        with torch.no_grad():
            if self.feature_smooth_map is None or self.feature_smooth_map["K"] != K:
                from scipy.spatial import cKDTree

                xyz = self.get_xyz
                xyz_np = xyz.detach().cpu().numpy()
                tree = cKDTree(xyz_np)
                _, nearest_k_idx = tree.query(xyz_np, k=K)
                nearest_k_idx = torch.tensor(nearest_k_idx, device=xyz.device)
                self.feature_smooth_map = {"K": K, "m": nearest_k_idx}

        normed_features = torch.nn.functional.normalize(self._point_features, dim = -1, p = 2)

        if dropout > 0 and dropout < 1:
            select_point = torch.randperm(K)[ : int(K*dropout)]

            select_idx = self.feature_smooth_map["m"][:, select_point]
            ret = normed_features[select_idx, :].mean(dim = 1)
        else:
            ret = normed_features[self.feature_smooth_map["m"], :].mean(dim = 1)

        return ret


class AppearanceOptModule(torch.nn.Module):
    """Appearance optimization module."""

    def __init__(
        self,
        n: int,
        feature_dim: int,
        embed_dim: int = 16,
        sh_degree: int = 3,
        mlp_width: int = 64,
        mlp_depth: int = 2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.sh_degree = sh_degree
        self.num_images = n
        self.embeds = torch.nn.Embedding(n, embed_dim)
        layers = []
        layers.append(
            torch.nn.Linear(embed_dim + feature_dim + (sh_degree + 1) ** 2, mlp_width)
        )
        layers.append(torch.nn.ReLU(inplace=True))
        for _ in range(mlp_depth - 1):
            layers.append(torch.nn.Linear(mlp_width, mlp_width))
            layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.Linear(mlp_width, 3))
        self.color_head = torch.nn.Sequential(*layers)

    def forward(
        self, features: torch.Tensor, embed_ids: torch.Tensor, dirs: torch.Tensor, sh_degree: int
        ) -> torch.Tensor:
        """Adjust appearance based on embeddings.

        Args:
            features: (N, feature_dim)
            embed_ids: (C,)
            dirs: (C, N, 3)

        Returns:
            colors: (C, N, 3)
        """

        C, N = dirs.shape[:2]
        # Camera embeddings
        if embed_ids is None:
            embeds = torch.zeros(C, self.embed_dim, device=features.device)
        else:
            embeds:torch.Tensor = self.embeds(embed_ids)  # [C, D2]
            embeds = embeds.mean(0, True)

        embeds = embeds[:, None, :].expand(-1, N, -1)  # [C, N, D2]
        # GS features
        features = features[None, :, :].expand(C, -1, -1)  # [C, N, D1]
        # View directions
        dirs = F.normalize(dirs, dim=-1)  # [C, N, 3]
        num_bases_to_use = (sh_degree + 1) ** 2
        num_bases = (self.sh_degree + 1) ** 2
        sh_bases = torch.zeros(C, N, num_bases, device=features.device)  # [C, N, K]
        sh_bases[:, :, :num_bases_to_use] = _eval_sh_bases_fast(num_bases_to_use, dirs)
        # Get colors
        if self.embed_dim > 0:
            h = torch.cat([embeds, features, sh_bases], dim=-1)  # [C, N, D1 + D2 + K]
        else:
            h = torch.cat([features, sh_bases], dim=-1)
        colors = self.color_head(h)
        return colors





def _eval_sh_bases_fast(basis_dim: int, dirs: torch.Tensor):
    """
    Evaluate spherical harmonics bases at unit direction for high orders
    using approach described by
    Efficient Spherical Harmonic Evaluation, Peter-Pike Sloan, JCGT 2013
    https://jcgt.org/published/0002/02/06/


    :param basis_dim: int SH basis dim. Currently, only 1-25 square numbers supported
    :param dirs: torch.Tensor (..., 3) unit directions

    :return: torch.Tensor (..., basis_dim)

    See reference C++ code in https://jcgt.org/published/0002/02/06/code.zip
    """
    result = torch.empty(
        (*dirs.shape[:-1], basis_dim), dtype=dirs.dtype, device=dirs.device
    )

    result[..., 0] = 0.2820947917738781

    if basis_dim <= 1:
        return result

    x, y, z = dirs.unbind(-1)

    fTmpA = -0.48860251190292
    result[..., 2] = -fTmpA * z
    result[..., 3] = fTmpA * x
    result[..., 1] = fTmpA * y

    if basis_dim <= 4:
        return result

    z2 = z * z
    fTmpB = -1.092548430592079 * z
    fTmpA = 0.5462742152960395
    fC1 = x * x - y * y
    fS1 = 2 * x * y
    result[..., 6] = 0.9461746957575601 * z2 - 0.3153915652525201
    result[..., 7] = fTmpB * x
    result[..., 5] = fTmpB * y
    result[..., 8] = fTmpA * fC1
    result[..., 4] = fTmpA * fS1

    if basis_dim <= 9:
        return result

    fTmpC = -2.285228997322329 * z2 + 0.4570457994644658
    fTmpB = 1.445305721320277 * z
    fTmpA = -0.5900435899266435
    fC2 = x * fC1 - y * fS1
    fS2 = x * fS1 + y * fC1
    result[..., 12] = z * (1.865881662950577 * z2 - 1.119528997770346)
    result[..., 13] = fTmpC * x
    result[..., 11] = fTmpC * y
    result[..., 14] = fTmpB * fC1
    result[..., 10] = fTmpB * fS1
    result[..., 15] = fTmpA * fC2
    result[..., 9] = fTmpA * fS2

    if basis_dim <= 16:
        return result

    fTmpD = z * (-4.683325804901025 * z2 + 2.007139630671868)
    fTmpC = 3.31161143515146 * z2 - 0.47308734787878
    fTmpB = -1.770130769779931 * z
    fTmpA = 0.6258357354491763
    fC3 = x * fC2 - y * fS2
    fS3 = x * fS2 + y * fC2
    result[..., 20] = 1.984313483298443 * z2 * (
        1.865881662950577 * z2 - 1.119528997770346
    ) + -1.006230589874905 * (0.9461746957575601 * z2 - 0.3153915652525201)
    result[..., 21] = fTmpD * x
    result[..., 19] = fTmpD * y
    result[..., 22] = fTmpC * fC1
    result[..., 18] = fTmpC * fS1
    result[..., 23] = fTmpB * fC2
    result[..., 17] = fTmpB * fS2
    result[..., 24] = fTmpA * fC3
    result[..., 16] = fTmpA * fS3
    return result
