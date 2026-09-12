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

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C
import warnings
warnings.simplefilter('once')

def finite_differencing_gradient(func, input_tensor, epsilon=1e-5):
    output_pos = func(input_tensor + epsilon)
    output_neg = func(input_tensor - epsilon)
    
    # Estimate gradient
    grad_est = (output_pos - output_neg) / (2 * epsilon)
    
    return grad_est

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    viewmatrix,
    projmatrix,
    raster_settings,
    contrastive_features = None,
    sort_order_in = None,
    sort_order_out = None,
    max_per_pixel_depth = None
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        viewmatrix,
        projmatrix,
        raster_settings,
        contrastive_features,
        sort_order_in,
        sort_order_out,
        max_per_pixel_depth
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        viewmatrix,
        projmatrix,
        raster_settings,
        contrastive_features,
        sort_order_in,
        sort_order_out,
        max_per_pixel_depth
    ):
        if sort_order_in is None or sort_order_out is None:
            sort_order_in = torch.empty((0,)).long()
            sort_order_out = torch.empty((0,)).long()
        
        if contrastive_features is None:
            contrastive_features = torch.empty((0,))

        if max_per_pixel_depth is None:
            max_per_pixel_depth = torch.empty((0,)).float()
        else:
            warnings.warn("Max per pixel depth has been set. This is not considered in the backward pass, so it should only be used for visualization and not training.", UserWarning)

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            viewmatrix,
            projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.filter_radius,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug,
            raster_settings.normalize_gaussians,
            sort_order_in,
            sort_order_out,
            max_per_pixel_depth,
            contrastive_features
        )
        
        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, color, depth, alpha, contrastive_img, radii,\
                    geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, depth, alpha, contrastive_img, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(opacities, colors_precomp, means3D, 
                              scales, rotations, cov3Ds_precomp, 
                              radii, sh, geomBuffer, binningBuffer, 
                              imgBuffer, alpha, viewmatrix, projmatrix,
                              contrastive_features)
        return color, radii, depth, alpha, contrastive_img

    @staticmethod
    def backward(ctx, grad_color, grad_radii, grad_depth, grad_alpha, grad_contrastive_img):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        opacities, colors_precomp, means3D, scales, rotations,\
            cov3Ds_precomp, radii, sh, geomBuffer, \
            binningBuffer, imgBuffer, alpha, viewmatrix, \
            projmatrix, contrastive_features = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D, 
                radii, 
                colors_precomp, 
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                viewmatrix, 
                projmatrix, 
                contrastive_features,
                raster_settings.tanfovx, 
                raster_settings.tanfovy,
                raster_settings.filter_radius, 
                raster_settings.normalize_gaussians,
                grad_color,
                grad_depth,
                grad_alpha,
                grad_contrastive_img,
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                alpha,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations, grad_viewmatrix, grad_projmatrix = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, \
             grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations, \
             grad_viewmatrix, grad_projmatrix, grad_contrastive_f = _C.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            grad_viewmatrix,
            grad_projmatrix,
            None,
            None if grad_contrastive_f.numel() == 0 else grad_contrastive_f,
            None,
            None,
            None
        )
        return grads


class AllTensorRasterizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bg, scale_modifier, tanfovx, tanfovy,
                image_height, image_width, sh_degree,
                campos, prefiltered, debug,
                means3D, means2D, sh, colors_precomp,
                opacities, scales, rotations, cov3Ds_precomp,
                viewmatrix, projmatrix):
        
        args = (
            bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            scale_modifier.item(),
            cov3Ds_precomp,
            viewmatrix,
            projmatrix,
            # raster_settings.viewmatrix,
            # raster_settings.projmatrix,
            tanfovx.item(),
            tanfovy.item(),
            image_height.item(),
            image_width.item(),
            sh,
            sh_degree.item(),
            campos,
            bool(prefiltered.item()),
            bool(debug.item())
        )
        
        result = _C.rasterize_gaussians(*args)
        num_rendered, color, depth, alpha, radii, geomBuffer, binningBuffer, imgBuffer = result
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(bg, scale_modifier, tanfovx, tanfovy,
                              image_height, image_width, sh_degree,
                              campos, prefiltered, debug,
                              opacities, colors_precomp, means3D, scales, rotations, cov3Ds_precomp,
                              radii, sh, geomBuffer, binningBuffer, imgBuffer, alpha, viewmatrix, projmatrix)

        return result


    @staticmethod
    def backward(ctx, grad_color, grad_radii, grad_depth, grad_alpha):


        bg, scale_modifier, tanfovx, tanfovy,\
        image_height, image_width, sh_degree,\
        campos, prefiltered, debug,\
        opacities, colors_precomp, means3D, scales, rotations, cov3Ds_precomp,\
        radii, sh, geomBuffer, binningBuffer, imgBuffer, alpha, viewmatrix, projmatrix = ctx.saved_tensors

        num_rendered = ctx.num_rendered

        args = (bg,
                means3D, 
                radii, 
                colors_precomp, 
                scales, 
                rotations, 
                scale_modifier, 
                cov3Ds_precomp, 
                viewmatrix, 
                projmatrix, 
                tanfovx, 
                tanfovy, 
                grad_color,
                grad_depth,
                grad_alpha,
                sh, 
                sh_degree, 
                campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                alpha,
                debug)

        result = _C.rasterize_gaussians_backward(*args)

        grad_bg = torch.zeros_like(bg)
        sg = torch.zeros_like(grad_bg.flatten()[0])

        grad_means2D, grad_colors_precomp, grad_opacities,\
            grad_means3D, grad_cov3Ds_precomp, grad_sh,\
            grad_scales, grad_rotations, grad_viewmatrix, grad_projmatrix = result

        return grad_bg, sg, sg, sg, sg, sg, sg, torch.zeros_like(campos), sg, sg, grad_means3D,\
               grad_means2D, grad_sh, grad_colors_precomp, grad_opacities, grad_scales, \
               grad_rotations, grad_cov3Ds_precomp, grad_viewmatrix, grad_projmatrix

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    # viewmatrix : torch.Tensor
    # projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool
    filter_radius: float
    normalize_gaussians: bool


class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, opacities, shs = None, colors_precomp = None, 
                scales = None, rotations = None, cov3D_precomp = None, viewmatrix = None, projmatrix = None,
                contrastive_features = None, sort_order_in = None, sort_order_out = None, max_per_pixel_depth=None):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            viewmatrix,
            projmatrix,
            raster_settings,
            contrastive_features,
            sort_order_in,
            sort_order_out,
            max_per_pixel_depth
        )



def SortKeys(in_data: torch.Tensor, msb: int = -1):
    return _C.sort_keys(in_data, msb)


def MakeKeys(size: int, points_xy: torch.Tensor,
             depths: torch.Tensor, offsets: torch.Tensor,
             radii: torch.Tensor, grid: torch.Tensor):
    
    return _C.make_keys(size, points_xy, depths, offsets, radii, grid)


def GetExactSortOrder(num_rendered: int,
                      means3D,
                      means2D,
                      shs,
                      opacities,
                      scales, 
                      rotations,
                      viewmatrix,
                      projmatrix,
                      raster_settings):
    
    sort_order_in = torch.arange(num_rendered).cuda().long()
    sort_order_out = torch.zeros_like(sort_order_in)
    
    
    rasterize_gaussians(means3D, means2D, shs, torch.empty((0,)).to(means3D), opacities,
                        scales, rotations, torch.empty((0,)).to(means3D), viewmatrix, projmatrix,
                        raster_settings, sort_order_in, sort_order_out)
    
    return sort_order_out