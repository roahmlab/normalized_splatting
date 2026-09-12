/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include <fstream>
#include <string>
#include <functional>

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
					 torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& colors,
	const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
	const float filter_radius,
	const int image_height,
	const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool debug,
	const bool normalize_gaussians,
	torch::Tensor& sort_order_in,
	torch::Tensor& sort_order_out,
	const torch::Tensor& max_per_pixel_depth,
	torch::Tensor& contrastive_features)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }
  
  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor out_depth = torch::full({1, H, W}, 0.0, float_opts);
  torch::Tensor out_alpha = torch::full({1, H, W}, 0.0, float_opts);
  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  torch::Tensor contrastive_features_out = contrastive_features.numel() == 0
			? torch::empty({0,}, float_opts)
			:	torch::full({NUM_CONTRASTIVE_CHANNELS, H, W}, 0.0, float_opts);

  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);

  int rendered = 0;
  if(P != 0)
  {
	  int M = 0;
	  if(sh.size(0) != 0)
	  {
			M = sh.size(1);
		}

		rendered = CudaRasterizer::Rasterizer::forward(
			geomFunc,
			binningFunc,
			imgFunc,
			P, degree, M,
			background.contiguous().data<float>(),
			W, H,
			means3D.contiguous().data<float>(),
			sh.contiguous().data_ptr<float>(),
			colors.contiguous().data<float>(), 
			opacity.contiguous().data<float>(), 
			scales.contiguous().data_ptr<float>(),
			scale_modifier,
			rotations.contiguous().data_ptr<float>(),
			cov3D_precomp.contiguous().data<float>(), 
			viewmatrix.contiguous().data<float>(), 
			projmatrix.contiguous().data<float>(),
			campos.contiguous().data<float>(),
			tan_fovx,
			tan_fovy,
			filter_radius,
			prefiltered,
			out_color.contiguous().data<float>(),
			out_depth.contiguous().data<float>(),
			out_alpha.contiguous().data<float>(),
			radii.contiguous().data<int>(),
			normalize_gaussians,
			sort_order_in.numel() == 0? nullptr : sort_order_in.contiguous().data<int64_t>(),
			sort_order_out.numel() == 0? nullptr : sort_order_out.contiguous().data<int64_t>(),
			nullptr, nullptr, nullptr,
			max_per_pixel_depth.numel() == 0? nullptr : max_per_pixel_depth.contiguous().data<float>(),
			contrastive_features.numel() == 0? nullptr : contrastive_features.contiguous().data<float>(),
			contrastive_features_out.contiguous().data<float>(),
			debug);
	}

  return std::make_tuple(rendered, out_color, out_depth, out_alpha, contrastive_features_out,
												 radii, geomBuffer, binningBuffer, imgBuffer);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, 
				   torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
					 torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
 	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
	const torch::Tensor& colors,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const torch::Tensor& contrastive_features,
	const float tan_fovx,
	const float tan_fovy,
	const float filter_radius,
	const bool normalize_gaussians,
	const torch::Tensor& dL_dout_color,
	const torch::Tensor& dL_dout_depth,
	const torch::Tensor& dL_dout_alpha,
	const torch::Tensor& dL_dcontrastive_img,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const torch::Tensor& alphas,
	const bool debug) 
{

  const int P = means3D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);
  
  int M = 0;
  if(sh.size(0) != 0)
  {	
	M = sh.size(1);
  }

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
  torch::Tensor dL_dviewmatrix = torch::zeros({P, 4, 4}, means3D.options());
  torch::Tensor dL_dprojmatrix = torch::zeros({P, 4, 4}, means3D.options());
  torch::Tensor dL_dt = torch::zeros({P, 3}, means3D.options());
	torch::Tensor dL_dnorm = torch::zeros({P, 1}, means3D.options());

	torch::Tensor dL_dcontrastive_features = contrastive_features.numel() == 0 
		? torch::zeros_like(contrastive_features)
		: torch::zeros({P, NUM_CONTRASTIVE_CHANNELS}, means3D.options());


  if(P != 0)
  {  
	  CudaRasterizer::Rasterizer::backward(P, degree, M, R,
			background.contiguous().data<float>(),
			W, H, 
			means3D.contiguous().data<float>(),
			sh.contiguous().data<float>(),
			colors.contiguous().data<float>(),
			alphas.contiguous().data<float>(),
			scales.data_ptr<float>(),
			scale_modifier,
			rotations.data_ptr<float>(),
			cov3D_precomp.contiguous().data<float>(),
			viewmatrix.contiguous().data<float>(),
			projmatrix.contiguous().data<float>(),
			campos.contiguous().data<float>(),
			contrastive_features.numel() == 0? nullptr : contrastive_features.contiguous().data<float>(),
			tan_fovx,
			tan_fovy,
			filter_radius,
			normalize_gaussians,
			radii.contiguous().data<int>(),
			reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
			reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
			reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
			dL_dout_color.contiguous().data<float>(),
			dL_dout_depth.contiguous().data<float>(),
			dL_dout_alpha.contiguous().data<float>(),
			contrastive_features.numel() == 0? nullptr : dL_dcontrastive_img.contiguous().data<float>(),
			dL_dmeans2D.contiguous().data<float>(),
			dL_dconic.contiguous().data<float>(),  
			dL_dopacity.contiguous().data<float>(),
			dL_dcolors.contiguous().data<float>(),
			dL_dmeans3D.contiguous().data<float>(),
			dL_dcov3D.contiguous().data<float>(),
			dL_dsh.contiguous().data<float>(),
			dL_dscales.contiguous().data<float>(),
			dL_drotations.contiguous().data<float>(),
			dL_dviewmatrix.contiguous().data<float>(),
			dL_dprojmatrix.contiguous().data<float>(),
			dL_dt.contiguous().data<float>(),
			dL_dnorm.contiguous().data<float>(),
			dL_dcontrastive_features.contiguous().data<float>(),
			debug);
  }

	// torch::save(dL_dout_color, "grads/dL_dout_color.pt"); 
	// torch::save(dL_dout_depth, "grads/dL_dout_depth.pt"); 
	// torch::save(dL_dout_alpha, "grads/dL_dout_alpha.pt"); 
	// torch::save(dL_dmeans2D, "grads/dL_dmeans2D.pt"); 
	// torch::save(dL_dconic, "grads/dL_dconic.pt"); 
	// torch::save(dL_dopacity, "grads/dL_dopacity.pt"); 
	// torch::save(dL_dcolors, "grads/dL_dcolors.pt"); 
	// torch::save(dL_dmeans3D, "grads/dL_dmeans3D.pt"); 
	// torch::save(dL_dcov3D, "grads/dL_dcov3D.pt"); 
	// torch::save(dL_dsh, "grads/dL_dsh.pt"); 
	// torch::save(dL_dscales, "grads/dL_dscales.pt"); 
	// torch::save(dL_drotations, "grads/dL_drotations.pt"); 
	// torch::save(dL_dviewmatrix, "grads/dL_dviewmatrix.pt"); 
	// torch::save(dL_dprojmatrix, "grads/dL_dprojmatrix.pt"); 
	// torch::save(dL_dt, "grads/dL_dt.pt"); 
	// torch::save(dL_dnorm, "grads/dL_dnorm.pt");

	// std::cout << dL_dnorm.min().item() << "," << dL_dnorm.mean().item()<< ","
	// 					<< dL_dnorm.median().item() << "," << dL_dnorm.max().item() << std::endl;

  return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh,
		dL_dscales, dL_drotations, dL_dviewmatrix, dL_dprojmatrix, dL_dcontrastive_features);
}

torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix)
{ 
  const int P = means3D.size(0);
  
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
  if(P != 0)
  {
	CudaRasterizer::Rasterizer::markVisible(P,
		means3D.contiguous().data<float>(),
		viewmatrix.contiguous().data<float>(),
		projmatrix.contiguous().data<float>(),
		present.contiguous().data<bool>());
  }
  
  return present;
}

torch::Tensor sortKeys(torch::Tensor& keys, int end_bit /* = -1 */) {
	int64_t* data = keys.flatten().contiguous().data<int64_t>();
	const uint32_t size = static_cast<uint32_t>(keys.numel());

	torch::Tensor sort_order = torch::arange({(int)size}).to(keys).contiguous();

	int64_t* out_data = sort_order.data<int64_t>();

	CudaRasterizer::Rasterizer::sortKeys(size, data, out_data, end_bit);

	return sort_order;
}

std::tuple<torch::Tensor, torch::Tensor> makeKeys(const uint32_t size, torch::Tensor& points_xy,
																									torch::Tensor& depths, torch::Tensor& offsets, 
																									torch::Tensor& radii, torch::Tensor& grid)
{
	const int num_rendered = offsets.index({-1}).item<int>();
	torch::Tensor keys_unsorted = torch::zeros({(int)num_rendered}).to(points_xy).to(torch::kInt64);
	torch::Tensor vals_unsorted = torch::zeros({(int)num_rendered}).to(points_xy).to(torch::kInt32);

	dim3 grid_dim3(grid[0].item<int>(), grid[1].item<int>(), grid[2].item<int>());
	
	CudaRasterizer::Rasterizer::makeKeys(size,
																			 (float2*)points_xy.contiguous().data<float>(),
																			 depths.contiguous().data<float>(),
																			 (uint32_t*)offsets.contiguous().data<int32_t>(),
																			 radii.contiguous().data<int>(),
																			 grid_dim3,
																			 (uint64_t*)keys_unsorted.contiguous().data<int64_t>(),
																			 (uint32_t*)vals_unsorted.contiguous().data<int32_t>());

	return {keys_unsorted, vals_unsorted};
}