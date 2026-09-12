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

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>

namespace CudaRasterizer
{
	class Rasterizer
	{
	public:

		static void markVisible(
			int P,
			float* means3D,
			float* viewmatrix,
			float* projmatrix,
			bool* present);

		static int forward(
			std::function<char* (size_t)> geometryBuffer,   
			std::function<char* (size_t)> binningBuffer,   
			std::function<char* (size_t)> imageBuffer,   
			const int P, int D, int M,   
			const float* background,   
			const int width, int height,   
			const float* means3D,   
			const float* shs,   
			const float* colors_precomp,   
			const float* opacities,   
			const float* scales,   
			const float scale_modifier,   
			const float* rotations,   
			const float* cov3D_precomp,   
			const float* viewmatrix,   
			const float* projmatrix,   
			const float* cam_pos,   
			const float tan_fovx,float tan_fovy,   
			const float filter_radius,   
			const bool prefiltered,   
			float* out_color,   
			float* out_depth,   
			float* out_alpha,   
			int* radii = nullptr,
			bool normalize_gaussians = false,
			int64_t* sort_order_in = nullptr,   
			int64_t* sort_order_out = nullptr,   
			const int* rects = nullptr,   
			const float* boxmin = nullptr,   
			const float* boxmax = nullptr,
			const float* max_per_pixel_depth = nullptr,
			const float* contrastive_features = nullptr,
			float* out_contrastive_features = nullptr,
			const bool debug = false); 

		static void backward(
			const int P, int D, int M, int R,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* alphas,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* campos,
			const float* f_contrastive,
			const float tan_fovx, float tan_fovy,
			const float filter_radius,
			const bool normalize_gaussians,
			const int* radii,
			char* geom_buffer,
			char* binning_buffer,
			char* img_buffer,
			const float* dL_dpix,
			const float* dL_ddepths,
			const float* dL_dalphas,
			const float* dL_dcontrastive_img,
			float* dL_dmean2D,
			float* dL_dconic,
			float* dL_dopacity,
			float* dL_dcolor,
			float* dL_dmean3D,
			float* dL_dcov3D,
			float* dL_dsh,
			float* dL_dscale,
			float* dL_drot,
			float* dL_dviewmat,
			float* dL_dprojmat,
			float* dL_dt,
			float* dL_dnorm,
			float* dL_dcontrastive_features,
			bool debug);
	
		static void sortKeys(const uint32_t size, const int64_t* data, int64_t* sort_order, int end_bit = -1);

		static void makeKeys(const uint32_t size,
												 const float2* points_xy,
												 const float* depths,
												 const uint32_t* offsets,
												 int* radii,
												 dim3 grid,
												 uint64_t* gaussian_keys_unsorted,
												 uint32_t* gaussian_values_unsorted);

	};

};

#endif