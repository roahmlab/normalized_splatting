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

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"


#include <torch/extension.h>

void saveTensor(void* d_ptr, c10::ScalarType type, torch::IntArrayRef shape, const std::string& file_path)
{
	auto options = torch::TensorOptions().dtype(type).device(torch::kCUDA, 0);
	torch::Tensor tensor = torch::from_blob(d_ptr, shape, options);
	torch::save(tensor, file_path);
}

void saveFloatTensor(float* d_ptr, torch::IntArrayRef shape, const std::string& file_path)
{
	saveTensor(d_ptr, torch::kFloat32, shape, file_path);
}

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
	int P,
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii,
	dim3 grid,
	int2* rects=nullptr)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if (radii[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;

		if(rects == nullptr)
			getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);
		else
			getRect(points_xy[idx], rects[idx], rect_min, rect_max, grid);

		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values 
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth. 
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				uint64_t key = y * grid.x + x;
				key <<= 32;
				key |= *((uint32_t*)&depths[idx]);
				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx;
				off++;
			}
		}
	}
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.cov3D, P * 6, 128);
	obtain(chunk, geom.conic_opacity, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	obtain(chunk, geom.normalizing_const, P, 128);
	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, N, 128);
	return img;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::Rasterizer::forward(
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
	const float tan_fovx, float tan_fovy,  
	const float filter_radius,  
	const bool prefiltered,  
	float* out_color,  
	float* out_depth,  
	float* out_alpha,  
	int* radii,
	bool normalize_gaussians,
	int64_t* sort_order_in,  
	int64_t* sort_order_out,  
	const int* rects,  
	const float* boxmin,  
	const float* boxmax,  
	const float* max_per_pixel_depth,
	const float* contrastive_features,
	float* out_contrastive_features,
	const bool debug)
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	float3 minn = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
	float3 maxx = { FLT_MAX, FLT_MAX, FLT_MAX };
	if (boxmin != nullptr)
	{
		minn = *((float3*)boxmin);
		maxx = *((float3*)boxmax);
	}

	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	CHECK_CUDA(FORWARD::preprocess(
		P, D, M,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		shs,
		geomState.clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		filter_radius,
		radii,
		geomState.means2D,
		geomState.depths,
		geomState.cov3D,
		geomState.rgb,
		geomState.conic_opacity,
		tile_grid,
		geomState.tiles_touched,
		geomState.normalizing_const,
		prefiltered,
		normalize_gaussians,
		(int2*)rects,
		minn,
		maxx
	), debug)

	// saveFloatTensor(geomState.cov3D, {P, 6}, "debug/cov3D.pt");
	// saveFloatTensor((float*)geomState.means2D, {P, 2}, "debug/means2D.pt");
	// saveFloatTensor(geomState.depths, {P, 1}, "debug/depths.pt");
	// saveFloatTensor(geomState.rgb, {P, 3}, "debug/rgb.pt");
	// saveFloatTensor((float*)geomState.conic_opacity, {P, 4}, "debug/conic_opacity.pt");
	// saveTensor(radii, torch::kInt32, {P, 1}, "debug/radii.pt");


	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)

	// saveFloatTensor((float*)geomState.point_offsets, {P,}, "debug/point_offsets.pt");

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	int num_rendered;
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

	// printf("Num Rendered: %d\n", num_rendered);

	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	duplicateWithKeys << <(P + 255) / 256, 256 >> > (
		P,
		geomState.means2D,
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		tile_grid)
	CHECK_CUDA(, debug)

	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// saveFloatTensor(geomState.depths, {P,}, "debug/depths.pt");
	
	// saveTensor(binningState.point_list_keys_unsorted, torch::kInt64, {num_rendered,1}, "debug/point_list_keys_unsorted.pt");
	// saveTensor(binningState.point_list_unsorted, torch::kInt32, {num_rendered,1}, "debug/point_list_keys_unsorted.pt");

	// Sort complete list of (duplicated) Gaussian indices by keys

	if(sort_order_in != nullptr && sort_order_out != nullptr) {
		size_t scratch_size;
		CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
			nullptr,
			scratch_size,
			binningState.point_list_keys_unsorted, binningState.point_list_keys,
			sort_order_in, sort_order_out,
			num_rendered, 0, 32 + bit), debug)

		char* scratch_space;
		cudaMalloc(&scratch_space, scratch_size);

		CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
			scratch_space,
			scratch_size,
			binningState.point_list_keys_unsorted, binningState.point_list_keys,
			sort_order_in, sort_order_out,
			num_rendered, 0, 32 + bit), debug)

			CHECK_CUDA(cudaFree(scratch_space), debug);

			return num_rendered;
	} else {
			CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
				binningState.list_sorting_space,
				binningState.sorting_size,
				binningState.point_list_keys_unsorted, binningState.point_list_keys,
				binningState.point_list_unsorted, binningState.point_list,
				num_rendered, 0, 32 + bit), debug)
	}

	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(, debug);

	// Let each tile blend its range of Gaussians independently in parallel
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;

	CHECK_CUDA(FORWARD::render(
		tile_grid, block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		geomState.means2D,
		feature_ptr,
		geomState.depths,
		geomState.conic_opacity,
		geomState.normalizing_const,
		out_alpha,
		imgState.n_contrib,
		background,
		out_color,
		out_depth,
		max_per_pixel_depth,
		contrastive_features,
		out_contrastive_features), debug);

	return num_rendered;
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
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
	bool debug)
{
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
	const float* depth_ptr = geomState.depths;
	CHECK_CUDA(BACKWARD::render(
		tile_grid,
		block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		background,
		geomState.means2D,
		geomState.conic_opacity,
		geomState.normalizing_const,
		geomState.cov3D,
		color_ptr,
		depth_ptr,
		alphas,
		f_contrastive,
		imgState.n_contrib,
		viewmatrix,
		dL_dpix,
		dL_ddepths,
		dL_dalphas,
		dL_dcontrastive_img,
		focal_x, focal_y,
		normalize_gaussians,
		means3D,
		(float3*)dL_dmean2D,
		(float4*)dL_dconic,
		dL_dopacity,
		dL_dcolor,
		(float3*) dL_dt, // add new
		(float3*)dL_dmean3D,
		dL_dnorm,
		dL_dcontrastive_features), debug)
		
	// float* dL_dnorm_host = new float[P*sizeof(float)];
	// cudaMemcpy(dL_dnorm_host, dL_dnorm, P*sizeof(float), cudaMemcpyDeviceToHost);

	// int max_idx;
	// float max_d = 0;
	// for(int i = 0; i < P; ++i) {
	// 	if(fabs(dL_dnorm_host[i]) >= max_d) {
	// 		max_idx = i;
	// 		max_d = dL_dnorm_host[i];
	// 	}
	// }
	// std::cout << "Max ID: " << max_idx << "," << max_d << std::endl;

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
	CHECK_CUDA(BACKWARD::preprocess(P, D, M,
		(float3*)means3D,
		radii,
		shs,
		geomState.clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		cov3D_ptr,
		viewmatrix,
		projmatrix,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		filter_radius,
		normalize_gaussians,
		(glm::vec3*)campos,
		(float3*)dL_dmean2D,
		dL_dconic,
		(float3*) dL_dt, // add new
		dL_dnorm,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_dsh,
		(glm::vec3*)dL_dscale,
		(glm::vec4*)dL_drot,
		(glm::mat4*)dL_dviewmat,
		(glm::mat4*)dL_dprojmat), debug)
}


void CudaRasterizer::Rasterizer::sortKeys(const uint32_t size, const int64_t* data, int64_t* sort_order, int end_bit)
{
	if(end_bit < 0) {
		end_bit = sizeof(int64_t) * 8;
	}
	size_t temp_storage_bytes;

	int64_t* keys_out_scratch;
	CHECK_CUDA(cudaMalloc(&keys_out_scratch, size * sizeof(int64_t)), true)

	int64_t* vals_in_clone;
	CHECK_CUDA(cudaMalloc(&vals_in_clone, size * sizeof(int64_t)), true);
	CHECK_CUDA(cudaMemcpy(vals_in_clone, sort_order, size * sizeof(int64_t), cudaMemcpyDeviceToDevice), true);

	// first call does no work, just stores the correct size to temp_storage_bytes
	cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes, (uint64_t*)data, (uint64_t*)keys_out_scratch, vals_in_clone,
																						sort_order, size, 0, end_bit);

	uint8_t* scratch_space;
	CHECK_CUDA(cudaMalloc(&scratch_space, temp_storage_bytes), true);

	// This one actually does the work
	cub::DeviceRadixSort::SortPairs(scratch_space, temp_storage_bytes, (uint64_t*)data, (uint64_t*)keys_out_scratch, vals_in_clone,
																						sort_order, size, 0, end_bit);
	


	CHECK_CUDA(cudaFree(keys_out_scratch), true);
	CHECK_CUDA(cudaFree(vals_in_clone), true);
	CHECK_CUDA(cudaFree(scratch_space), true);

}

void CudaRasterizer::Rasterizer::makeKeys(const uint32_t size,
																					const float2* points_xy,
																					const float* depths,
																					const uint32_t* offsets,
																					int* radii,
																					dim3 grid,
																					uint64_t* gaussian_keys_unsorted,
																					uint32_t* gaussian_values_unsorted)
{
	duplicateWithKeys <<<(size + 255) / 256, 256 >>> (
		size,
		points_xy,
		depths,
		offsets,
		gaussian_keys_unsorted,
		gaussian_values_unsorted,
		radii,
		grid);
	CHECK_CUDA(, true)
}
