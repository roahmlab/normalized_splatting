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

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

__device__
void printmat3(glm::mat3 M, char* label) {
	printf("===== %s =====:\n%e, %e,, %e\n%e, %e, %e\n, %e, %e, %e\n========\n", label,
	M[0][0], M[0][1], M[0][2], M[1][0], M[1][1], M[1][2], M[2][0], M[2][1], M[2][2]);
}

// Should this matter? no.
// does it matter? apparently.
__device__
glm::vec3 vec3max(const glm::vec3& input) {
	return glm::vec3((input.x > 0.0f) ? input.x : 0.0f,
									 (input.y > 0.0f) ? input.y : 0.0f,
									 (input.z > 0.0f) ? input.z : 0.0f);
}

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ __noinline__ 
glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, 
																   glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	
	return vec3max(result);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy,
															 const float* cov3D, const float* viewmatrix,
															 const float filter_radius,
															 const bool normalize_gaussians, int idx, float* norm_const)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	const float tzsq = t.z * t.z;
	const float tysq = t.y * t.y;
	const float txsq = t.x * t.x;
	const float tnorm = sqrt(txsq + tysq + tzsq);
	
	const glm::mat3 J = normalize_gaussians 
										?	glm::mat3(
													focal_x / t.z, 0.0f, -(focal_x * t.x) / tzsq,
													0.0f, focal_y / t.z, -(focal_y * t.y) / tzsq,
													t.x/(tnorm), t.y/(tnorm), t.z/(tnorm))
										: glm::mat3(
													focal_x / t.z, 0.0f, -(focal_x * t.x) / tzsq,
													0.0f, focal_y / t.z, -(focal_y * t.y) / tzsq,
													0.0f, 0.0f, 0.0f);



	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov3D_transf = glm::transpose(T) * glm::transpose(Vrk) * T;

	cov3D_transf[0][0] += filter_radius;
	cov3D_transf[1][1] += filter_radius;
	// printf("Radius: %f, covx: %f, covz: %f\n", filter_radius, cov3D_transf[0][0], cov3D_transf[1][1]);

	if(norm_const) {
		const float C00 = cov3D_transf[0][0];
		const float C01 = cov3D_transf[0][1];
		const float C10 = cov3D_transf[1][0];
		const float C11 = cov3D_transf[1][1];

		const float det_2d = C00 * C11 - C01 * C10;

		*norm_const = normalize_gaussians ? (glm::determinant(J) / (sqrt(det_2d) * 2 * M_PI)) : 1;
	}

	// if(idx % 100 == 0) {
	// 	printf("==== cov2d idx: %d ====\nVrk: [%f, %f %f; %f %f %f; %f %f %f]\ncov2D:[%f %f %f]\nnorm_const:%f\n",
	// 		idx,
	// 		Vrk[0][0], Vrk[0][1], Vrk[0][2], Vrk[1][0], Vrk[1][1], Vrk[1][2], Vrk[2][0], Vrk[2][1], Vrk[2][2],
	// 		cov3D_transf[0][0], cov3D_transf[0][1], cov3D_transf[1][1],
	// 		*norm_const);
	// }


	return {cov3D_transf[0][0], cov3D_transf[0][1], cov3D_transf[1][1]};
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	const float filter_radius,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	float* normalizing_consts,
	bool prefiltered,
	const bool normalize_gaussians,
	int2* rects,
	const float3 boxmin,
	const float3 boxmax)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view)) {
		return;
	}

	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	
	if (p_orig.x < boxmin.x || p_orig.y < boxmin.y || p_orig.z < boxmin.z ||
		p_orig.x > boxmax.x || p_orig.y > boxmax.y || p_orig.z > boxmax.z)
		return;
	
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	float norm_const;
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix,
														filter_radius, normalize_gaussians, idx, &norm_const);

	// Invert covariance (EWA algorithm)
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f) {
		return;
	}
	float det_inv = 1.f / det;

	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	normalizing_consts[idx] = norm_const;

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	uint2 rect_min, rect_max;

	if (rects == nullptr) 	// More conservative
	{
		getRect(point_image, my_radius, rect_min, rect_max, grid);
	}
	else // Slightly more aggressive, might need a math cleanup
	{
		const int2 my_rect = { (int)ceil(3.f * sqrt(cov.x)), (int)ceil(3.f * sqrt(cov.z)) };
		rects[idx] = my_rect;
		getRect(point_image, my_rect, rect_min, rect_max, grid);
	}

	// getRect(point_image, my_radius, rect_min, rect_max, grid);

	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0){
		return;
	}

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr)
	{
		// if(idx == 0)
		// printf("================================================\n");
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		// if(idx == 0)
		// 	printf("Result: [%f, %f, %f]\n", result.x, result.y, result.x);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z; // TODO: make this more accurate using rotation and scale matrix!
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] };
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);

}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float* __restrict__ depths,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ normalizing_consts,
	float* __restrict__ out_alpha,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	float* __restrict__ out_depth,
	const float* __restrict__ max_per_pixel_depth,
	const float* __restrict__ contrastive_features,
	float* __restrict__ out_contrastive_features)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	const bool pp = pix.x == 599 && pix.y == 432;

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_normalizing_consts[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };
	float D = 0;

	float f_contrastive[NUM_CONTRASTIVE_CHANNELS] = {0};

	const bool should_print = false && pp;

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			collected_normalizing_consts[block.thread_rank()] = normalizing_consts[coll_id];
		}

		// Fetches BLOCK_SIZE gaussians, then waits for them all to be fetched.
		block.sync();



		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];

			const float depth = depths[collected_id[j]];

			if(max_per_pixel_depth != nullptr && depth > max_per_pixel_depth[pix_id]) {
				done = true;
				continue;
			}

			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			const float det_inv_cov = (con_o.x * con_o.z) - (con_o.y * con_o.y);

			// Full Normalization
			// const float normalization = sqrt(det_inv_cov) * M_1_PI * 0.5;
			
			// Numerically Sensible Normalization
			const float normalization = collected_normalizing_consts[j];

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, normalization * con_o.w * exp(power));
			
			if (alpha < 1.0f / 255.0f){
				if(should_print) {
					printf("==contributor: %d SKIPPED with ALpha: %.15e\ncoll_id:%d\nxy: [%e, %e]\npix:[%u,%u]\nd:[%e,%e]\ncon_o:[%e,%e,%e,%e]\npower: %e\n"
							 "dic:%e\nn:%e\na:%e\nT: %e\n",
							 contributor-1,alpha,collected_id[j],xy.x,xy.y,pix.x,pix.y,d.x,d.y,con_o.x,con_o.y,con_o.z,con_o.w,power,
							 det_inv_cov,normalization,alpha,T);
				}
				continue;
			}
			float test_T = T * (1 - alpha); // visibility
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}
			if(should_print) {
				printf("=====contributor: %d\ncoll_id:%d\nxy: [%e, %e]\npix:[%u,%u]\nd:[%e,%e]\ncon_o:[%e,%e,%e,%e]\npower: %e\n"
							 "dic:%e\nn:%e\na:%e\nT:%e\ntest_T:%e\n",
							 contributor-1, collected_id[j],xy.x,xy.y,pix.x,pix.y,d.x,d.y,con_o.x,con_o.y,con_o.z,con_o.w,power,
							 det_inv_cov,normalization,alpha,T,test_T);
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++){
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;
				if(should_print) {
					printf("CH: %d, Features: %e, Contrib: %e, Current: %e\n", ch, features[collected_id[j] * CHANNELS + ch],
								features[collected_id[j] * CHANNELS + ch] * alpha * T, C[ch]);
				}	
			}

			if(contrastive_features != nullptr) {
				for(int ch = 0; ch < NUM_CONTRASTIVE_CHANNELS; ++ch) {
					f_contrastive[ch] += contrastive_features[collected_id[j] * NUM_CONTRASTIVE_CHANNELS + ch] * alpha * T;
				}
			}

			
			D += depth * alpha * T; // weight: w = alpha * T

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++){
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
			if(should_print) {
				printf("ch: %d, pix_id: %d, C[ch]: %e, T: %e, bg: %e, result: %e\n", ch, pix_id, C[ch], T, bg_color[ch], out_color[ch*H*W + pix_id]);
			}
		}
		if(out_alpha)
			out_alpha[pix_id] = 1 - T;
		
		if(out_depth)
			out_depth[pix_id] = D;
    
		for (int ch = 0; out_contrastive_features != nullptr && f_contrastive != nullptr && ch < NUM_CONTRASTIVE_CHANNELS; ++ch) {
			out_contrastive_features[ch*H*W + pix_id] = f_contrastive[ch];
		}
		
		if(should_print) {
			printf("pix_id: %d, out_alpha: %e\n", pix_id, out_alpha[pix_id]);
			printf("pix_id: %d, out_depth: %e\n", pix_id, out_depth[pix_id]);
		}
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float* depths,
	const float4* conic_opacity,
	const float* normalizing_consts,
	float* out_alpha,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	float* out_depth,
	const float* max_per_pixel_depth,
	const float* contrastive_features,
	float* out_contrastive_features)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		depths,
		conic_opacity,
		normalizing_consts,
		out_alpha,
		n_contrib,
		bg_color,
		out_color,
		out_depth,
		max_per_pixel_depth,
		contrastive_features,
		out_contrastive_features);
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	const float filter_radius,
	int* radii,
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	float* normalizing_consts,
	bool prefiltered,
	bool normalize_gaussians,
	int2* rects,
	const float3 minn,
	const float3 maxx)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		filter_radius,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		normalizing_consts,
		prefiltered,
		normalize_gaussians,
		rects,
		minn,
		maxx
		);
}