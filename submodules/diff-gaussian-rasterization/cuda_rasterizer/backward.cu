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

#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;


// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
__device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, const bool* clamped, const glm::vec3* dL_dcolor, glm::vec3* dL_dmeans, glm::vec3* dL_dshs)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Gaussian to write SH gradients to
	glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = SH_C0;
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (
					SH_C3[0] * sh[9] * 3.f * 2.f * xy +
					SH_C3[1] * sh[10] * yz +
					SH_C3[2] * sh[11] * -2.f * xy +
					SH_C3[3] * sh[12] * -3.f * 2.f * xz +
					SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh[14] * 2.f * xz +
					SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dRGBdy += (
					SH_C3[0] * sh[9] * 3.f * (xx - yy) +
					SH_C3[1] * sh[10] * xz +
					SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh[12] * -3.f * 2.f * yz +
					SH_C3[4] * sh[13] * -2.f * xy +
					SH_C3[5] * sh[14] * -2.f * yz +
					SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dRGBdz += (
					SH_C3[1] * sh[10] * xy +
					SH_C3[2] * sh[11] * 4.f * 2.f * yz +
					SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh[13] * 4.f * 2.f * xz +
					SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	// Account for normalization of direction
	float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}



		/* Leaving this here for easy finding

		clear;

		% Conic: Inverse of 2D covariance
		syms cx cy cz 'real';
		C = [cx cy; cy cz];

		% Following notation from https://aalexan3.math.ncsu.edu/articles/gaussian_marginals.pdf
		syms("Sig_mn", [2,1], 'real');
		syms Sig_nn root2pi 'real';

		Snn_inv = Sig_nn - Sig_mn' * C * Sig_mn;

		N = root2pi * sqrt(Snn_inv);

		syms dx dy 'real';

		x = [dx; dy];

		G = N * exp(-0.5 * x' * C * x);

		dG_dcx = simplify(diff(G, cx));
		dG_dcy = simplify(diff(G, cy));
		dG_dcz = simplify(diff(G, cz));

		dG_ddelx = simplify(diff(G, dx));
		dG_ddely = simplify(diff(G, dy));

		dG_dsigmn_1 = simplify(diff(G, Sig_mn1));
		dG_dsigmn_2 = simplify(diff(G, Sig_mn2));

		dG_dsignn = simplify(diff(G, Sig_nn));


		% Derivatives of mean wrt J
		syms tx ty tz fx fy fz M 'real';

		n = sqrt(tx^2 + ty^2 + tz^2);

		phi = [
			tx * fx / tz;
			ty * fy / tz;
			n/M;
		];

		J  = jacobian(phi, [tx ty tz]);
		det_J = simplify(det(J));
		ddetJ_dtx = simplify(diff(det_J, tx));
		ddetJ_dty = simplify(diff(det_J, ty));
		ddetJ_dtz = simplify(diff(det_J, tz));
		*/


// Backward version of INVERSE 2D covariance matrix computation
// (due to length launched as separate kernel before other 
// backward steps contained in preprocess)
__global__ void computeCov2DCUDA(int P,
	const float3* means,
	const int* radii,
	const float* cov3Ds,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	const float filter_radius,
	const float* view_matrix,
	const float* dL_dconics,
	const float3* dL_dt_prev,
	const float* dL_dnorm,
	const bool normalize_gaussians,
	float3* dL_dmeans,
	float* dL_dcov,
	glm::mat4* dL_dviewmat)
{

	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	const bool pp = idx == 354833;
	const bool should_print = false && pp;

	// Reading location of 3D covariance for this Gaussian
	const float* cov3D = cov3Ds + 6 * idx;

	// Fetch gradients, recompute 2D covariance and relevant 
	// intermediate forward results needed in the backward.
	float3 mean = means[idx];
	float3 dL_dconic = { dL_dconics[4 * idx], dL_dconics[4 * idx + 1], dL_dconics[4 * idx + 3] };
	float3 t = transformPoint4x3(mean, view_matrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;
	
	const float x_grad_mul = txtz < -limx || txtz > limx ? 0 : 1;
	const float y_grad_mul = tytz < -limy || tytz > limy ? 0 : 1;

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
		view_matrix[0], view_matrix[4], view_matrix[8],
		view_matrix[1], view_matrix[5], view_matrix[9],
		view_matrix[2], view_matrix[6], view_matrix[10]);

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 T = W * J;

	glm::mat3 cov3D_transf = glm::transpose(T) * Vrk * T;
	const float sqrt2Pi = sqrt(M_PI * 2.0f);


	// Use helper variables for 2D covariance entries. More compact.
	const float C00 = cov3D_transf[0][0] += filter_radius;
	const float C01 = cov3D_transf[0][1];
	const float C10 = cov3D_transf[1][0];
	const float C11 = cov3D_transf[1][1] += filter_radius;

	const float det_2d = C00 * C11 - C01 * C01;

	// Setup for normalization term: norm = det(J) / (sqrt(det_2d)*2pi)
	const float dL_ddetJ = normalize_gaussians ? (dL_dnorm[idx]/(sqrt(det_2d)*2*M_PI)) : 0;

	// printf("det2D: %e\ndL_ddetJ: %e\n", det_2d, dL_ddetJ);

	//dL_dcij = dL_dnorm * dnorm_dC_ij <excludes component-specific terms>
	const float det2d_32 = pow(det_2d, 1.5);
	const float dL_dCij_mul = dL_dnorm[idx] * glm::determinant(J) / (4.*M_PI*det2d_32);

	float dL_dC00 = normalize_gaussians ? (-C11 * dL_dCij_mul) : 0;
	float dL_dC01 = normalize_gaussians ? (2 * C10  * dL_dCij_mul) : 0;
	float dL_dC02 = 0;
	float dL_dC11 = normalize_gaussians ? (-C00 * dL_dCij_mul) : 0;
	float dL_dC12 = 0;
	float dL_dC22 = 0;

	float denom2inv = 1.0f / ((det_2d * det_2d) + 0.0000001f);
	
	if (denom2inv != 0)
	{
		// Gradients of loss w.r.t. entries of 2D covariance matrix,
		// given gradients of loss w.r.t. conic matrix (inverse covariance matrix).
		// e.g., dL / dC00 = dL / d_conic00 * d_conic00 / C00
		dL_dC00 += denom2inv * (-C11 * C11 * dL_dconic.x + 2 * C01 * C11 * dL_dconic.y + (det_2d - C00 * C11) * dL_dconic.z);
		dL_dC01 += 2 * denom2inv * (C01 * C11 * dL_dconic.x - (det_2d + 2 * C01 * C01) * dL_dconic.y + C00 * C01 * dL_dconic.z);
		dL_dC11 += denom2inv * (-C00 * C00 * dL_dconic.z + 2 * C00 * C01 * dL_dconic.y + (det_2d - C00 * C11) * dL_dconic.x);
	}


	const glm::mat3 dL_dcov3d_transf(
		dL_dC00, dL_dC01, dL_dC02,
		0, dL_dC11, dL_dC12,
		0, 0, dL_dC22);
	
	// Derivative with respect to pre-transformed 3D covariance
	const glm::mat3 dL_dcov3d = T * dL_dcov3d_transf * glm::transpose(T);


	atomicAdd(&dL_dcov[6 * idx + 0], dL_dcov3d[0][0]);
	atomicAdd(&dL_dcov[6 * idx + 1], dL_dcov3d[0][1] + dL_dcov3d[1][0]);
	atomicAdd(&dL_dcov[6 * idx + 2], dL_dcov3d[0][2] + dL_dcov3d[2][0]);
	atomicAdd(&dL_dcov[6 * idx + 3], dL_dcov3d[1][1]);
	atomicAdd(&dL_dcov[6 * idx + 4], dL_dcov3d[1][2] + dL_dcov3d[2][1]);
	atomicAdd(&dL_dcov[6 * idx + 5], dL_dcov3d[2][2]);

	const glm::mat3 cov3D_halftransf = Vrk * T;
	const glm::mat3 dL_dcov3d_halftransf = T * dL_dcov3d_transf;
	
	const glm::mat3 dL_dT = cov3D_halftransf * glm::transpose(dL_dcov3d_transf) + Vrk * dL_dcov3d_halftransf;

	// if(idx == 354833){
	// 	printf("===cov3D_halftransf===:\n %e, %e, %e\n%e, %e, %e\n%e, %e, %e\n",
	// 		cov3D_halftransf[0][0], cov3D_halftransf[0][1], cov3D_halftransf[0][2],
	// 		cov3D_halftransf[1][0], cov3D_halftransf[1][1], cov3D_halftransf[1][2],
	// 		cov3D_halftransf[2][0], cov3D_halftransf[2][1], cov3D_halftransf[2][2]);

	// 	printf("===dL_dcov3d_halftransf===:\n %e, %e, %e\n%e, %e, %e\n%e, %e, %e\n",
	// 		dL_dcov3d_halftransf[0][0], dL_dcov3d_halftransf[0][1], dL_dcov3d_halftransf[0][2],
	// 		dL_dcov3d_halftransf[1][0], dL_dcov3d_halftransf[1][1], dL_dcov3d_halftransf[1][2],
	// 		dL_dcov3d_halftransf[2][0], dL_dcov3d_halftransf[2][1], dL_dcov3d_halftransf[2][2]);

	// 	printf("===dL_dT===:\n %e, %e, %e\n%e, %e, %e\n%e, %e, %e\n",
	// 		dL_dT[0][0], dL_dT[0][1], dL_dT[0][2],
	// 		dL_dT[1][0], dL_dT[1][1], dL_dT[1][2],
	// 		dL_dT[2][0], dL_dT[2][1], dL_dT[2][2]);

	// }

	// Gradients of loss w.r.t. entries of Jacobian matrix, coming from T = W * J, then cov = TJT'
	// T = W * J
	glm::mat3 dL_dJ = glm::transpose(W) * dL_dT;
	float dL_dJ00 = dL_dJ[0][0]; 
	float dL_dJ01 = dL_dJ[0][1]; 
	float dL_dJ02 = dL_dJ[0][2];
	float dL_dJ10 = dL_dJ[1][0]; 
	float dL_dJ11 = dL_dJ[1][1];
	float dL_dJ12 = dL_dJ[1][2];
	float dL_dJ20 = dL_dJ[2][0];
	float dL_dJ21 = dL_dJ[2][1];
	float dL_dJ22 = dL_dJ[2][2];

	// Gradients of loss w.r.t. entries of W matrix
	// T = W * J
	glm::mat3 dL_dW = dL_dT * glm::transpose(J);

	dL_dJ00 += dL_ddetJ * (J[1][1]*J[2][2] - J[1][2]*J[2][1]);
	dL_dJ01 += dL_ddetJ * (J[1][2]*J[2][0] - J[1][0]*J[2][2]);
	dL_dJ02 += dL_ddetJ * (J[1][0]*J[2][1] - J[1][1]*J[2][0]);
	dL_dJ10 += dL_ddetJ * (J[0][2]*J[2][1] - J[0][1]*J[2][2]);
	dL_dJ11 += dL_ddetJ * (J[0][0]*J[2][2] - J[0][2]*J[2][0]);
	dL_dJ12 += dL_ddetJ * (J[0][1]*J[2][0] - J[0][0]*J[2][1]);
	dL_dJ20 += dL_ddetJ * (J[0][1]*J[1][2] - J[0][2]*J[1][1]);
	dL_dJ21 += dL_ddetJ * (J[0][2]*J[1][0] - J[0][0]*J[1][2]);
	dL_dJ22 += dL_ddetJ * (J[0][0]*J[1][1] - J[0][1]*J[1][0]);

	// add dL_dviewmat // This part is minor. Can't tell its impact from ablation study in depth loss only localization experiment
	dL_dviewmat[idx] += glm::mat4(dL_dW[0][0], dL_dW[1][0], dL_dW[2][0], 0,
																dL_dW[0][1], dL_dW[1][1], dL_dW[2][1], 0,
																dL_dW[0][2], dL_dW[1][2], dL_dW[2][2], 0,
																0, 0, 0, 0);
	
	/// Original

	float dL_dtx, dL_dty, dL_dtz;

	if(normalize_gaussians) {
		// Third row in jacobian included. generated with gen_derivatives.py
		const float tzcube = t.z * tzsq;
		const float norm4 = (txsq + tysq + tzsq)*(txsq + tysq + tzsq);
		const float norm72 = pow(txsq + tysq + tzsq, 7./2.);
		const float norm32 = pow(txsq + tysq + tzsq, 3./2.);

		// NOTE [seth]: if moving focal back out of J, also delete here
		dL_dtx = (-dL_dJ02*focal_x*norm72 + dL_dJ20*tzsq*(tysq + tzsq)*norm4 - t.x*tzsq*(dL_dJ21*t.y + dL_dJ22*t.z)*norm4)/(tzsq*norm72);
		dL_dty = (-dL_dJ12*focal_y*norm72 + dL_dJ21*tzsq*(txsq + tzsq)*norm4 - t.y*tzsq*(dL_dJ20*t.x + dL_dJ22*t.z)*norm4)/(tzsq*norm72);
		dL_dtz = -dL_dJ00*focal_x/tzsq + 2*dL_dJ02*focal_x*t.x/tzcube - dL_dJ11*focal_y/tzsq + 2*dL_dJ12*focal_y*t.y/tzcube - dL_dJ20*t.x*t.z/norm32 - dL_dJ21*t.y*t.z/norm32 + dL_dJ22*txsq/norm32 + dL_dJ22*tysq/norm32;
	} else {
		float tz = 1.f / t.z;
		float tz2 = tz * tz;
		float tz3 = tz2 * tz;
		dL_dtx = x_grad_mul * -focal_x * tz2 * dL_dJ02;
		dL_dty = y_grad_mul * -focal_y * tz2 * dL_dJ12;
		dL_dtz = -focal_x * tz2 * dL_dJ00 - focal_y * tz2 * dL_dJ11 + (2 * focal_x * t.x) * tz3 * dL_dJ02 + (2 * focal_y * t.y) * tz3 * dL_dJ12;
	}

	if(should_print){
		printf("dL_dt before addition: %e %e %e\n", dL_dtx, dL_dty,dL_dtz);
	}

	// add previous dL_dtz from renderCUDA depth loss
	dL_dtx += dL_dt_prev[idx].x;
	dL_dty += dL_dt_prev[idx].y;
	dL_dtz += dL_dt_prev[idx].z; // from depth loss // This part is important. Key component to make pose estimation converge in depth loss only localization experiment

	// Account for transformation of mean to t
	// t = transformPoint4x3(mean, view_matrix);
	float3 dL_dmean = transformVec4x3Transpose({ dL_dtx, dL_dty, dL_dtz }, view_matrix);

	// add dL_dviewmat
	glm::vec4 dL_dt(dL_dtx, dL_dty, dL_dtz, 0);
	glm::vec4 dt_dv(mean.x, mean.y, mean.z, 1);
	glm::mat4 dL_dv = glm::outerProduct(dL_dt, dt_dv);
	dL_dviewmat[idx] += dL_dv;

	// printf("dL_dT: %e %e %e\n", dL_dt.x, dL_dt.y, dL_dt.z);
	
	// printmat4(dL_dviewmat[idx], "dL_dviewmat");

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the covariance matrix.
	// Additional mean gradient is accumulated in BACKWARD::preprocess.
	dL_dmeans[idx] = dL_dmean;

	if(should_print) {
		printf("\n===================Start================\n");
		printf("# Inputs\n");
    printf("mean3d = torch.tensor([%.10e, %.10e, %.10e]).requires_grad_(True)\n", mean.x, mean.y, mean.z);
    printf("cov3d = torch.tensor([[%.10e, %.10e, %.10e],[%.10e, %.10e, %.10e], [%.10e, %.10e, %.10e]]).requires_grad_(True)\n",
           cov3D[0], cov3D[1], cov3D[2], cov3D[1], cov3D[3], cov3D[4], cov3D[2], cov3D[4], cov3D[5]);
    printf("viewmatrix = torch.tensor([%.10e, %.10e, %.10e, %.10e, %.10e, %.10e, %.10e, %.10e, %.10e, %.10e, %.10e, %.10e, %.10e, %.10e, %.10e, %.10e]).view(4, 4).requires_grad_(True)\n",
           view_matrix[0], view_matrix[1], view_matrix[2], view_matrix[3],
           view_matrix[4], view_matrix[5], view_matrix[6], view_matrix[7],
           view_matrix[8], view_matrix[9], view_matrix[10], view_matrix[11],
           view_matrix[12], view_matrix[13], view_matrix[14], view_matrix[15]);
    printf("focal_x, focal_y = %.10e, %.10e\n", focal_x, focal_y);
    printf("tan_fovx, tan_fovy = %.10e, %.10e\n", tan_fovx, tan_fovy);
    printf("dL_dconic = torch.tensor([%.10e, %.10e, %.10e])\n", dL_dconics[4 * idx], dL_dconics[4 * idx + 1], dL_dconics[4 * idx + 3]);
    printf("dL_dt_prev = torch.tensor([%.10e, %.10e, %.10e])\n", dL_dt_prev[idx].x, dL_dt_prev[idx].y, dL_dt_prev[idx].z);
    printf("dL_dnorm = torch.tensor([%.10e])\n", dL_dnorm[idx]);

		printf("\n#Intermediate Vals\n");
    // After each intermediate calculation:
    printf("cuda_t = torch.tensor([%.10e, %.10e, %.10e])\n", t.x, t.y, t.z);
    printf("cuda_J = torch.tensor([[%.10e, %.10e, %.10e], [%.10e, %.10e, %.10e], [%.10e, %.10e, %.10e]])\n",
           J[0][0], J[0][1], J[0][2], J[1][0], J[1][1], J[1][2], J[2][0], J[2][1], J[2][2]);
    printf("cuda_W = torch.tensor([[%.10e, %.10e, %.10e], [%.10e, %.10e, %.10e], [%.10e, %.10e, %.10e]])\n",
           W[0][0], W[0][1], W[0][2], W[1][0], W[1][1], W[1][2], W[2][0], W[2][1], W[2][2]);
    printf("cuda_cov3D_transf = torch.tensor([[%.10e, %.10e, %.10e], [%.10e, %.10e, %.10e], [%.10e, %.10e, %.10e]])\n",
           cov3D_transf[0][0], cov3D_transf[0][1], cov3D_transf[0][2],
           cov3D_transf[1][0], cov3D_transf[1][1], cov3D_transf[1][2],
           cov3D_transf[2][0], cov3D_transf[2][1], cov3D_transf[2][2]);
    printf("cuda_det2d = torch.tensor(%.10e)\n", det_2d);
    printf("cuda_dL_dcov3d_transf = torch.tensor([[%.10e, %.10e, %.10e], [%.10e, %.10e, %.10e], [%.10e, %.10e, %.10e]])\n",
           dL_dcov3d_transf[0][0], dL_dcov3d_transf[0][1], dL_dcov3d_transf[0][2],
           dL_dcov3d_transf[1][0], dL_dcov3d_transf[1][1], dL_dcov3d_transf[1][2],
           dL_dcov3d_transf[2][0], dL_dcov3d_transf[2][1], dL_dcov3d_transf[2][2]);

    printf("cuda_dL_dT = torch.tensor([[%.10e, %.10e, %.10e], [%.10e, %.10e, %.10e], [%.10e, %.10e, %.10e]])\n",
           dL_dT[0][0], dL_dT[0][1], dL_dT[0][2], dL_dT[1][0], dL_dT[1][1], dL_dT[1][2], dL_dT[2][0], dL_dT[2][1], dL_dT[2][2]);
    printf("cuda_dL_dJ = torch.tensor([[%.10e, %.10e, %.10e], [%.10e, %.10e, %.10e], [%.10e, %.10e, %.10e]])\n",
           dL_dJ00, dL_dJ01, dL_dJ02, dL_dJ10, dL_dJ11, dL_dJ12, dL_dJ20, dL_dJ21, dL_dJ22);
    printf("cuda_dL_dW = torch.tensor([[%.10e, %.10e, %.10e], [%.10e, %.10e, %.10e], [%.10e, %.10e, %.10e]])\n",
           dL_dW[0][0], dL_dW[0][1], dL_dW[0][2],
           dL_dW[1][0], dL_dW[1][1], dL_dW[1][2],
           dL_dW[2][0], dL_dW[2][1], dL_dW[2][2]);

    printf("cuda_dL_dtx = torch.tensor(%.10e)\n", dL_dtx);
    printf("cuda_dL_dty = torch.tensor(%.10e)\n", dL_dty);
    printf("cuda_dL_dtz = torch.tensor(%.10e)\n", dL_dtz);
    printf("cuda_T = torch.tensor([[%.10e, %.10e, %.10e], [%.10e, %.10e, %.10e], [%.10e, %.10e, %.10e]])\n",
           T[0][0], T[0][1], T[0][2],
           T[1][0], T[1][1], T[1][2],
           T[2][0], T[2][1], T[2][2]);

		printf("det_J =  %e\n", glm::determinant(J));
		printf("dL_ddetJ =  %e\n", dL_ddetJ);

		printf("#==========Outputs========\n");
		printf("cuda_dL_dmean = torch.tensor([%.10e, %.10e, %.10e])\n", dL_dmean.x, dL_dmean.y, dL_dmean.z);
    
		printf("cuda_dL_dcov3d = torch.tensor([[%.10e, %.10e, %.10e], [%.10e, %.10e, %.10e], [%.10e, %.10e, %.10e]])\n",
           dL_dcov3d[0][0], dL_dcov3d[0][1], dL_dcov3d[0][2],
           dL_dcov3d[1][0], dL_dcov3d[1][1], dL_dcov3d[1][2],
           dL_dcov3d[2][0], dL_dcov3d[2][1], dL_dcov3d[2][2]);
		printf("cuda_dL_dviewmat = torch.tensor([[%.10e, %.10e, %.10e, %.10e], [%.10e, %.10e, %.10e, %.10e], [%.10e, %.10e, %.10e, %.10e], [%.10e, %.10e, %.10e, %.10e]])\n",
           dL_dviewmat[idx][0][0], dL_dviewmat[idx][1][0], dL_dviewmat[idx][2][0], dL_dviewmat[idx][3][0],
           dL_dviewmat[idx][0][1], dL_dviewmat[idx][1][1], dL_dviewmat[idx][2][1], dL_dviewmat[idx][3][1],
           dL_dviewmat[idx][0][2], dL_dviewmat[idx][1][2], dL_dviewmat[idx][2][2], dL_dviewmat[idx][3][2],
           dL_dviewmat[idx][0][3], dL_dviewmat[idx][1][3], dL_dviewmat[idx][2][3], dL_dviewmat[idx][3][3]);
	}	
}

// Backward pass for the conversion of scale and rotation to a 
// 3D covariance matrix for each Gaussian. 
__device__ void computeCov3D(int idx, const glm::vec3 scale, 
														 float mod, const glm::vec4 rot,
														 const float* dL_dcov3Ds,
														 glm::vec3* dL_dscales,
														 glm::vec4* dL_drots)
{
	// Recompute (intermediate) results for the 3D covariance computation.
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 S = glm::mat3(1.0f);

	glm::vec3 s = mod * scale;
	S[0][0] = s.x;
	S[1][1] = s.y;
	S[2][2] = s.z;

	glm::mat3 M = S * R;

	const float* dL_dcov3D = dL_dcov3Ds + 6 * idx;

	glm::vec3 dunc(dL_dcov3D[0], dL_dcov3D[3], dL_dcov3D[5]);
	glm::vec3 ounc = 0.5f * glm::vec3(dL_dcov3D[1], dL_dcov3D[2], dL_dcov3D[4]);

	// Convert per-element covariance loss gradients to matrix form
	glm::mat3 dL_dSigma = glm::mat3(
		dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
		0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
		0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]
	);

	// Compute loss gradient w.r.t. matrix M
	// dSigma_dM = 2 * M
	glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

	glm::mat3 Rt = glm::transpose(R);
	glm::mat3 dL_dMt = glm::transpose(dL_dM);

	// Gradients of loss w.r.t. scale
	glm::vec3* dL_dscale = dL_dscales + idx;
	dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]);
	dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);
	dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]);

	dL_dMt[0] *= s.x;
	dL_dMt[1] *= s.y;
	dL_dMt[2] *= s.z;

	// Gradients of loss w.r.t. normalized quaternion
	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

	// Gradients of loss w.r.t. unnormalized quaternion
	float4* dL_drot = (float4*)(dL_drots + idx);
	*dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };//dnormvdv(float4{ rot.x, rot.y, rot.z, rot.w }, float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w });
}

// Backward pass of the preprocessing steps, except
// for the covariance computation and inversion
// (those are handled by a previous kernel call)
template<int C>
__global__ void preprocessCUDA(
	int P, int D, int M,
	const float3* means,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* proj,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	glm::vec3* dL_dmeans,
	float* dL_dcolor,
	float* dL_dcov3D,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot,
	glm::mat4* dL_dprojmat)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	float3 m = means[idx];

	// Taking care of gradients from the screenspace points
	float4 m_hom = transformPoint4x4(m, proj);
	float m_w = 1.0f / (m_hom.w + 0.0000001f);

	// Compute loss gradient w.r.t. 3D means due to gradients of 2D means
	// from rendering procedure
	glm::vec3 dL_dmean;
	float mul1 = (proj[0] * m.x + proj[4] * m.y + proj[8] * m.z + proj[12]) * m_w * m_w;
	float mul2 = (proj[1] * m.x + proj[5] * m.y + proj[9] * m.z + proj[13]) * m_w * m_w;
	dL_dmean.x = (proj[0] * m_w - proj[3] * mul1) * dL_dmean2D[idx].x + (proj[1] * m_w - proj[3] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.y = (proj[4] * m_w - proj[7] * mul1) * dL_dmean2D[idx].x + (proj[5] * m_w - proj[7] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.z = (proj[8] * m_w - proj[11] * mul1) * dL_dmean2D[idx].x + (proj[9] * m_w - proj[11] * mul2) * dL_dmean2D[idx].y;

	// That's the second part of the mean gradient. Previous computation
	// of cov2D and following SH conversion also affects it.
	dL_dmeans[idx] += dL_dmean;

	float dL_dM0 = m.x * m_w * dL_dmean2D[idx].x;
	float dL_dM4 = m.y * m_w * dL_dmean2D[idx].x;
	float dL_dM8 = m.z * m_w * dL_dmean2D[idx].x;
	float dL_dM12 = m_w * dL_dmean2D[idx].x;

	float dL_dM1 = m.x * m_w * dL_dmean2D[idx].y;
	float dL_dM5 = m.y * m_w * dL_dmean2D[idx].y;
	float dL_dM9 = m.z * m_w * dL_dmean2D[idx].y;
	float dL_dM13 = m_w * dL_dmean2D[idx].y;
 
	float dL_dM3 = - m_hom.x * m.x * m_w * m_w * dL_dmean2D[idx].x - m_hom.y * m.x * m_w * m_w * dL_dmean2D[idx].y;
	float dL_dM7 = - m_hom.x * m.y * m_w * m_w * dL_dmean2D[idx].x - m_hom.y * m.y * m_w * m_w * dL_dmean2D[idx].y;
	float dL_dM11 = - m_hom.x * m.z * m_w * m_w * dL_dmean2D[idx].x - m_hom.y * m.z * m_w * m_w * dL_dmean2D[idx].y;
	float dL_dM15 = - m_hom.x * 1 * m_w * m_w * dL_dmean2D[idx].x - m_hom.y * 1 * m_w * m_w * dL_dmean2D[idx].y;
 
	dL_dprojmat[idx] += glm::mat4(dL_dM0, dL_dM1, 0, dL_dM3,
							 	  dL_dM4, dL_dM5, 0, dL_dM7,
							 	  dL_dM8, dL_dM9, 0, dL_dM11,
							 	  dL_dM12, dL_dM13, 0, dL_dM15);

	// Compute gradient updates due to computing colors from SHs
	if (shs)
		computeColorFromSH(idx, D, M, (glm::vec3*)means, *campos, shs, clamped, (glm::vec3*)dL_dcolor, (glm::vec3*)dL_dmeans, (glm::vec3*)dL_dsh);

	// Compute gradient updates due to computing covariance from scale/rotation
	if (scales)
		computeCov3D(idx, scales[idx], scale_modifier, rotations[idx], dL_dcov3D, dL_dscale, dL_drot);

}

// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ normalizing_consts,
	const float* __restrict__ cov3Ds,
	const float* __restrict__ colors,
	const float* __restrict__ depths,
	const float* __restrict__ alphas,
	const float* __restrict__ f_contrastive,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ viewmatrix,
	const float* __restrict__ dL_dpixels,
	const float* __restrict__ dL_ddepths,
	const float* __restrict__ dL_dalphas,
	const float* __restrict__ dL_dcontrastive_img,
	const float focal_x, const float focal_y,
	const bool normalize_gaussians,
	const float3* __restrict__ means3D,
	float3* __restrict__ dL_dmean2D, // looks wrong
	float4* __restrict__ dL_dconic2D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors,
	float3* __restrict__ dL_dt, // add new
	float3* __restrict__ dL_dmean3D,
	float* __restrict__ dL_dnorm,
	float* __restrict__ dL_dcontrastive_features)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };

	const bool inside = pix.x < W&& pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_colors[C * BLOCK_SIZE];
	__shared__ float collected_depths[BLOCK_SIZE];
	__shared__ float collected_normalizing_consts[BLOCK_SIZE];
	__shared__ float collected_cov3Ds[6*BLOCK_SIZE];
	// __shared__ float collected_contrastive_f[NUM_CONTRASTIVE_CHANNELS*BLOCK_SIZE];

	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors. 
	const float T_final = inside ? (1 - alphas[pix_id]) : 0;
	float T = T_final;

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float accum_rec[C] = { 0 };
	float dL_dpixel[C];
	float accum_depth_rec = 0;
	float dL_ddepth;
	float accum_alpha_rec = 0;
	float dL_dalpha;
	if (inside) {
		for (int i = 0; i < C; i++)
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];
		dL_ddepth = dL_ddepths[pix_id];
		dL_dalpha = dL_dalphas[pix_id];
	}

	float last_alpha = 0;
	float last_color[C] = { 0 };
	float last_depth = 0;
	float accum_contrastive_rec[NUM_CONTRASTIVE_CHANNELS] = {0};
	float last_contrastive[NUM_CONTRASTIVE_CHANNELS] = {0};
	
	// Gradient of pixel coordinate w.r.t. normalized 
	// screen-space viewport corrdinates (-1 to 1)
	const float ddelx_dx = 0.5 * W;
	const float ddely_dy = 0.5 * H;
  
	// (197, 154): idx=90
	const bool pp = pix.x == 599 && pix.y == 432;

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			for (int i = 0; i < C; i++)
				collected_colors[i * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + i];

			// if(f_contrastive) {
			// 	for(int i = 0; i < NUM_CONTRASTIVE_CHANNELS; ++i) {
			// 		collected_contrastive_f[i*BLOCK_SIZE + block.thread_rank()] =
			// 			 f_contrastive[coll_id*NUM_CONTRASTIVE_CHANNELS + i];
			// 	}
			// }
			collected_depths[block.thread_rank()] = depths[coll_id];
			collected_normalizing_consts[block.thread_rank()] = normalizing_consts[coll_id];
			for(int i = 0; i < 6; ++i) {
				// TODO (seth): I don't understand the above indexing in color sections...
				collected_cov3Ds[block.thread_rank()*6 + i] = cov3Ds[coll_id * 6 + i];
			}
		}
		block.sync();

		const bool should_print = false && pp;

		// Iterate over Gaussians
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			contributor--;
			if (contributor >= last_contributor) {
				if(should_print) {
					printf("== [BACKWARD] contributor: %d SKIPPED; last contributor is %d\n", contributor, last_contributor);
				}
				continue;
			}
				

			// Compute blending values, as before.
			const float2 xy = collected_xy[j];
			const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			const float4 con_o = collected_conic_opacity[j];
			const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

		
			///// Compute Normalization /////

			/// Numerically Sensible Normalization ///
			const float normalization = collected_normalizing_consts[j];
		
			const float G = exp(power);

			const float alpha = min(0.99f, con_o.w * G * normalization);
			const bool alpha_saturated = alpha == 0.99f;
			const float alpha_unsaturated_f = 1 - (float)alpha_saturated;

			if (alpha < 1.0f / 255.0f)
				continue;

			T = T / (1.f - alpha);

			if(should_print) {
				printf("===== [BACKWARD] contributor: %d\ncoll_id:%d\nxy: [%e, %e]\npix:[%u,%u]\nd:[%e,%e]\ncon_o:[%e,%e,%e,%e]\npower: %e\n"
							 "n:%e\na:%e\nT:%e\n",
							 contributor, collected_id[j],xy.x,xy.y,pix.x,pix.y,d.x,d.y,con_o.x,con_o.y,con_o.z,con_o.w,power,
							 normalization,alpha,T);
			}


			const float dchannel_dcolor = alpha * T;
			const float ddepth_dlocaldepth = alpha * T;

			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			float dL_dopa = 0.0f;
			const int global_id = collected_id[j];

			// Compute dL_dopa. If alpha was saturated, then this is always 0.
			for (int ch = 0; ch < C; ch++)
			{
				const float c = collected_colors[ch * BLOCK_SIZE + j];
				// Update last color (to be used in the next iteration)
				accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				last_color[ch] = c;

				const float dL_dchannel = dL_dpixel[ch];
				dL_dopa += alpha_unsaturated_f * (c - accum_rec[ch]) * dL_dchannel;

				// Update the gradients w.r.t. color of the Gaussian. 
				// Atomic, since this pixel is just one of potentially
				// many that were affected by this Gaussian.
				atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
			}

			if (dL_dcontrastive_img && f_contrastive) {
				for (int ch = 0; ch < NUM_CONTRASTIVE_CHANNELS; ch++) {
					const float f_c = f_contrastive[global_id*NUM_CONTRASTIVE_CHANNELS + i];
					accum_contrastive_rec[ch] = last_alpha * last_contrastive[ch] + (1.f - last_alpha) * accum_contrastive_rec[ch];
					last_contrastive[ch] = f_c;

					const float dL_dcontrastive = dL_dcontrastive_img[ch * H * W + pix_id];
					dL_dopa += alpha_unsaturated_f * (f_c - accum_contrastive_rec[ch]) * dL_dcontrastive;

					atomicAdd(&(dL_dcontrastive_features[global_id * NUM_CONTRASTIVE_CHANNELS + ch]), dchannel_dcolor * dL_dcontrastive);
				}
			}

			// Propagate gradients from pixel depth to opacity
			const float c_d = collected_depths[j];
			accum_depth_rec = last_alpha * last_depth + (1.f - last_alpha) * accum_depth_rec;
			last_depth = c_d;
			dL_dopa += alpha_unsaturated_f * (c_d - accum_depth_rec) * dL_ddepth;

			// Propagate gradients from pixel alpha (weights_sum) to opacity
			accum_alpha_rec = last_alpha + (1.f - last_alpha) * accum_alpha_rec;
			dL_dopa += - alpha_unsaturated_f * (alpha - accum_alpha_rec) * dL_dalpha;
			
			if(should_print) {
				printf("dL_dopa contrib from pixel alpha - alpha %e, alpha_accum_rec %e, dL_dalpha %e, contrib %e, result %e\n",
				alpha, accum_alpha_rec, dL_dalpha, - (alpha - accum_alpha_rec) * dL_dalpha, dL_dopa);
			}

			dL_dopa *= T;
			if(should_print)
				printf("Mult dL_dopa by T: %e, result: %e\n", T, dL_dopa);
			// Update last alpha (to be used in the next iteration)
			last_alpha = alpha;

			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			float bg_dot_dpixel = 0;
			for (int i = 0; i < C; i++)
				bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
			dL_dopa += alpha_unsaturated_f * (-T_final / (1.f - alpha)) * bg_dot_dpixel;

			// Propagate gradients from pixel depth to gaussian mean along z-axis: tz
			const float dL_dlocaldepth = ddepth_dlocaldepth * dL_ddepth;
			const float dL_dtz_ = dL_dlocaldepth;


			// Helpful reusable temporary variables
			const float dL_dG = con_o.w * dL_dopa * normalization;

			const float gdx = G * d.x;
			const float gdy = G * d.y;

			///////////// dG_ddelx /////////////

			//// Original /////
			const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
			const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

			atomicAdd(&(dL_dnorm[global_id]), normalize_gaussians ? (dL_dopa * con_o.w * G) : 0);


			// Update gradients w.r.t. 2D mean position of the Gaussian
			atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx * ddelx_dx);
			atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely * ddely_dy);

			if(should_print) {
				printf("G: %e\ncon_o.w: %e\nnormalization: %e\ndL_dopa: %e\n dL_dG: %e\ndG_ddelx: %e\ndG_ddely: %e\ndL_dmean2D.x_contrib: %e\ndL_dmean2D.y_contrib: %e\n"
							 "dL_dmean2D.x: %e\ndL_dmean2D.y: %e\n", 
				G, con_o.w, normalization, dL_dopa, dL_dG, dG_ddelx, dG_ddely, dL_dG * dG_ddelx * ddelx_dx, dL_dG * dG_ddely * ddely_dy,
				dL_dmean2D[global_id].x, dL_dmean2D[global_id].y);
			}
			////////////// dL/dConic ////////////////////

			// Update gradients w.r.t. 2D covariance (2x2 matrix, symmetric)
			atomicAdd(&dL_dconic2D[global_id].x, -0.5f * gdx * d.x * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].y, -0.5f * gdx * d.y * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].w, -0.5f * gdy * d.y * dL_dG);

			// Update gradients w.r.t. opacity of the Gaussian
			atomicAdd(&(dL_dopacity[global_id]), G * normalization * dL_dopa);

			// Update gradients w.r.t. t.z of the Gaussian
			atomicAdd(&(dL_dt[global_id].z), dL_dtz_);

		}
	}
}

void BACKWARD::preprocess(
	int P, int D, int M,
	const float3* means3D,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* cov3Ds,
	const float* viewmatrix,
	const float* projmatrix,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	const float filter_radius,
	const bool normalize_gaussians,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	const float* dL_dconic,
	const float3* dL_dt,
	const float* dL_dnorm,
	glm::vec3* dL_dmean3D,
	float* dL_dcolor,
	float* dL_dcov3D,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot,
	glm::mat4* dL_dviewmat,
	glm::mat4* dL_dprojmat)
{
	
	// Propagate gradients for the path of 2D conic matrix computation. 
	// Somewhat long, thus it is its own kernel rather than being part of 
	// "preprocess". When done, loss gradient w.r.t. 3D means has been
	// modified and gradient w.r.t. 3D covariance matrix has been computed.	
	computeCov2DCUDA << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		radii,
		cov3Ds,
		focal_x,
		focal_y,
		tan_fovx,
		tan_fovy,
		filter_radius,
		viewmatrix,
		dL_dconic,
		dL_dt,
		dL_dnorm,
		normalize_gaussians,
		(float3*)dL_dmean3D,
		dL_dcov3D,
		dL_dviewmat);

	// Propagate gradients for remaining steps: finish 3D mean gradients,
	// propagate color gradients to SH (if desireD), propagate 3D covariance
	// matrix gradients to scale and rotation.
	preprocessCUDA<NUM_CHANNELS> << < (P + 255) / 256, 256 >> > (
		P, D, M,
		(float3*)means3D,
		radii,
		shs,
		clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		projmatrix,
		campos,
		(float3*)dL_dmean2D,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_dsh,
		dL_dscale,
		dL_drot,
		(glm::mat4*)dL_dprojmat);
}

void BACKWARD::render(
	const dim3 grid, const dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float* bg_color,
	const float2* means2D,
	const float4* conic_opacity,
	const float* normalizing_consts,
	const float* covs3D,
	const float* colors,
	const float* depths,
	const float* alphas,
	const float* f_contrastive,
	const uint32_t* n_contrib,
	const float* viewmatrix,
	const float* dL_dpixels,
	const float* dL_ddepths,
	const float* dL_dalphas,
	const float* dL_dcontrastive_img,
	const float focal_x, const float focal_y,
	const bool normalize_gaussians,
	const float* means3D,
	float3* dL_dmean2D,
	float4* dL_dconic2D,
	float* dL_dopacity,
	float* dL_dcolors,
	float3* dL_dt, // add new
	float3* dL_dmean3D,
	float* dL_dnorm,
	float* dL_dcontrastive_features)
{
	// printf("Start: %e %e", dL_dcov3D[0], normalizing_consts[0]);
	renderCUDA<NUM_CHANNELS> << <grid, block >> >(
		ranges,
		point_list,
		W, H,
		bg_color,
		means2D,
		conic_opacity,
		normalizing_consts,
		covs3D,
		colors,
		depths,
		alphas,
		f_contrastive,
		n_contrib,
		viewmatrix,
		dL_dpixels,
		dL_ddepths,
		dL_dalphas,
		dL_dcontrastive_img,
		focal_x, focal_y,
		normalize_gaussians,
		(float3*)means3D,
		dL_dmean2D,
		dL_dconic2D,
		dL_dopacity,
		dL_dcolors,
		dL_dt, // add new
		dL_dmean3D,
		dL_dnorm,
		dL_dcontrastive_features);
}