#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cub/cub.cuh>

#include "block_match.h"


namespace block_match
{
	__host__ void
	print_peak_positions(const int2* peak_positions, const float* peaks, int total_patches)
	{
		std::cout << std::endl << "Peak Positions and Values:" << std::endl;
		for (int i = 0; i < total_patches; i++)
		{
			int2 pos = peak_positions[i];
			float val = peaks[i];
			printf("Peak %d: Position (%d, %d), Value: %f\n", i, pos.x, pos.y, val);
		}
		std::cout << std::endl;
	}

namespace kernels {
	
	constexpr uint PEAK_CANDIDATE_COUNT = 32; // Maximum number of peaks per block

	// Warp reduce to find the maximum value and its position
	// Treating the float2 position as a single 64-bit integer for the intrinsics
	__inline__ __device__ void
	warp_reduce_max(float* val, double * pos)
	{
		static constexpr unsigned mask = 0xffffffffu;
		static constexpr int warp_size = 32;
		#pragma unroll
		for (int offset = warp_size / 2; offset > 0; offset /= 2)
		{
			float v2 = __shfl_down_sync(mask, *val, offset);
			double  p2 = __shfl_down_sync(mask, *pos, offset);
			if (v2 > *val)
			{
				*val = v2;
				*pos = p2;
			}
		}
	}

		// Each warp takes a 8x4 block and returns the peak position and value 
	__global__ void
	find_peaks_kernel(const float* d_corr_map, NppiSize dims, int line_step, float* peak_values, int2* peak_positions, uint peak_count);


	// Test the prominance and sharpness of the peaks, set any that fail to zero.
	__global__ void
	test_peaks(const float* d_corr_map, float2* d_motion_map, NppiSize dims, int corr_line_step, 
			   int2 * peak_positions, float* peak_values, int2 no_shift_pos,
			   float min_sharpness, float rel_threshold, float abs_threshold);



	__device__ constexpr float P5[150] = {
		0.0285714f,  0.0285714f,  0.0285714f,  0.0285714f,  0.0285714f,
		-0.0142857f, -0.0142857f, -0.0142857f, -0.0142857f, -0.0142857f,
		-0.0285714f, -0.0285714f, -0.0285714f, -0.0285714f, -0.0285714f,
		-0.0142857f, -0.0142857f, -0.0142857f, -0.0142857f, -0.0142857f,
		0.0285714f,  0.0285714f,  0.0285714f,  0.0285714f,  0.0285714f,

		0.0285714f, -0.0142857f, -0.0285714f, -0.0142857f,  0.0285714f,
		0.0285714f, -0.0142857f, -0.0285714f, -0.0142857f,  0.0285714f,
		0.0285714f, -0.0142857f, -0.0285714f, -0.0142857f,  0.0285714f,
		0.0285714f, -0.0142857f, -0.0285714f, -0.0142857f,  0.0285714f,
		0.0285714f, -0.0142857f, -0.0285714f, -0.0142857f,  0.0285714f,

		0.04f,       0.02f,       0.0f,      -0.02f,     -0.04f,
		0.02f,       0.01f,       0.0f,      -0.01f,     -0.02f,
		0.0f,        0.0f,        0.0f,       0.0f,       0.0f,
		-0.02f,      -0.01f,       0.0f,       0.01f,      0.02f,
		-0.04f,      -0.02f,       0.0f,       0.02f,      0.04f,

		-0.04f, -0.04f, -0.04f, -0.04f, -0.04f,
		-0.02f, -0.02f, -0.02f, -0.02f, -0.02f,
		0.0f,   0.0f,   0.0f,   0.0f,   0.0f,
		0.02f,  0.02f,  0.02f,  0.02f,  0.02f,
		0.04f,  0.04f,  0.04f,  0.04f,  0.04f,

		-0.04f, -0.02f,  0.0f,   0.02f,  0.04f,
		-0.04f, -0.02f,  0.0f,   0.02f,  0.04f,
		-0.04f, -0.02f,  0.0f,   0.02f,  0.04f,
		-0.04f, -0.02f,  0.0f,   0.02f,  0.04f,
		-0.04f, -0.02f,  0.0f,   0.02f,  0.04f,

		-0.0742857f,  0.0114286f,  0.04f,      0.0114286f, -0.0742857f,
		0.0114286f,  0.0971429f,  0.125714f,  0.0971429f,  0.0114286f,
		0.04f,       0.125714f,   0.154286f,  0.125714f,   0.04f,
		0.0114286f,  0.0971429f,  0.125714f,  0.0971429f,  0.0114286f,
		-0.0742857f,  0.0114286f,  0.04f,      0.0114286f, -0.0742857f
	};


	__device__ constexpr float P3[54] = {
		1/6.f, -1/3.f,  1/6.f,
		1/6.f, -1/3.f,  1/6.f,
		1/6.f, -1/3.f,  1/6.f,
		
		1/6.f,  1/6.f,  1/6.f,
		-1/3.f, -1/3.f, -1/3.f,
		1/6.f,  1/6.f,  1/6.f,

		1/4.f, 0.f, -1/4.f,
		0.f, 0.f, 0.f,
		-1/4.f, 0.f, 1/4.f,

		-1/6.f, 0.f, 1/6.f,
		-1/6.f, 0.f, 1/6.f,
		-1/6.f, 0.f, 1/6.f,

		-1/6.f, -1/6.f, -1/6.f,
		0.f, 0.f, 0.f,
		1/6.f,  1/6.f,  1/6.f,

		-1/9.f, 2/9.f, -1/9.f,
		2/9.f, 5/9.f, 2/9.f,
		-1/9.f, 2/9.f, -1/9.f
	};


}
}