#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cub/cub.cuh>

#include "block_match.h"


namespace block_match::kernels
{

	
	// Warp reduce to find the maximum value and its position
	// Treating the int2 position as a single 64-bit integer for the intrinsics
	__inline__ __device__ void
	warp_reduce_max(float* val, i64* pos)
	{
		static constexpr unsigned mask = 0xffffffffu;
		static constexpr int warp_size = 32;
		//#pragma unroll
		for (int offset = warp_size / 2; offset > 0; offset /= 2)
		{
			float v2 = __shfl_down_sync(mask, *val, offset);
			i64  p2 = __shfl_down_sync(mask, *pos, offset);
			if (v2 > *val)
			{
				*val = v2;
				*pos = p2;
			}
		}
	}

	// Each warp takes a 8x4 block and returns the peak position and value 
	__global__ void
	find_local_peaks_kernel(const float* d_corr_map, NppiSize dims, int row_pitch, float threshold, float* peak_values, int2* peak_positions);

	// Test the prominance and sharpness of the peaks, set any that fail to zero.
	__global__ void
	test_peaks(const float* d_corr_map, NppiSize dims, int line_step, int2* peak_positions, float* peak_value, float min_prominence, float max_width);


}