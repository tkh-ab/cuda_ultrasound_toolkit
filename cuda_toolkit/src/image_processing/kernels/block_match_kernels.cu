#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include "block_match_kernels.cuh"

int2
block_match::select_peak(const float* d_corr_map, NppiSize dims, const NccMotionParameters& params, Npp8u* d_scratch_buffer, NppStreamContext stream_context, int2 no_shift_pos, int line_step)
{

	struct Stats
	{
		int2 peak_pos;
		float peak_value;
		int2 min_peak_pos;
		float min_peak_value;
		double corr_mean;
		double corr_std;
	};

	// The scratch buffer is sized for the NCC so we have extra space.
	u8 *scratch_stack = d_scratch_buffer;
	Stats *d_stats = (Stats *)scratch_stack;
	scratch_stack += sizeof(Stats);

	NppStatus status = nppiMaxIndx_32f_C1R_Ctx(d_corr_map, line_step, dims, scratch_stack, &d_stats->peak_value, &d_stats->peak_pos.x, &d_stats->peak_pos.y, stream_context);

	if (status != NPP_SUCCESS)
	{
		std::cerr << "NPP error '"<< status <<"' during peak detection." << std::endl;
		return { INT_MIN, INT_MIN };
	}

	status = nppiMean_StdDev_32f_C1R_Ctx(d_corr_map, line_step, dims, scratch_stack, &d_stats->corr_mean, &d_stats->corr_std, stream_context);

	if (status != NPP_SUCCESS)
	{
		std::cerr << "NPP error '"<< status <<"' during mean/stddev calculation." << std::endl;
		return { INT_MIN, INT_MIN };
	}

	Stats c_stats = sample_value<Stats>(d_stats);
	float no_shift_value = sample_value<float>(d_corr_map + no_shift_pos.y * dims.width + no_shift_pos.x);

	int2 true_peak_pos = c_stats.peak_pos;
	float true_peak_value = c_stats.peak_value;

	double corr_variance = c_stats.corr_std * c_stats.corr_std;

	// If the peak isn't much higher than the no-shift value, we reject motion
	if (true_peak_value < (no_shift_value * params.correlation_threshold) || corr_variance < params.min_patch_variance)
	{
		true_peak_pos = { 0, 0 };
	}
	return true_peak_pos;
}


__host__ int2
block_match::find_peaks(const float* d_corr_map, NppiSize dims, const NccMotionParameters& params, int line_step, int2 no_shift_pos, u8* d_scratch_buffer)
{
	int2 motion_vector = { 0, 0 };
	int row_pitch = line_step / sizeof(float);
	int no_shift_offset = no_shift_pos.y * row_pitch + no_shift_pos.x;

	float no_shift_value = sample_value<float>(d_corr_map + no_shift_offset);
	float threshold = abs(no_shift_value) * params.correlation_threshold;

	//threshold = threshold > 0.0f ? threshold : 0.0f;

	uint patch_cols = (uint)ceilf((float)dims.width / (float)Peak_Detect_Block_Dims.x);
	uint patch_rows = (uint)ceilf((float)dims.height / (float)Peak_Detect_Block_Dims.y);

	uint total_patches = patch_cols * patch_rows;

	// Based and arena-pilled
	
	int2* d_peak_positions = (int2*)d_scratch_buffer;
	float* d_peak_values = (float*)d_scratch_buffer + total_patches * sizeof(int2);

	dim3 block_dims = { Peak_Detect_Block_Dims.x, Peak_Detect_Block_Dims.y, 1 };
	dim3 grid_dims = { patch_cols, patch_rows, 1 };

	kernels::find_local_peaks_kernel<<<grid_dims, block_dims>>>(d_corr_map, dims, row_pitch, threshold, d_peak_values, d_peak_positions);

	volatile cudaError_t err = cudaDeviceSynchronize();
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA error during peak detection synchronization: " << err << std::endl;
		return motion_vector;
	}

	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA error during peak detection: " << err << std::endl << std::endl;
		return no_shift_pos;
	}

	block_dims = { WARP_SIZE, 1, 1 };
	grid_dims = { (uint)ceilf((float)total_patches / (float)WARP_SIZE), 1, 1 };

	float min_prominence = 0.1f;
	float min_sharpness = params.min_patch_variance;

	thrust::device_ptr<float> d_peaks_ptr(d_peak_values);
	thrust::device_ptr<int2> d_positions_ptr(d_peak_positions);

	kernels::test_peaks<<<grid_dims, block_dims>>>(d_corr_map, dims, row_pitch, d_peak_positions, d_peak_values, total_patches, min_sharpness);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaSuccess)
	{
		std::cerr << "CUDA error during peak testing: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
		return motion_vector;
	}

	thrust::sort_by_key(d_peaks_ptr, d_peaks_ptr + total_patches, d_positions_ptr, thrust::greater<float>());

	if(sample_value<float>(d_peak_values) < threshold)
	{
		return motion_vector; // No valid peaks found
	}

	int2 peak_pos = sample_value<int2>(d_peak_positions);
	return SUB_V2(peak_pos, no_shift_pos); // Return the motion vector relative to the no-shift position

}


__global__ void
block_match::kernels::find_local_peaks_kernel(const float* d_corr_map, NppiSize dims, int line_step, float threshold, float* peak_values, int2* peak_positions)
{
	int2 pixel_pos = { static_cast<int>(threadIdx.x + blockIdx.x * Peak_Detect_Block_Dims.x),
					   static_cast<int>(threadIdx.y + blockIdx.y * Peak_Detect_Block_Dims.y) };

	if( pixel_pos.x >= dims.width || pixel_pos.y >= dims.height )
	{
		return; // Out of bounds
	}

	float value = d_corr_map[pixel_pos.y * line_step + pixel_pos.x];

	// Thread 0 in the warp will have the maximum value
	warp_reduce_max(&value, reinterpret_cast<i64*>(&pixel_pos));

	if( threadIdx.x == 0 && threadIdx.y == 0 )
	{
		if( value < threshold )
		{
			value = -1.0f;
		}
		peak_values[blockIdx.x + blockIdx.y * gridDim.x] = value;
		peak_positions[blockIdx.x + blockIdx.y * gridDim.x] = pixel_pos;
	}

}


__global__ void
block_match::kernels::test_peaks(const float* d_corr_map, NppiSize dims, int line_step, int2* peak_positions, float* peak_values, uint peak_count, float min_sharpness)
{
	constexpr float P5[150] = {
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

	constexpr int2 patch_margins = { 2, 2 };
	constexpr int patch_width = patch_margins.x * 2 + 1;
	constexpr int Total_Samples = 25; // 5x5 polynomial fit

	int peak_id = blockIdx.x * blockDim.x + threadIdx.x;

	if( peak_id >= peak_count )
	{
		return; // Out of bounds
	}

	int2 peak_pos = peak_positions[peak_id];
	int peak_offset = peak_pos.y * line_step + peak_pos.x;

	float peak_value = peak_values[peak_id];
	if( peak_value < 0.0f )
	{
		return; // Invalid peak, skip processing
	}

	float values[Total_Samples] = { 0.0f };

	// Load the 5x5 patch around the peak position
	int i = 0;
	int n = 0;
	#pragma unroll
	for(int y = -patch_margins.y; y <= patch_margins.y; y++)
	{
		if (y + peak_pos.y < 0 || y + peak_pos.y >= dims.height)
		{
			i += patch_width;
			continue; // Skip rows outside the image bounds
		}
		#pragma unroll
		for(int x = -patch_margins.x; x <= patch_margins.x; x++)
		{
			if (x + peak_pos.x < 0 || x + peak_pos.x >= dims.width)
			{
				i++;
				continue; // Skip columns outside the image bounds
			}
			values[i++] = d_corr_map[peak_offset + y * line_step + x];
		}
	}

	// Generate the polynomial fit (f is unused)
	float coeff[6] = { 0.0f };
	#pragma unroll
	for(int i = 0; i < 6; i++)
	{
		float sum = 0.0f;
		#pragma unroll
		for(int j = 0; j < Total_Samples; j++)
		{
			sum += P5[i * Total_Samples + j] * values[j];
		}
		coeff[i] = sum;
	}

	float sharpness[2] = { 0.0f, 0.0f };

	// v/(a^2 + b^2 + c^2 - 2ab)
	float root = sqrtf( coeff[0] * coeff[0] + coeff[1] * coeff[1] + coeff[2] * coeff[2] - 2 * coeff[0] * coeff[1] );

	sharpness[0] = coeff[0] + coeff[1] + root;
	sharpness[1] = coeff[0] + coeff[1] - root;

	float max_sharpness = fmaxf(abs(sharpness[0]),abs(sharpness[1]));

	float width = sqrt( coeff[5] / (max_sharpness * 0.5f) );

	float calculated_peak = coeff[5];

	if(max_sharpness < min_sharpness || sharpness[0] > 0.0f || sharpness[1] > 0.0f)
	{
		peak_values[peak_id] = -1.0f; // Mark as invalid
	}

	// if(peak_id == 0)
	// 	printf("Peak ID: %d, Position: (%d, %d), Value: %f, Width: %f\n", peak_id, peak_pos.x, peak_pos.y, peak_value, width);
	

	return;

}
