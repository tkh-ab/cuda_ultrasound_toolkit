#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include "block_match_kernels.cuh"

static __host__ void
print_peak_positions(const int2* peak_positions, const float* peaks, int total_patches)
{
	std::cout << std::endl << "Peak Positions and Values:" << std::endl;
	for(int i = 0; i < total_patches; i++)
	{
		int2 pos = peak_positions[i];
		float val = peaks[i];
		printf("Peak %d: Position (%d, %d), Value: %f\n", i, pos.x, pos.y, val);
	}
	std::cout << std::endl;
}

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

	// status = nppiMinIndx_32f_C1R_Ctx(d_corr_map, line_step, dims, scratch_stack, &d_stats->min_peak_value, &d_stats->min_peak_pos.x, &d_stats->min_peak_pos.y, stream_context);

	// if (status != NPP_SUCCESS)
	// {
	// 	std::cerr << "NPP error '"<< status <<"' during peak detection." << std::endl;
	// 	return { INT_MIN, INT_MIN };
	// }

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

	// if(abs(c_stats.min_peak_value) > abs(c_stats.peak_value))
	// {
	// 	true_peak_value = abs(c_stats.min_peak_value);
	// 	true_peak_pos = c_stats.min_peak_pos;
	// }

	double corr_variance = c_stats.corr_std * c_stats.corr_std;

	// If the peak isn't much higher than the no-shift value, we reject motion
	if (true_peak_value < (no_shift_value * params.correlation_threshold))
	{
		true_peak_pos = { 0, 0 };
		return true_peak_pos;
	}

	// If the correlation map is too uniform we can't trust the peak
	// if( corr_variance < params.min_patch_variance )
	// {
	// 	return { 0, 0 };
	// }
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

	// Print all parameter of the call
	//std::cout << "Grid Dims: " << grid_dims.x << " x " << grid_dims.y << ", Block Dims: " << block_dims.x << " x " << block_dims.y << std::endl;
	//std::cout << "Total Patches: " << total_patches << std::endl;
	//std::cout << "Threshold: " << threshold << ", No Shift Value: " << no_shift_value << std::endl;
	//std::cout << "Row Pitch: " << row_pitch << ", Line Step: " << line_step << std::endl;

	//// And pointer address sanity check
	//std::cout << "d_corr_map: " << d_corr_map << ", d_peak_values: " << d_peak_values << ", d_peak_positions: " << d_peak_positions << std::endl;

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

	block_dims = { 1, 1, 1 };
	grid_dims = {total_patches, 1, 1};

	float min_prominence = 0.05f;
	float min_sharpness = params.min_patch_variance;

	thrust::device_ptr<float> d_peaks_ptr(d_peak_values);
	thrust::device_ptr<int2> d_positions_ptr(d_peak_positions);
	thrust::sort_by_key(d_peaks_ptr, d_peaks_ptr + total_patches, d_positions_ptr, thrust::greater<float>());

	kernels::test_peaks<<<grid_dims, block_dims>>>(d_corr_map, dims, row_pitch, d_peak_positions, d_peak_values, min_prominence, min_sharpness);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaSuccess)
	{
		std::cerr << "CUDA error during peak testing: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
		return motion_vector;
	}

	
	thrust::sort_by_key(d_peaks_ptr, d_peaks_ptr + total_patches, d_positions_ptr, thrust::greater<float>());
	 //float* cpu_peak_values = new float[total_patches];
	 //cudaMemcpy(cpu_peak_values, d_peak_values, total_patches * sizeof(float), cudaMemcpyDeviceToHost);
	 //int2* cpu_peak_positions = new int2[total_patches];
	 //cudaMemcpy(cpu_peak_positions, d_peak_positions, total_patches * sizeof(int2), cudaMemcpyDeviceToHost);

	 ////print_peak_positions(cpu_peak_positions, cpu_peak_values, total_patches);
	 //delete[] cpu_peak_values;
	 //delete[] cpu_peak_positions;

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
	int2 pixel_pos = { threadIdx.x + blockIdx.x * Peak_Detect_Block_Dims.x, threadIdx.y + blockIdx.y * Peak_Detect_Block_Dims.y };

	if( pixel_pos.x >= dims.width || pixel_pos.y >= dims.height )
	{
		return; // Out of bounds
	}

	float value = d_corr_map[pixel_pos.y * line_step + pixel_pos.x];

	// Thread 0 in the warp will have the maximum value
	warp_reduce_max(&value, reinterpret_cast<i64*>(&pixel_pos));

	i64* pos = reinterpret_cast<i64*>(&pixel_pos);

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
block_match::kernels::test_peaks(const float* d_corr_map, NppiSize dims, int line_step, int2* peak_positions, float* peak_values, float min_prominence, float min_sharpness)
{
	constexpr float P[150] = {
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
	int peak_id = blockIdx.x;

	int2 peak_pos = peak_positions[peak_id];
	int peak_offset = peak_pos.y * line_step + peak_pos.x;

	float peak_value = peak_values[peak_id];
	if( peak_value <= 0.0f )
	{
		return;
	}

	float values[25] = { 0.0f };
	float border_rms = 0.0f;

	// Load the 5x5 patch around the peak position
	int i = 0;
	#pragma unroll
	for(int y = -patch_margins.y; y <= patch_margins.y; y++)
	{
		#pragma unroll
		for(int x = -patch_margins.x; x <= patch_margins.x; x++)
		{
			values[i] = d_corr_map[peak_offset + y * line_step + x];

			if( abs(x) == patch_margins.x || abs(y) == patch_margins.y)
			{
				border_rms += values[i] * values[i]; // Accumulate the border values for RMS calculation
			}
			i++;
		}
	}

	// Average value at the edge of this patch
	// Using for a quick and dirty prominance metric
	border_rms = sqrtf(border_rms/16);
	float prominance = (peak_value - border_rms) / peak_value;

	

	// Generate the polynomial fit (f is unused)
	float coeff[5] = { 0.0f };
	#pragma unroll
	for(int i = 0; i < 5; i++)
	{
		float sum = 0.0f;
		#pragma unroll
		for(int j = 0; j < 25; j++)
		{
			sum += P[i * 25 + j] * values[j];
		}
		coeff[i] = sum;
	}

	float sharpness[2] = { 0.0f, 0.0f };

	// v/(a^2 + b^2 + c^2 - 2ab)
	float root = sqrtf( coeff[0] * coeff[0] + coeff[1] * coeff[1] + coeff[2] * coeff[2] - 2 * coeff[0] * coeff[1] );

	sharpness[0] = coeff[0] + coeff[1] + root;
	sharpness[1] = coeff[0] + coeff[1] - root;

	float max_sharpness = fmaxf(abs(sharpness[0]),abs(sharpness[1]));

	if (prominance < min_prominence)
	{
		peak_values[peak_id] = -1.0f; // Mark as invalid
		return;
	}

	if(max_sharpness < min_sharpness)
	{
		peak_values[peak_id] = -1.0f; // Mark as invalid
		return;
	}

	// if(peak_id == 0)
	// 	printf("Peak ID: %d, Position: (%d, %d), Value: %f, Sharpness: %f, Prominence: %f\n", peak_id, peak_pos.x, peak_pos.y, peak_value, max_sharpness, prominance);
	

	return;

}
