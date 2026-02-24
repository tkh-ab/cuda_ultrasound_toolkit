#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include "block_match_kernels.cuh"

// Single block of 256 threads, each thread processes 8 values
__global__ void
block_match::kernels::find_peaks_kernel(const float* d_corr_map, NppiSize dims, int line_step, float* peak_values, int2* peak_positions, uint peak_count)
{
	static constexpr uint Vals_Per_Thread = 8;
	static constexpr uint Threads_Per_Block = 256;

	using BlockLoad = cub::BlockLoad<float, Threads_Per_Block, Vals_Per_Thread, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
	using BlockRadixSort = cub::BlockRadixSort<float, Threads_Per_Block, Vals_Per_Thread, int>;

    __shared__ typename BlockLoad::TempStorage load_storage;
	__shared__ typename BlockRadixSort::TempStorage sort_storage;

	uint total_values = dims.width * dims.height;

	float values[Vals_Per_Thread];
	int indicies[Vals_Per_Thread];

	for (int i = 0; i < Vals_Per_Thread; i++)
	{
		indicies[i] = i + threadIdx.x * Vals_Per_Thread;
	}

    BlockLoad(load_storage).Load(d_corr_map, values, total_values, -1.0f);

	// for (int i = 0; i < Vals_Per_Thread; i++)
	// {
	// 	if (values[i] > 1.0f)
	// 	{
	// 		values[i] = -1.0f;
	// 	}
	// }


	__syncthreads();

	BlockRadixSort(sort_storage).SortDescendingBlockedToStriped(values, indicies);
	__syncthreads();

	if (threadIdx.x < PEAK_CANDIDATE_COUNT)
	{
		peak_values[threadIdx.x] = values[0];
		int2 peak_pos = { indicies[threadIdx.x] % dims.width, indicies[threadIdx.x] / dims.width };
		peak_positions[threadIdx.x] = peak_pos;
	}


}


__global__ void
block_match::kernels::test_peaks(const float* d_corr_map, float4* d_motion_map, NppiSize dims, int line_step, int2* peak_positions, float* peak_values, int2 no_shift_pos, float min_sharpness,  float rel_threshold, float abs_threshold)
{
	
	// constexpr int2 Patch_Margins = { 2, 2 };
	// constexpr int Patch_Width = Patch_Margins.x * 2 + 1;
	// constexpr int Total_Samples = 25; // 5x5 polynomial fit

	constexpr int2 Patch_Margins = { 1, 1 };
	constexpr int Patch_Width = Patch_Margins.x * 2 + 1;
	constexpr int Total_Samples = 9; // 5x5 polynomial fit


	int peak_id = threadIdx.x;
	

	int2 peak_pos = peak_positions[peak_id];
	int peak_offset = peak_pos.y * line_step + peak_pos.x;

	int no_shift_offset = no_shift_pos.y * line_step + no_shift_pos.x;

	float values[Total_Samples] = { 0.0f };

	// Load the 5x5 patch around the peak position
	int i = 0;
	int n = 0;
	float value = 0.0f;
	#pragma unroll
	for(int y = -Patch_Margins.y; y <= Patch_Margins.y; y++)
	{
		if (y + peak_pos.y < 0 || y + peak_pos.y >= dims.height)
		{
			i += Patch_Width;
			continue; // Skip rows outside the image bounds
		}
		#pragma unroll
		for(int x = -Patch_Margins.x; x <= Patch_Margins.x; x++)
		{
			if (x + peak_pos.x < 0 || x + peak_pos.x >= dims.width)
			{
				i++;
				continue; // Skip columns outside the image bounds
			}
			values[i++] = d_corr_map[peak_offset + y * line_step + x];
		}
	}

	
	// Generate the polynomial fit 
	float coeff[6] = { 0.0f };
	//#pragma unroll
	for(int i = 0; i < 6; i++)
	{
		float sum = 0.0f;
		//#pragma unroll
		for(int j = 0; j < Total_Samples; j++)
		{
			coeff[i] += P3[i * Total_Samples + j] * values[j];
		}
		//coeff[i] = sum;
	}

	float sharpness[2] = { 0.0f, 0.0f };

	// Curvature of the paraboloid 
	// v/(a^2 + b^2 + c^2 - 2ab)
	float root = sqrtf( coeff[0] * coeff[0] + coeff[1] * coeff[1] + coeff[2] * coeff[2] - 2 * coeff[0] * coeff[1] );
	sharpness[0] = coeff[0] + coeff[1] + root;
	sharpness[1] = coeff[0] + coeff[1] - root;

	float max_sharpness = fmaxf(abs(sharpness[0]),abs(sharpness[1]));

	//float width = sqrt( coeff[5] / (max_sharpness * 0.5f) );
	float peak = values[4]; // Center value of the patch
	float calculated_peak = coeff[5];

	float denominator = coeff[2] * coeff[2] - 4 * coeff[1] * coeff[0];
	float2 sub_pixel_offset = { 0.0f, 0.0f };
	if (denominator != 0.0f)
	{
		sub_pixel_offset.x = (2 * coeff[1] * coeff[3] - coeff[2] * coeff[4]) / denominator;
		sub_pixel_offset.y = (2 * coeff[0] * coeff[4] - coeff[2] * coeff[3]) / denominator;
	}

	sub_pixel_offset.x = CLAMP(sub_pixel_offset.x, -1.0f, 1.0f);
	sub_pixel_offset.y = CLAMP(sub_pixel_offset.y, -1.0f, 1.0f);

	float2 total_offset = ADD_V2(sub_pixel_offset, make_float2(peak_pos.x, peak_pos.y));

	//float2 total_offset = make_float2(peak_pos.x, peak_pos.y);

	// if(calculated_peak < peak)
	// {
	// 	printf("Calculated peak %f < peak %f at position (%d, %d)\n", calculated_peak, peak, peak_pos.x, peak_pos.y);
	// }
	//peak = calculated_peak;
	float no_shift_peak = d_corr_map[no_shift_offset];
	float threshold = abs(no_shift_peak) * rel_threshold;

	if(max_sharpness < min_sharpness || sharpness[0] > 0.0f || sharpness[1] > 0.0f || peak < threshold || peak > 1.0f)
	{
		peak = -1.0f;
	}

	warp_reduce_max(&peak, reinterpret_cast<double*>(&total_offset));

	if (threadIdx.x == 0)
	{
		if (peak > abs_threshold)
		{
			total_offset = SUB_V2(total_offset, no_shift_pos);
		}
		else
		{
			total_offset = make_float2(0.0f, 0.0f);
		}
		*d_motion_map = make_float4(total_offset.x, total_offset.y, peak, no_shift_peak);
	}
	return;

}




bool
block_match::block_match_pipeline(const float* d_source, const float* d_template, float4* d_motion_map,
									NppiSize src_roi, NppiSize tpl_roi,
									int src_line_step, int tpl_line_step, PipelineCtx& ctx,
									int2 no_shift_index, const NccMotionParameters& params)
{
	NppiSize valid_corr_dims = { .width = src_roi.width - tpl_roi.width + 1, 
										.height = src_roi.height - tpl_roi.height + 1 };

	int corr_line_step = valid_corr_dims.width * sizeof(float);
	int row_pitch = corr_line_step / sizeof(float);

	uint patch_cols = UINT_DIV_CEIL((uint)valid_corr_dims.width, Peak_Detect_Block_Dims.x);
	uint patch_rows = UINT_DIV_CEIL((uint)valid_corr_dims.height, Peak_Detect_Block_Dims.y);

	// Pad the buffers so a full warp can be used
	uint total_peaks = patch_cols * patch_rows;

	uint warp_count = UINT_DIV_CEIL(total_peaks, WARP_SIZE);
	uint padded_total = warp_count * WARP_SIZE;

	int2* d_peak_positions = (int2*)ctx.d_scratch_buffer;
	float* d_peak_values = (float*)ctx.d_scratch_buffer + padded_total * sizeof(int2);
	
	NppStatus status = nppiCrossCorrValid_NormLevel_32f_C1R_Ctx(d_source, src_line_step, src_roi, 
													d_template, tpl_line_step, tpl_roi, 
													ctx.d_corr_map, corr_line_step, 
													ctx.d_scratch_buffer, ctx.stream_context);

	if (status != NPP_SUCCESS)
	{
		std::cerr << "NPP error '"<< status <<"' during cross-correlation." << std::endl;
		return false;
	}

	dim3 find_peaks_grid = { 1, 1, 1 };
	dim3 find_peaks_block = { 256, 1, 1 };
	kernels::find_peaks_kernel<<<find_peaks_grid, find_peaks_block, 0, ctx.stream>>>(ctx.d_corr_map, valid_corr_dims, row_pitch, d_peak_values, d_peak_positions, total_peaks);

	dim3 test_peaks_grid = { 1, 1, 1 };
	dim3 test_peaks_block = { WARP_SIZE, 1, 1 };
	kernels::test_peaks<<<test_peaks_grid, test_peaks_block, 0, ctx.stream>>>(
							ctx.d_corr_map, d_motion_map, valid_corr_dims, row_pitch, 
							d_peak_positions, d_peak_values, no_shift_index,
							params.min_patch_variance, params.rel_cor_threshold, params.abs_cor_threshold);
	
	return true;
}
