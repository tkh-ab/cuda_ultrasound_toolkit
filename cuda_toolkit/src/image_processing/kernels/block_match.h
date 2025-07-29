#pragma once
#include <npp.h>
#include <span>
#include "../../defs.h"


namespace block_match
{
	static constexpr dim3 Peak_Detect_Block_Dims = { 8, 4, 1 };

	struct PipelineCtx {
		NppStreamContext stream_context;
		cudaStream_t stream;

		u8* d_scratch_buffer = nullptr;
		size_t scratch_buffer_size = 0;

		float* d_corr_map = nullptr;
		size_t corr_map_size = 0;
	};

	__host__ inline int2
	find_peak(const float* d_corr_map, NppiSize dims)
	{
		float max_val = -1.0f;
		int2 max_pos = { 0, 0 };

		for (int y = 0; y < dims.height; ++y)
		{
			for (int x = 0; x < dims.width; ++x)
			{
				float val = abs(sample_value<float>(d_corr_map + y * dims.width + x));
				if (val > max_val)
				{
					max_val = val;
					max_pos = { x, y };
				}
			}
		}

		return max_pos;
	}


	/**
	 * Breaks the image into 8x4 blocks, runs find_local_peaks_kernel for each block, 
	 * each warp gets one block and finds the peak, if the peak is below the threshold it is rejected
	 * 
	 * Then each peak is tested for prominence and sharpness, if it fails it is removed from the list
	 * 
	 * The final list is sorted by peak value and its position is returned.
	 * 
	 * If there are no valid peaks it returns no_shift_pos.
	 */
	__host__ int2
	find_peaks(const float* d_corr_map, NppiSize dims, const NccMotionParameters& params, int line_step, int2 no_shift_pos, u8* d_scratch_buffer, cudaStream_t stream );

	
	__host__ int2
	select_peak(const float* d_corr_map, NppiSize dims, const NccMotionParameters& params, Npp8u* d_scratch_buffer, NppStreamContext stream_context, int2 no_shift_pos, int line_step);

	__host__ bool
	block_match_pipeline(const float* d_source, const float* d_template, const int2* motion_map,
						 NppiSize src_roi, NppiSize tpl_roi,
						 int src_line_step, int tpl_line_step, PipelineCtx& ctx,
						 int2 no_shift_index, const NccMotionParameters& params);

};